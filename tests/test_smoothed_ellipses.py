"""Per-frame smoothed uncertainty ellipses on the overlay and replay.

v0.53.0: each smoothed export row carries its own 95% ellipse —
the Rauch-Tung-Striebel smoother's posterior covariance at that
step, not the live estimate's — and the ward-map overlay draws
one dashed ellipse per smoothed point while the replay animates
one per frame, in each device's own colour, with a legend entry.
A moving-average smoothing carries no smoothed ellipse, since a
moving average has no covariance, and draws none.
"""

import csv
import math
import os
import re
import statistics
import tempfile
import unittest

from beacon.beacon import Beacon
from beacon.export import (CSV_FIELDS, SMOOTHED_CSV_FIELDS,
                           export_smoothed_multi_track_csv,
                           export_smoothed_track_csv,
                           export_smoothed_track_json, export_track_csv,
                           export_track_json, load_track_csv,
                           load_track_json, multi_track_rows,
                           smooth_track_rows, track_rows)
from beacon.filter import KalmanFilter
from beacon.map_overlay import (MARGIN, SVG_HEIGHT, SVG_WIDTH, _project,
                                render_map_overlay)
from beacon.multi_device import MultiDeviceTracker
from beacon.replay import render_track_replay
from beacon.simulate import rectangular_walk, simulate_walk
from beacon.tracking import LiveTracker
from beacon.uncertainty import (PositionUncertainty,
                                ellipse_from_covariance)
from beacon import PositioningEngine

# Four corner beacons in a 20 x 12 m ward, matching the CLI defaults.
BEACONS = [
    Beacon("A", 0.0, 0.0),
    Beacon("B", 20.0, 0.0),
    Beacon("C", 20.0, 12.0),
    Beacon("D", 0.0, 12.0),
]


def _one_walk_track(seed=7):
    """The default ward walk, live-tracked, with its ground truth."""
    steps = simulate_walk(rectangular_walk(20.0, 12.0, 10), BEACONS,
                          noise_db=2.0, seed=seed)
    tracker = LiveTracker(PositioningEngine(BEACONS))
    positions = [tracker.feed(step["rssi"], float(i))
                 for i, step in enumerate(steps)]
    return positions, [step["true"] for step in steps]


def _two_walk_tracks():
    """The multi-device demo walks, fed through one tracker."""
    forward = simulate_walk(rectangular_walk(20.0, 12.0, 10), BEACONS,
                            noise_db=2.0, seed=7)
    reverse = simulate_walk(
        list(reversed(rectangular_walk(20.0, 12.0, 10))), BEACONS,
        noise_db=2.0, seed=11)
    tracker = MultiDeviceTracker(BEACONS)
    for i, step in enumerate(forward):
        tracker.feed("tag-1", step["rssi"], float(i))
    for i, step in enumerate(reverse):
        tracker.feed("tag-2", step["rssi"], float(i))
    truths = {
        "tag-1": [step["true"] for step in forward],
        "tag-2": [step["true"] for step in reverse],
    }
    return tracker.tracks(), truths


def _row(x, y, semi_major=1.0, floor=0, t=0.0, device=None):
    """A hand-built row with no smoothed ellipse fields, as a
    smoothed file written before v0.53.0 would load."""
    row = {
        "t": t, "x": x, "y": y, "floor": floor,
        "floor_confidence": 1.0,
        "ellipse_semi_major_m": semi_major,
        "ellipse_semi_minor_m": semi_major / 2.0,
        "ellipse_angle_deg": 30.0,
        "beacons_used": ["A", "B", "C", "D"],
        "beacons_excluded": [], "coverage": 1.0,
        "drift_alarms": [], "drift_classifications": [],
    }
    if device is not None:
        row["device"] = device
    return row


def _unproject(px, py, width_m=20.0, height_m=12.0):
    """SVG pixels back to ward metres, the inverse of _project."""
    scale = min((SVG_WIDTH - 2 * MARGIN) / width_m,
                (SVG_HEIGHT - 2 * MARGIN) / height_m)
    return ((px - MARGIN) / scale, height_m - (py - MARGIN) / scale)


def _mean_semi_major(rows, key="ellipse_semi_major_m"):
    return statistics.fmean(row[key] for row in rows)


def _inside_count(rows, truth, key_major="ellipse_semi_major_m",
                  key_minor="ellipse_semi_minor_m",
                  key_angle="ellipse_angle_deg"):
    """How many true positions fall inside the rows' ellipses."""
    inside = 0
    for row, (tx, ty) in zip(rows, truth):
        ellipse = PositionUncertainty(
            row[key_major], row[key_minor], row[key_angle])
        inside += ellipse.contains(tx, ty, (row["x"], row["y"]))
    return inside


class EllipseFromCovarianceTest(unittest.TestCase):
    def test_isotropic_covariance(self):
        ellipse = ellipse_from_covariance([[1.0, 0.0], [0.0, 1.0]])
        # A unit circle scaled by sqrt(5.991): the 95% radius.
        self.assertAlmostEqual(ellipse.semi_major, math.sqrt(5.991),
                               places=6)
        self.assertAlmostEqual(ellipse.semi_minor, math.sqrt(5.991),
                               places=6)
        self.assertAlmostEqual(ellipse.angle_deg, 0.0, places=6)

    def test_anisotropic_covariance(self):
        ellipse = ellipse_from_covariance([[4.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(ellipse.semi_major, 2 * math.sqrt(5.991),
                               places=6)
        self.assertAlmostEqual(ellipse.semi_minor, math.sqrt(5.991),
                               places=6)
        self.assertAlmostEqual(ellipse.angle_deg, 0.0, places=6)

    def test_rotated_covariance(self):
        # diag(4, 1) rotated by 30 degrees: the major axis points
        # along the rotated x-axis.
        theta = math.radians(30.0)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cov = [[4 * cos_t ** 2 + sin_t ** 2,
                (4 - 1) * cos_t * sin_t],
               [(4 - 1) * cos_t * sin_t,
                4 * sin_t ** 2 + cos_t ** 2]]
        ellipse = ellipse_from_covariance(cov)
        self.assertAlmostEqual(ellipse.semi_major, 2 * math.sqrt(5.991),
                               places=6)
        self.assertAlmostEqual(ellipse.semi_minor, math.sqrt(5.991),
                               places=6)
        self.assertAlmostEqual(ellipse.angle_deg, 30.0, places=6)

    def test_degenerate_covariance_raises(self):
        with self.assertRaises(ValueError):
            ellipse_from_covariance([[0.0, 0.0], [0.0, 0.0]])
        with self.assertRaises(ValueError):
            ellipse_from_covariance([[1.0, 0.0], [0.0, -1.0]])

    def test_non_symmetric_covariance_raises(self):
        with self.assertRaises(ValueError):
            ellipse_from_covariance([[1.0, 0.5], [0.25, 1.0]])


class SmoothWithCovarianceTest(unittest.TestCase):
    def test_positions_match_smooth(self):
        kalman = KalmanFilter()
        series = [(1.0, 2.0), (1.5, 2.5), (2.0, 3.0), (2.5, 3.5),
                  (3.0, 4.0)]
        for position in series:
            kalman.update(position)
        with_cov = kalman.smooth_with_covariance()
        self.assertEqual(len(with_cov), len(series))
        self.assertEqual([(x, y) for x, y, _ in with_cov],
                         kalman.smooth())

    def test_covariances_are_valid(self):
        kalman = KalmanFilter()
        series = [(1.0, 2.0), (1.5, 2.5), (2.0, 3.0), (2.5, 3.5),
                  (3.0, 4.0)]
        for position in series:
            kalman.update(position)
        for _x, _y, cov in kalman.smooth_with_covariance():
            self.assertEqual(cov[0][1], cov[1][0])
            self.assertGreater(cov[0][0], 0.0)
            self.assertGreater(cov[1][1], 0.0)
            # Every smoothed covariance draws a real ellipse.
            ellipse = ellipse_from_covariance(cov)
            self.assertGreater(ellipse.semi_major, 0.0)
            self.assertGreater(ellipse.semi_minor, 0.0)
            self.assertGreaterEqual(ellipse.semi_major,
                                    ellipse.semi_minor)
        # The last smoothed covariance is the last filtered one:
        # nothing comes after it to revise it with.
        last_cov = kalman.smooth_with_covariance()[-1][2]
        filtered = kalman._history[-1][3]
        self.assertAlmostEqual(last_cov[0][0], filtered[0][0],
                               places=9)
        self.assertAlmostEqual(last_cov[1][1], filtered[1][1],
                               places=9)

    def test_empty_filter_returns_empty(self):
        self.assertEqual(KalmanFilter().smooth_with_covariance(), [])


class SmoothedEllipseExportTest(unittest.TestCase):
    def test_rts_rows_carry_smoothed_ellipse(self):
        positions, _ = _one_walk_track(seed=7)
        raw = track_rows(positions)
        smoothed = smooth_track_rows(raw)
        self.assertEqual(len(smoothed), 40)
        for row in smoothed:
            self.assertIsNotNone(row["smoothed_ellipse_semi_major_m"])
            self.assertIsNotNone(row["smoothed_ellipse_semi_minor_m"])
            self.assertIsNotNone(row["smoothed_ellipse_angle_deg"])
            self.assertGreater(row["smoothed_ellipse_semi_major_m"],
                               0.0)
            self.assertGreaterEqual(
                row["smoothed_ellipse_semi_major_m"],
                row["smoothed_ellipse_semi_minor_m"])
            self.assertGreaterEqual(
                row["smoothed_ellipse_angle_deg"], -90.0)
            self.assertLessEqual(
                row["smoothed_ellipse_angle_deg"], 90.0)
        # The smoother's ellipse is much tighter than the live
        # geometry's: 1.90 m against 6.32 m on the demo walk.
        self.assertAlmostEqual(_mean_semi_major(raw), 6.32, places=2)
        self.assertAlmostEqual(
            _mean_semi_major(smoothed, "smoothed_ellipse_semi_major_m"),
            1.90, places=2)
        # And the live ellipse fields are untouched by smoothing.
        self.assertEqual(
            [row["ellipse_semi_major_m"] for row in smoothed],
            [row["ellipse_semi_major_m"] for row in raw])

    def test_moving_average_rows_carry_no_smoothed_ellipse(self):
        positions, _ = _one_walk_track(seed=7)
        smoothed = smooth_track_rows(track_rows(positions),
                                     method="moving_average")
        for row in smoothed:
            self.assertIsNone(row["smoothed_ellipse_semi_major_m"])
            self.assertIsNone(row["smoothed_ellipse_semi_minor_m"])
            self.assertIsNone(row["smoothed_ellipse_angle_deg"])

    def test_json_round_trip_preserves_smoothed_ellipse(self):
        positions, _ = _one_walk_track(seed=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "smoothed.json")
            export_smoothed_track_json(positions, path)
            loaded = load_track_json(path)
            self.assertEqual(loaded,
                             smooth_track_rows(track_rows(positions)))
            # A raw export loads without the keys at all.
            raw_path = os.path.join(tmpdir, "raw.json")
            export_track_json(positions, raw_path)
            raw_loaded = load_track_json(raw_path)
            self.assertEqual(raw_loaded, track_rows(positions))
            self.assertNotIn("smoothed_ellipse_semi_major_m",
                             raw_loaded[0])

    def test_csv_round_trip_preserves_smoothed_ellipse(self):
        positions, _ = _one_walk_track(seed=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "smoothed.csv")
            export_smoothed_track_csv(positions, path)
            with open(path, encoding="utf-8") as handle:
                header = handle.readline().strip().split(",")
            self.assertEqual(header, list(SMOOTHED_CSV_FIELDS))
            loaded = load_track_csv(path)
            self.assertEqual(loaded,
                             smooth_track_rows(track_rows(positions)))
            # The CSV and JSON smoothed exports load to the same
            # rows, ellipse and all.
            json_path = os.path.join(tmpdir, "smoothed.json")
            export_smoothed_track_json(positions, json_path)
            self.assertEqual(loaded, load_track_json(json_path))

    def test_raw_csv_header_unchanged(self):
        positions, _ = _one_walk_track(seed=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "raw.csv")
            export_track_csv(positions, path)
            with open(path, encoding="utf-8") as handle:
                header = handle.readline().strip()
            self.assertEqual(header, ",".join(CSV_FIELDS))
            self.assertEqual(load_track_csv(path),
                             track_rows(positions))

    def test_old_smoothed_csv_loads_without_ellipse(self):
        # A smoothed CSV written before v0.53.0 has the raw
        # header — no smoothed ellipse columns — and loads with
        # no smoothed ellipse, the same way a raw file does.
        positions, _ = _one_walk_track(seed=7)
        smoothed = smooth_track_rows(track_rows(positions))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "old-smoothed.csv")
            from beacon.export import _row_to_csv
            with open(path, "w", encoding="utf-8",
                      newline="") as handle:
                writer = csv.DictWriter(handle,
                                        fieldnames=list(CSV_FIELDS))
                writer.writeheader()
                for row in smoothed:
                    writer.writerow(_row_to_csv(row))
            loaded = load_track_csv(path)
            self.assertNotIn("smoothed_ellipse_semi_major_m",
                             loaded[0])
            self.assertEqual([row["x"] for row in loaded],
                             [row["x"] for row in smoothed])

    def test_multi_device_smoothed_ellipses(self):
        tracks, _ = _two_walk_tracks()
        smoothed = smooth_track_rows(multi_track_rows(tracks))
        for device in ("tag-1", "tag-2"):
            device_rows = [row for row in smoothed
                           if row["device"] == device]
            self.assertEqual(len(device_rows), 40)
            self.assertTrue(all(
                row["smoothed_ellipse_semi_major_m"] is not None
                for row in device_rows))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "smoothed-multi.csv")
            export_smoothed_multi_track_csv(tracks, path)
            self.assertEqual(load_track_csv(path), smoothed)


class SmoothedEllipseMeasuredTest(unittest.TestCase):
    def test_demo_walk_smoothed_ellipse_numbers(self):
        # The smoothed ellipse is the smoother's own posterior,
        # and its 95% claim is only as good as the filter's noise
        # model: the smoother now assumes the measured 2 m
        # measurement noise instead of the filter's 1 m default,
        # so on the demo walk 30 of 40 true positions fall inside
        # the smoothed ellipse, against 33 of 40 inside the raw
        # geometry ellipse — the smoothed ellipse is still
        # tighter than the raw one (1.90 m against 6.32 m
        # semi-major), but it now reads as wide as the track's
        # actual accuracy instead of tighter than it. Passing
        # measurement_noise_m=1.0 reproduces the old overconfident
        # numbers: 20 of 40 inside on seed 7, 1.14 m semi-major.
        positions, truth = _one_walk_track(seed=7)
        raw = track_rows(positions)
        smoothed = smooth_track_rows(raw)
        self.assertEqual(
            _inside_count(smoothed, truth,
                          "smoothed_ellipse_semi_major_m",
                          "smoothed_ellipse_semi_minor_m",
                          "smoothed_ellipse_angle_deg"), 30)
        self.assertEqual(_inside_count(raw, truth), 33)
        old = smooth_track_rows(raw, measurement_noise_m=1.0)
        self.assertEqual(
            _inside_count(old, truth,
                          "smoothed_ellipse_semi_major_m",
                          "smoothed_ellipse_semi_minor_m",
                          "smoothed_ellipse_angle_deg"), 20)
        self.assertAlmostEqual(
            _mean_semi_major(old, "smoothed_ellipse_semi_major_m"),
            1.14, places=2)
        # The filter's noise model is isotropic, so the smoothed
        # ellipse is near-circular on every seed — it carries
        # almost no orientation, unlike the raw geometry ellipse.
        for seed in (1, 2, 3, 7, 11):
            seed_positions, seed_truth = _one_walk_track(seed=seed)
            seed_smoothed = smooth_track_rows(
                track_rows(seed_positions))
            self.assertAlmostEqual(
                _mean_semi_major(seed_smoothed,
                                 "smoothed_ellipse_semi_major_m"),
                1.90, places=2)
            self.assertAlmostEqual(
                _mean_semi_major(seed_smoothed,
                                 "smoothed_ellipse_semi_minor_m"),
                1.90, places=2)
            inside = _inside_count(
                seed_smoothed, seed_truth,
                "smoothed_ellipse_semi_major_m",
                "smoothed_ellipse_semi_minor_m",
                "smoothed_ellipse_angle_deg")
            self.assertGreaterEqual(inside, 21)
            self.assertLessEqual(inside, 37)


class SmoothedEllipseOverlayTest(unittest.TestCase):
    def test_overlay_draws_smoothed_ellipses(self):
        positions, _ = _one_walk_track(seed=7)
        raw = track_rows(positions)
        smoothed = smooth_track_rows(raw)
        svg = render_map_overlay(raw, BEACONS, smoothed_rows=smoothed)
        # One dashed smoothed ellipse per smoothed point, and
        # the raw ellipses are still all there.
        self.assertEqual(svg.count('data-smoothed-ellipse="true"'), 40)
        self.assertEqual(svg.count("<ellipse"), 80)
        self.assertEqual(svg.count('stroke-dasharray="4 2"'), 40)
        self.assertIn("smoothed 95% ellipse (dashed)", svg)

    def test_overlay_smoothed_ellipse_centres_are_the_smoothed_points(self):
        positions, _ = _one_walk_track(seed=7)
        raw = track_rows(positions)
        smoothed = smooth_track_rows(raw)
        svg = render_map_overlay(raw, BEACONS, smoothed_rows=smoothed)
        match = re.search(
            r'<ellipse cx="([\d.]+)" cy="([\d.]+)"[^>]*'
            r'data-smoothed-ellipse="true"', svg)
        self.assertIsNotNone(match, "no smoothed ellipse in the SVG")
        px, py = _unproject(float(match.group(1)),
                            float(match.group(2)))
        self.assertAlmostEqual(px, smoothed[0]["x"], places=1)
        self.assertAlmostEqual(py, smoothed[0]["y"], places=1)
        # And its radii are the smoothed export's ellipse, not
        # the raw one: the smoothed ellipse is far smaller.
        radii = re.search(
            r'<ellipse cx="[\d.]+" cy="[\d.]+" rx="([\d.]+)" '
            r'ry="([\d.]+)"[^>]*data-smoothed-ellipse="true"', svg)
        scale = min((SVG_WIDTH - 2 * MARGIN) / 20.0,
                    (SVG_HEIGHT - 2 * MARGIN) / 12.0)
        self.assertAlmostEqual(
            float(radii.group(1)) / scale,
            smoothed[0]["smoothed_ellipse_semi_major_m"], places=1)

    def test_overlay_no_ellipse_without_smoothed_fields(self):
        # Hand-built smoothed rows, as a pre-v0.53.0 file loads:
        # the dashed trail and dots draw, but no smoothed
        # ellipse and no legend line for one.
        raw = [_row(2.0, 2.0, t=0.0), _row(5.0, 5.0, t=1.0)]
        smoothed = [_row(2.5, 2.5, t=0.0), _row(5.0, 5.0, t=1.0)]
        svg = render_map_overlay(raw, BEACONS, smoothed_rows=smoothed)
        self.assertNotIn('data-smoothed-ellipse="true"', svg)
        self.assertNotIn("smoothed 95% ellipse", svg)
        self.assertEqual(svg.count("<ellipse"), 2)

    def test_overlay_moving_average_draws_no_smoothed_ellipse(self):
        positions, _ = _one_walk_track(seed=7)
        raw = track_rows(positions)
        smoothed = smooth_track_rows(raw, method="moving_average")
        svg = render_map_overlay(raw, BEACONS, smoothed_rows=smoothed)
        self.assertNotIn('data-smoothed-ellipse="true"', svg)
        self.assertNotIn("smoothed 95% ellipse", svg)
        self.assertIn("smoothed track (dashed)", svg)


class SmoothedEllipseReplayTest(unittest.TestCase):
    def test_replay_animates_smoothed_ellipses(self):
        positions, _ = _one_walk_track(seed=7)
        raw = track_rows(positions)
        smoothed = smooth_track_rows(raw)
        svg = render_track_replay(raw, BEACONS, smoothed_rows=smoothed)
        self.assertEqual(svg.count('data-smoothed-ellipse="true"'), 40)
        # Each smoothed ellipse is visible only on its frame,
        # like the raw ellipse — frame 0's carries the visibility
        # animation, and the legend keys it.
        match = re.search(
            r'data-smoothed-ellipse="true" data-frame="0">'
            r'<animate attributeName="visibility"', svg)
        self.assertIsNotNone(match, "frame 0 smoothed ellipse "
                                    "has no visibility animation")
        self.assertIn("smoothed 95% ellipse (dashed)", svg)
        # The raw ellipses are still all there too.
        self.assertEqual(svg.count("<ellipse"), 80)

    def test_replay_single_point_smoothed_ellipse_is_static(self):
        raw = [_row(2.0, 2.0, t=0.0)]
        smoothed = [dict(_row(2.5, 2.5, t=0.0),
                         smoothed_ellipse_semi_major_m=1.0,
                         smoothed_ellipse_semi_minor_m=0.5,
                         smoothed_ellipse_angle_deg=0.0)]
        svg = render_track_replay(raw, BEACONS, smoothed_rows=smoothed)
        self.assertEqual(svg.count('data-smoothed-ellipse="true"'), 1)
        element = svg.split('data-smoothed-ellipse="true"')[1]
        self.assertNotIn("<animate",
                         element.split("</ellipse>")[0])

    def test_replay_moving_average_draws_no_smoothed_ellipse(self):
        positions, _ = _one_walk_track(seed=7)
        raw = track_rows(positions)
        smoothed = smooth_track_rows(raw, method="moving_average")
        svg = render_track_replay(raw, BEACONS, smoothed_rows=smoothed)
        self.assertNotIn('data-smoothed-ellipse="true"', svg)
        self.assertNotIn("smoothed 95% ellipse", svg)
        self.assertIn("smoothed track (dashed)", svg)


if __name__ == "__main__":
    unittest.main()
