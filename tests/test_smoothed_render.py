"""Smoothed track on the ward-map overlay and the replay.

v0.49.0: the smoothed export's rows draw as a dashed series in
each device's own colour on the static overlay and as a hollow
dot gliding over a dashed trail in the replay — the export's own
(x, y), never a second smoothing, so neither render can disagree
with the export. A single-device track with smoothing off renders
byte-identical to before.
"""

import math
import os
import re
import statistics
import tempfile
import unittest

from beacon.beacon import Beacon
from beacon.cli import main as cli_main
from beacon.export import (export_smoothed_multi_track_json,
                           export_smoothed_track_json, load_track_json,
                           multi_track_rows, smooth_track_rows,
                           track_rows)
from beacon.map_overlay import (
    DEVICE_COLOURS,
    MARGIN,
    SVG_HEIGHT,
    SVG_WIDTH,
    TRACK_COLOUR,
    _project,
    _smoothed_series,
    render_map_overlay,
)
from beacon.multi_device import MultiDeviceTracker
from beacon.replay import render_track_replay
from beacon.simulate import rectangular_walk, simulate_walk
from beacon.tracking import LiveTracker
from beacon import PositioningEngine

# Four corner beacons in a 20 x 12 m ward, matching the CLI defaults.
BEACONS = [
    Beacon("A", 0.0, 0.0),
    Beacon("B", 20.0, 0.0),
    Beacon("C", 20.0, 12.0),
    Beacon("D", 0.0, 12.0),
]


def _row(x, y, semi_major=1.0, floor=0, t=0.0, device=None):
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


def _two_devices():
    """Two three-frame tracks on different paths, raw and smoothed.

    The smoothed points sit a little inside the raw ones, as a
    real smoothing would: the smoothed trail is visibly steadier
    without leaving the ward.
    """
    raw = [
        _row(2.0, 2.0, t=0.0, device="tag-1"),
        _row(5.0, 5.0, t=1.0, device="tag-1"),
        _row(8.0, 8.0, t=2.0, device="tag-1"),
        _row(18.0, 10.0, t=0.0, device="tag-2"),
        _row(15.0, 7.0, t=1.0, device="tag-2"),
        _row(12.0, 4.0, t=2.0, device="tag-2"),
    ]
    smoothed = [
        _row(2.5, 2.5, t=0.0, device="tag-1"),
        _row(5.0, 5.0, t=1.0, device="tag-1"),
        _row(7.5, 7.5, t=2.0, device="tag-1"),
        _row(17.5, 9.5, t=0.0, device="tag-2"),
        _row(15.0, 7.0, t=1.0, device="tag-2"),
        _row(12.5, 4.5, t=2.0, device="tag-2"),
    ]
    return raw, smoothed


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


def _mean_error(rows, truth):
    """Mean Euclidean error of row (x, y) against ground truth."""
    return statistics.fmean(
        math.hypot(row["x"] - point[0], row["y"] - point[1])
        for row, point in zip(rows, truth))


def _unproject(px, py, width_m=20.0, height_m=12.0):
    """SVG pixels back to ward metres, the inverse of _project."""
    scale = min((SVG_WIDTH - 2 * MARGIN) / width_m,
                (SVG_HEIGHT - 2 * MARGIN) / height_m)
    return ((px - MARGIN) / scale, height_m - (py - MARGIN) / scale)


def _smoothed_polyline_points(svg):
    """The (x, y) ward metres of the overlay's dashed trail."""
    match = re.search(
        r'<polyline points="([^"]+)"[^>]*data-smoothed="true"', svg)
    assert match is not None, "no smoothed polyline in the SVG"
    points = []
    for pair in match.group(1).split(" "):
        px, py = pair.split(",")
        points.append(_unproject(float(px), float(py)))
    return points


class SmoothedOverlayTest(unittest.TestCase):
    def test_overlay_draws_smoothed_series(self):
        raw, smoothed = _two_devices()
        svg = render_map_overlay(raw, BEACONS, smoothed_rows=smoothed)
        # One dashed trail per device, in that device's own
        # colour, with a hollow dot per smoothed point — and the
        # raw trails, dots and ellipses are still all there.
        self.assertEqual(svg.count('data-smoothed="true"'), 8)
        self.assertEqual(svg.count('stroke-dasharray="6 3"'), 2)
        for device, colour in (("tag-1", DEVICE_COLOURS[0]),
                               ("tag-2", DEVICE_COLOURS[1])):
            self.assertIn(
                f'stroke="{colour}" stroke-width="2" '
                f'stroke-dasharray="6 3" data-smoothed="true" '
                f'data-device="{device}"', svg)
        self.assertEqual(svg.count('data-trail="true"'), 2)
        self.assertEqual(svg.count("<ellipse"), 6)
        self.assertEqual(svg.count('data-index="'), 6)
        # The legend keys each device's smoothed series.
        self.assertIn("tag-1 smoothed track (dashed)", svg)
        self.assertIn("tag-2 smoothed track (dashed)", svg)

    def test_overlay_smoothed_points_are_the_export_points(self):
        # The dashed trail's points, unprojected, are the smoothed
        # export's (x, y) — the overlay draws the export, it does
        # not re-derive it. tag-1's trail is the first dashed
        # polyline in device order.
        raw, smoothed = _two_devices()
        svg = render_map_overlay(raw, BEACONS, smoothed_rows=smoothed)
        points = _smoothed_polyline_points(svg)
        expected = [(row["x"], row["y"]) for row in smoothed
                    if row["device"] == "tag-1"]
        self.assertEqual(len(points), len(expected))
        for (px, py), (ex, ey) in zip(points, expected):
            self.assertAlmostEqual(px, ex, places=1)
            self.assertAlmostEqual(py, ey, places=1)

    def test_overlay_without_smoothing_renders_as_before(self):
        raw, smoothed = _two_devices()
        plain = render_map_overlay(raw, BEACONS)
        # The no-smoothing call is byte-identical to the call
        # with smoothed_rows=None, and carries no smoothed
        # element at all.
        self.assertEqual(
            plain, render_map_overlay(raw, BEACONS, smoothed_rows=None))
        self.assertNotIn("smoothed", plain)
        self.assertNotIn('data-smoothed="true"', plain)
        # Adding the smoothed series changes only the smoothed
        # elements: the raw trails, dots and ellipses are the
        # same count as before.
        with_smoothed = render_map_overlay(
            raw, BEACONS, smoothed_rows=smoothed)
        self.assertEqual(with_smoothed.count('data-trail="true"'),
                         plain.count('data-trail="true"'))
        self.assertEqual(with_smoothed.count('data-index="'),
                         plain.count('data-index="'))
        self.assertEqual(with_smoothed.count("<ellipse"),
                         plain.count("<ellipse"))

    def test_single_device_smoothed_series(self):
        raw = [_row(2.0, 2.0, t=0.0), _row(5.0, 5.0, t=1.0)]
        smoothed = [_row(2.5, 2.5, t=0.0), _row(5.0, 5.0, t=1.0)]
        svg = render_map_overlay(raw, BEACONS, smoothed_rows=smoothed)
        # One dashed trail in the replay's dot colour — distinct
        # from the raw track colour — and no device attributes
        # anywhere, as on a single-device track.
        self.assertEqual(svg.count('stroke-dasharray="6 3"'), 1)
        self.assertIn(
            f'stroke="{DEVICE_COLOURS[0]}" stroke-width="2" '
            f'stroke-dasharray="6 3" data-smoothed="true"', svg)
        self.assertNotIn("data-device=", svg)
        self.assertIn("smoothed track (dashed)", svg)
        # And with smoothing off the single-device render is the
        # one it always was.
        self.assertEqual(render_map_overlay(raw, BEACONS),
                         render_map_overlay(raw, BEACONS,
                                            smoothed_rows=None))
        self.assertIn(f'stroke="{TRACK_COLOUR}"',
                      render_map_overlay(raw, BEACONS))

    def test_smoothed_series_alignment(self):
        raw, smoothed = _two_devices()
        aligned = _smoothed_series(raw, smoothed)
        self.assertEqual([device for device, _ in aligned],
                         ["tag-1", "tag-2"])
        self.assertEqual([len(rows) for _, rows in aligned], [3, 3])
        self.assertIsNone(_smoothed_series(raw, None))

    def test_smoothed_mismatch_raises(self):
        raw, smoothed = _two_devices()
        # A device the raw track never had.
        extra = smoothed + [_row(1.0, 1.0, t=0.0, device="tag-3")]
        with self.assertRaises(ValueError):
            render_map_overlay(raw, BEACONS, smoothed_rows=extra)
        # A device missing from the smoothed rows.
        missing = [row for row in smoothed
                   if row["device"] == "tag-1"]
        with self.assertRaises(ValueError):
            render_map_overlay(raw, BEACONS, smoothed_rows=missing)
        # A different step count for one device.
        short = smoothed[:-1]
        with self.assertRaises(ValueError):
            render_map_overlay(raw, BEACONS, smoothed_rows=short)
        # Different timestamps for one device.
        shifted = [dict(row) for row in smoothed]
        shifted[1]["t"] = 1.5
        with self.assertRaises(ValueError):
            render_map_overlay(raw, BEACONS, smoothed_rows=shifted)
        # No smoothed points on the plotted floor.
        other_floor = [dict(row, floor=1) for row in smoothed]
        with self.assertRaises(ValueError):
            render_map_overlay(raw, BEACONS, smoothed_rows=other_floor)

    def test_rendered_smoothed_track_measures_like_the_export(self):
        # The two-device demo walk: the dashed trails on the
        # overlay, unprojected, score the same as the smoothed
        # export's rows — overall 1.94 m raw to 1.25 m smoothed
        # over the 80 located steps — so the map shows the
        # improvement the export measured, not a different one.
        tracks, truths = _two_walk_tracks()
        raw = multi_track_rows(tracks)
        smoothed = smooth_track_rows(raw)
        svg = render_map_overlay(raw, BEACONS, smoothed_rows=smoothed)
        for device, truth in truths.items():
            match = re.search(
                r'<polyline points="([^"]+)"[^>]*data-smoothed="true" '
                rf'data-device="{device}"', svg)
            self.assertIsNotNone(match, f"no smoothed trail for {device}")
            points = []
            for pair in match.group(1).split(" "):
                px, py = pair.split(",")
                points.append(_unproject(float(px), float(py)))
            device_raw = [row for row in raw
                          if row["device"] == device]
            device_smoothed = [row for row in smoothed
                               if row["device"] == device]
            self.assertEqual(len(points), len(device_smoothed))
            rendered_error = statistics.fmean(
                math.hypot(px - tx, py - ty)
                for (px, py), (tx, ty) in zip(points, truth))
            export_error = _mean_error(device_smoothed, truth)
            raw_error = _mean_error(device_raw, truth)
            self.assertAlmostEqual(rendered_error, export_error,
                                   delta=0.05)
            self.assertLess(rendered_error, raw_error)
        overall_raw = statistics.fmean(
            math.hypot(row["x"] - point[0], row["y"] - point[1])
            for device, truth in truths.items()
            for row, point in zip(
                [row for row in raw if row["device"] == device], truth))
        overall_smoothed = statistics.fmean(
            math.hypot(row["x"] - point[0], row["y"] - point[1])
            for device, truth in truths.items()
            for row, point in zip(
                [row for row in smoothed if row["device"] == device],
                truth))
        self.assertAlmostEqual(overall_raw, 1.94, places=2)
        self.assertAlmostEqual(overall_smoothed, 1.25, places=2)


class SmoothedReplayTest(unittest.TestCase):
    def test_replay_draws_smoothed_dot_and_trail(self):
        raw, smoothed = _two_devices()
        svg = render_track_replay(raw, BEACONS, smoothed_rows=smoothed)
        # One hollow smoothed dot and one dashed smoothed trail
        # per device, in that device's own colour.
        self.assertEqual(svg.count('data-smoothed-dot="true"'), 2)
        self.assertEqual(svg.count('data-smoothed-trail="true"'), 2)
        for device, colour in (("tag-1", DEVICE_COLOURS[0]),
                               ("tag-2", DEVICE_COLOURS[1])):
            self.assertIn(
                f'stroke="{colour}" stroke-width="1.5" '
                f'data-smoothed-dot="true" data-device="{device}"', svg)
            self.assertIn(
                f'stroke="{colour}" stroke-width="2" '
                f'stroke-dasharray="6 3" '
                f'data-smoothed-trail="true" data-device="{device}"',
                svg)
        # The legend keys each device's smoothed series.
        self.assertIn("tag-1 smoothed track (dashed)", svg)
        self.assertIn("tag-2 smoothed track (dashed)", svg)

    def test_replay_smoothed_dot_follows_the_export(self):
        # tag-1's smoothed dot animates through the smoothed
        # export's projected x positions, on the same key times
        # as the raw dot — the two dots stay in step, and the gap
        # between them is the smoothing.
        raw, smoothed = _two_devices()
        svg = render_track_replay(raw, BEACONS, smoothed_rows=smoothed)
        match = re.search(
            r'data-smoothed-dot="true" data-device="tag-1">'
            r'<animate attributeName="cx" values="([^"]+)" '
            r'keyTimes="([^"]+)"', svg)
        self.assertIsNotNone(match, "no smoothed dot animation")
        expected_xs = ";".join(
            f"{_project(row['x'], row['y'], 38.0, 20.0, 12.0)[0]:.1f}"
            for row in smoothed if row["device"] == "tag-1")
        self.assertEqual(match.group(1), expected_xs)
        raw_match = re.search(
            r'data-dot="true" data-device="tag-1">'
            r'<animate attributeName="cx" values="[^"]+" '
            r'keyTimes="([^"]+)"', svg)
        self.assertEqual(match.group(2), raw_match.group(1))
        # The smoothed trail is drawn whole, not animated: it
        # reads as where the smoothed path ends up.
        self.assertNotIn("data-smoothed-trail=\"true\"><animate", svg)

    def test_replay_without_smoothing_renders_as_before(self):
        raw, smoothed = _two_devices()
        plain = render_track_replay(raw, BEACONS)
        self.assertEqual(
            plain, render_track_replay(raw, BEACONS,
                                       smoothed_rows=None))
        self.assertNotIn("smoothed", plain)
        self.assertNotIn('data-smoothed-dot="true"', plain)
        with_smoothed = render_track_replay(
            raw, BEACONS, smoothed_rows=smoothed)
        self.assertEqual(with_smoothed.count('data-dot="true"'),
                         plain.count('data-dot="true"'))
        self.assertEqual(with_smoothed.count("<ellipse"),
                         plain.count("<ellipse"))

    def test_replay_smoothed_mismatch_raises(self):
        raw, smoothed = _two_devices()
        short = smoothed[:-1]
        with self.assertRaises(ValueError):
            render_track_replay(raw, BEACONS, smoothed_rows=short)
        extra = smoothed + [_row(1.0, 1.0, t=0.0, device="tag-3")]
        with self.assertRaises(ValueError):
            render_track_replay(raw, BEACONS, smoothed_rows=extra)

    def test_single_point_smoothed_dot_is_static(self):
        raw = [_row(2.0, 2.0, t=0.0)]
        smoothed = [_row(2.5, 2.5, t=0.0)]
        svg = render_track_replay(raw, BEACONS, smoothed_rows=smoothed)
        self.assertEqual(svg.count('data-smoothed-dot="true"'), 1)
        self.assertNotIn('data-smoothed-trail="true"', svg)
        self.assertNotIn("<animate", svg.split(
            'data-smoothed-dot="true"')[1].split("</circle>")[0])


class SmoothedRenderCliTest(unittest.TestCase):
    def test_cli_map_smoothed_flag(self):
        tracks, _ = _two_walk_tracks()
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "raw.json")
            smoothed_path = os.path.join(tmpdir, "smoothed.json")
            export_smoothed_multi_track_json(tracks, smoothed_path)
            from beacon.export import export_multi_track_json
            export_multi_track_json(tracks, raw_path)
            out = os.path.join(tmpdir, "map.svg")
            self.assertEqual(cli_main(
                ["--map-overlay", raw_path, "--map-smoothed",
                 smoothed_path, "--map-output", out]), 0)
            with open(out, encoding="utf-8") as handle:
                svg = handle.read()
            self.assertIn('data-smoothed="true"', svg)
            self.assertIn("tag-1 smoothed track (dashed)", svg)
            # The same flag on the replay draws the smoothed dot.
            replay_out = os.path.join(tmpdir, "replay.svg")
            self.assertEqual(cli_main(
                ["--map-overlay", raw_path, "--map-smoothed",
                 smoothed_path, "--map-replay",
                 "--map-output", replay_out]), 0)
            with open(replay_out, encoding="utf-8") as handle:
                replay = handle.read()
            self.assertIn('data-smoothed-dot="true"', replay)
            self.assertIn('data-smoothed-trail="true"', replay)

    def test_cli_map_smoothed_from_single_device_exports(self):
        positions, truth = _one_walk_track(seed=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "raw.json")
            smoothed_path = os.path.join(tmpdir, "smoothed.json")
            from beacon.export import export_track_json
            export_track_json(positions, raw_path)
            export_smoothed_track_json(positions, smoothed_path)
            out = os.path.join(tmpdir, "map.svg")
            self.assertEqual(cli_main(
                ["--map-overlay", raw_path, "--map-smoothed",
                 smoothed_path, "--map-output", out]), 0)
            with open(out, encoding="utf-8") as handle:
                svg = handle.read()
            self.assertIn("smoothed track (dashed)", svg)
            # The rendered smoothed trail scores the smoothed
            # mean, not the raw one: 1.45 m against 2.16 m.
            points = _smoothed_polyline_points(svg)
            self.assertEqual(len(points), 40)
            rendered_error = statistics.fmean(
                math.hypot(px - tx, py - ty)
                for (px, py), (tx, ty) in zip(points, truth))
            self.assertAlmostEqual(rendered_error, 1.45, delta=0.05)
            self.assertLess(
                rendered_error,
                _mean_error(track_rows(positions), truth))


if __name__ == "__main__":
    unittest.main()
