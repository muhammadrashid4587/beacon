"""Tests for the smoothed-track export (beacon 0.48.0)."""

import io
import json
import math
import os
import statistics
import tempfile
import unittest

from beacon import (Beacon, LiveTracker, MultiDeviceTracker,
                    PositioningEngine)
from beacon.export import (export_smoothed_multi_track_csv,
                           export_smoothed_multi_track_json,
                           export_smoothed_track_csv,
                           export_smoothed_track_json, load_track_csv,
                           load_track_json, multi_track_rows,
                           smooth_track_rows, smoothed_multi_track_to_dict,
                           smoothed_track_to_dict, track_rows)
from beacon.multi_device import main as multi_device_main
from beacon.simulate import rectangular_walk, simulate_walk
from beacon.tracking import main as tracking_main

# Four corner beacons in a 20 x 12 m ward, matching the CLI defaults.
BEACONS = [
    Beacon("A", 0.0, 0.0),
    Beacon("B", 20.0, 0.0),
    Beacon("C", 20.0, 12.0),
    Beacon("D", 0.0, 12.0),
]


def _one_walk_track(seed=7):
    """The default ward walk, live-tracked, with its ground truth.

    Returns (positions, truth): one LivePosition (or None) per
    step, and one (x, y) per step.
    """
    steps = simulate_walk(rectangular_walk(20.0, 12.0, 10), BEACONS,
                          noise_db=2.0, seed=seed)
    tracker = LiveTracker(PositioningEngine(BEACONS))
    positions = [tracker.feed(step["rssi"], float(i))
                 for i, step in enumerate(steps)]
    return positions, [step["true"] for step in steps]


def _two_walk_tracks():
    """The multi-device demo walks, fed through one tracker.

    Returns (tracks, truths): tag-1 forward, seed 7; tag-2 the
    reverse walk, seed 11.
    """
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


class SmoothedExportTest(unittest.TestCase):
    def test_rts_cuts_mean_error_single_device(self):
        # The seeded demo track, seed 7: the README's raw mean
        # 2.16 m against the smoothed 1.29 m.
        positions, truth = _one_walk_track(seed=7)
        raw = track_rows(positions)
        smoothed = smooth_track_rows(raw)
        self.assertEqual(len(smoothed), len(raw))
        self.assertAlmostEqual(_mean_error(raw, truth), 2.16, places=2)
        self.assertAlmostEqual(_mean_error(smoothed, truth), 1.29,
                               places=2)
        self.assertLess(_mean_error(smoothed, truth),
                        _mean_error(raw, truth))

    def test_rts_cuts_mean_error_every_seed(self):
        # Smoothing helps on every seed, not just the demo one.
        for seed, raw_mean, smoothed_mean in (
                (1, 1.97, 1.15), (2, 2.57, 1.60), (3, 2.63, 1.72),
                (7, 2.16, 1.29), (11, 1.83, 1.04)):
            positions, truth = _one_walk_track(seed=seed)
            raw = track_rows(positions)
            smoothed = smooth_track_rows(raw)
            self.assertAlmostEqual(_mean_error(raw, truth), raw_mean,
                                   places=2)
            self.assertAlmostEqual(_mean_error(smoothed, truth),
                                   smoothed_mean, places=2)
            self.assertLess(_mean_error(smoothed, truth),
                            _mean_error(raw, truth))

    def test_smoothed_rows_keep_everything_but_xy(self):
        positions, _ = _one_walk_track(seed=7)
        raw = track_rows(positions)
        smoothed = smooth_track_rows(raw)
        for raw_row, smoothed_row in zip(raw, smoothed):
            for key, value in raw_row.items():
                if key in ("x", "y"):
                    continue
                self.assertEqual(smoothed_row[key], value,
                                 f"{key} changed by smoothing")
        # The input rows are not modified.
        self.assertNotEqual(raw[5]["x"], smoothed[5]["x"])

    def test_moving_average_opt_out(self):
        # The moving-average method runs and returns the same
        # rows, but on a track that is already live-filtered it
        # adds lag and measures worse than the raw track — which
        # is why RTS is the default, not the moving average.
        positions, truth = _one_walk_track(seed=7)
        raw = track_rows(positions)
        averaged = smooth_track_rows(raw, method="moving_average")
        self.assertEqual(len(averaged), len(raw))
        self.assertAlmostEqual(_mean_error(averaged, truth), 2.93,
                               places=2)
        self.assertGreater(_mean_error(averaged, truth),
                           _mean_error(raw, truth))
        rts = smooth_track_rows(raw)
        self.assertLess(_mean_error(rts, truth),
                        _mean_error(averaged, truth))

    def test_multi_device_smoothed_per_device(self):
        # Each device's smoothed track is scored against its own
        # truth, and each improves — and the smoothing never
        # smears one tag's walk into the other's.
        tracks, truths = _two_walk_tracks()
        raw = multi_track_rows(tracks)
        smoothed = smooth_track_rows(raw)
        self.assertEqual(len(smoothed), len(raw))
        self.assertEqual([row["device"] for row in smoothed],
                         [row["device"] for row in raw])
        for device_id, truth in truths.items():
            device_raw = [row for row in raw
                          if row["device"] == device_id]
            device_smoothed = [row for row in smoothed
                               if row["device"] == device_id]
            self.assertEqual(len(device_smoothed), len(device_raw))
            self.assertLess(_mean_error(device_smoothed, truth),
                            _mean_error(device_raw, truth))
        self.assertAlmostEqual(
            _mean_error([row for row in smoothed
                         if row["device"] == "tag-1"], truths["tag-1"]),
            1.29, places=2)
        self.assertAlmostEqual(
            _mean_error([row for row in smoothed
                         if row["device"] == "tag-2"], truths["tag-2"]),
            0.98, places=2)

    def test_interleaved_rows_smooth_per_device(self):
        # Combined rows need not arrive grouped by device: the
        # smoothing still runs per device, in each device's own
        # order, and the rows come back in their original order.
        tracks, truths = _two_walk_tracks()
        grouped = multi_track_rows(tracks)
        interleaved = [row for pair in zip(
            grouped[:40], grouped[40:]) for row in pair]
        smoothed = smooth_track_rows(interleaved)
        self.assertEqual([row["device"] for row in smoothed],
                         [row["device"] for row in interleaved])
        grouped_smoothed = smooth_track_rows(grouped)
        for device_id in ("tag-1", "tag-2"):
            interleaved_device = [row for row in smoothed
                                  if row["device"] == device_id]
            grouped_device = [row for row in grouped_smoothed
                              if row["device"] == device_id]
            self.assertEqual(interleaved_device, grouped_device)

    def test_floor_change_splits_the_run(self):
        # A series that changes floor is split at the change,
        # the way the live filter resets on a floor change —
        # smoothing never runs across two planes.
        rows = []
        for i in range(6):
            rows.append({"t": float(i), "x": float(i), "y": 1.0,
                         "floor": 0 if i < 3 else 1, "device": None})
        smoothed = smooth_track_rows(rows)
        self.assertEqual(len(smoothed), 6)
        self.assertEqual([row["floor"] for row in smoothed],
                         [0, 0, 0, 1, 1, 1])
        # Each three-step run smooths on its own: smoothing the
        # two runs as separate calls gives exactly the combined
        # result, so no information crosses the floor change.
        first = smooth_track_rows(rows[:3])
        second = smooth_track_rows(rows[3:])
        self.assertEqual(
            [(row["x"], row["y"]) for row in smoothed],
            [(row["x"], row["y"]) for row in first + second])

    def test_single_row_unchanged(self):
        positions, _ = _one_walk_track(seed=7)
        raw = track_rows(positions)[:1]
        smoothed = smooth_track_rows(raw)
        # The single point has nothing to smooth against, so its
        # position and every recorded field are unchanged — only
        # the smoothed ellipse is added: with one step the
        # smoother's posterior is the filter's initial 1 m^2
        # covariance, so the ellipse is a sqrt(5.991) m circle.
        self.assertEqual(len(smoothed), 1)
        self.assertEqual(smoothed[0]["x"], raw[0]["x"])
        self.assertEqual(smoothed[0]["y"], raw[0]["y"])
        for key in raw[0]:
            self.assertEqual(smoothed[0][key], raw[0][key])
        self.assertAlmostEqual(
            smoothed[0]["smoothed_ellipse_semi_major_m"],
            math.sqrt(5.991), places=6)
        self.assertAlmostEqual(
            smoothed[0]["smoothed_ellipse_semi_minor_m"],
            math.sqrt(5.991), places=6)
        self.assertAlmostEqual(
            smoothed[0]["smoothed_ellipse_angle_deg"], 0.0, places=6)

    def test_empty_rows_raise(self):
        with self.assertRaises(ValueError):
            smooth_track_rows([])

    def test_unknown_method_raises(self):
        positions, _ = _one_walk_track(seed=7)
        with self.assertRaises(ValueError):
            smooth_track_rows(track_rows(positions),
                              method="savitzky_golay")

    def test_bad_window_raises(self):
        positions, _ = _one_walk_track(seed=7)
        with self.assertRaises(ValueError):
            smooth_track_rows(track_rows(positions),
                              method="moving_average", window=0)

    def test_smoothed_json_round_trip(self):
        positions, truth = _one_walk_track(seed=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "smoothed.json")
            export_smoothed_track_json(positions, path)
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            self.assertTrue(data["smoothed"])
            self.assertEqual(data["smoothing_method"], "rts")
            self.assertEqual(data["steps"], 40)
            self.assertEqual(data["located"], 40)
            self.assertEqual(data["coverage"], 1.0)
            loaded = load_track_json(path)
            expected = smooth_track_rows(track_rows(positions))
            self.assertEqual(loaded, expected)
            self.assertAlmostEqual(_mean_error(loaded, truth), 1.29,
                                   places=2)

    def test_smoothed_csv_round_trip(self):
        positions, truth = _one_walk_track(seed=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "smoothed.csv")
            export_smoothed_track_csv(positions, path)
            loaded = load_track_csv(path)
            expected = smooth_track_rows(track_rows(positions))
            self.assertEqual(loaded, expected)
            self.assertAlmostEqual(_mean_error(loaded, truth), 1.29,
                                   places=2)

    def test_smoothed_multi_json_round_trip(self):
        tracks, truths = _two_walk_tracks()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "smoothed-multi.json")
            export_smoothed_multi_track_json(tracks, path)
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            self.assertTrue(data["smoothed"])
            self.assertEqual(data["steps"], 80)
            self.assertEqual(data["located"], 80)
            loaded = load_track_json(path)
            expected = smooth_track_rows(multi_track_rows(tracks))
            self.assertEqual(loaded, expected)
            for device_id, truth in truths.items():
                device_rows = [row for row in loaded
                               if row["device"] == device_id]
                self.assertLess(
                    _mean_error(device_rows, truth),
                    _mean_error(
                        [row for row in multi_track_rows(tracks)
                         if row["device"] == device_id], truth))

    def test_smoothed_multi_csv_round_trip(self):
        tracks, _ = _two_walk_tracks()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "smoothed-multi.csv")
            export_smoothed_multi_track_csv(tracks, path)
            loaded = load_track_csv(path)
            self.assertEqual(
                loaded, smooth_track_rows(multi_track_rows(tracks)))

    def test_smoothed_dict_header(self):
        positions, _ = _one_walk_track(seed=7)
        data = smoothed_track_to_dict(positions)
        raw = track_rows(positions)
        self.assertTrue(data["smoothed"])
        self.assertEqual(data["smoothing_method"], "rts")
        self.assertEqual(data["steps"], len(positions))
        self.assertEqual(data["located"], len(raw))
        self.assertEqual(data["coverage"], 1.0)
        self.assertEqual(data["rows"],
                         smooth_track_rows(raw))
        multi = smoothed_multi_track_to_dict(_two_walk_tracks()[0])
        self.assertTrue(multi["smoothed"])
        self.assertEqual(multi["steps"], 80)

    def test_tracker_smoothed_export_methods(self):
        tracker = MultiDeviceTracker(BEACONS)
        # Feed the demo walks directly, as the CLI does.
        forward = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                                BEACONS, noise_db=2.0, seed=7)
        reverse = simulate_walk(
            list(reversed(rectangular_walk(20.0, 12.0, 10))), BEACONS,
            noise_db=2.0, seed=11)
        for i, step in enumerate(forward):
            tracker.feed("tag-1", step["rssi"], float(i))
        for i, step in enumerate(reverse):
            tracker.feed("tag-2", step["rssi"], float(i))
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "smoothed.json")
            csv_path = os.path.join(tmpdir, "smoothed.csv")
            tracker.export_smoothed_track_json(json_path)
            tracker.export_smoothed_track_csv(csv_path)
            self.assertEqual(load_track_json(json_path),
                             load_track_csv(csv_path))
            self.assertEqual(
                load_track_json(json_path),
                smooth_track_rows(multi_track_rows(tracker.tracks())))

    def test_multi_device_cli_smoothed_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "raw.json")
            smoothed_path = os.path.join(tmpdir, "smoothed.json")
            smoothed_csv = os.path.join(tmpdir, "smoothed.csv")
            sink = io.StringIO()
            self.assertEqual(multi_device_main(
                ["--demo", "--export-json", raw_path,
                 "--export-smoothed-json", smoothed_path,
                 "--export-smoothed-csv", smoothed_csv],
                output_stream=sink), 0)
            raw = load_track_json(raw_path)
            smoothed = load_track_json(smoothed_path)
            self.assertEqual(len(raw), 80)
            self.assertEqual(len(smoothed), 80)
            self.assertEqual({row["device"] for row in smoothed},
                             {"tag-1", "tag-2"})
            self.assertEqual(load_track_csv(smoothed_csv), smoothed)
            self.assertNotEqual(
                [(row["x"], row["y"]) for row in smoothed],
                [(row["x"], row["y"]) for row in raw])

    def test_tracking_cli_smoothed_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            smoothed_path = os.path.join(tmpdir, "smoothed.json")
            sink = io.StringIO()
            self.assertEqual(tracking_main(
                ["--demo", "--export-smoothed-json", smoothed_path],
                output_stream=sink), 0)
            loaded = load_track_json(smoothed_path)
            self.assertEqual(len(loaded), 40)
            self.assertAlmostEqual(
                _mean_error(loaded, [step["true"] for step in
                                     simulate_walk(
                                         rectangular_walk(20.0, 12.0, 10),
                                         BEACONS, noise_db=2.0, seed=7)]),
                1.29, places=2)


if __name__ == "__main__":
    unittest.main()
