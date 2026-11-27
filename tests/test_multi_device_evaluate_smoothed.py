"""Tests for the smoothed-track accuracy report (beacon 0.50.0)."""

import io
import json
import math
import os
import tempfile
import unittest

from beacon import (Beacon, MultiDeviceTracker,
                    evaluate_multi_device_track,
                    evaluate_smoothed_multi_device_track,
                    format_multi_device_report)
from beacon.export import smoothed_multi_track_to_dict
from beacon.multi_device import main as multi_main
from beacon.multi_device import run_demo
from beacon.simulate import rectangular_walk, simulate_walk

# Four corner beacons in a 20 x 12 m ward, matching the CLI defaults.
BEACONS = [
    Beacon("A", 0.0, 0.0),
    Beacon("B", 20.0, 0.0),
    Beacon("C", 20.0, 12.0),
    Beacon("D", 0.0, 12.0),
]


def _two_walk_tracks_and_truths():
    """The multi-device demo tracks and truths, scored raw and smoothed.

    Tag-1 forward, seed 7; tag-2 the reverse walk, seed 11.
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


class EvaluateSmoothedTrackTest(unittest.TestCase):
    def test_demo_smoothed_report_numbers(self):
        # The README's demo numbers: the smoothed track cuts the
        # overall mean from 1.94 m to 1.14 m, tag-1 from 2.16 m
        # to 1.29 m and tag-2 from 1.72 m to 0.98 m — the same
        # numbers the smoothed export measures.
        tracks, truths = _two_walk_tracks_and_truths()
        report = evaluate_smoothed_multi_device_track(tracks, truths)
        self.assertAlmostEqual(
            report.by_device["tag-1"].mean_error_m, 1.29, places=2)
        self.assertAlmostEqual(
            report.by_device["tag-1"].median_error_m, 1.01, places=2)
        self.assertAlmostEqual(
            report.by_device["tag-1"].p90_error_m, 2.73, places=2)
        self.assertAlmostEqual(
            report.by_device["tag-1"].max_error_m, 3.37, places=2)
        self.assertAlmostEqual(
            report.by_device["tag-2"].mean_error_m, 0.98, places=2)
        self.assertAlmostEqual(
            report.by_device["tag-2"].median_error_m, 0.80, places=2)
        self.assertAlmostEqual(
            report.by_device["tag-2"].p90_error_m, 1.93, places=2)
        self.assertAlmostEqual(
            report.by_device["tag-2"].max_error_m, 2.85, places=2)
        self.assertAlmostEqual(report.overall.mean_error_m, 1.14,
                               places=2)
        self.assertAlmostEqual(report.overall.median_error_m, 0.90,
                               places=2)
        self.assertAlmostEqual(report.overall.p90_error_m, 2.40,
                               places=2)
        self.assertAlmostEqual(report.overall.max_error_m, 3.37,
                               places=2)
        self.assertEqual(report.overall.steps, 80)
        self.assertEqual(report.overall.located, 80)
        self.assertEqual(report.overall.coverage, 1.0)

    def test_smoothed_report_scores_the_exported_rows(self):
        # The report must score exactly the positions the smoothed
        # export writes — the same rows the overlay and the replay
        # draw — so scoring the exported rows directly gives the
        # same means, per device and pooled.
        tracks, truths = _two_walk_tracks_and_truths()
        report = evaluate_smoothed_multi_device_track(tracks, truths)
        data = smoothed_multi_track_to_dict(tracks)
        self.assertTrue(data["smoothed"])
        all_errors = []
        for device_id, truth in truths.items():
            rows = [row for row in data["rows"]
                    if row["device"] == device_id]
            errors = [math.hypot(row["x"] - point[0],
                                 row["y"] - point[1])
                      for row, point in zip(rows, truth)]
            self.assertAlmostEqual(
                sum(errors) / len(errors),
                report.by_device[device_id].mean_error_m, places=2)
            all_errors.extend(errors)
        self.assertAlmostEqual(sum(all_errors) / len(all_errors),
                               report.overall.mean_error_m, places=2)

    def test_raw_and_smoothed_share_the_scorer(self):
        # The same tracks scored raw and smoothed: the step
        # counts, located counts and coverage are identical, only
        # the errors move — the two reports cannot disagree about
        # what was scored.
        tracks, truths = _two_walk_tracks_and_truths()
        raw = evaluate_multi_device_track(tracks, truths)
        smoothed = evaluate_smoothed_multi_device_track(tracks, truths)
        for device_id in ("tag-1", "tag-2"):
            self.assertEqual(smoothed.by_device[device_id].steps,
                             raw.by_device[device_id].steps)
            self.assertEqual(smoothed.by_device[device_id].located,
                             raw.by_device[device_id].located)
            self.assertEqual(smoothed.by_device[device_id].coverage,
                             raw.by_device[device_id].coverage)
            self.assertLess(smoothed.by_device[device_id].mean_error_m,
                            raw.by_device[device_id].mean_error_m)

    def test_moving_average_method_is_worse(self):
        # The moving-average opt-out adds lag on an already
        # live-filtered track, so it scores worse than the raw
        # track — the same verdict the smoothed export gave.
        tracks, truths = _two_walk_tracks_and_truths()
        raw = evaluate_multi_device_track(tracks, truths)
        smoothed = evaluate_smoothed_multi_device_track(
            tracks, truths, method="moving_average")
        self.assertGreater(smoothed.overall.mean_error_m,
                           raw.overall.mean_error_m)
        self.assertAlmostEqual(smoothed.overall.mean_error_m, 2.74,
                               places=2)

    def test_unknown_method_raises(self):
        tracks, truths = _two_walk_tracks_and_truths()
        with self.assertRaises(ValueError):
            evaluate_smoothed_multi_device_track(
                tracks, truths, method="spline")

    def test_nothing_located_raises_like_raw(self):
        tracks, truths = _two_walk_tracks_and_truths()
        empty = {device_id: [None] * len(positions)
                 for device_id, positions in tracks.items()}
        with self.assertRaises(ValueError):
            evaluate_smoothed_multi_device_track(empty, truths)

    def test_unlocated_step_keeps_coverage(self):
        # A step too thin to locate stays unlocated in the
        # smoothed report: the coverage is the raw track's, and
        # the smoothed report never claims a step the tracker
        # never located.
        tracks, truths = _two_walk_tracks_and_truths()
        starved = {device_id: list(positions)
                   for device_id, positions in tracks.items()}
        starved["tag-1"][5] = None
        raw = evaluate_multi_device_track(starved, truths)
        smoothed = evaluate_smoothed_multi_device_track(starved, truths)
        self.assertEqual(raw.by_device["tag-1"].located, 39)
        self.assertEqual(smoothed.by_device["tag-1"].located, 39)
        self.assertAlmostEqual(smoothed.by_device["tag-1"].coverage,
                               39 / 40)
        self.assertEqual(smoothed.overall.located, 79)
        self.assertAlmostEqual(smoothed.overall.coverage, 79 / 80)


class EvaluateSmoothedCliTest(unittest.TestCase):
    def _report_lines(self, output):
        return [line for line in output.splitlines()
                if line.startswith(("tag-1:", "tag-2:", "overall:"))]

    def test_demo_evaluate_and_evaluate_smoothed_print_both(self):
        out = io.StringIO()
        self.assertEqual(
            multi_main(argv=["--demo", "--evaluate",
                            "--evaluate-smoothed"],
                       output_stream=out), 0)
        self.assertIn("Smoothed track (Rauch-Tung-Striebel):",
                      out.getvalue())
        lines = self._report_lines(out.getvalue())
        # Three raw lines, then three smoothed lines.
        self.assertEqual(len(lines), 6)
        self.assertIn("mean error: 2.16 m", lines[0])
        self.assertIn("mean error: 1.94 m", lines[2])
        self.assertIn("mean error: 1.29 m", lines[3])
        self.assertIn("mean error: 0.98 m", lines[4])
        self.assertIn("mean error: 1.14 m", lines[5])

    def test_demo_evaluate_smoothed_alone_prints_only_smoothed(self):
        out = io.StringIO()
        self.assertEqual(
            multi_main(argv=["--demo", "--evaluate-smoothed"],
                       output_stream=out), 0)
        lines = self._report_lines(out.getvalue())
        self.assertEqual(len(lines), 3)
        self.assertIn("mean error: 1.14 m", lines[2])
        self.assertNotIn("mean error: 1.94 m", out.getvalue())

    def test_demo_evaluate_smoothed_json_writes_report_dict(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "smoothed-report.json")
            self.assertEqual(
                multi_main(argv=["--demo", "--evaluate-smoothed-json",
                                path],
                           output_stream=out), 0)
            with open(path, encoding="utf-8") as handle:
                written = json.load(handle)
        self.assertTrue(written["smoothed"])
        self.assertEqual(written["smoothing_method"], "rts")
        self.assertEqual(set(written) - {"smoothed", "smoothing_method"},
                         {"by_device", "overall"})
        self.assertAlmostEqual(
            written["by_device"]["tag-1"]["mean_error_m"], 1.29,
            places=2)
        self.assertAlmostEqual(
            written["by_device"]["tag-2"]["mean_error_m"], 0.98,
            places=2)
        self.assertAlmostEqual(written["overall"]["mean_error_m"],
                               1.14, places=2)
        self.assertEqual(written["overall"]["steps"], 80)

    def test_evaluate_smoothed_and_json_together(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = os.path.join(tmp, "report.json")
            smoothed_path = os.path.join(tmp, "smoothed.json")
            self.assertEqual(
                multi_main(argv=["--demo", "--evaluate",
                                "--evaluate-json", raw_path,
                                "--evaluate-smoothed",
                                "--evaluate-smoothed-json",
                                smoothed_path],
                           output_stream=out), 0)
            with open(raw_path, encoding="utf-8") as handle:
                raw = json.load(handle)
            with open(smoothed_path, encoding="utf-8") as handle:
                smoothed = json.load(handle)
        self.assertIn("Smoothed track (Rauch-Tung-Striebel):",
                      out.getvalue())
        self.assertAlmostEqual(raw["overall"]["mean_error_m"], 1.94,
                               places=2)
        self.assertAlmostEqual(smoothed["overall"]["mean_error_m"],
                               1.14, places=2)

    def test_evaluate_smoothed_needs_demo(self):
        with self.assertRaises(SystemExit) as caught:
            multi_main(argv=["--evaluate-smoothed"],
                       output_stream=io.StringIO())
        self.assertEqual(caught.exception.code, 2)

    def test_evaluate_smoothed_json_needs_demo(self):
        with self.assertRaises(SystemExit) as caught:
            multi_main(argv=["--evaluate-smoothed-json", "report.json"],
                       output_stream=io.StringIO())
        self.assertEqual(caught.exception.code, 2)

    def test_evaluate_smoothed_composes_with_drift(self):
        # Beacon A 12 dB too loud on tag-1's walk only: the
        # smoothed report still shows tag-1 worse than tag-2,
        # and better than its own raw mean.
        out = io.StringIO()
        self.assertEqual(
            multi_main(argv=["--demo", "--evaluate", "--evaluate-smoothed",
                            "--drift", "--drift-beacon", "A:12"],
                       output_stream=out), 0)
        lines = self._report_lines(out.getvalue())
        self.assertEqual(len(lines), 6)
        self.assertIn("mean error: 0.98 m", lines[4])
        self.assertNotIn("mean error: 2.16 m", lines[3])
        self.assertNotIn("mean error: 1.94 m", lines[5])


if __name__ == "__main__":
    unittest.main()
