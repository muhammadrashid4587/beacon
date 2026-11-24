"""Tests for the multi-device track accuracy report CLI (beacon 0.47.0)."""

import io
import json
import os
import tempfile
import unittest

from beacon import (Beacon, MultiDeviceTracker, MultiDeviceTrackReport,
                    TrackReport, evaluate_multi_device_track,
                    format_multi_device_report)
from beacon.export import load_track_json
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


def _two_walk_report():
    """The multi-device demo report, scored from the demo walks.

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
    return evaluate_multi_device_track(tracker.tracks(), truths)


class FormatMultiDeviceReportTest(unittest.TestCase):
    def test_demo_report_lines(self):
        # The README's demo numbers, in the same fields as the
        # single-device demo summary: steps, located, coverage,
        # mean, median, p90 and worst — one line per device plus
        # the pooled overall line.
        text = format_multi_device_report(_two_walk_report())
        lines = text.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("tag-1: steps: 40  located: 40  coverage: 1.00  "
                      "mean error: 2.16 m  median: 2.08 m  "
                      "p90: 4.14 m  worst: 6.70 m", lines[0])
        self.assertIn("tag-2: steps: 40  located: 40  coverage: 1.00  "
                      "mean error: 1.72 m  median: 1.69 m  "
                      "p90: 2.94 m  worst: 3.25 m", lines[1])
        self.assertIn("overall: steps: 80  located: 80  coverage: 1.00  "
                      "mean error: 1.94 m  median: 1.91 m  "
                      "p90: 3.10 m  worst: 6.70 m", lines[2])

    def test_floor_accuracy_appended_when_present(self):
        report = MultiDeviceTrackReport(
            by_device={
                "tag-1": TrackReport(
                    steps=4, located=4, coverage=1.0, mean_error_m=1.0,
                    median_error_m=1.0, p90_error_m=1.0, max_error_m=1.0,
                    floor_accuracy=1.0),
            },
            overall=TrackReport(
                steps=4, located=4, coverage=1.0, mean_error_m=1.0,
                median_error_m=1.0, p90_error_m=1.0, max_error_m=1.0,
                floor_accuracy=1.0))
        text = format_multi_device_report(report)
        self.assertIn("floor accuracy: 1.00", text)
        self.assertEqual(text.count("floor accuracy: 1.00"), 2)

    def test_no_floor_accuracy_without_floors(self):
        self.assertNotIn("floor accuracy",
                         format_multi_device_report(_two_walk_report()))


class EvaluateCliTest(unittest.TestCase):
    def test_demo_evaluate_prints_report(self):
        out = io.StringIO()
        self.assertEqual(
            multi_main(argv=["--demo", "--evaluate"],
                       output_stream=out), 0)
        lines = out.getvalue().splitlines()
        # The stream lines first (80 JSON lines), then a blank
        # line, then the three report lines.
        report_lines = [line for line in lines
                        if line.startswith(("tag-1:", "tag-2:",
                                            "overall:"))]
        self.assertEqual(len(report_lines), 3)
        self.assertIn("mean error: 2.16 m", report_lines[0])
        self.assertIn("mean error: 1.72 m", report_lines[1])
        self.assertIn("mean error: 1.94 m", report_lines[2])
        self.assertIn("steps: 80", report_lines[2])

    def test_demo_evaluate_json_writes_report_dict(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.json")
            self.assertEqual(
                multi_main(argv=["--demo", "--evaluate-json", path],
                           output_stream=out), 0)
            with open(path, encoding="utf-8") as handle:
                written = json.load(handle)
        self.assertEqual(
            set(written), {"by_device", "overall"})
        self.assertAlmostEqual(
            written["by_device"]["tag-1"]["mean_error_m"], 2.16,
            places=2)
        self.assertAlmostEqual(
            written["by_device"]["tag-2"]["mean_error_m"], 1.72,
            places=2)
        self.assertAlmostEqual(written["overall"]["mean_error_m"],
                               1.94, places=2)
        self.assertEqual(written["overall"]["steps"], 80)

    def test_evaluate_and_evaluate_json_together(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.json")
            self.assertEqual(
                multi_main(argv=["--demo", "--evaluate",
                                "--evaluate-json", path],
                           output_stream=out), 0)
            with open(path, encoding="utf-8") as handle:
                written = json.load(handle)
        self.assertIn("overall:", out.getvalue())
        self.assertAlmostEqual(written["overall"]["mean_error_m"],
                               1.94, places=2)

    def test_evaluate_needs_demo(self):
        with self.assertRaises(SystemExit) as caught:
            multi_main(argv=["--evaluate"],
                       output_stream=io.StringIO())
        self.assertEqual(caught.exception.code, 2)

    def test_evaluate_json_needs_demo(self):
        with self.assertRaises(SystemExit) as caught:
            multi_main(argv=["--evaluate-json", "report.json"],
                       output_stream=io.StringIO())
        self.assertEqual(caught.exception.code, 2)

    def test_evaluate_with_drift_beacon_scores_the_drift(self):
        # Beacon A 12 dB too loud on tag-1's walk only: tag-1's
        # error grows while tag-2 stays at its healthy mean.
        out = io.StringIO()
        self.assertEqual(
            multi_main(argv=["--demo", "--evaluate", "--drift",
                            "--drift-beacon", "A:12"],
                       output_stream=out), 0)
        report_lines = [line for line in out.getvalue().splitlines()
                        if line.startswith(("tag-1:", "tag-2:",
                                            "overall:"))]
        self.assertIn("mean error: 1.72 m", report_lines[1])
        self.assertNotIn("mean error: 2.16 m", report_lines[0])
        self.assertNotIn("mean error: 1.94 m", report_lines[2])

    def test_evaluate_composes_with_export(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "both.json")
            self.assertEqual(
                multi_main(argv=["--demo", "--evaluate",
                                "--export-json", json_path],
                           output_stream=out), 0)
            rows = load_track_json(json_path)
        self.assertIn("overall:", out.getvalue())
        self.assertEqual(len(rows), 80)


class RunDemoTruthsTest(unittest.TestCase):
    def test_return_truths(self):
        out = io.StringIO()
        tracker, truths = run_demo(5.0, 10.0, 3, 2.0, out,
                                   return_truths=True)
        self.assertEqual(set(truths), {"tag-1", "tag-2"})
        self.assertEqual(len(truths["tag-1"]), 40)
        self.assertEqual(len(truths["tag-2"]), 40)
        self.assertEqual(set(tracker.tracks()), {"tag-1", "tag-2"})
        report = evaluate_multi_device_track(tracker.tracks(), truths)
        self.assertAlmostEqual(report.overall.mean_error_m, 1.94,
                               places=2)

    def test_default_return_is_tracker_only(self):
        out = io.StringIO()
        tracker = run_demo(5.0, 10.0, 3, 2.0, out)
        self.assertEqual(set(tracker.tracks()), {"tag-1", "tag-2"})


if __name__ == "__main__":
    unittest.main()
