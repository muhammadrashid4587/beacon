"""Tests for the smoothed-track accuracy report in the single-device
demo summary (beacon 0.52.0)."""

import contextlib
import io
import math
import unittest

from beacon import (Beacon, PositioningEngine, evaluate_live_track,
                    evaluate_smoothed_live_track, format_report_line)
from beacon.cli import main as cli_main
from beacon.export import smoothed_track_to_dict
from beacon.simulate import rectangular_walk, simulate_walk
from beacon.tracking import LivePosition

# Four corner beacons in a 20 x 12 m ward, matching the CLI defaults.
BEACONS = [
    Beacon("A", 0.0, 0.0),
    Beacon("B", 20.0, 0.0),
    Beacon("C", 20.0, 12.0),
    Beacon("D", 0.0, 12.0),
]


def _demo_track_and_truth(seed=7, filter_name="kalman"):
    """The CLI demo walk as live positions, with its ground truth.

    The same walk ``python -m beacon.cli`` prints: the default
    ward, 2 dB RSSI noise, 40 steps. The engine gives no ellipse
    here, and the walk is single-floor by construction, so the
    positions carry no uncertainty and floor 0 at confidence 1.0
    — exactly what the CLI builds for its summary.
    """
    steps = simulate_walk(rectangular_walk(20.0, 12.0, 10), BEACONS,
                          noise_db=2.0, seed=seed)
    engine = PositioningEngine(BEACONS, filter=filter_name)
    positions = []
    for i, step in enumerate(steps):
        x, y = engine.estimate(step["rssi"])
        positions.append(LivePosition(
            t=float(i), x=x, y=y, floor=0, floor_confidence=1.0,
            uncertainty=None, heard=tuple(sorted(step["rssi"]))))
    return positions, [step["true"] for step in steps]


class EvaluateSmoothedLiveTrackTest(unittest.TestCase):
    def test_demo_smoothed_report_numbers(self):
        # The README's demo numbers: the smoothed track cuts the
        # mean from 2.16 m to 1.45 m — the same numbers tag-1
        # scores in the multi-device smoothed report, since it is
        # the same forward walk, seed 7.
        positions, truth = _demo_track_and_truth()
        report = evaluate_smoothed_live_track(positions, truth)
        self.assertAlmostEqual(report.mean_error_m, 1.45, places=2)
        self.assertAlmostEqual(report.median_error_m, 1.20, places=2)
        self.assertAlmostEqual(report.p90_error_m, 3.10, places=2)
        self.assertAlmostEqual(report.max_error_m, 3.90, places=2)
        self.assertEqual(report.steps, 40)
        self.assertEqual(report.located, 40)
        self.assertEqual(report.coverage, 1.0)
        self.assertIsNone(report.floor_accuracy)

    def test_smoothed_report_scores_the_exported_rows(self):
        # The report must score exactly the positions the smoothed
        # export writes — the same rows the overlay and the replay
        # draw — so scoring the exported rows directly gives the
        # same mean.
        positions, truth = _demo_track_and_truth()
        report = evaluate_smoothed_live_track(positions, truth)
        data = smoothed_track_to_dict(positions)
        self.assertTrue(data["smoothed"])
        errors = [math.hypot(row["x"] - point[0],
                             row["y"] - point[1])
                  for row, point in zip(data["rows"], truth)]
        self.assertAlmostEqual(sum(errors) / len(errors),
                               report.mean_error_m, places=2)

    def test_raw_and_smoothed_share_the_scorer(self):
        # The same track scored raw and smoothed: the step counts,
        # located counts and coverage are identical, only the
        # errors move — the two reports cannot disagree about what
        # was scored.
        positions, truth = _demo_track_and_truth()
        raw = evaluate_live_track(positions, truth)
        smoothed = evaluate_smoothed_live_track(positions, truth)
        self.assertEqual(smoothed.steps, raw.steps)
        self.assertEqual(smoothed.located, raw.located)
        self.assertEqual(smoothed.coverage, raw.coverage)
        self.assertLess(smoothed.mean_error_m, raw.mean_error_m)
        self.assertAlmostEqual(raw.mean_error_m, 2.16, places=2)

    def test_smoothing_helps_on_every_seed(self):
        # Seeds 1, 2, 3, 7, 11: the smoothed mean is below the raw
        # mean on each — the improvement is not a seed-7 fluke.
        for seed in (1, 2, 3, 7, 11):
            positions, truth = _demo_track_and_truth(seed=seed)
            raw = evaluate_live_track(positions, truth)
            smoothed = evaluate_smoothed_live_track(positions, truth)
            self.assertLess(smoothed.mean_error_m, raw.mean_error_m)

    def test_moving_average_method_is_worse(self):
        # The moving-average opt-out adds lag on an already
        # live-filtered track, so it scores worse than the raw
        # track — the same verdict the multi-device report gave.
        positions, truth = _demo_track_and_truth()
        raw = evaluate_live_track(positions, truth)
        smoothed = evaluate_smoothed_live_track(
            positions, truth, method="moving_average")
        self.assertGreater(smoothed.mean_error_m, raw.mean_error_m)
        self.assertAlmostEqual(smoothed.mean_error_m, 2.93, places=2)

    def test_unknown_method_raises(self):
        positions, truth = _demo_track_and_truth()
        with self.assertRaises(ValueError):
            evaluate_smoothed_live_track(positions, truth,
                                         method="spline")

    def test_nothing_located_raises_like_raw(self):
        positions, truth = _demo_track_and_truth()
        empty = [None] * len(positions)
        with self.assertRaises(ValueError):
            evaluate_smoothed_live_track(empty, truth)

    def test_unlocated_step_keeps_coverage(self):
        # A step too thin to locate stays unlocated in the
        # smoothed report: the coverage is the raw track's, and
        # the smoothed report never claims a step the tracker
        # never located.
        positions, truth = _demo_track_and_truth()
        starved = list(positions)
        starved[5] = None
        raw = evaluate_live_track(starved, truth)
        smoothed = evaluate_smoothed_live_track(starved, truth)
        self.assertEqual(raw.located, 39)
        self.assertEqual(smoothed.located, 39)
        self.assertAlmostEqual(smoothed.coverage, 39 / 40)


class SmoothedDemoSummaryCliTest(unittest.TestCase):
    def _run_demo(self, argv):
        # The CLI prints to real stdout (no output_stream hook),
        # so capture it with redirect_stdout.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli_main(argv=argv), 0)
        return out.getvalue()

    def test_demo_prints_raw_and_smoothed_summaries(self):
        output = self._run_demo(["--steps", "10", "--noise", "2.0",
                                 "--seed", "7"])
        # The raw summary line is unchanged.
        self.assertIn("steps: 40  mean error: 2.16 m  p95: 4.71 m  "
                      "max: 6.70 m", output)
        # And the smoothed summary follows it, in the same fields
        # as the multi-device smoothed report.
        self.assertIn("Smoothed track (Rauch-Tung-Striebel):", output)
        self.assertIn(
            "smoothed: steps: 40  located: 40  coverage: 1.00  "
            "mean error: 1.45 m  median: 1.20 m  p90: 3.10 m  "
            "worst: 3.90 m", output)

    def test_demo_smoothed_line_uses_the_shared_formatter(self):
        # The CLI line is the shared formatter's output for the
        # same report, so the demo summary cannot disagree with
        # the library report.
        positions, truth = _demo_track_and_truth()
        report = evaluate_smoothed_live_track(positions, truth)
        output = self._run_demo(["--steps", "10", "--noise", "2.0",
                                 "--seed", "7"])
        self.assertIn(format_report_line("smoothed", report), output)

    def test_demo_with_moving_average_filter_prints_smoothed(self):
        # The moving-average engine still gets a smoothed summary:
        # smoothing its laggy estimates helps a little (2.73 m to
        # 2.60 m) but stays worse than the Kalman default.
        output = self._run_demo(["--steps", "10", "--noise", "2.0",
                                 "--seed", "7",
                                 "--filter", "moving_average"])
        self.assertIn("steps: 40  mean error: 2.73 m", output)
        self.assertIn("Smoothed track (Rauch-Tung-Striebel):", output)
        self.assertIn("mean error: 2.60 m", output)

    def test_demo_smoothed_beats_raw_on_other_seeds(self):
        for seed in (1, 2, 3, 11):
            output = self._run_demo(["--steps", "10", "--noise", "2.0",
                                     "--seed", str(seed)])
            raw_line = [line for line in output.splitlines()
                        if line.startswith("steps:")][0]
            smoothed_line = [line for line in output.splitlines()
                             if line.startswith("smoothed:")][0]
            raw_mean = float(raw_line.split("mean error: ")[1]
                             .split(" m")[0])
            smoothed_mean = float(smoothed_line.split("mean error: ")[1]
                                  .split(" m")[0])
            self.assertLess(smoothed_mean, raw_mean)


if __name__ == "__main__":
    unittest.main()
