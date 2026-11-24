"""Tests for the multi-device track accuracy report (beacon 0.46.0)."""

import math
import unittest

from beacon import (Beacon, LiveTracker, MultiDeviceTracker,
                    PositioningEngine, evaluate_live_track,
                    evaluate_multi_device_track)
from beacon.simulate import rectangular_walk, simulate_walk

# Four corner beacons in a 20 x 12 m ward, matching the CLI defaults.
BEACONS = [
    Beacon("A", 0.0, 0.0),
    Beacon("B", 20.0, 0.0),
    Beacon("C", 20.0, 12.0),
    Beacon("D", 0.0, 12.0),
]

FULL = {"A": -62.1, "B": -71.4, "C": -68.0, "D": -59.8}


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


class MultiDeviceReportTest(unittest.TestCase):
    def test_two_walk_demo_report(self):
        # The README's hand-measured per-device means, now in code:
        # tag-1 forward 2.16 m, tag-2 reverse 1.72 m, pooled 1.94 m.
        tracks, truths = _two_walk_tracks()
        report = evaluate_multi_device_track(tracks, truths)
        self.assertEqual(set(report.by_device), {"tag-1", "tag-2"})
        self.assertAlmostEqual(report.by_device["tag-1"].mean_error_m,
                               2.16, places=2)
        self.assertAlmostEqual(report.by_device["tag-2"].mean_error_m,
                               1.72, places=2)
        self.assertEqual(report.by_device["tag-1"].steps, 40)
        self.assertEqual(report.by_device["tag-2"].steps, 40)
        self.assertEqual(report.by_device["tag-1"].coverage, 1.0)
        self.assertEqual(report.by_device["tag-2"].coverage, 1.0)
        overall = report.overall
        self.assertEqual(overall.steps, 80)
        self.assertEqual(overall.located, 80)
        self.assertEqual(overall.coverage, 1.0)
        self.assertAlmostEqual(overall.mean_error_m, 1.94, places=2)
        self.assertAlmostEqual(overall.median_error_m, 1.91, places=2)
        self.assertAlmostEqual(overall.p90_error_m, 3.10, places=2)
        self.assertAlmostEqual(overall.max_error_m, 6.70, places=2)
        self.assertIsNone(overall.floor_accuracy)

    def test_overall_pools_errors_not_averages(self):
        # The overall p90 and worst step come from the pooled error
        # list, not from averaging the per-device figures: the
        # overall max is the worst device's max, and the overall
        # p90 sits between the two devices' p90s.
        tracks, truths = _two_walk_tracks()
        report = evaluate_multi_device_track(tracks, truths)
        tag1 = report.by_device["tag-1"]
        tag2 = report.by_device["tag-2"]
        self.assertEqual(report.overall.max_error_m,
                         max(tag1.max_error_m, tag2.max_error_m))
        self.assertGreaterEqual(report.overall.p90_error_m,
                                min(tag1.p90_error_m, tag2.p90_error_m))
        self.assertLessEqual(report.overall.p90_error_m,
                             max(tag1.p90_error_m, tag2.p90_error_m))
        # And the pooled mean is the located-weighted mean of the
        # per-device means.
        pooled = ((tag1.mean_error_m * tag1.located
                   + tag2.mean_error_m * tag2.located)
                  / report.overall.located)
        self.assertAlmostEqual(report.overall.mean_error_m, pooled,
                               places=9)

    def test_one_bad_device_does_not_hide_in_the_overall(self):
        # A device that walks badly keeps its own bad report even
        # when the other device is fine, and the overall mean sits
        # between them instead of reporting either one alone.
        tracks, truths = _two_walk_tracks()
        report = evaluate_multi_device_track(tracks, truths)
        tag1 = report.by_device["tag-1"]
        tag2 = report.by_device["tag-2"]
        self.assertGreater(tag1.mean_error_m, tag2.mean_error_m)
        self.assertGreater(report.overall.mean_error_m,
                           tag2.mean_error_m)
        self.assertLess(report.overall.mean_error_m,
                        tag1.mean_error_m)

    def test_thin_device_lowers_its_own_and_overall_coverage(self):
        # A device fed one reading too thin to locate keeps its
        # own coverage below 1.0, and the overall coverage counts
        # that step too — the other device stays at 1.0.
        tracker = MultiDeviceTracker(BEACONS)
        tracker.feed("tag-1", {"A": -62.1}, 0.0)  # too thin: None
        positions = [tracker.feed("tag-1", FULL, float(i))
                     for i in range(1, 6)]
        truths = {
            "tag-1": [(10.0, 6.0)] * 6,
            "tag-2": [(10.0, 6.0)] * 5,
        }
        tracks = {"tag-1": [None] + positions,
                  "tag-2": [tracker.feed("tag-2", FULL, float(i))
                            for i in range(5)]}
        report = evaluate_multi_device_track(tracks, truths)
        self.assertAlmostEqual(report.by_device["tag-1"].coverage,
                               5 / 6)
        self.assertEqual(report.by_device["tag-2"].coverage, 1.0)
        self.assertAlmostEqual(report.overall.coverage, 10 / 11)
        self.assertEqual(report.overall.steps, 11)
        self.assertEqual(report.overall.located, 10)

    def test_to_dict(self):
        tracks, truths = _two_walk_tracks()
        report = evaluate_multi_device_track(tracks, truths)
        as_dict = report.to_dict()
        self.assertEqual(set(as_dict), {"by_device", "overall"})
        self.assertEqual(set(as_dict["by_device"]), {"tag-1", "tag-2"})
        self.assertEqual(as_dict["by_device"]["tag-1"],
                         report.by_device["tag-1"].to_dict())
        self.assertEqual(as_dict["overall"], report.overall.to_dict())

    def test_empty_tracks_raise(self):
        with self.assertRaises(ValueError):
            evaluate_multi_device_track({}, {})

    def test_device_sets_must_match(self):
        tracks, truths = _two_walk_tracks()
        with self.assertRaises(ValueError):
            evaluate_multi_device_track(
                tracks, {"tag-1": truths["tag-1"]})
        with self.assertRaises(ValueError):
            evaluate_multi_device_track(
                {"tag-1": tracks["tag-1"]}, truths)

    def test_per_device_length_mismatch_raises(self):
        tracks, truths = _two_walk_tracks()
        truths["tag-1"] = truths["tag-1"][:-1]
        with self.assertRaises(ValueError):
            evaluate_multi_device_track(tracks, truths)

    def test_device_with_nothing_located_raises(self):
        tracks = {"tag-1": [None, None, None]}
        truths = {"tag-1": [(10.0, 6.0)] * 3}
        with self.assertRaises(ValueError):
            evaluate_multi_device_track(tracks, truths)

    def test_mixed_truth_within_a_device_raises(self):
        tracker = MultiDeviceTracker(BEACONS)
        positions = [tracker.feed("tag-1", FULL, float(i))
                     for i in range(3)]
        tracks = {"tag-1": positions}
        truths = {"tag-1": [(10.0, 6.0), (10.0, 6.0),
                            (10.0, 6.0, 0)]}
        with self.assertRaises(ValueError):
            evaluate_multi_device_track(tracks, truths)

    def test_floor_accuracy_only_when_every_device_has_floors(self):
        # Both devices with floor truth: the overall floor
        # accuracy is the pooled share of located steps on the
        # right floor. One device without floors: the overall
        # figure is None, while that device's own report is None
        # and the other device's keeps its own accuracy.
        tracker = MultiDeviceTracker(BEACONS)
        positions = [tracker.feed("tag-1", FULL, float(i))
                     for i in range(3)]
        tracks = {"tag-1": positions, "tag-2": positions}
        floored = {"tag-1": [(10.0, 6.0, 0)] * 3,
                   "tag-2": [(10.0, 6.0, 0)] * 3}
        report = evaluate_multi_device_track(tracks, floored)
        self.assertEqual(report.overall.floor_accuracy, 1.0)
        self.assertEqual(report.by_device["tag-1"].floor_accuracy, 1.0)
        mixed = {"tag-1": [(10.0, 6.0, 0)] * 3,
                 "tag-2": [(10.0, 6.0)] * 3}
        report = evaluate_multi_device_track(tracks, mixed)
        self.assertIsNone(report.overall.floor_accuracy)
        self.assertEqual(report.by_device["tag-1"].floor_accuracy, 1.0)
        self.assertIsNone(report.by_device["tag-2"].floor_accuracy)

    def test_single_device_report_matches_evaluate_live_track(self):
        # One device through the multi-device scorer is the same
        # report the single-device scorer gives, overall included.
        tracks, truths = _two_walk_tracks()
        single = {"tag-1": tracks["tag-1"]}
        single_truth = {"tag-1": truths["tag-1"]}
        report = evaluate_multi_device_track(single, single_truth)
        self.assertEqual(report.by_device["tag-1"],
                         evaluate_live_track(tracks["tag-1"],
                                             truths["tag-1"]))
        self.assertEqual(report.overall,
                         evaluate_live_track(tracks["tag-1"],
                                             truths["tag-1"]))

    def test_single_device_track_still_works_after_refactor(self):
        # The refactor of evaluate_live_track must not have
        # disturbed the live track path itself.
        tracker = LiveTracker(PositioningEngine(BEACONS))
        live = tracker.feed(FULL, 0.0)
        self.assertIsNotNone(live)
        self.assertTrue(math.isfinite(live.x) and math.isfinite(live.y))


if __name__ == "__main__":
    unittest.main()
