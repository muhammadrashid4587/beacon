"""Calibrated smoother noise model (beacon 0.54.0).

The smoother's 95% ellipse is only honest when the filter's
measurement-noise model matches the estimates it is fed: the
filter's 1 m default understated the ~2 m live error, so the
smoothed ellipse covered far fewer than 95% of the true
positions. The smoothing path now defaults to the measured
2 m (``MEASURED_MEASUREMENT_NOISE_M``) and takes it as a
tunable ``measurement_noise_m`` parameter — the standard
deviation in metres, squared into the filter's variance —
through the export, the reports and the tracker methods.
"""

import math
import os
import statistics
import tempfile
import unittest

from beacon import (MEASURED_MEASUREMENT_NOISE_M, Beacon,
                    PositioningEngine, evaluate_smoothed_live_track,
                    evaluate_smoothed_multi_device_track)
from beacon.export import (export_smoothed_track_json, smooth_track_rows,
                           track_rows)
from beacon.filter import KalmanFilter
from beacon.multi_device import MultiDeviceTracker
from beacon.simulate import rectangular_walk, simulate_walk
from beacon.tracking import LiveTracker
from beacon.uncertainty import PositionUncertainty, ellipse_from_covariance

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


def _mean_semi_major(rows, key="smoothed_ellipse_semi_major_m"):
    return statistics.fmean(row[key] for row in rows)


def _inside_count(rows, truth):
    """How many true positions fall inside the smoothed ellipses."""
    inside = 0
    for row, (tx, ty) in zip(rows, truth):
        ellipse = PositionUncertainty(
            row["smoothed_ellipse_semi_major_m"],
            row["smoothed_ellipse_semi_minor_m"],
            row["smoothed_ellipse_angle_deg"])
        inside += ellipse.contains(tx, ty, (row["x"], row["y"]))
    return inside


class MeasuredNoiseConstantTest(unittest.TestCase):
    def test_measured_noise_is_two_metres(self):
        # The measured value the smoothing path defaults to: the
        # live estimates on the demo walk err by about 2 m.
        self.assertEqual(MEASURED_MEASUREMENT_NOISE_M, 2.0)


class KalmanNoiseWidensCovarianceTest(unittest.TestCase):
    def test_wider_noise_widens_smoothed_covariance(self):
        # The same series through the filter with 1 m and 2 m
        # measurement noise: the honest noise model reports a
        # wider posterior, so the ellipse it draws is wider too.
        series = [(1.0, 2.0), (1.5, 2.5), (2.0, 3.0), (2.5, 3.5),
                  (3.0, 4.0), (3.5, 4.5), (4.0, 5.0)]

        def _mean_major(variance):
            kalman = KalmanFilter(measurement_noise=variance)
            for position in series:
                kalman.update(position)
            return statistics.fmean(
                ellipse_from_covariance(cov).semi_major
                for _x, _y, cov in kalman.smooth_with_covariance())

        self.assertGreater(_mean_major(4.0), _mean_major(1.0))


class SmoothingNoiseParameterTest(unittest.TestCase):
    def test_default_is_the_per_step_noise(self):
        # The default smoothing is the per-step one: the
        # smoothed ellipse averages 1.82 m semi-major on the
        # demo walk — a little tighter than the 1.90 m the
        # scalar 2 m gave, since weak-geometry steps are
        # smoothed on a larger noise and strong ones on a
        # smaller one — and not the 1.14 m the 1 m default gave.
        positions, _ = _one_walk_track(seed=7)
        smoothed = smooth_track_rows(track_rows(positions))
        self.assertAlmostEqual(_mean_semi_major(smoothed), 1.82,
                               places=2)

    def test_explicit_one_metre_reproduces_old_ellipse(self):
        # The noise is tunable, and 1.0 m reproduces the old
        # overconfident numbers exactly: 1.14 m semi-major and
        # 20 of 40 true positions inside on seed 7.
        positions, truth = _one_walk_track(seed=7)
        old = smooth_track_rows(track_rows(positions),
                                measurement_noise_m=1.0)
        self.assertAlmostEqual(_mean_semi_major(old), 1.14, places=2)
        self.assertEqual(_inside_count(old, truth), 20)

    def test_default_covers_better_on_every_seed(self):
        # The per-step ellipse covers more true positions than
        # the 1 m one on every seed — the fix is not a seed-7
        # fluke — while staying tighter than the raw ellipse.
        for seed in (1, 2, 3, 7, 11):
            positions, truth = _one_walk_track(seed=seed)
            raw = track_rows(positions)
            old = smooth_track_rows(raw, measurement_noise_m=1.0)
            new = smooth_track_rows(raw)
            self.assertGreater(_inside_count(new, truth),
                               _inside_count(old, truth))
            self.assertLess(_mean_semi_major(new),
                            _mean_semi_major(raw, "ellipse_semi_major_m"))

    def test_non_positive_noise_raises(self):
        positions, _ = _one_walk_track(seed=7)
        rows = track_rows(positions)
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                smooth_track_rows(rows, measurement_noise_m=bad)

    def test_moving_average_ignores_noise(self):
        # The moving-average method has no noise model, so the
        # parameter changes nothing there — same positions, and
        # still no smoothed ellipse.
        positions, _ = _one_walk_track(seed=7)
        rows = track_rows(positions)
        one = smooth_track_rows(rows, method="moving_average",
                                measurement_noise_m=1.0)
        two = smooth_track_rows(rows, method="moving_average",
                                measurement_noise_m=2.0)
        self.assertEqual(one, two)
        self.assertTrue(all(
            row["smoothed_ellipse_semi_major_m"] is None
            for row in one))


class NoisePlumbingTest(unittest.TestCase):
    def test_evaluate_passes_noise_through(self):
        # The report scores the same positions the export writes
        # at the same noise: 1.0 m reproduces the old 1.45 m mean,
        # the default scores the per-step 1.29 m, and the scalar
        # 2.0 m reproduces the v0.54.0/v0.55.0 1.34 m.
        positions, truth = _one_walk_track(seed=7)
        old = evaluate_smoothed_live_track(
            positions, truth, measurement_noise_m=1.0)
        new = evaluate_smoothed_live_track(positions, truth)
        calibrated = evaluate_smoothed_live_track(
            positions, truth, measurement_noise_m=2.0)
        self.assertAlmostEqual(old.mean_error_m, 1.45, places=2)
        self.assertAlmostEqual(new.mean_error_m, 1.29, places=2)
        self.assertAlmostEqual(calibrated.mean_error_m, 1.34,
                               places=2)

    def test_multi_device_evaluate_passes_noise_through(self):
        tracks, truths = _two_walk_tracks()
        old = evaluate_smoothed_multi_device_track(
            tracks, truths, measurement_noise_m=1.0)
        new = evaluate_smoothed_multi_device_track(tracks, truths)
        calibrated = evaluate_smoothed_multi_device_track(
            tracks, truths, measurement_noise_m=2.0)
        self.assertAlmostEqual(old.overall.mean_error_m, 1.25,
                               places=2)
        self.assertAlmostEqual(new.overall.mean_error_m, 1.14,
                               places=2)
        self.assertAlmostEqual(calibrated.overall.mean_error_m, 1.13,
                               places=2)

    def test_smoothed_export_passes_noise_through(self):
        # The JSON export carries the same ellipse at the same
        # noise as smoothing the rows directly.
        positions, _ = _one_walk_track(seed=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            old_path = os.path.join(tmpdir, "old.json")
            export_smoothed_track_json(positions, old_path,
                                       measurement_noise_m=1.0)
            from beacon.export import load_track_json
            old = load_track_json(old_path)
            self.assertAlmostEqual(_mean_semi_major(old), 1.14,
                                   places=2)
            new_path = os.path.join(tmpdir, "new.json")
            export_smoothed_track_json(positions, new_path)
            new = load_track_json(new_path)
            self.assertAlmostEqual(_mean_semi_major(new), 1.82,
                                   places=2)
            self.assertEqual(
                new, smooth_track_rows(track_rows(positions)))

    def test_tracker_export_methods_pass_noise_through(self):
        # The multi-device tracker's smoothed export takes the
        # same noise parameter and agrees with smoothing the
        # rows directly at that noise.
        tracks, truths = _two_walk_tracks()
        from beacon.export import load_track_json, multi_track_rows
        tracker = MultiDeviceTracker(BEACONS)
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
            path = os.path.join(tmpdir, "smoothed.json")
            tracker.export_smoothed_track_json(
                path, measurement_noise_m=1.0)
            loaded = load_track_json(path)
            expected = smooth_track_rows(
                multi_track_rows(tracker.tracks()),
                measurement_noise_m=1.0)
            self.assertEqual(loaded, expected)


if __name__ == "__main__":
    unittest.main()
