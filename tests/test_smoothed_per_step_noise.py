"""Per-step geometry-aware smoother noise (beacon 0.56.0).

One isotropic number cannot capture geometry-dependent error:
the smoother's noise was calibrated to the measured 2 m, but
seeds 2 and 3 still covered only 26 and 21 of 40 inside the
smoothed ellipse, because a step located by beacons clustered
on one side errs more than a step pinned down from all sides.
The smoothing path now defaults to a per-step noise: each step
is fed with the RMS sigma of its own live confidence ellipse
(see :func:`beacon.uncertainty.ellipse_rms_sigma`), so the
filter trusts geometrically strong steps more than weak ones.
A step with no live ellipse falls back to the measured 2 m
for that step, and passing an explicit ``measurement_noise_m``
selects one scalar noise for every step instead — 1.0 m
reproduces the pre-0.54.0 numbers exactly, 2.0 m the
v0.54.0/v0.55.0 calibrated ones. The moving-average opt-out
has no noise model, so it ignores the per-step noise too.
"""

import math
import statistics
import unittest

from beacon import (MEASURED_MEASUREMENT_NOISE_M, Beacon,
                    PositioningEngine, evaluate_smoothed_live_track,
                    evaluate_smoothed_multi_device_track)
from beacon.export import (multi_track_rows, smooth_track_rows,
                           track_rows)
from beacon.filter import KalmanFilter
from beacon.multi_device import MultiDeviceTracker
from beacon.simulate import rectangular_walk, simulate_walk
from beacon.tracking import LiveTracker
from beacon.uncertainty import (PositionUncertainty, ellipse_rms_sigma)

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


def _mean_error(rows, truth):
    return statistics.fmean(
        math.hypot(row["x"] - tx, row["y"] - ty)
        for row, (tx, ty) in zip(rows, truth))


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


def _step_sigmas(rows):
    """Each row's per-step noise, from its live ellipse."""
    return [ellipse_rms_sigma(PositionUncertainty(
                row["ellipse_semi_major_m"],
                row["ellipse_semi_minor_m"],
                row["ellipse_angle_deg"]))
            for row in rows]


class EllipseRmsSigmaTest(unittest.TestCase):
    def test_circular_ellipse_is_its_axis_over_chi2(self):
        # A circular ellipse's RMS sigma is its semi-axis divided
        # by sqrt(5.991): the 95% interval of a 1-D Gaussian with
        # that standard deviation.
        ellipse = PositionUncertainty(2.45, 2.45, 0.0)
        self.assertAlmostEqual(ellipse_rms_sigma(ellipse), 1.0,
                               places=2)

    def test_stretched_ellipse_averages_its_axes(self):
        # The RMS of the two axes' variances: a 6.32 m by 2.59 m
        # ellipse — the demo walk's mean live ellipse — is about
        # 1.97 m, the measured error the scalar noise was
        # calibrated to, so the per-step noise varies the noise
        # per step instead of changing its average.
        ellipse = PositionUncertainty(6.32, 2.59, 0.0)
        self.assertAlmostEqual(ellipse_rms_sigma(ellipse), 1.97,
                               places=2)

    def test_orientation_does_not_change_sigma(self):
        # The RMS is over the axes, so rotating the ellipse
        # changes nothing — the per-step noise is a magnitude,
        # not a direction.
        for angle in (-80.0, -30.0, 0.0, 45.0, 90.0):
            self.assertAlmostEqual(
                ellipse_rms_sigma(PositionUncertainty(6.0, 3.0, angle)),
                ellipse_rms_sigma(PositionUncertainty(6.0, 3.0, 0.0)),
                places=9)

    def test_non_positive_axis_raises(self):
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                ellipse_rms_sigma(PositionUncertainty(bad, 1.0, 0.0))
            with self.assertRaises(ValueError):
                ellipse_rms_sigma(PositionUncertainty(1.0, bad, 0.0))


class KalmanPerStepNoiseTest(unittest.TestCase):
    def test_per_step_override_matches_constructor_noise(self):
        # Feeding every step the constructor's own variance
        # through the override is identical to not overriding:
        # the override only changes the noise when it differs.
        series = [(1.0, 2.0), (1.5, 2.5), (2.0, 3.0), (2.5, 3.5),
                  (3.0, 4.0), (3.5, 4.5), (4.0, 5.0)]
        plain = KalmanFilter(measurement_noise=4.0)
        for position in series:
            plain.update(position)
        stepped = KalmanFilter(measurement_noise=4.0)
        for position in series:
            stepped.update(position, measurement_noise=4.0)
        self.assertEqual(plain.smooth(), stepped.smooth())

    def test_per_step_override_changes_the_track(self):
        # A step fed with a much larger variance is trusted
        # less, so the filter leans on its velocity model there
        # and the smoothed track moves — the override is real,
        # not ignored.
        series = [(1.0, 2.0), (1.5, 2.5), (2.0, 3.0), (2.5, 3.5),
                  (3.0, 4.0), (3.5, 4.5), (4.0, 5.0)]
        plain = KalmanFilter(measurement_noise=4.0)
        for position in series:
            plain.update(position)
        stepped = KalmanFilter(measurement_noise=4.0)
        for i, position in enumerate(series):
            stepped.update(position,
                            measurement_noise=100.0 if i == 3 else 4.0)
        self.assertNotEqual(plain.smooth(), stepped.smooth())

    def test_non_positive_per_step_noise_raises(self):
        kalman = KalmanFilter()
        kalman.update((1.0, 2.0))
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                kalman.update((1.5, 2.5), measurement_noise=bad)


class PerStepNoiseTest(unittest.TestCase):
    def test_step_noise_varies_with_geometry(self):
        # The per-step sigmas on the demo walk are not one
        # number: they run from about 1.2 m on geometrically
        # strong steps to about 3.7 m on weak ones, averaging
        # about 2.0 m — the measured error the scalar noise was
        # calibrated to.
        positions, _ = _one_walk_track(seed=7)
        sigmas = _step_sigmas(track_rows(positions))
        self.assertEqual(len(sigmas), 40)
        self.assertLess(min(sigmas), 1.5)
        self.assertGreater(max(sigmas), 3.0)
        self.assertAlmostEqual(statistics.fmean(sigmas), 2.0,
                               places=1)

    def test_default_improves_mean_error_on_every_seed(self):
        # The per-step noise cuts the smoothed mean error on
        # every seed against the scalar 2 m — the geometry
        # weighting helps where the walk is weak, not just on
        # the demo seed.
        for seed, scalar_mean, per_step_mean in (
                (1, 1.26, 1.15), (2, 1.64, 1.60), (3, 1.92, 1.72),
                (7, 1.34, 1.29), (11, 1.11, 1.04)):
            positions, truth = _one_walk_track(seed=seed)
            rows = track_rows(positions)
            scalar = smooth_track_rows(rows, measurement_noise_m=2.0)
            per_step = smooth_track_rows(rows)
            self.assertAlmostEqual(_mean_error(scalar, truth),
                                   scalar_mean, places=2)
            self.assertAlmostEqual(_mean_error(per_step, truth),
                                   per_step_mean, places=2)
            self.assertLess(_mean_error(per_step, truth),
                            _mean_error(scalar, truth))

    def test_default_covers_more_overall(self):
        # Total coverage across seeds 1, 2, 3, 7, 11 rises from
        # 147 to 152 of 200 — seed 3's 21 of 40 becomes 25 —
        # while seed 2 stays at 26 of 40: the interval is still
        # an approximation, just a per-step one now.
        total_scalar = total_per_step = 0
        per_seed = {}
        for seed in (1, 2, 3, 7, 11):
            positions, truth = _one_walk_track(seed=seed)
            rows = track_rows(positions)
            scalar = _inside_count(
                smooth_track_rows(rows, measurement_noise_m=2.0),
                truth)
            per_step = _inside_count(smooth_track_rows(rows), truth)
            per_seed[seed] = (scalar, per_step)
            total_scalar += scalar
            total_per_step += per_step
        self.assertEqual((total_scalar, total_per_step), (147, 152))
        self.assertEqual(per_seed[2], (26, 26))
        self.assertEqual(per_seed[3], (21, 25))

    def test_explicit_scalar_still_reproduces_old_numbers(self):
        # The scalar path is untouched: 1.0 m reproduces the
        # pre-0.54.0 mean error and ellipse, 2.0 m the
        # v0.54.0/v0.55.0 calibrated ones, on the demo walk.
        positions, truth = _one_walk_track(seed=7)
        rows = track_rows(positions)
        old = smooth_track_rows(rows, measurement_noise_m=1.0)
        calibrated = smooth_track_rows(rows, measurement_noise_m=2.0)
        self.assertAlmostEqual(_mean_error(old, truth), 1.45,
                               places=2)
        self.assertAlmostEqual(_mean_error(calibrated, truth), 1.34,
                               places=2)
        self.assertAlmostEqual(
            statistics.fmean(
                row["smoothed_ellipse_semi_major_m"] for row in old),
            1.14, places=2)
        self.assertAlmostEqual(
            statistics.fmean(
                row["smoothed_ellipse_semi_major_m"]
                for row in calibrated),
            1.90, places=2)

    def test_missing_ellipse_falls_back_to_measured_noise(self):
        # A step with no live ellipse (degenerate geometry)
        # smooths on the measured 2 m for that step instead of
        # failing — and a track with no ellipses at all smooths
        # exactly like the scalar 2 m.
        positions, truth = _one_walk_track(seed=7)
        rows = track_rows(positions)
        starved = [dict(row, ellipse_semi_major_m=None,
                        ellipse_semi_minor_m=None,
                        ellipse_angle_deg=None)
                   for row in rows]
        # The smoothed positions and ellipses match the scalar
        # 2 m exactly — the live ellipse fields differ, since the
        # starved rows carry none, but smoothing leaves those
        # fields untouched, so they are compared separately.
        fallback = smooth_track_rows(starved)
        scalar = smooth_track_rows(rows, measurement_noise_m=2.0)
        for left, right in zip(fallback, scalar):
            self.assertEqual((left["x"], left["y"]),
                             (right["x"], right["y"]))
            self.assertEqual(
                left["smoothed_ellipse_semi_major_m"],
                right["smoothed_ellipse_semi_major_m"])
        partial = [dict(row) for row in rows]
        partial[10] = dict(partial[10], ellipse_semi_major_m=None,
                           ellipse_semi_minor_m=None,
                           ellipse_angle_deg=None)
        smoothed = smooth_track_rows(partial)
        self.assertEqual(len(smoothed), 40)
        self.assertIsNotNone(
            smoothed[10]["smoothed_ellipse_semi_major_m"])

    def test_moving_average_ignores_per_step_noise(self):
        # The moving-average method has no noise model, so the
        # per-step default changes nothing there — same positions
        # as the scalar, and still no smoothed ellipse.
        positions, _ = _one_walk_track(seed=7)
        rows = track_rows(positions)
        per_step = smooth_track_rows(rows, method="moving_average")
        scalar = smooth_track_rows(rows, method="moving_average",
                                   measurement_noise_m=2.0)
        self.assertEqual(per_step, scalar)
        self.assertTrue(all(
            row["smoothed_ellipse_semi_major_m"] is None
            for row in per_step))

    def test_multi_device_overall_is_flat(self):
        # Honest limit: the two-device overall mean is flat
        # (1.13 m to 1.14 m) — tag-1 improves 1.34 m to 1.29 m
        # but the reverse walk gets slightly worse, 0.92 m to
        # 0.98 m — so the per-step noise is not a universal win,
        # it is a better default on the forward walks.
        tracks, truths = _two_walk_tracks()
        report = evaluate_smoothed_multi_device_track(tracks, truths)
        self.assertAlmostEqual(
            report.by_device["tag-1"].mean_error_m, 1.29, places=2)
        self.assertAlmostEqual(
            report.by_device["tag-2"].mean_error_m, 0.98, places=2)
        self.assertAlmostEqual(report.overall.mean_error_m, 1.14,
                               places=2)

    def test_single_device_report_uses_per_step_by_default(self):
        # The report scores the per-step smoothing by default:
        # 1.29 m on the demo walk, not the scalar 1.34 m.
        positions, truth = _one_walk_track(seed=7)
        report = evaluate_smoothed_live_track(positions, truth)
        self.assertAlmostEqual(report.mean_error_m, 1.29, places=2)


if __name__ == "__main__":
    unittest.main()
