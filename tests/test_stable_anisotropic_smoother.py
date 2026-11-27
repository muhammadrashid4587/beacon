"""Numerically stable anisotropic smoother (beacon 0.57.0).

The v0.56.0 run tried a full anisotropic smoother — each step fed
its whole live-ellipse covariance as the measurement noise — and
abandoned it: 33–37 of 40 smoothed covariances went degenerate per
demo walk, because the Rauch-Tung-Striebel backward subtraction
drives the minor eigenvalue non-positive when the filter is
overconfident along one axis, and the forward pass's short-form
covariance update ``(I - K H) P`` had already lost symmetry and
positive-definiteness in the last bits. This version fixes the
numerics — a Joseph-form covariance update, symmetrised
covariances at every step, and an eigenvalue floor on the smoothed
position block — so the anisotropic smoother runs with zero
degenerate covariances.

The honest limit, measured on the demo walks across seeds
1, 2, 3, 7, 11: the stable anisotropic smoother still does NOT
beat the per-step isotropic noise. Mean error 1.60 m against
1.36 m, coverage 103 of 200 true positions inside the smoothed
95% ellipse against 152 of 200. Blending the anisotropic
covariance halfway toward its own isotropic RMS (0.5) gives
1.38 m and 143 of 200; a quarter blend gives 1.36 m and 150 of
200 — still not beating the isotropic default. The smoother is
fed live-filtered positions, whose error is dominated by the
filter's own dynamics and lag, not by the instantaneous
trilateration geometry — so the raw ellipse's shape misweights
the steps, trusting the "strong" axis too much in a direction
the filtered error does not follow. The default therefore stays
the per-step isotropic noise; the anisotropic path is available
at the filter level, numerically stable, for a space where the
geometry does follow the filtered error.
"""

import math
import statistics
import unittest

from beacon import (MEASURED_MEASUREMENT_NOISE_M, Beacon,
                    PositioningEngine, ellipse_covariance,
                    ellipse_from_covariance)
from beacon.export import track_rows
from beacon.filter import KalmanFilter, _floor_position_block
from beacon.simulate import rectangular_walk, simulate_walk
from beacon.tracking import LiveTracker
from beacon.uncertainty import PositionUncertainty, ellipse_rms_sigma

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


def _smooth_anisotropic(rows):
    """Smooth one row list with each step's full ellipse covariance.

    The filter-level anisotropic path: each step's live confidence
    ellipse becomes the measurement covariance for that update,
    falling back to the measured noise where a step carries no
    ellipse. Returns the smoothed (x, y, ellipse) triples, with
    None for a step whose smoothed covariance was degenerate.
    """
    kalman = KalmanFilter(
        measurement_noise=MEASURED_MEASUREMENT_NOISE_M ** 2)
    for row in rows:
        major = row.get("ellipse_semi_major_m")
        minor = row.get("ellipse_semi_minor_m")
        angle = row.get("ellipse_angle_deg")
        if major is None or minor is None or angle is None:
            kalman.update((row["x"], row["y"]))
        else:
            kalman.update(
                (row["x"], row["y"]),
                measurement_covariance=ellipse_covariance(
                    PositionUncertainty(major, minor, angle)))
    smoothed = []
    for x, y, cov in kalman.smooth_with_covariance():
        try:
            ellipse = ellipse_from_covariance(cov)
        except ValueError:
            ellipse = None
        smoothed.append((x, y, ellipse))
    return smoothed


def _mean_error(smoothed, truth):
    return statistics.fmean(
        math.hypot(x - tx, y - ty)
        for (x, y, _), (tx, ty) in zip(smoothed, truth))


def _inside_count(smoothed, truth):
    inside = degenerate = 0
    for (x, y, ellipse), (tx, ty) in zip(smoothed, truth):
        if ellipse is None:
            degenerate += 1
            continue
        inside += ellipse.contains(tx, ty, (x, y))
    return inside, degenerate


class EllipseCovarianceTest(unittest.TestCase):
    def test_circular_ellipse_is_unit_covariance(self):
        # A circular ellipse with semi-axis sqrt(5.991) has
        # variance 1.0 on each axis.
        cov = ellipse_covariance(
            PositionUncertainty(math.sqrt(5.991), math.sqrt(5.991),
                                0.0))
        self.assertAlmostEqual(cov[0][0], 1.0, places=6)
        self.assertAlmostEqual(cov[1][1], 1.0, places=6)
        self.assertAlmostEqual(cov[0][1], 0.0, places=6)

    def test_round_trip_through_ellipse(self):
        # The covariance behind an ellipse rebuilds the same
        # ellipse: the two functions are exact inverses.
        ellipse = PositionUncertainty(6.32, 2.59, 35.0)
        rebuilt = ellipse_from_covariance(ellipse_covariance(ellipse))
        self.assertAlmostEqual(rebuilt.semi_major,
                               ellipse.semi_major, places=6)
        self.assertAlmostEqual(rebuilt.semi_minor,
                               ellipse.semi_minor, places=6)
        self.assertAlmostEqual(rebuilt.angle_deg, ellipse.angle_deg,
                               places=6)

    def test_rotation_puts_variance_on_the_axes(self):
        # A 90-degree ellipse swaps the variances onto the other
        # diagonal entries.
        cov = ellipse_covariance(PositionUncertainty(6.32, 2.59, 90.0))
        self.assertAlmostEqual(cov[0][0], 2.59 ** 2 / 5.991, places=6)
        self.assertAlmostEqual(cov[1][1], 6.32 ** 2 / 5.991, places=6)
        self.assertAlmostEqual(cov[0][1], 0.0, places=6)

    def test_non_positive_axis_raises(self):
        with self.assertRaises(ValueError):
            ellipse_covariance(PositionUncertainty(0.0, 1.0, 0.0))
        with self.assertRaises(ValueError):
            ellipse_covariance(PositionUncertainty(1.0, -1.0, 0.0))


class MeasurementCovarianceUpdateTest(unittest.TestCase):
    def test_diagonal_covariance_matches_scalar_override(self):
        # A diagonal covariance with equal entries is exactly
        # the scalar override with that variance: same filtered
        # positions and same smoothed covariances.
        positions, _ = _one_walk_track(seed=7)
        series = [(position.x, position.y) for position in positions]
        scalar = KalmanFilter(measurement_noise=4.0)
        for position in series:
            scalar.update(position, measurement_noise=4.0)
        matrix = KalmanFilter(measurement_noise=4.0)
        for position in series:
            matrix.update(position,
                          measurement_covariance=[[4.0, 0.0],
                                                  [0.0, 4.0]])
        self.assertEqual(scalar.smooth(), matrix.smooth())
        self.assertEqual(scalar.smooth_with_covariance(),
                         matrix.smooth_with_covariance())

    def test_anisotropic_covariance_changes_the_track(self):
        # A stretched covariance weights the axes differently,
        # so the smoothed track is not the isotropic one.
        positions, _ = _one_walk_track(seed=7)
        series = [(position.x, position.y) for position in positions]
        isotropic = KalmanFilter(measurement_noise=4.0)
        for position in series:
            isotropic.update(position)
        anisotropic = KalmanFilter(measurement_noise=4.0)
        for position in series:
            anisotropic.update(
                position,
                measurement_covariance=[[1.0, 0.0], [0.0, 9.0]])
        self.assertNotEqual(isotropic.smooth(), anisotropic.smooth())

    def test_non_symmetric_covariance_raises(self):
        kalman = KalmanFilter()
        kalman.update((1.0, 1.0))
        with self.assertRaises(ValueError):
            kalman.update((2.0, 2.0),
                          measurement_covariance=[[1.0, 0.5],
                                                  [0.6, 1.0]])

    def test_non_positive_definite_covariance_raises(self):
        kalman = KalmanFilter()
        kalman.update((1.0, 1.0))
        # Negative determinant: not a covariance at all.
        with self.assertRaises(ValueError):
            kalman.update((2.0, 2.0),
                          measurement_covariance=[[1.0, 2.0],
                                                  [2.0, 1.0]])
        # Zero variance on one axis: degenerate.
        with self.assertRaises(ValueError):
            kalman.update((2.0, 2.0),
                          measurement_covariance=[[1.0, 0.0],
                                                  [0.0, 0.0]])

    def test_noise_and_covariance_together_raise(self):
        kalman = KalmanFilter()
        kalman.update((1.0, 1.0))
        with self.assertRaises(ValueError):
            kalman.update((2.0, 2.0), measurement_noise=4.0,
                          measurement_covariance=[[4.0, 0.0],
                                                  [0.0, 4.0]])


class FloorPositionBlockTest(unittest.TestCase):
    def test_healthy_block_passes_through(self):
        cov = [[2.0, 0.5, 0.0, 0.0],
               [0.5, 1.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]]
        fixed, repaired = _floor_position_block(cov)
        self.assertFalse(repaired)
        self.assertEqual(fixed, cov)

    def test_degenerate_block_is_repaired(self):
        # Minor eigenvalue -1: no ellipse can be drawn from it.
        cov = [[1.0, 2.0, 0.0, 0.0],
               [2.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]]
        fixed, repaired = _floor_position_block(cov)
        self.assertTrue(repaired)
        ellipse = ellipse_from_covariance(
            [[fixed[0][0], fixed[0][1]], [fixed[1][0], fixed[1][1]]])
        self.assertGreater(ellipse.semi_minor, 0.0)
        self.assertAlmostEqual(ellipse.semi_minor,
                               math.sqrt(5.991e-6), places=6)
        # The rest of the state covariance is untouched.
        self.assertEqual(fixed[2], cov[2])
        self.assertEqual(fixed[3], cov[3])


class StableAnisotropicSmootherTest(unittest.TestCase):
    def test_no_degenerate_smoothed_covariance_on_any_seed(self):
        # The v0.56.0 failure, fixed: the short-form update plus
        # the plain backward subtraction degenerated 33–37 of 40
        # smoothed covariances per demo walk. The Joseph form,
        # symmetrisation and eigenvalue floor keep all 40
        # healthy on every seed.
        for seed in (1, 2, 3, 7, 11):
            positions, truth = _one_walk_track(seed=seed)
            smoothed = _smooth_anisotropic(track_rows(positions))
            _, degenerate = _inside_count(smoothed, truth)
            self.assertEqual(degenerate, 0, f"seed {seed}")

    def test_anisotropic_numbers_are_honestly_worse(self):
        # The stable anisotropic smoother measures worse than
        # the per-step isotropic noise it was meant to replace,
        # on both figures that matter — so the default stays
        # isotropic, and this test pins the honest limit so a
        # future change cannot silently claim otherwise.
        per_seed_mean = {}
        total_inside = 0
        for seed, expected_mean in ((1, 1.37), (2, 1.59), (3, 2.05),
                                    (7, 1.85), (11, 1.14)):
            positions, truth = _one_walk_track(seed=seed)
            smoothed = _smooth_anisotropic(track_rows(positions))
            per_seed_mean[seed] = _mean_error(smoothed, truth)
            self.assertAlmostEqual(per_seed_mean[seed], expected_mean,
                                   places=2)
            inside, _ = _inside_count(smoothed, truth)
            total_inside += inside
        self.assertEqual(total_inside, 103)
        # Pooled mean 1.60 m against the isotropic default's
        # 1.36 m, coverage 103 of 200 against 152 of 200.
        self.assertAlmostEqual(
            statistics.fmean(per_seed_mean.values()), 1.60, places=2)
        self.assertLess(total_inside, 152)


if __name__ == "__main__":
    unittest.main()
