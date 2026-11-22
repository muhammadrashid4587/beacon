"""Tests for beacon: trilateration, RSSI model, engine, simulator."""

import math
import random
import statistics
import unittest

from beacon import (
    Beacon,
    MovingAverageFilter,
    PositioningEngine,
    calibrate_beacon,
    calibrate_survey,
    distance_to_rssi,
    fit_path_loss,
    rssi_to_distance,
    simulate_walk,
    trilaterate,
)
from beacon.simulate import rectangular_walk

# Four corner beacons in a 20 x 12 m ward, matching the CLI defaults.
BEACONS = [
    Beacon("A", 0.0, 0.0),
    Beacon("B", 20.0, 0.0),
    Beacon("C", 20.0, 12.0),
    Beacon("D", 0.0, 12.0),
]


def _distances_to(point, beacons):
    return [math.hypot(point[0] - b.x, point[1] - b.y) for b in beacons]


class TestTrilateration(unittest.TestCase):
    def test_recovers_known_position_three_beacons(self):
        point = (7.0, 4.0)
        got = trilaterate(
            [(b.x, b.y) for b in BEACONS[:3]], _distances_to(point, BEACONS[:3]))
        self.assertAlmostEqual(got[0], point[0], places=6)
        self.assertAlmostEqual(got[1], point[1], places=6)

    def test_recovers_known_position_four_beacons(self):
        point = (13.5, 8.25)
        got = trilaterate(
            [(b.x, b.y) for b in BEACONS], _distances_to(point, BEACONS))
        self.assertAlmostEqual(got[0], point[0], places=6)
        self.assertAlmostEqual(got[1], point[1], places=6)

    def test_too_few_beacons_raises(self):
        with self.assertRaises(ValueError):
            trilaterate([(0.0, 0.0), (10.0, 0.0)], [5.0, 5.0])

    def test_collinear_beacons_raise(self):
        with self.assertRaises(ValueError):
            trilaterate([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)],
                        [5.0, 5.0, 15.0])


class TestRssiModel(unittest.TestCase):
    def test_round_trips_distance(self):
        for d in (0.5, 1.0, 2.5, 5.0, 10.0, 18.0):
            rssi = distance_to_rssi(d)
            back = rssi_to_distance(rssi)
            self.assertAlmostEqual(back, d, delta=d * 0.001)

    def test_farther_means_weaker(self):
        self.assertGreater(distance_to_rssi(1.0), distance_to_rssi(10.0))

    def test_nonpositive_distance_raises(self):
        with self.assertRaises(ValueError):
            distance_to_rssi(0.0)


class TestEngine(unittest.TestCase):
    def test_noiseless_estimate_is_exact(self):
        engine = PositioningEngine(BEACONS)
        point = (9.0, 6.0)
        obs = {
            b.id: distance_to_rssi(math.hypot(point[0] - b.x, point[1] - b.y))
            for b in BEACONS
        }
        got = engine.estimate_raw(obs)
        self.assertAlmostEqual(got[0], point[0], places=6)
        self.assertAlmostEqual(got[1], point[1], places=6)

    def test_unknown_beacon_raises(self):
        engine = PositioningEngine(BEACONS)
        with self.assertRaises(ValueError):
            engine.estimate_raw({"ZZ": -70.0})

    def test_too_few_beacons_raises(self):
        with self.assertRaises(ValueError):
            PositioningEngine(BEACONS[:2])


class TestFilter(unittest.TestCase):
    def test_window_one_is_identity(self):
        f = MovingAverageFilter(window=1)
        self.assertEqual(f.update((3.0, 4.0)), (3.0, 4.0))

    def test_averages_last_window(self):
        f = MovingAverageFilter(window=3)
        f.update((0.0, 0.0))
        f.update((3.0, 0.0))
        self.assertEqual(f.update((6.0, 0.0)), (3.0, 0.0))

    def test_reset_forgets_history(self):
        f = MovingAverageFilter(window=5)
        f.update((10.0, 10.0))
        f.reset()
        self.assertEqual(f.update((2.0, 4.0)), (2.0, 4.0))


class TestSimulator(unittest.TestCase):
    def test_simulated_walk_stays_within_error_bound(self):
        # Default ward walk, 2 dB RSSI noise, seed fixed for reproducibility.
        path = rectangular_walk(20.0, 12.0, 10)
        steps = simulate_walk(path, BEACONS, noise_db=2.0, seed=7)
        engine = PositioningEngine(BEACONS)
        errors = []
        for step in steps:
            est = engine.estimate(step["rssi"])
            true = step["true"]
            errors.append(math.hypot(est[0] - true[0], est[1] - true[1]))
        # Measured across seeds 1, 2, 3, 7, 11: mean 2.30-2.90 m.
        # Bound is ~1.5x the worst measured seed, so the test guards
        # against regressions, not against the model itself.
        self.assertLess(statistics.mean(errors), 4.0)

    def test_smoothing_reduces_jitter(self):
        # Stationary device at (10, 6) with noisy RSSI: the filtered
        # estimates must jitter less than the raw ones.
        point = (10.0, 6.0)
        steps = simulate_walk([point] * 60, BEACONS, noise_db=3.0, seed=11)
        engine = PositioningEngine(BEACONS)
        raw, smooth = [], []
        for step in steps:
            raw.append(engine.estimate_raw(step["rssi"]))
            smooth.append(engine.estimate(step["rssi"]))
        raw_jitter = statistics.pstdev([p[0] for p in raw]) + statistics.pstdev(
            [p[1] for p in raw])
        smooth_jitter = statistics.pstdev([p[0] for p in smooth]) + statistics.pstdev(
            [p[1] for p in smooth])
        self.assertLess(smooth_jitter, raw_jitter)

    def test_seed_reproduces_run(self):
        path = rectangular_walk(20.0, 12.0, 5)
        a = simulate_walk(path, BEACONS, seed=42)
        b = simulate_walk(path, BEACONS, seed=42)
        self.assertEqual(a, b)


class TestCalibration(unittest.TestCase):
    def test_fit_recovers_exact_parameters_noiseless(self):
        samples = [(d, distance_to_rssi(d, -65.0, 3.0))
                   for d in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)]
        rssi_at_1m, n = fit_path_loss(samples)
        self.assertAlmostEqual(rssi_at_1m, -65.0, places=9)
        self.assertAlmostEqual(n, 3.0, places=9)

    def test_fit_with_noise_recovers_approximately(self):
        rng = random.Random(3)
        samples = [(d, distance_to_rssi(d, -59.0, 2.0) + rng.gauss(0.0, 1.0))
                   for d in (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 18.0)]
        rssi_at_1m, n = fit_path_loss(samples)
        self.assertAlmostEqual(rssi_at_1m, -59.0, delta=2.0)
        self.assertAlmostEqual(n, 2.0, delta=0.3)

    def test_fit_rejects_too_few_samples(self):
        with self.assertRaises(ValueError):
            fit_path_loss([(1.0, -59.0)])

    def test_fit_rejects_one_distinct_distance(self):
        with self.assertRaises(ValueError):
            fit_path_loss([(2.0, -65.0), (2.0, -66.0), (2.0, -64.0)])

    def test_fit_rejects_non_decreasing_rssi(self):
        with self.assertRaises(ValueError):
            fit_path_loss([(1.0, -70.0), (2.0, -68.0), (4.0, -65.0)])

    def test_fit_rejects_nonpositive_distance(self):
        with self.assertRaises(ValueError):
            fit_path_loss([(0.0, -59.0), (2.0, -65.0)])

    def test_calibrate_beacon_returns_new_beacon(self):
        beacon = Beacon("A", 0.0, 0.0)
        samples = [((d, 0.0), distance_to_rssi(d, -65.0, 3.0))
                   for d in (1.0, 2.0, 4.0, 8.0, 16.0)]
        fitted = calibrate_beacon(beacon, samples)
        self.assertAlmostEqual(fitted.rssi_at_1m, -65.0, places=9)
        self.assertAlmostEqual(fitted.path_loss_exponent, 3.0, places=9)
        # The original beacon is unchanged (frozen dataclass).
        self.assertEqual(beacon.rssi_at_1m, -59.0)
        self.assertEqual(beacon.path_loss_exponent, 2.0)

    def test_calibrate_survey_recovers_true_parameters(self):
        # True ward: beacons at -65 dBm / n=3.0. A noiseless survey
        # walk must recover those constants.
        true_beacons = [Beacon(b.id, b.x, b.y, -65.0, 3.0) for b in BEACONS]
        path = rectangular_walk(20.0, 12.0, 10)
        survey = simulate_walk(path, true_beacons, noise_db=0.0,
                               path_loss_exponent=3.0, seed=7)
        calibrated = calibrate_survey(BEACONS, survey)
        for beacon in calibrated:
            self.assertAlmostEqual(beacon.rssi_at_1m, -65.0, places=6)
            self.assertAlmostEqual(beacon.path_loss_exponent, 3.0, places=6)

    def test_calibrate_survey_leaves_unsurveyed_beacon_unchanged(self):
        true_beacons = [Beacon(b.id, b.x, b.y, -65.0, 3.0) for b in BEACONS]
        path = rectangular_walk(20.0, 12.0, 10)
        survey = simulate_walk(path, true_beacons, noise_db=0.0,
                               path_loss_exponent=3.0, seed=7)
        # Drop every observation of beacon "D" from the survey.
        for step in survey:
            step["rssi"].pop("D", None)
        calibrated = {b.id: b for b in calibrate_survey(BEACONS, survey)}
        self.assertAlmostEqual(calibrated["A"].rssi_at_1m, -65.0, places=6)
        self.assertEqual(calibrated["D"].rssi_at_1m, -59.0)
        self.assertEqual(calibrated["D"].path_loss_exponent, 2.0)

    def test_engine_uses_per_beacon_exponents(self):
        # Beacons with different exponents, noiseless observations
        # generated per beacon: the engine must use each beacon's own
        # exponent to recover the position exactly.
        mixed = [
            Beacon("A", 0.0, 0.0, -59.0, 2.0),
            Beacon("B", 20.0, 0.0, -62.0, 2.5),
            Beacon("C", 20.0, 12.0, -65.0, 3.0),
            Beacon("D", 0.0, 12.0, -60.0, 2.2),
        ]
        engine = PositioningEngine(mixed)
        point = (9.0, 6.0)
        obs = {
            b.id: distance_to_rssi(
                math.hypot(point[0] - b.x, point[1] - b.y),
                b.rssi_at_1m, b.path_loss_exponent)
            for b in mixed
        }
        got = engine.estimate_raw(obs)
        self.assertAlmostEqual(got[0], point[0], places=6)
        self.assertAlmostEqual(got[1], point[1], places=6)

    def test_engine_global_override_still_works(self):
        # Passing path_loss_exponent to the engine overrides every
        # beacon's own exponent (the pre-calibration behaviour).
        mixed = [Beacon(b.id, b.x, b.y, -59.0, 3.0) for b in BEACONS]
        engine = PositioningEngine(mixed, path_loss_exponent=2.0)
        point = (9.0, 6.0)
        obs = {
            b.id: distance_to_rssi(math.hypot(point[0] - b.x, point[1] - b.y))
            for b in BEACONS
        }
        got = engine.estimate_raw(obs)
        self.assertAlmostEqual(got[0], point[0], places=6)
        self.assertAlmostEqual(got[1], point[1], places=6)

    def test_calibration_improves_accuracy(self):
        # A ward whose true constants differ from the defaults:
        # uncalibrated error is large; calibrating from a noiseless
        # survey walk brings it back under the 4.0 m guard.
        true_beacons = [Beacon(b.id, b.x, b.y, -65.0, 3.0) for b in BEACONS]
        path = rectangular_walk(20.0, 12.0, 10)
        survey = simulate_walk(path, true_beacons, noise_db=0.0,
                               path_loss_exponent=3.0, seed=7)
        calibrated = calibrate_survey(BEACONS, survey)
        steps = simulate_walk(path, true_beacons, noise_db=2.0,
                              path_loss_exponent=3.0, seed=7)
        default_engine = PositioningEngine(BEACONS)
        calibrated_engine = PositioningEngine(calibrated)
        default_errors, calibrated_errors = [], []
        for step in steps:
            true = step["true"]
            default_errors.append(math.hypot(
                *(a - b for a, b in zip(
                    default_engine.estimate(step["rssi"]), true))))
            calibrated_errors.append(math.hypot(
                *(a - b for a, b in zip(
                    calibrated_engine.estimate(step["rssi"]), true))))
        self.assertLess(statistics.mean(calibrated_errors), 4.0)
        self.assertLess(statistics.mean(calibrated_errors),
                        statistics.mean(default_errors))


if __name__ == "__main__":
    unittest.main()
