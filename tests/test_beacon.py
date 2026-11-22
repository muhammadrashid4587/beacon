"""Tests for beacon: trilateration, RSSI model, engine, simulator."""

import math
import random
import statistics
import unittest

from beacon import (
    Beacon,
    FingerprintMap,
    KalmanFilter,
    MovingAverageFilter,
    PositioningEngine,
    build_fingerprint_map,
    calibrate_beacon,
    calibrate_survey,
    distance_to_rssi,
    fit_path_loss,
    grid_points,
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

    def test_multistart_avoids_diverged_refinement(self):
        # Noisy distances (3 dB RSSI noise) whose linear solution is
        # so far off that a single Gauss-Newton run from it diverges
        # to (655, -3843), ~3900 m from the true (21.06, 0.96).
        # The multi-start refinement must stay in the ward.
        got = trilaterate([(b.x, b.y) for b in BEACONS],
                          [17.81, 1.21, 8.6, 51.42])
        self.assertLess(math.hypot(got[0] - 21.06, got[1] - 0.96), 15.0)

    def test_multistart_picks_lowest_residual(self):
        # On noisy distances, the returned point must have the lowest
        # distance residual of any single-seed refinement.
        from beacon.trilateration import _gauss_newton, _residual_sum
        rng = random.Random(17)
        true = (12.0, 5.0)
        distances = []
        for b in BEACONS:
            d = math.hypot(true[0] - b.x, true[1] - b.y)
            distances.append(
                rssi_to_distance(distance_to_rssi(d) + rng.gauss(0.0, 3.0)))
        beacons = [(b.x, b.y) for b in BEACONS]
        got = trilaterate(beacons, distances)
        got_residual = _residual_sum(beacons, distances, got)
        seeds = [(10.0, 6.0),
                 (sum(x for x, _ in beacons) / len(beacons),
                  sum(y for _, y in beacons) / len(beacons))]
        seeds.extend(beacons)
        for seed in seeds:
            candidate = _gauss_newton(beacons, distances, seed)
            self.assertLessEqual(
                got_residual, _residual_sum(beacons, distances, candidate))


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


class TestKalmanFilter(unittest.TestCase):
    def test_first_update_returns_measurement(self):
        f = KalmanFilter()
        self.assertEqual(f.update((3.0, 4.0)), (3.0, 4.0))

    def test_stationary_device_stays_put(self):
        f = KalmanFilter()
        for _ in range(20):
            got = f.update((10.0, 6.0))
        self.assertAlmostEqual(got[0], 10.0, places=9)
        self.assertAlmostEqual(got[1], 6.0, places=9)

    def test_tracks_constant_velocity(self):
        # Device moving 0.5 m/step along x, noisy measurements:
        # the filter must converge onto the true position.
        rng = random.Random(5)
        f = KalmanFilter()
        errors = []
        for step in range(1, 61):
            true = (0.5 * step, 6.0)
            got = f.update((true[0] + rng.gauss(0.0, 1.0),
                            true[1] + rng.gauss(0.0, 1.0)))
            if step > 40:
                errors.append(math.hypot(got[0] - true[0],
                                         got[1] - true[1]))
        # Converged (last 20 steps): mean error below the raw
        # measurements' ~1.25 m (1 m noise per axis, Euclidean).
        self.assertLess(statistics.mean(errors), 1.0)
        vx, vy = f.velocity
        self.assertAlmostEqual(vx, 0.5, delta=0.2)
        self.assertAlmostEqual(vy, 0.0, delta=0.2)

    def test_less_lag_than_moving_average_on_a_walk(self):
        # Constant-velocity walk, noiseless measurements: the Kalman
        # filter's steady-state lag must be smaller than the moving
        # average's (window 5 lags ~2 steps behind a moving device).
        ma = MovingAverageFilter(window=5)
        kf = KalmanFilter()
        ma_err = kf_err = 0.0
        for step in range(1, 41):
            true = (0.5 * step, 6.0)
            ma_err = abs(ma.update(true)[0] - true[0])
            kf_err = abs(kf.update(true)[0] - true[0])
        self.assertLess(kf_err, ma_err)

    def test_reduces_jitter_like_moving_average(self):
        # Stationary device at (10, 6) with noisy measurements:
        # filtered positions must jitter less than the raw ones.
        rng = random.Random(9)
        f = KalmanFilter()
        raw, smooth = [], []
        for _ in range(60):
            meas = (10.0 + rng.gauss(0.0, 1.0), 6.0 + rng.gauss(0.0, 1.0))
            raw.append(meas)
            smooth.append(f.update(meas))
        raw_jitter = statistics.pstdev([p[0] for p in raw]) + statistics.pstdev(
            [p[1] for p in raw])
        smooth_jitter = statistics.pstdev([p[0] for p in smooth]) + statistics.pstdev(
            [p[1] for p in smooth])
        self.assertLess(smooth_jitter, raw_jitter)

    def test_reset_forgets_state(self):
        f = KalmanFilter()
        for _ in range(10):
            f.update((10.0, 6.0))
        f.reset()
        self.assertEqual(f.update((2.0, 4.0)), (2.0, 4.0))
        self.assertEqual(f.velocity, (0.0, 0.0))

    def test_nonpositive_noise_raises(self):
        with self.assertRaises(ValueError):
            KalmanFilter(process_noise=0.0)
        with self.assertRaises(ValueError):
            KalmanFilter(measurement_noise=-1.0)


class TestEngineKalmanFilter(unittest.TestCase):
    def test_default_filter_is_kalman(self):
        # The Kalman filter is the default since 0.5.0; the moving
        # average is only an opt-out.
        engine = PositioningEngine(BEACONS)
        self.assertEqual(engine.filter, "kalman")
        self.assertIsInstance(engine._filter, KalmanFilter)

    def test_default_engine_accuracy_across_seeds(self):
        # Regression guard for the new default: on the existing walk
        # fixtures the default (Kalman) engine must stay under the
        # 4.0 m mean-error bound on every seed.
        path = rectangular_walk(20.0, 12.0, 10)
        for seed in (1, 2, 3, 7, 11):
            steps = simulate_walk(path, BEACONS, noise_db=2.0, seed=seed)
            engine = PositioningEngine(BEACONS)
            errors = []
            for step in steps:
                est = engine.estimate(step["rssi"])
                true = step["true"]
                errors.append(math.hypot(est[0] - true[0],
                                         est[1] - true[1]))
            self.assertLess(statistics.mean(errors), 4.0,
                            f"default engine mean error too high on seed {seed}")

    def test_moving_average_opt_out_still_available(self):
        # The old default is still reachable as an explicit opt-out.
        engine = PositioningEngine(BEACONS, filter="moving_average")
        self.assertEqual(engine.filter, "moving_average")
        self.assertIsInstance(engine._filter, MovingAverageFilter)
        path = rectangular_walk(20.0, 12.0, 10)
        steps = simulate_walk(path, BEACONS, noise_db=2.0, seed=7)
        errors = []
        for step in steps:
            est = engine.estimate(step["rssi"])
            true = step["true"]
            errors.append(math.hypot(est[0] - true[0], est[1] - true[1]))
        self.assertLess(statistics.mean(errors), 4.0)

    def test_engine_kalman_option_tracks_walk(self):
        path = rectangular_walk(20.0, 12.0, 10)
        steps = simulate_walk(path, BEACONS, noise_db=2.0, seed=7)
        engine = PositioningEngine(BEACONS, filter="kalman")
        errors = []
        for step in steps:
            est = engine.estimate(step["rssi"])
            true = step["true"]
            errors.append(math.hypot(est[0] - true[0], est[1] - true[1]))
        self.assertLess(statistics.mean(errors), 4.0)

    def test_engine_unknown_filter_raises(self):
        with self.assertRaises(ValueError):
            PositioningEngine(BEACONS, filter="particle")

    def test_kalman_beats_moving_average_across_seeds(self):
        # Multi-seed accuracy comparison (2 dB RSSI noise, default
        # ward walk): the Kalman filter must beat the moving average
        # on every seed, not just seed 7.
        path = rectangular_walk(20.0, 12.0, 10)
        for seed in (1, 2, 3, 7, 11):
            steps = simulate_walk(path, BEACONS, noise_db=2.0, seed=seed)
            ma = PositioningEngine(BEACONS, filter="moving_average")
            kf = PositioningEngine(BEACONS, filter="kalman")
            ma_err, kf_err = [], []
            for step in steps:
                true = step["true"]
                ma_err.append(math.hypot(
                    *(a - b for a, b in zip(ma.estimate(step["rssi"]), true))))
                kf_err.append(math.hypot(
                    *(a - b for a, b in zip(kf.estimate(step["rssi"]), true))))
            self.assertLess(statistics.mean(kf_err),
                            statistics.mean(ma_err),
                            f"Kalman did not beat moving average on seed {seed}")


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
        # Measured across seeds 1, 2, 3, 7, 11 with the Kalman
        # default: mean 1.83-2.63 m. Bound is ~1.5x the worst
        # measured seed, so the test guards against regressions,
        # not against the model itself.
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


class TestFingerprinting(unittest.TestCase):
    def _noiseless_map(self, spacing=2.0):
        cells = grid_points(20.0, 12.0, spacing)
        survey = simulate_walk(cells, BEACONS, noise_db=0.0, seed=7)
        return build_fingerprint_map(BEACONS, survey)

    def test_grid_points_cover_ward(self):
        cells = grid_points(20.0, 12.0, 2.0)
        self.assertEqual(len(cells), 60)  # 10 x 6 at 2 m spacing, 1 m margin
        self.assertIn((1.0, 1.0), cells)
        self.assertIn((19.0, 11.0), cells)
        # No cell sits on a corner beacon (distance zero is undefined).
        for cell in cells:
            for beacon in BEACONS:
                self.assertGreater(math.hypot(cell[0] - beacon.x,
                                              cell[1] - beacon.y), 0.0)

    def test_grid_points_reject_bad_arguments(self):
        with self.assertRaises(ValueError):
            grid_points(0.0, 12.0, 2.0)
        with self.assertRaises(ValueError):
            grid_points(20.0, 12.0, 0.0)
        with self.assertRaises(ValueError):
            grid_points(20.0, 12.0, 2.0, margin=6.0)

    def test_noiseless_locate_recovers_exact_cell(self):
        radio_map = self._noiseless_map()
        for cell in radio_map.cells:
            obs = {
                b.id: distance_to_rssi(math.hypot(cell[0] - b.x,
                                                  cell[1] - b.y))
                for b in BEACONS
            }
            self.assertEqual(radio_map.locate(obs, k=1), cell)
            # The exact-match shortcut returns the cell for k=3 too.
            self.assertEqual(radio_map.locate(obs, k=3), cell)

    def test_off_grid_point_lands_within_one_cell(self):
        radio_map = self._noiseless_map()
        point = (10.0, 6.0)
        obs = {
            b.id: distance_to_rssi(math.hypot(point[0] - b.x,
                                              point[1] - b.y))
            for b in BEACONS
        }
        got = radio_map.locate(obs, k=3)
        self.assertLess(math.hypot(got[0] - point[0], got[1] - point[1]), 2.0)

    def test_noisy_accuracy_comparable_to_trilateration(self):
        # Survey and walk both at 2 dB RSSI noise: the k=3
        # fingerprint estimate must stay under a 3.0 m mean error
        # (trilateration measures ~2.2 m on the same walks).
        cells = grid_points(20.0, 12.0, 2.0)
        survey = simulate_walk(cells, BEACONS, noise_db=2.0, seed=7)
        radio_map = build_fingerprint_map(BEACONS, survey)
        path = rectangular_walk(20.0, 12.0, 10)
        for seed in (1, 2, 3, 7, 11):
            steps = simulate_walk(path, BEACONS, noise_db=2.0, seed=seed)
            errors = []
            for step in steps:
                est = radio_map.locate(step["rssi"], k=3)
                true = step["true"]
                errors.append(math.hypot(est[0] - true[0], est[1] - true[1]))
            self.assertLess(statistics.mean(errors), 3.0,
                            f"fingerprint mean error too high on seed {seed}")

    def test_k3_beats_k1_on_noisy_data(self):
        cells = grid_points(20.0, 12.0, 2.0)
        survey = simulate_walk(cells, BEACONS, noise_db=2.0, seed=7)
        radio_map = build_fingerprint_map(BEACONS, survey)
        path = rectangular_walk(20.0, 12.0, 10)
        steps = simulate_walk(path, BEACONS, noise_db=2.0, seed=7)
        err1, err3 = [], []
        for step in steps:
            true = step["true"]
            err1.append(math.hypot(
                *(a - b for a, b in zip(radio_map.locate(step["rssi"], k=1), true))))
            err3.append(math.hypot(
                *(a - b for a, b in zip(radio_map.locate(step["rssi"], k=3), true))))
        self.assertLess(statistics.mean(err3), statistics.mean(err1))

    def test_repeated_samples_at_a_cell_are_averaged(self):
        radio_map = FingerprintMap(["A", "B"])
        radio_map.add((5.0, 5.0), {"A": -60.0, "B": -70.0})
        radio_map.add((5.0, 5.0), {"A": -62.0, "B": -72.0})
        self.assertEqual(len(radio_map), 1)
        self.assertEqual(radio_map.fingerprint((5.0, 5.0)),
                         {"A": -61.0, "B": -71.0})
        # The averaged vector matches exactly.
        self.assertEqual(radio_map.locate({"A": -61.0, "B": -71.0}, k=1),
                         (5.0, 5.0))

    def test_missing_beacon_does_not_raise(self):
        radio_map = self._noiseless_map()
        point = (9.0, 5.0)
        obs = {
            b.id: distance_to_rssi(math.hypot(point[0] - b.x,
                                              point[1] - b.y))
            for b in BEACONS[:3]  # beacon "D" unheard
        }
        got = radio_map.locate(obs, k=3)
        self.assertLess(math.hypot(got[0] - point[0], got[1] - point[1]), 4.0)

    def test_unknown_beacon_raises(self):
        radio_map = self._noiseless_map()
        with self.assertRaises(ValueError):
            radio_map.locate({"ZZ": -70.0})
        with self.assertRaises(ValueError):
            radio_map.add((1.0, 1.0), {"ZZ": -70.0})

    def test_empty_map_raises(self):
        radio_map = FingerprintMap(["A", "B", "C", "D"])
        with self.assertRaises(ValueError):
            radio_map.locate({"A": -60.0})

    def test_bad_k_raises(self):
        radio_map = self._noiseless_map()
        obs = {b.id: -60.0 for b in BEACONS}
        with self.assertRaises(ValueError):
            radio_map.locate(obs, k=0)
        # k larger than the map just uses every cell.
        self.assertEqual(len(radio_map.nearest(obs, k=1000)),
                         len(radio_map))

    def test_nearest_is_sorted_by_rssi_distance(self):
        radio_map = self._noiseless_map()
        point = (9.0, 5.0)
        obs = {
            b.id: distance_to_rssi(math.hypot(point[0] - b.x,
                                              point[1] - b.y))
            for b in BEACONS
        }
        neighbours = radio_map.nearest(obs, k=3)
        distances = [d for _, d in neighbours]
        self.assertEqual(distances, sorted(distances))
        # The nearest cell is a neighbour of (9, 5) on the 2 m grid.
        nearest = neighbours[0][0]
        self.assertLess(math.hypot(nearest[0] - 9.0, nearest[1] - 5.0), 2.0)


if __name__ == "__main__":
    unittest.main()
