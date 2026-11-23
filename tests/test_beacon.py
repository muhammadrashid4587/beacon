"""Tests for beacon: trilateration, RSSI model, engine, simulator."""

import json
import math
import os
import random
import statistics
import tempfile
import unittest

from beacon import (
    Beacon,
    BeaconHealth,
    FingerprintMap,
    FloorEstimate,
    KalmanFilter,
    LayoutScore,
    LivePosition,
    LiveTracker,
    ExcludedBeacon,
    MovingAverageFilter,
    MultiFloorFingerprintMap,
    PlacementAdvice,
    Position,
    PositioningEngine,
    advise_placement,
    assess_beacon_health,
    build_fingerprint_map,
    build_multifloor_fingerprint_map,
    calibrate_beacon,
    calibrate_survey,
    distance_to_rssi,
    DriftMonitor,
    estimate_floor,
    estimate_uncertainty,
    evaluate_live_track,
    export_track_csv,
    export_track_json,
    fit_path_loss,
    grid_points,
    healthy_beacons,
    load_track_csv,
    load_track_for_overlay,
    load_track_json,
    placement_uncertainty,
    PositionUncertainty,
    render_map_overlay,
    rssi_to_distance,
    score_layout,
    simulate_multifloor_walk,
    simulate_walk,
    Survey,
    track_rows,
    TrackReport,
    trilaterate,
    two_floor_walk,
    write_map_overlay,
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


class TestTrajectorySmoothing(unittest.TestCase):
    def test_smoother_beats_filter_across_seeds(self):
        # A recorded walk can be revised with the observations that
        # came after each step, so the RTS-smoothed path must beat
        # the causal filtered path on every seed, not just seed 7.
        path = rectangular_walk(20.0, 12.0, 10)
        for seed in (1, 2, 3, 7, 11):
            steps = simulate_walk(path, BEACONS, noise_db=2.0, seed=seed)
            filtered = PositioningEngine(BEACONS)
            ferr = []
            for step in steps:
                est = filtered.estimate(step["rssi"])
                ferr.append(math.hypot(est[0] - step["true"][0],
                                       est[1] - step["true"][1]))
            smoothed = PositioningEngine(BEACONS).smooth_walk(
                [step["rssi"] for step in steps])
            serr = [math.hypot(p[0] - step["true"][0],
                               p[1] - step["true"][1])
                    for p, step in zip(smoothed, steps)]
            # Measured smoothed means: 1.18-1.83 m against filtered
            # 1.83-2.63 m. Bound is ~2x the worst measured seed.
            self.assertLess(statistics.mean(serr), 4.0,
                            f"smoothed mean error too high on seed {seed}")
            self.assertLess(statistics.mean(serr), statistics.mean(ferr),
                            f"smoother did not beat filter on seed {seed}")

    def test_last_smoothed_equals_last_filtered(self):
        # Nothing comes after the last step to revise it with, so
        # the last smoothed position is the last filtered one.
        path = rectangular_walk(20.0, 12.0, 10)
        steps = simulate_walk(path, BEACONS, noise_db=2.0, seed=7)
        filtered = PositioningEngine(BEACONS)
        last_filtered = None
        for step in steps:
            last_filtered = filtered.estimate(step["rssi"])
        smoothed = PositioningEngine(BEACONS).smooth_walk(
            [step["rssi"] for step in steps])
        self.assertEqual(len(smoothed), len(steps))
        self.assertAlmostEqual(smoothed[-1][0], last_filtered[0])
        self.assertAlmostEqual(smoothed[-1][1], last_filtered[1])

    def test_empty_walk_returns_empty(self):
        self.assertEqual(PositioningEngine(BEACONS).smooth_walk([]), [])

    def test_single_step_returns_the_estimate(self):
        step = simulate_walk([(10.0, 6.0)], BEACONS, noise_db=0.0, seed=1)[0]
        smoothed = PositioningEngine(BEACONS).smooth_walk([step["rssi"]])
        self.assertEqual(len(smoothed), 1)
        self.assertAlmostEqual(smoothed[0][0], 10.0)
        self.assertAlmostEqual(smoothed[0][1], 6.0)

    def test_stationary_jitter_reduced_further(self):
        # Smoothing a stationary device must jitter less than the
        # already-filtered estimates.
        steps = simulate_walk([(10.0, 6.0)] * 60, BEACONS,
                              noise_db=3.0, seed=11)
        filtered = PositioningEngine(BEACONS)
        filt = [filtered.estimate(step["rssi"]) for step in steps]
        smoothed = PositioningEngine(BEACONS).smooth_walk(
            [step["rssi"] for step in steps])
        filt_jitter = (statistics.pstdev([p[0] for p in filt])
                       + statistics.pstdev([p[1] for p in filt]))
        sm_jitter = (statistics.pstdev([p[0] for p in smoothed])
                     + statistics.pstdev([p[1] for p in smoothed]))
        self.assertLess(sm_jitter, filt_jitter)

    def test_moving_average_engine_rejects_smooth_walk(self):
        # The moving average keeps no forward-pass history, so there
        # is nothing for the RTS recursion to run over.
        engine = PositioningEngine(BEACONS, filter="moving_average")
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 5), BEACONS,
                              noise_db=2.0, seed=7)
        with self.assertRaises(ValueError):
            engine.smooth_walk([step["rssi"] for step in steps])

    def test_floor_changing_walk_is_rejected(self):
        # A floor change resets the filter mid-walk, so only the
        # final segment's history would survive — reject instead of
        # silently returning a partial smoothing.
        floors = ([Beacon("A", 0.0, 0.0, floor=0),
                   Beacon("B", 20.0, 0.0, floor=0),
                   Beacon("C", 20.0, 12.0, floor=0),
                   Beacon("D", 0.0, 12.0, floor=0),
                   Beacon("E", 0.0, 0.0, floor=1),
                   Beacon("F", 20.0, 0.0, floor=1),
                   Beacon("G", 20.0, 12.0, floor=1),
                   Beacon("H", 0.0, 12.0, floor=1)])
        steps = simulate_multifloor_walk(
            two_floor_walk(20.0, 12.0, 5), floors, noise_db=2.0, seed=7)
        with self.assertRaises(ValueError):
            PositioningEngine(floors).smooth_walk(
                [step["rssi"] for step in steps])

    def test_reset_clears_smoother_history(self):
        filt = KalmanFilter()
        filt.update((10.0, 6.0))
        filt.update((11.0, 6.0))
        self.assertEqual(len(filt.smooth()), 2)
        filt.reset()
        self.assertEqual(filt.smooth(), [])


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


class TestMultiFloor(unittest.TestCase):
    """Multi-floor positioning: floor vote, within-floor accuracy,
    and the honest edge cases (adjacent-floor bleed, missing floor
    beacons, low-confidence votes, too few beacons on a floor)."""

    # Two floors over the same 20 x 12 m ward: four corner beacons
    # per floor, same (x, y), ids A-D downstairs and E-H upstairs.
    FLOORS = [
        Beacon("A", 0.0, 0.0, floor=0),
        Beacon("B", 20.0, 0.0, floor=0),
        Beacon("C", 20.0, 12.0, floor=0),
        Beacon("D", 0.0, 12.0, floor=0),
        Beacon("E", 0.0, 0.0, floor=1),
        Beacon("F", 20.0, 0.0, floor=1),
        Beacon("G", 20.0, 12.0, floor=1),
        Beacon("H", 0.0, 12.0, floor=1),
    ]

    def test_beacon_default_floor_is_zero(self):
        self.assertEqual(Beacon("A", 0.0, 0.0).floor, 0)
        self.assertEqual(Beacon("A", 0.0, 0.0, floor=2).floor, 2)

    def test_noiseless_vote_picks_correct_floor_with_confidence(self):
        # Noiseless observations from the middle of the ward on each
        # floor: the vote must pick that floor with high confidence,
        # even though the other floor's beacons are also heard
        # (attenuated through the slab).
        for floor in (0, 1):
            steps = simulate_multifloor_walk(
                [(10.0, 6.0, floor)], self.FLOORS, noise_db=0.0, seed=7)
            vote = estimate_floor(steps[0]["rssi"], self.FLOORS)
            self.assertEqual(vote.floor, floor)
            self.assertGreater(vote.confidence, 0.8)
            self.assertIsInstance(vote, FloorEstimate)
            # Both floors scored (the other floor's beacons were
            # heard, just weakly), so the vote is a comparison, not
            # a default.
            self.assertEqual(set(vote.scores), {0, 1})

    def test_cross_floor_attenuation_weakens_rssi(self):
        # Same (x, y), one floor apart from a ground-floor beacon:
        # the beacon must read weaker from upstairs than from its
        # own floor (3D distance plus the slab attenuation).
        down = simulate_multifloor_walk(
            [(10.0, 6.0, 0)], self.FLOORS, noise_db=0.0, seed=7)[0]
        up = simulate_multifloor_walk(
            [(10.0, 6.0, 1)], self.FLOORS, noise_db=0.0, seed=7)[0]
        self.assertLess(up["rssi"]["A"], down["rssi"]["A"])

    def test_two_floor_walk_accuracy_and_floor_classification(self):
        # A walk downstairs then upstairs, 2 dB RSSI noise: the floor
        # must be right on at least 95% of steps on every seed, and
        # the within-floor (x, y) error must stay under the same
        # 4.0 m guard as the single-floor engine.
        path = two_floor_walk(20.0, 12.0, 10)
        for seed in (1, 2, 3, 7, 11):
            steps = simulate_multifloor_walk(path, self.FLOORS,
                                             noise_db=2.0, seed=seed)
            engine = PositioningEngine(self.FLOORS)
            floor_ok = 0
            errors = []
            for step in steps:
                pos = engine.estimate_with_floor(step["rssi"])
                true_x, true_y, true_floor = step["true"]
                self.assertIsInstance(pos, Position)
                if pos.floor == true_floor:
                    floor_ok += 1
                errors.append(math.hypot(pos.x - true_x, pos.y - true_y))
            self.assertGreaterEqual(floor_ok / len(steps), 0.95,
                                    f"floor accuracy too low on seed {seed}")
            self.assertLess(statistics.mean(errors), 4.0,
                            f"within-floor error too high on seed {seed}")

    def test_within_floor_error_comparable_to_single_floor(self):
        # Same walk, same seeds, single-floor engine vs multi-floor:
        # localising within the voted floor must not cost more than
        # 1.0 m of mean error over the single-floor baseline (measured
        # differences are -0.3 to +0.5 m).
        path = two_floor_walk(20.0, 12.0, 10)
        single = [b for b in self.FLOORS if b.floor == 0]
        for seed in (1, 2, 3, 7, 11):
            multi_steps = simulate_multifloor_walk(
                path, self.FLOORS, noise_db=2.0, seed=seed)
            single_steps = simulate_walk(
                rectangular_walk(20.0, 12.0, 10), single,
                noise_db=2.0, seed=seed)
            multi_engine = PositioningEngine(self.FLOORS)
            single_engine = PositioningEngine(single)
            multi_err, single_err = [], []
            for step in multi_steps:
                pos = multi_engine.estimate_with_floor(step["rssi"])
                tx, ty, _ = step["true"]
                multi_err.append(math.hypot(pos.x - tx, pos.y - ty))
            for step in single_steps:
                est = single_engine.estimate(step["rssi"])
                single_err.append(math.hypot(
                    est[0] - step["true"][0], est[1] - step["true"][1]))
            self.assertLess(statistics.mean(multi_err),
                            statistics.mean(single_err) + 1.0,
                            f"multi-floor much worse than single on seed {seed}")

    def test_floor_with_no_beacons_heard_cannot_win(self):
        # Only ground-floor beacons heard (the upstairs beacons are
        # out of range): the vote must pick floor 0 with confidence
        # 1.0, and the engine must report floor 0.
        steps = simulate_multifloor_walk(
            [(10.0, 6.0, 0)], self.FLOORS, noise_db=0.0, seed=7)
        obs = {k: v for k, v in steps[0]["rssi"].items()
               if k in ("A", "B", "C", "D")}
        vote = estimate_floor(obs, self.FLOORS)
        self.assertEqual(vote.floor, 0)
        self.assertAlmostEqual(vote.confidence, 1.0)
        engine = PositioningEngine(self.FLOORS)
        self.assertEqual(engine.estimate_with_floor(obs).floor, 0)

    def test_low_confidence_vote_keeps_current_floor(self):
        # The engine is tracking on floor 0; then every beacon is
        # heard at the same strength from both floors, so the vote
        # is a 50/50 coin toss. Hysteresis must keep floor 0 rather
        # than flickering.
        engine = PositioningEngine(self.FLOORS)
        steps = simulate_multifloor_walk(
            [(10.0, 6.0, 0)], self.FLOORS, noise_db=0.0, seed=7)
        self.assertEqual(engine.estimate_with_floor(steps[0]["rssi"]).floor, 0)
        ambiguous = {b.id: -70.0 for b in self.FLOORS}
        vote = estimate_floor(ambiguous, self.FLOORS)
        self.assertAlmostEqual(vote.confidence, 0.5, places=6)
        self.assertEqual(engine.estimate_with_floor(ambiguous).floor, 0)
        self.assertEqual(engine.current_floor, 0)

    def test_floor_change_resets_filter(self):
        # The device walks downstairs, then appears upstairs at the
        # far corner. A floor change must reset the filter, so the
        # first upstairs estimate lands near the true upstairs point
        # instead of being dragged back toward the downstairs track.
        engine = PositioningEngine(self.FLOORS)
        down = simulate_multifloor_walk(
            [(10.0, 6.0, 0)] * 20, self.FLOORS, noise_db=2.0, seed=7)
        for step in down:
            engine.estimate_with_floor(step["rssi"])
        up = simulate_multifloor_walk(
            [(2.0, 2.0, 1)], self.FLOORS, noise_db=0.0, seed=7)
        pos = engine.estimate_with_floor(up[0]["rssi"])
        self.assertEqual(pos.floor, 1)
        self.assertLess(math.hypot(pos.x - 2.0, pos.y - 2.0), 4.0)

    def test_too_few_beacons_on_floor_falls_back(self):
        # Only two upstairs beacons heard, plus one downstairs: the
        # vote still picks floor 1 (two votes beat one), but
        # trilateration needs three beacons on the floor, so the
        # engine must fall back to all heard beacons and still return
        # a position on floor 1 instead of raising. The (x, y) is
        # degraded — the cross-floor distance is inflated by the
        # slab — so this asserts the floor and a finite position,
        # not accuracy.
        steps = simulate_multifloor_walk(
            [(10.0, 6.0, 1)], self.FLOORS, noise_db=0.0, seed=7)
        obs = {k: v for k, v in steps[0]["rssi"].items()
               if k in ("E", "F", "C")}
        engine = PositioningEngine(self.FLOORS)
        pos = engine.estimate_with_floor(obs)
        self.assertEqual(pos.floor, 1)
        self.assertTrue(math.isfinite(pos.x) and math.isfinite(pos.y))

    def test_estimate_still_returns_xy_tuple(self):
        # Backwards compatibility: estimate() returns (x, y) and
        # estimate_with_floor() returns the Position with the floor.
        engine = PositioningEngine(self.FLOORS)
        steps = simulate_multifloor_walk(
            [(9.0, 5.0, 1)], self.FLOORS, noise_db=0.0, seed=7)
        xy = engine.estimate(steps[0]["rssi"])
        self.assertIsInstance(xy, tuple)
        self.assertEqual(len(xy), 2)
        pos = engine.estimate_with_floor(steps[0]["rssi"])
        self.assertEqual(pos.floor, 1)
        self.assertAlmostEqual(pos.x, xy[0])
        self.assertAlmostEqual(pos.y, xy[1])

    def test_reset_forgets_floor(self):
        engine = PositioningEngine(self.FLOORS)
        steps = simulate_multifloor_walk(
            [(10.0, 6.0, 1)], self.FLOORS, noise_db=0.0, seed=7)
        engine.estimate_with_floor(steps[0]["rssi"])
        self.assertEqual(engine.current_floor, 1)
        engine.reset()
        self.assertIsNone(engine.current_floor)

    def test_estimate_floor_rejects_unknown_beacon(self):
        with self.assertRaises(ValueError):
            estimate_floor({"ZZ": -70.0}, self.FLOORS)

    def test_estimate_floor_rejects_empty_observations(self):
        with self.assertRaises(ValueError):
            estimate_floor({}, self.FLOORS)

    def test_bad_floor_switch_confidence_raises(self):
        with self.assertRaises(ValueError):
            PositioningEngine(self.FLOORS, floor_switch_confidence=0.0)
        with self.assertRaises(ValueError):
            PositioningEngine(self.FLOORS, floor_switch_confidence=1.5)

    def test_multifloor_walk_is_reproducible(self):
        path = two_floor_walk(20.0, 12.0, 5)
        a = simulate_multifloor_walk(path, self.FLOORS, seed=42)
        b = simulate_multifloor_walk(path, self.FLOORS, seed=42)
        self.assertEqual(a, b)


class TestMultiFloorFingerprinting(unittest.TestCase):
    """Multi-floor fingerprinting: per-floor radio maps, floor
    classification from the maps themselves, and within-floor
    accuracy comparable to single-floor fingerprinting."""

    # Reuse the two-floor fixture from TestMultiFloor: four corner
    # beacons per floor over the same 20 x 12 m ward.
    FLOORS = TestMultiFloor.FLOORS

    def _two_floor_map(self, noise_db=0.0, spacing=2.0, seed=7):
        cells = grid_points(20.0, 12.0, spacing)
        path = ([(x, y, 0) for x, y in cells]
                + [(x, y, 1) for x, y in cells])
        survey = simulate_multifloor_walk(path, self.FLOORS,
                                          noise_db=noise_db, seed=seed)
        return build_multifloor_fingerprint_map(self.FLOORS, survey)

    def test_maps_cover_both_floors(self):
        radio_map = self._two_floor_map()
        self.assertEqual(radio_map.floors, [0, 1])
        self.assertEqual(len(radio_map.maps[0]), 60)
        self.assertEqual(len(radio_map.maps[1]), 60)
        self.assertEqual(len(radio_map), 120)
        # Each floor's map only knows its own beacons.
        self.assertEqual(radio_map.maps[0].beacon_ids,
                         ["A", "B", "C", "D"])
        self.assertEqual(radio_map.maps[1].beacon_ids,
                         ["E", "F", "G", "H"])

    def test_noiseless_locate_recovers_exact_cell_and_floor(self):
        radio_map = self._two_floor_map()
        for floor in (0, 1):
            for cell in radio_map.maps[floor].cells:
                steps = simulate_multifloor_walk(
                    [(cell[0], cell[1], floor)], self.FLOORS,
                    noise_db=0.0, seed=7)
                pos = radio_map.locate_with_floor(steps[0]["rssi"], k=1)
                self.assertIsInstance(pos, Position)
                self.assertEqual(pos.floor, floor)
                self.assertEqual((pos.x, pos.y), cell)
                # An exact fingerprint match is confident.
                self.assertGreater(pos.floor_confidence, 0.8)

    def test_noiseless_vote_picks_correct_floor_with_confidence(self):
        # Noiseless observations from mid-ward on each floor: the
        # floor whose map matches best must win with high
        # confidence, and both floors must have scored.
        radio_map = self._two_floor_map()
        for floor in (0, 1):
            steps = simulate_multifloor_walk(
                [(10.0, 6.0, floor)], self.FLOORS, noise_db=0.0, seed=7)
            got, confidence, distances = radio_map.classify_floor(
                steps[0]["rssi"])
            self.assertEqual(got, floor)
            self.assertGreater(confidence, 0.8)
            self.assertEqual(set(distances), {0, 1})
            # The winning floor matches far better than the other:
            # the wrong floor's beacons were heard through the slab.
            self.assertLess(distances[floor], distances[1 - floor] / 3)

    def test_two_floor_walk_accuracy_and_floor_classification(self):
        # A walk downstairs then upstairs, 2 dB RSSI noise, surveyed
        # with a noisy radio map: the floor must be right on at
        # least 95% of steps on every seed, and the within-floor
        # (x, y) error must stay under the same 3.0 m guard as
        # single-floor fingerprinting.
        radio_map = self._two_floor_map(noise_db=2.0)
        path = two_floor_walk(20.0, 12.0, 10)
        for seed in (1, 2, 3, 7, 11):
            steps = simulate_multifloor_walk(path, self.FLOORS,
                                             noise_db=2.0, seed=seed)
            floor_ok = 0
            errors = []
            for step in steps:
                pos = radio_map.locate_with_floor(step["rssi"], k=3)
                true_x, true_y, true_floor = step["true"]
                if pos.floor == true_floor:
                    floor_ok += 1
                errors.append(math.hypot(pos.x - true_x, pos.y - true_y))
            self.assertGreaterEqual(floor_ok / len(steps), 0.95,
                                    f"floor accuracy too low on seed {seed}")
            self.assertLess(statistics.mean(errors), 3.0,
                            f"within-floor error too high on seed {seed}")

    def test_within_floor_error_comparable_to_single_floor(self):
        # Same walk, same seeds, single-floor fingerprinting vs
        # multi-floor: classifying the floor first and looking up
        # within that floor's map must not cost more than 1.0 m of
        # mean error over the single-floor baseline.
        multi_map = self._two_floor_map(noise_db=2.0)
        cells = grid_points(20.0, 12.0, 2.0)
        single_survey = simulate_walk(cells, BEACONS, noise_db=2.0, seed=7)
        single_map = build_fingerprint_map(BEACONS, single_survey)
        path = two_floor_walk(20.0, 12.0, 10)
        single = [b for b in self.FLOORS if b.floor == 0]
        for seed in (1, 2, 3, 7, 11):
            multi_steps = simulate_multifloor_walk(
                path, self.FLOORS, noise_db=2.0, seed=seed)
            single_steps = simulate_walk(
                rectangular_walk(20.0, 12.0, 10), single,
                noise_db=2.0, seed=seed)
            multi_err, single_err = [], []
            for step in multi_steps:
                pos = multi_map.locate_with_floor(step["rssi"], k=3)
                tx, ty, _ = step["true"]
                multi_err.append(math.hypot(pos.x - tx, pos.y - ty))
            for step in single_steps:
                est = single_map.locate(step["rssi"], k=3)
                single_err.append(math.hypot(
                    est[0] - step["true"][0], est[1] - step["true"][1]))
            self.assertLess(statistics.mean(multi_err),
                            statistics.mean(single_err) + 1.0,
                            f"multi-floor much worse than single on seed {seed}")

    def test_locate_returns_xy_tuple(self):
        # Backwards-compatible shape: locate() returns (x, y) and
        # locate_with_floor() returns the Position with the floor.
        radio_map = self._two_floor_map()
        steps = simulate_multifloor_walk(
            [(9.0, 5.0, 1)], self.FLOORS, noise_db=0.0, seed=7)
        xy = radio_map.locate(steps[0]["rssi"])
        self.assertIsInstance(xy, tuple)
        self.assertEqual(len(xy), 2)
        pos = radio_map.locate_with_floor(steps[0]["rssi"])
        self.assertEqual(pos.floor, 1)
        self.assertAlmostEqual(pos.x, xy[0])
        self.assertAlmostEqual(pos.y, xy[1])

    def test_single_floor_map_reports_confidence_one(self):
        # One floor only: the classification is trivial and the
        # confidence is 1.0.
        cells = grid_points(20.0, 12.0, 2.0)
        survey = simulate_walk(cells, BEACONS, noise_db=0.0, seed=7)
        radio_map = MultiFloorFingerprintMap(
            {0: build_fingerprint_map(BEACONS, survey)})
        steps = simulate_walk([(9.0, 5.0)], BEACONS, noise_db=0.0, seed=7)
        pos = radio_map.locate_with_floor(steps[0]["rssi"])
        self.assertEqual(pos.floor, 0)
        self.assertEqual(pos.floor_confidence, 1.0)

    def test_partial_hearing_still_classifies(self):
        # Only the upstairs beacons heard: the downstairs map
        # shares no beacons with the observation, so it is skipped
        # and floor 1 still wins.
        radio_map = self._two_floor_map()
        steps = simulate_multifloor_walk(
            [(10.0, 6.0, 1)], self.FLOORS, noise_db=0.0, seed=7)
        obs = {k: v for k, v in steps[0]["rssi"].items()
               if k in ("E", "F", "G", "H")}
        pos = radio_map.locate_with_floor(obs)
        self.assertEqual(pos.floor, 1)
        self.assertEqual(pos.floor_confidence, 1.0)

    def test_empty_map_raises(self):
        with self.assertRaises(ValueError):
            MultiFloorFingerprintMap({})
        with self.assertRaises(ValueError):
            MultiFloorFingerprintMap({0: FingerprintMap(["A", "B"])})

    def test_no_survey_steps_for_a_floor_raises(self):
        # Survey only walked floor 0, but beacons exist on floor 1.
        cells = grid_points(20.0, 12.0, 2.0)
        path = [(x, y, 0) for x, y in cells]
        survey = simulate_multifloor_walk(path, self.FLOORS,
                                          noise_db=0.0, seed=7)
        with self.assertRaises(ValueError):
            build_multifloor_fingerprint_map(self.FLOORS, survey)

    def test_unknown_beacon_raises(self):
        radio_map = self._two_floor_map()
        with self.assertRaises(ValueError):
            radio_map.locate_with_floor({"ZZ": -70.0})
        with self.assertRaises(ValueError):
            radio_map.classify_floor({"ZZ": -70.0})

    def test_empty_observations_raise(self):
        radio_map = self._two_floor_map()
        with self.assertRaises(ValueError):
            radio_map.locate_with_floor({})


class TestBeaconHealth(unittest.TestCase):
    def _survey(self, beacons, noise_db=2.0, seed=7):
        return simulate_walk(rectangular_walk(20, 12, 10), beacons,
                             noise_db=noise_db, seed=seed)

    def test_healthy_survey_reports_all_ok(self):
        report = assess_beacon_health(BEACONS, self._survey(BEACONS))
        self.assertEqual([r.status for r in report], ["ok"] * 4)
        for entry in report:
            self.assertLess(abs(entry.mean_residual_db), 2.0)
            self.assertEqual(entry.samples, entry.expected_samples)

    def test_drifted_beacon_flagged_with_residual(self):
        # Beacon A transmits 10 dB hotter than its calibrated
        # constant says; the survey is generated with the drifted
        # beacon, the assessment uses the calibrated one.
        drifted = [Beacon("A", 0.0, 0.0, rssi_at_1m=-49.0)] + BEACONS[1:]
        report = assess_beacon_health(BEACONS, self._survey(drifted))
        by_id = {r.id: r for r in report}
        self.assertEqual(by_id["A"].status, "drifting")
        self.assertAlmostEqual(by_id["A"].mean_residual_db, 10.0, delta=1.0)
        for other in ("B", "C", "D"):
            self.assertEqual(by_id[other].status, "ok")

    def test_dead_beacon_flagged(self):
        # Beacon A is absent from the survey entirely: never heard.
        report = assess_beacon_health(
            BEACONS, self._survey(BEACONS[1:]))
        by_id = {r.id: r for r in report}
        self.assertEqual(by_id["A"].status, "dead")
        self.assertIsNone(by_id["A"].mean_residual_db)
        self.assertEqual(by_id["A"].samples, 0)
        for other in ("B", "C", "D"):
            self.assertEqual(by_id[other].status, "ok")

    def test_healthy_beacons_excludes_unhealthy(self):
        drifted = [Beacon("A", 0.0, 0.0, rssi_at_1m=-49.0)] + BEACONS[1:]
        report = assess_beacon_health(BEACONS, self._survey(drifted))
        healthy = healthy_beacons(BEACONS, report)
        self.assertEqual([b.id for b in healthy], ["B", "C", "D"])
        # The filtered engine still localises from the healthy set.
        engine = PositioningEngine(healthy)
        x, y = engine.estimate_raw({"B": -71.4, "C": -68.0, "D": -59.8})
        self.assertTrue(math.isfinite(x) and math.isfinite(y))

    def test_excluding_drifted_beacon_improves_accuracy(self):
        # With the drifting beacon in the engine, the estimate is
        # pulled far off; without it, it lands near the true point.
        # A 15 dB drift (transmit power up) underestimates A's
        # distance ~5.6x, so the bad estimate is metres off.
        drifted = [Beacon("A", 0.0, 0.0, rssi_at_1m=-44.0)] + BEACONS[1:]
        survey = self._survey(drifted, noise_db=0.0)
        report = assess_beacon_health(BEACONS, survey)
        healthy = healthy_beacons(BEACONS, report)
        # survey[10] is a known point on the walk; use its true pos.
        true = survey[10]["true"]
        obs = dict(survey[10]["rssi"])
        bad = PositioningEngine(BEACONS).estimate_raw(obs)
        good = PositioningEngine(healthy).estimate_raw(
            {k: v for k, v in obs.items() if k != "A"})
        bad_err = math.hypot(bad[0] - true[0], bad[1] - true[1])
        good_err = math.hypot(good[0] - true[0], good[1] - true[1])
        self.assertGreater(bad_err, 5.0)
        self.assertLess(good_err, 1.0)

    def test_empty_survey_raises(self):
        with self.assertRaises(ValueError):
            assess_beacon_health(BEACONS, [])

    def test_bad_threshold_raises(self):
        with self.assertRaises(ValueError):
            assess_beacon_health(BEACONS, self._survey(BEACONS),
                                 drift_threshold_db=0.0)

    def test_multifloor_survey_raises(self):
        survey = simulate_multifloor_walk(two_floor_walk(20, 12, 10),
                                          BEACONS, noise_db=0.0, seed=7)
        with self.assertRaises(ValueError):
            assess_beacon_health(BEACONS, survey)

    def test_mismatched_report_raises(self):
        report = assess_beacon_health(BEACONS, self._survey(BEACONS))
        with self.assertRaises(ValueError):
            healthy_beacons(BEACONS[:3], report)


class TestSurveyExport(unittest.TestCase):
    def _survey(self, noise_db=0.0):
        cells = grid_points(20.0, 12.0, spacing=2.0)
        steps = simulate_walk(cells, BEACONS, noise_db=noise_db, seed=7)
        return Survey.build(BEACONS, steps)

    def test_round_trip_is_lossless(self):
        survey = self._survey(noise_db=2.0)
        restored = Survey.from_dict(survey.to_dict())
        # The serialised forms are identical, so nothing was lost.
        self.assertEqual(restored.to_dict(), survey.to_dict())
        self.assertEqual(restored.beacons, survey.beacons)
        self.assertEqual(restored.steps, survey.steps)

    def test_imported_map_locates_identically(self):
        survey = self._survey(noise_db=2.0)
        restored = Survey.from_dict(survey.to_dict())
        observation = {"A": -62.1, "B": -71.4, "C": -68.0, "D": -59.8}
        self.assertEqual(restored.fingerprint_map().locate(observation),
                         survey.fingerprint_map().locate(observation))

    def test_calibration_survives_round_trip(self):
        # Beacons whose true constants differ from the defaults, so
        # the export has fitted values worth carrying.
        beacons = [Beacon(b.id, b.x, b.y, rssi_at_1m=-65.0,
                          path_loss_exponent=3.0) for b in BEACONS]
        cells = grid_points(20.0, 12.0, spacing=2.0)
        steps = simulate_walk(cells, beacons, noise_db=0.0,
                              path_loss_exponent=3.0, seed=7)
        survey = Survey.build(beacons, steps)
        calibrated = survey.calibrated_beacons()
        self.assertAlmostEqual(calibrated[0].rssi_at_1m, -65.0, places=6)
        self.assertAlmostEqual(calibrated[0].path_loss_exponent, 3.0,
                               places=6)
        exported = Survey.build(calibrated, survey.steps)
        restored = Survey.from_dict(exported.to_dict())
        self.assertEqual(restored.beacons, calibrated)

    def test_repeated_samples_keep_their_weight(self):
        # Two passes over the same cell, then export/import, then a
        # third pass: the average must weight all three samples
        # equally, not average the imported average with the new one.
        radio_map = FingerprintMap(["A", "B"])
        radio_map.add((5.0, 5.0), {"A": -60.0, "B": -70.0})
        radio_map.add((5.0, 5.0), {"A": -62.0, "B": -72.0})
        restored = FingerprintMap.from_dict(radio_map.to_dict())
        restored.add((5.0, 5.0), {"A": -63.0, "B": -69.0})
        fingerprint = restored.fingerprint((5.0, 5.0))
        self.assertAlmostEqual(fingerprint["A"], (-60 - 62 - 63) / 3)
        self.assertAlmostEqual(fingerprint["B"], (-70 - 72 - 69) / 3)

    def test_multifloor_round_trip(self):
        beacons = [Beacon(b.id, b.x, b.y, floor=0) for b in BEACONS] + [
            Beacon("E", 0.0, 0.0, floor=1),
            Beacon("F", 20.0, 0.0, floor=1),
            Beacon("G", 20.0, 12.0, floor=1),
            Beacon("H", 0.0, 12.0, floor=1),
        ]
        cells = grid_points(20.0, 12.0, spacing=2.0)
        path = ([(x, y, 0) for x, y in cells]
                + [(x, y, 1) for x, y in cells])
        steps = simulate_multifloor_walk(path, beacons, noise_db=2.0,
                                         seed=7)
        survey = Survey.build(beacons, steps)
        restored = Survey.from_dict(survey.to_dict())
        self.assertEqual(restored.to_dict(), survey.to_dict())
        observation = {"A": -80.1, "B": -84.3, "C": -82.0, "D": -79.5,
                       "E": -62.4, "F": -70.8, "G": -68.2, "H": -60.1}
        self.assertEqual(
            restored.fingerprint_map().locate_with_floor(observation),
            survey.fingerprint_map().locate_with_floor(observation))

    def test_save_and_load_file_round_trip(self):
        survey = self._survey(noise_db=2.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ward-a-survey.json")
            survey.save(path)
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["format"],
                                 "beacon-survey")
            restored = Survey.load(path)
        self.assertEqual(restored.to_dict(), survey.to_dict())

    def test_survey_without_radio_map_round_trips(self):
        steps = simulate_walk(grid_points(20.0, 12.0, spacing=2.0),
                              BEACONS, noise_db=0.0, seed=7)
        survey = Survey(BEACONS, steps)
        restored = Survey.from_dict(survey.to_dict())
        self.assertIsNone(restored.radio_map)
        # The map rebuilds from the steps on demand.
        self.assertEqual(len(restored.fingerprint_map()), 60)

    def test_map_only_survey_locates_after_import(self):
        survey = self._survey(noise_db=0.0)
        map_only = Survey(survey.beacons, [],
                          radio_map=survey.fingerprint_map())
        restored = Survey.from_dict(map_only.to_dict())
        self.assertEqual(restored.steps, [])
        self.assertEqual(
            restored.fingerprint_map().locate(
                {"A": -59.0, "B": -79.0, "C": -79.0, "D": -59.0}),
            survey.fingerprint_map().locate(
                {"A": -59.0, "B": -79.0, "C": -79.0, "D": -59.0}))

    def test_wrong_format_raises(self):
        with self.assertRaises(ValueError):
            Survey.from_dict({"format": "not-a-survey", "version": 1})

    def test_unsupported_version_raises(self):
        data = self._survey().to_dict()
        data["version"] = 99
        with self.assertRaises(ValueError):
            Survey.from_dict(data)

    def test_empty_beacons_raises(self):
        data = self._survey().to_dict()
        data["beacons"] = []
        with self.assertRaises(ValueError):
            Survey.from_dict(data)

    def test_empty_survey_cannot_build_map(self):
        with self.assertRaises(ValueError):
            Survey(BEACONS, []).fingerprint_map()

    def test_calibrated_beacons_rejects_multifloor(self):
        beacons = [Beacon(b.id, b.x, b.y, floor=0) for b in BEACONS] + [
            Beacon("E", 0.0, 0.0, floor=1),
        ]
        steps = simulate_multifloor_walk([(5.0, 5.0, 0), (5.0, 5.0, 1)],
                                         beacons, noise_db=0.0, seed=7)
        with self.assertRaises(ValueError):
            Survey(beacons, steps).calibrated_beacons()


class TestDriftMonitor(unittest.TestCase):
    def test_healthy_walk_raises_no_alarm(self):
        monitor = DriftMonitor(BEACONS)
        steps = simulate_walk(rectangular_walk(20, 12, 10), BEACONS,
                              noise_db=2.0, seed=7)
        for step in steps:
            monitor.feed(step)
        self.assertEqual(monitor.alarms(), [])
        for beacon in BEACONS:
            self.assertFalse(monitor.is_alarming(beacon.id))

    def test_drifted_beacon_alarms_and_others_do_not(self):
        monitor = DriftMonitor(BEACONS)
        steps = simulate_walk(rectangular_walk(20, 12, 10), BEACONS,
                              noise_db=2.0, seed=7)
        for step in steps:
            step["rssi"]["A"] += 12.0  # beacon A reads 12 dB too loud
            monitor.feed(step)
        alarms = monitor.alarms()
        self.assertEqual([a.id for a in alarms], ["A"])
        self.assertAlmostEqual(alarms[0].mean_residual_db, 12.0, delta=2.0)
        self.assertEqual(alarms[0].samples, 20)

    def test_negative_drift_alarms_too(self):
        monitor = DriftMonitor(BEACONS)
        steps = simulate_walk(rectangular_walk(20, 12, 10), BEACONS,
                              noise_db=0.0, seed=7)
        for step in steps:
            step["rssi"]["C"] -= 8.0  # beacon C reads 8 dB too quiet
            monitor.feed(step)
        alarm = monitor.alarm("C")
        self.assertIsNotNone(alarm)
        self.assertAlmostEqual(alarm.mean_residual_db, -8.0, places=6)

    def test_single_bad_reading_does_not_alarm(self):
        monitor = DriftMonitor(BEACONS)
        monitor.observe((10.0, 6.0), {"A": -30.0, "B": -70.0,
                                      "C": -70.0, "D": -70.0})
        self.assertEqual(monitor.alarms(), [])

    def test_alarm_clears_when_beacon_recovers(self):
        monitor = DriftMonitor(BEACONS, window=10)
        drifted = simulate_walk(rectangular_walk(20, 12, 10), BEACONS,
                                noise_db=0.0, seed=7)
        for step in drifted:
            step["rssi"]["B"] += 12.0
            monitor.feed(step)
        self.assertTrue(monitor.is_alarming("B"))
        healthy = simulate_walk(rectangular_walk(20, 12, 10), BEACONS,
                                noise_db=0.0, seed=7)
        for step in healthy:
            monitor.feed(step)
        self.assertFalse(monitor.is_alarming("B"))
        self.assertEqual(monitor.alarms(), [])

    def test_rolling_window_forgets_old_drift(self):
        # A short drift that leaves the window stops alarming on its own.
        monitor = DriftMonitor(BEACONS, window=5, min_samples=5)
        steps = simulate_walk(rectangular_walk(20, 12, 10), BEACONS,
                              noise_db=0.0, seed=7)
        for step in steps[:5]:
            step["rssi"]["D"] += 12.0
            monitor.feed(step)
        self.assertTrue(monitor.is_alarming("D"))
        for step in steps[5:10]:
            monitor.feed(step)
        self.assertFalse(monitor.is_alarming("D"))

    def test_status_reports_rolling_mean(self):
        monitor = DriftMonitor(BEACONS)
        steps = simulate_walk([(10.0, 6.0)] * 6, BEACONS,
                              noise_db=0.0, seed=7)
        for step in steps:
            monitor.feed(step)
        state = monitor.status("A")
        self.assertIsNotNone(state)
        self.assertAlmostEqual(state.mean_residual_db, 0.0, places=6)
        self.assertEqual(state.samples, 6)
        self.assertIsNone(monitor.alarm("A"))

    def test_unheard_beacon_has_no_status(self):
        monitor = DriftMonitor(BEACONS)
        self.assertIsNone(monitor.status("A"))
        self.assertIsNone(monitor.alarm("A"))

    def test_unknown_beacon_in_observation_is_ignored(self):
        monitor = DriftMonitor(BEACONS)
        monitor.observe((10.0, 6.0), {"Z": -60.0})
        self.assertIsNone(monitor.status("A"))

    def test_unknown_beacon_status_raises(self):
        monitor = DriftMonitor(BEACONS)
        with self.assertRaises(ValueError):
            monitor.status("Z")

    def test_bad_parameters_raise(self):
        with self.assertRaises(ValueError):
            DriftMonitor([])
        with self.assertRaises(ValueError):
            DriftMonitor(BEACONS, window=0)
        with self.assertRaises(ValueError):
            DriftMonitor(BEACONS, drift_threshold_db=0)
        with self.assertRaises(ValueError):
            DriftMonitor(BEACONS, min_samples=0)
        with self.assertRaises(ValueError):
            DriftMonitor(BEACONS, window=5, min_samples=6)

    def test_multifloor_position_raises(self):
        monitor = DriftMonitor(BEACONS)
        with self.assertRaises(ValueError):
            monitor.observe((5.0, 5.0, 0), {"A": -60.0})


class TestPositionUncertainty(unittest.TestCase):
    def test_symmetric_layout_is_near_circular(self):
        # Four corner beacons of a square ward, device dead centre:
        # the geometry constrains x and y equally, so the ellipse
        # is a circle.
        square = [Beacon("A", 0.0, 0.0), Beacon("B", 12.0, 0.0),
                  Beacon("C", 12.0, 12.0), Beacon("D", 0.0, 12.0)]
        centre = (6.0, 6.0)
        distances = _distances_to(centre, square)
        unc = estimate_uncertainty(
            [b.position for b in square], centre, distances)
        self.assertLess(unc.semi_major / unc.semi_minor, 1.2)
        self.assertGreater(unc.semi_minor, 0.0)
        self.assertLess(unc.semi_major, 6.0)

    def test_one_sided_beacons_stretch_along_the_weak_axis(self):
        # Beacons clustered on the west wall, device 15 m east of
        # them: every beacon lies in roughly the same direction, so
        # the measurements pin down the east-west distance but
        # barely constrain north-south — the ellipse is long in y
        # and its major axis points along y.
        wall = [Beacon("A", 0.0, 0.0), Beacon("B", 0.0, 12.0),
                Beacon("C", 0.0, 6.0), Beacon("D", 2.0, 6.0)]
        point = (15.0, 6.0)
        distances = _distances_to(point, wall)
        unc = estimate_uncertainty(
            [b.position for b in wall], point, distances)
        self.assertGreater(unc.semi_major / unc.semi_minor, 2.0)
        self.assertGreater(abs(unc.angle_deg), 60.0)

    def test_axes_scale_linearly_with_rssi_noise(self):
        centre = (10.0, 6.0)
        distances = _distances_to(centre, BEACONS)
        positions = [b.position for b in BEACONS]
        small = estimate_uncertainty(positions, centre, distances,
                                     sigma_rssi_db=1.0)
        large = estimate_uncertainty(positions, centre, distances,
                                     sigma_rssi_db=2.0)
        self.assertAlmostEqual(large.semi_major,
                               2.0 * small.semi_major, places=9)
        self.assertAlmostEqual(large.semi_minor,
                               2.0 * small.semi_minor, places=9)

    def test_steeper_exponent_shrinks_the_ellipse(self):
        # A steeper path-loss exponent turns the same RSSI noise
        # into a smaller distance error.
        centre = (10.0, 6.0)
        distances = _distances_to(centre, BEACONS)
        positions = [b.position for b in BEACONS]
        shallow = estimate_uncertainty(positions, centre, distances,
                                       path_loss_exponents=[2.0] * 4)
        steep = estimate_uncertainty(positions, centre, distances,
                                     path_loss_exponents=[4.0] * 4)
        self.assertLess(steep.semi_major, shallow.semi_major)

    def test_true_position_lands_inside_the_95_ellipse(self):
        # Coverage check: with 2 dB RSSI noise, the true position
        # should sit inside the 95% ellipse in about 95% of trials.
        rng = random.Random(7)
        inside = 0
        trials = 200
        true = (10.0, 6.0)
        for _ in range(trials):
            obs = {b.id: distance_to_rssi(
                       math.hypot(true[0] - b.x, true[1] - b.y),
                       b.rssi_at_1m, b.path_loss_exponent)
                   + rng.gauss(0.0, 2.0) for b in BEACONS}
            engine = PositioningEngine(BEACONS)
            x, y, unc = engine.estimate_with_uncertainty(obs)
            if unc.contains(true[0], true[1], (x, y)):
                inside += 1
        self.assertGreaterEqual(inside / trials, 0.85)

    def test_bad_inputs_raise(self):
        centre = (10.0, 6.0)
        positions = [b.position for b in BEACONS]
        distances = _distances_to(centre, BEACONS)
        with self.assertRaises(ValueError):
            estimate_uncertainty(positions[:2], centre,
                                 distances[:2])
        with self.assertRaises(ValueError):
            estimate_uncertainty(positions, centre, distances,
                                 sigma_rssi_db=0.0)
        collinear = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        # The estimate sits on the beacon line itself, so every
        # Jacobian row points along x and the covariance is
        # singular — the cross-line uncertainty is unbounded.
        with self.assertRaises(ValueError):
            estimate_uncertainty(collinear, (10.0, 0.0),
                                 [10.0, 0.0, 10.0])

    def test_engine_method_matches_raw_estimate(self):
        true = (10.0, 6.0)
        obs = {b.id: distance_to_rssi(
                   math.hypot(true[0] - b.x, true[1] - b.y))
               for b in BEACONS}
        engine = PositioningEngine(BEACONS)
        x, y, unc = engine.estimate_with_uncertainty(obs)
        self.assertIsInstance(unc, PositionUncertainty)
        self.assertAlmostEqual(x, true[0], places=6)
        self.assertAlmostEqual(y, true[1], places=6)
        self.assertGreater(unc.semi_major, 0.0)
        self.assertGreaterEqual(unc.semi_major, unc.semi_minor)

    def test_engine_unknown_beacon_raises(self):
        engine = PositioningEngine(BEACONS)
        with self.assertRaises(ValueError):
            engine.estimate_with_uncertainty({"Z": -60.0})


class TestPlacementAdvisor(unittest.TestCase):
    def test_centre_ellipse_is_near_circular(self):
        unc = placement_uncertainty(BEACONS, 10.0, 6.0)
        self.assertIsInstance(unc, PositionUncertainty)
        self.assertGreater(unc.semi_major, 0.0)
        # Symmetric four-corner layout: the centre ellipse is
        # near-circular, within a factor of two on its axes.
        self.assertLess(unc.semi_major / unc.semi_minor, 2.0)

    def test_corner_ellipse_is_stretched_along_weak_axis(self):
        centre = placement_uncertainty(BEACONS, 10.0, 6.0)
        corner = placement_uncertainty(BEACONS, 1.0, 1.0)
        # Near a corner the beacons all lie on one side, so the
        # ellipse stretches: much more eccentric than at the
        # centre, and its minor axis is much tighter.
        self.assertGreater(corner.semi_major / corner.semi_minor, 5.0)
        self.assertLess(corner.semi_minor, centre.semi_minor)

    def test_placement_uncertainty_needs_three_beacons(self):
        with self.assertRaises(ValueError):
            placement_uncertainty(BEACONS[:2], 10.0, 6.0)

    def test_score_layout_corner_ward(self):
        score = score_layout(BEACONS, 20.0, 12.0, margin=0.5)
        self.assertIsInstance(score, LayoutScore)
        self.assertGreater(score.mean_semi_major, 0.0)
        self.assertGreaterEqual(score.worst_semi_major,
                                score.mean_semi_major)
        self.assertEqual(score.degenerate_points, 0)
        self.assertGreater(score.n_points, 100)
        self.assertIsNotNone(score.worst_point)

    def test_score_layout_needs_three_beacons(self):
        with self.assertRaises(ValueError):
            score_layout(BEACONS[:2], 20.0, 12.0)

    def test_advise_returns_n_beacons_inside_room(self):
        advice = advise_placement(20.0, 12.0, 4)
        self.assertIsInstance(advice, PlacementAdvice)
        self.assertEqual(len(advice.beacons), 4)
        self.assertEqual([b.id for b in advice.beacons],
                         ["P1", "P2", "P3", "P4"])
        positions = {(b.x, b.y) for b in advice.beacons}
        self.assertEqual(len(positions), 4)
        for beacon in advice.beacons:
            self.assertGreaterEqual(beacon.x, 0.0)
            self.assertLessEqual(beacon.x, 20.0)
            self.assertGreaterEqual(beacon.y, 0.0)
            self.assertLessEqual(beacon.y, 12.0)
        self.assertEqual(advice.score.degenerate_points, 0)

    def test_greedy_beats_corners_on_mean_uncertainty(self):
        corners = score_layout(BEACONS, 20.0, 12.0, margin=0.5)
        advice = advise_placement(20.0, 12.0, 4)
        self.assertLess(advice.score.mean_semi_major,
                        corners.mean_semi_major)

    def test_more_beacons_never_hurts_mean_uncertainty(self):
        four = advise_placement(20.0, 12.0, 4)
        six = advise_placement(20.0, 12.0, 6)
        self.assertLessEqual(six.score.mean_semi_major,
                             four.score.mean_semi_major)
        # Six beacons also beat the four-corner layout on the
        # worst point, not just the mean.
        corners = score_layout(BEACONS, 20.0, 12.0, margin=0.5)
        self.assertLess(six.score.worst_semi_major,
                        corners.worst_semi_major)

    def test_advise_rejects_too_few_beacons(self):
        with self.assertRaises(ValueError):
            advise_placement(20.0, 12.0, 2)

    def test_advise_rejects_more_beacons_than_spots(self):
        with self.assertRaises(ValueError):
            advise_placement(4.0, 4.0, 10)

    def test_advise_rejects_bad_dimensions(self):
        with self.assertRaises(ValueError):
            advise_placement(0.0, 12.0, 4)


class TestLiveTracking(unittest.TestCase):
    def _walk(self, steps_per_side=10, seed=7):
        return simulate_walk(rectangular_walk(20.0, 12.0, steps_per_side),
                             BEACONS, noise_db=2.0, seed=seed)

    def test_streaming_updates_move_the_estimate(self):
        tracker = LiveTracker(PositioningEngine(BEACONS))
        positions = []
        errors = []
        for i, step in enumerate(self._walk()):
            live = tracker.feed(step["rssi"], float(i))
            self.assertIsNotNone(live)
            positions.append((live.x, live.y))
            errors.append(math.hypot(live.x - step["true"][0],
                                     live.y - step["true"][1]))
        # The estimate follows the walk: it moves metres, not
        # centimetres, and stays within the usual accuracy guard.
        spread = max(math.hypot(x - positions[0][0], y - positions[0][1])
                     for x, y in positions)
        self.assertGreater(spread, 5.0)
        self.assertLess(statistics.mean(errors), 4.0)

    def test_position_carries_95_ellipse(self):
        tracker = LiveTracker(PositioningEngine(BEACONS))
        live = tracker.feed(self._walk()[0]["rssi"], 0.0)
        self.assertIsNotNone(live.uncertainty)
        self.assertGreater(live.uncertainty.semi_major, 0.0)
        self.assertGreaterEqual(live.uncertainty.semi_major,
                                live.uncertainty.semi_minor)
        self.assertGreater(live.uncertainty.semi_minor, 0.0)
        self.assertGreaterEqual(live.uncertainty.angle_deg, -90.0)
        self.assertLessEqual(live.uncertainty.angle_deg, 90.0)
        self.assertEqual(live.heard, ("A", "B", "C", "D"))
        self.assertEqual(live.floor, 0)

    def test_partial_scans_merge_across_beacons(self):
        tracker = LiveTracker(PositioningEngine(BEACONS))
        step = self._walk()[0]["rssi"]
        # A scan hearing only two beacons is not enough on its own,
        # but a second scan hearing the other two completes the set.
        self.assertIsNone(tracker.feed({"A": step["A"], "B": step["B"]}, 0.0))
        live = tracker.feed({"C": step["C"], "D": step["D"]}, 1.0)
        self.assertIsNotNone(live)
        self.assertEqual(live.heard, ("A", "B", "C", "D"))

    def test_stale_readings_age_out(self):
        tracker = LiveTracker(PositioningEngine(BEACONS), stale_after=5.0)
        step = self._walk()[0]["rssi"]
        tracker.feed(step, 0.0)
        # Ten seconds later only B, C, D are heard: A aged out of
        # the merged set, and the track continues on three beacons.
        live = tracker.feed({k: v for k, v in step.items() if k != "A"}, 10.0)
        self.assertIsNotNone(live)
        self.assertEqual(live.heard, ("B", "C", "D"))
        # A reading exactly at the stale boundary is still fresh.
        live = tracker.feed({"A": step["A"]}, 15.0)
        self.assertIsNotNone(live)
        self.assertEqual(live.heard, ("A", "B", "C", "D"))

    def test_beacon_dropping_out_mid_stream(self):
        tracker = LiveTracker(PositioningEngine(BEACONS))
        steps = self._walk()
        errors = []
        for i, step in enumerate(steps):
            rssi = dict(step["rssi"])
            if i >= 20:
                del rssi["D"]  # beacon D dies halfway through the walk
            live = tracker.feed(rssi, float(i))
            self.assertIsNotNone(live, f"no position at step {i}")
            errors.append(math.hypot(live.x - step["true"][0],
                                     live.y - step["true"][1]))
        self.assertLess(statistics.mean(errors), 4.0)
        self.assertEqual(tracker.last_position.heard, ("A", "B", "C"))

    def test_too_few_fresh_beacons_returns_none_and_keeps_last(self):
        tracker = LiveTracker(PositioningEngine(BEACONS), stale_after=5.0)
        step = self._walk()[0]["rssi"]
        self.assertIsNone(tracker.feed({"A": step["A"], "B": step["B"]}, 0.0))
        self.assertIsNone(tracker.last_position)
        live = tracker.feed(step, 1.0)
        self.assertIsNotNone(live)
        # Ten seconds later only A and B are heard and C, D aged
        # out: no position, but the last good one is still there.
        self.assertIsNone(
            tracker.feed({"A": step["A"], "B": step["B"]}, 11.0))
        self.assertEqual(tracker.last_position, live)

    def test_unknown_beacon_is_ignored(self):
        tracker = LiveTracker(PositioningEngine(BEACONS))
        step = self._walk()[0]["rssi"]
        live = tracker.feed({**step, "Z": -50.0}, 0.0)
        self.assertIsNotNone(live)
        self.assertEqual(live.heard, ("A", "B", "C", "D"))

    def test_timestamps_must_not_go_backwards(self):
        tracker = LiveTracker(PositioningEngine(BEACONS))
        step = self._walk()[0]["rssi"]
        tracker.feed(step, 5.0)
        with self.assertRaises(ValueError):
            tracker.feed(step, 4.0)

    def test_bad_parameters_raise(self):
        with self.assertRaises(ValueError):
            LiveTracker(PositioningEngine(BEACONS), stale_after=0.0)
        with self.assertRaises(ValueError):
            LiveTracker(PositioningEngine(BEACONS), sigma_rssi_db=0.0)

    def test_reset_forgets_the_track(self):
        tracker = LiveTracker(PositioningEngine(BEACONS))
        step = self._walk()[0]["rssi"]
        tracker.feed(step, 0.0)
        tracker.reset()
        self.assertIsNone(tracker.last_position)
        self.assertEqual(tracker.heard, ())
        live = tracker.feed(step, 0.0)
        self.assertIsNotNone(live)

    def test_stream_json_lines_round_trip(self):
        import io

        from beacon.tracking import main

        steps = self._walk()
        lines = "".join(
            json.dumps({"t": float(i), "rssi": step["rssi"]}) + "\n"
            for i, step in enumerate(steps[:5]))
        out = io.StringIO()
        self.assertEqual(main(argv=[], input_stream=io.StringIO(lines),
                              output_stream=out), 0)
        records = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(records), 5)
        for record, step in zip(records, steps[:5]):
            position = record["position"]
            self.assertIsNotNone(position)
            self.assertEqual(position["floor"], 0)
            self.assertIn("uncertainty", position)
            self.assertGreater(
                position["uncertainty"]["semi_major"], 0.0)
            self.assertLess(
                math.hypot(position["x"] - step["true"][0],
                           position["y"] - step["true"][1]), 4.0)

    def test_stream_emits_null_position_when_too_thin(self):
        import io

        from beacon.tracking import main

        step = self._walk()[0]["rssi"]
        lines = (json.dumps({"t": 0.0, "rssi": {"A": step["A"]}}) + "\n"
                  + json.dumps({"t": 1.0, "rssi": step}) + "\n")
        out = io.StringIO()
        self.assertEqual(main(argv=[], input_stream=io.StringIO(lines),
                              output_stream=out), 0)
        records = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertIsNone(records[0]["position"])
        self.assertEqual(records[0]["heard"], ["A"])
        self.assertIsNotNone(records[1]["position"])

    def test_bad_stream_lines_raise(self):
        from beacon.tracking import parse_line

        for bad in ("not json", "[]",
                    json.dumps({"rssi": {"A": -60.0}}),
                    json.dumps({"t": 0.0}),
                    json.dumps({"t": "0", "rssi": {"A": -60.0}}),
                    json.dumps({"t": 0.0, "rssi": {}})):
            with self.assertRaises(ValueError, msg=bad):
                parse_line(bad)

    def test_demo_emits_one_line_per_step(self):
        import io

        from beacon.tracking import main

        out = io.StringIO()
        self.assertEqual(main(argv=["--demo"], output_stream=out), 0)
        records = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(records), 40)
        self.assertTrue(all(r["position"] is not None for r in records))


class TestBeaconFailureFallback(unittest.TestCase):
    def _walk(self, steps_per_side=10, seed=7):
        return simulate_walk(rectangular_walk(20.0, 12.0, steps_per_side),
                             BEACONS, noise_db=2.0, seed=seed)

    def _track_with_dead_beacon(self, die_at=20):
        """Feed a walk in which beacon D dies at step ``die_at``;
        returns (tracker, positions, steps)."""
        tracker = LiveTracker(PositioningEngine(BEACONS))
        steps = self._walk()
        positions = []
        for i, step in enumerate(steps):
            rssi = dict(step["rssi"])
            if i >= die_at:
                del rssi["D"]
            positions.append(tracker.feed(rssi, float(i)))
        return tracker, positions, steps

    def test_dark_beacon_is_excluded_and_recorded(self):
        tracker, positions, _ = self._track_with_dead_beacon()
        # D is last heard at t=19, so it is not dark yet at t=25
        # (6 s unheard, inside the 10 s dark window) — the track
        # is still on three beacons, but nothing is called dark.
        self.assertEqual(positions[25].excluded, ())
        self.assertEqual(positions[25].heard, ("A", "B", "C"))
        # At t=30 D has been unheard for 11 s: it is excluded, with
        # the reason saying how long it has been quiet.
        live = positions[30]
        self.assertEqual([b.id for b in live.excluded], ["D"])
        self.assertIn("no reading for 11.0s", live.excluded[0].reason)
        self.assertEqual(live.heard, ("A", "B", "C"))
        # The exclusion is on the position itself and in its JSON
        # form, so a dashboard sees it without a second call.
        self.assertEqual(
            live.to_dict()["excluded"],
            [{"id": "D",
              "reason": "no reading for 11.0s (dark after 10.0s)"}])
        self.assertEqual([b.id for b in tracker.excluded], ["D"])

    def test_accuracy_degrades_gracefully_not_fails(self):
        _, positions, steps = self._track_with_dead_beacon()
        # Every step still locates, on three beacons once D is
        # dark — the dropout costs accuracy, it does not end the
        # track.
        self.assertTrue(all(p is not None for p in positions))
        report = evaluate_live_track(
            positions, [step["true"] for step in steps])
        self.assertEqual(report.coverage, 1.0)
        self.assertLess(report.mean_error_m, 4.0)
        # And the late steps — the ones running without D — are
        # still located, just less precisely than the early ones.
        late = [math.hypot(p.x - s["true"][0], p.y - s["true"][1])
                for p, s in zip(positions[30:], steps[30:])]
        self.assertLess(statistics.mean(late), 6.0)

    def test_never_heard_beacon_is_not_excluded(self):
        tracker = LiveTracker(PositioningEngine(BEACONS))
        steps = self._walk()
        # Only A, B, C are ever heard: D was never part of the
        # track, so there is no fallback to record for it.
        live = tracker.feed({k: v for k, v in steps[0]["rssi"].items()
                             if k != "D"}, 0.0)
        self.assertIsNotNone(live)
        self.assertEqual(live.excluded, ())
        self.assertEqual(tracker.excluded, ())

    def test_excluded_beacon_recovers_when_heard_again(self):
        tracker = LiveTracker(PositioningEngine(BEACONS))
        steps = self._walk()
        positions = []
        for i, step in enumerate(steps):
            rssi = dict(step["rssi"])
            if 20 <= i < 35:
                del rssi["D"]  # D dies, then comes back
            positions.append(tracker.feed(rssi, float(i)))
        # D is dark in between: at t=30 it had been quiet 11 s.
        self.assertEqual([b.id for b in positions[30].excluded], ["D"])
        # One hearing is not enough to trust it back: D stays
        # excluded after its first reading back, with the reason
        # counting the fresh readings still needed.
        self.assertEqual([b.id for b in positions[35].excluded], ["D"])
        self.assertEqual(positions[35].excluded[0].reason,
                         "heard again, 1 of 3 fresh readings to re-admit")
        self.assertEqual([b.id for b in positions[36].excluded], ["D"])
        self.assertEqual(positions[36].excluded[0].reason,
                         "heard again, 2 of 3 fresh readings to re-admit")
        # The third consecutive hearing re-admits it.
        self.assertEqual([b.id for b in positions[37].excluded], [])
        self.assertEqual(positions[39].heard, ("A", "B", "C", "D"))
        self.assertEqual([b.id for b in tracker.excluded], [])

    def test_readmit_after_one_restores_single_reading_recovery(self):
        tracker = LiveTracker(PositioningEngine(BEACONS),
                              readmit_after=1)
        steps = self._walk()
        positions = []
        for i, step in enumerate(steps):
            rssi = dict(step["rssi"])
            if 20 <= i < 35:
                del rssi["D"]
            positions.append(tracker.feed(rssi, float(i)))
        # With readmit_after=1 the first hearing back clears it.
        self.assertEqual([b.id for b in positions[30].excluded], ["D"])
        self.assertEqual([b.id for b in positions[35].excluded], [])

    def test_missed_reading_resets_readmit_streak(self):
        tracker = LiveTracker(PositioningEngine(BEACONS))
        steps = self._walk()
        positions = []
        for i, step in enumerate(steps):
            rssi = dict(step["rssi"])
            if 20 <= i < 35 or i == 36:
                del rssi["D"]  # dies, flickers once back, dies again
            positions.append(tracker.feed(rssi, float(i)))
        # One hearing at t=35 starts the streak, the miss at t=36
        # breaks it, so D is still excluded after two more
        # hearings — only three in a row bring it back.
        self.assertEqual([b.id for b in positions[35].excluded], ["D"])
        self.assertEqual(positions[35].excluded[0].reason,
                         "heard again, 1 of 3 fresh readings to re-admit")
        self.assertEqual(positions[36].excluded[0].reason,
                         "heard again, 0 of 3 fresh readings to re-admit")
        self.assertEqual([b.id for b in positions[38].excluded], ["D"])
        self.assertEqual([b.id for b in positions[39].excluded], [])

    def test_readmit_after_must_be_at_least_one(self):
        with self.assertRaises(ValueError):
            LiveTracker(PositioningEngine(BEACONS), readmit_after=0)

    def test_dropout_and_recovery_still_tracks(self):
        # D dies mid-walk and comes back: the track stays located
        # throughout, and the dropout costs accuracy, not coverage.
        tracker = LiveTracker(PositioningEngine(BEACONS))
        steps = self._walk()
        positions = []
        for i, step in enumerate(steps):
            rssi = dict(step["rssi"])
            if 20 <= i < 35:
                del rssi["D"]
            positions.append(tracker.feed(rssi, float(i)))
        self.assertTrue(all(p is not None for p in positions))
        report = evaluate_live_track(
            positions, [step["true"] for step in steps])
        self.assertEqual(report.coverage, 1.0)
        self.assertLess(report.mean_error_m, 4.0)

    def test_dark_after_must_not_be_shorter_than_stale_after(self):
        with self.assertRaises(ValueError):
            LiveTracker(PositioningEngine(BEACONS),
                        stale_after=5.0, dark_after=4.0)
        with self.assertRaises(ValueError):
            LiveTracker(PositioningEngine(BEACONS), dark_after=0.0)
        # Equal windows are allowed: a beacon ages out and counts
        # as dark on the same reading.
        tracker = LiveTracker(PositioningEngine(BEACONS),
                              stale_after=5.0, dark_after=5.0)
        step = self._walk()[0]["rssi"]
        tracker.feed(step, 0.0)
        live = tracker.feed({k: v for k, v in step.items() if k != "A"},
                            6.0)
        self.assertEqual([b.id for b in live.excluded], ["A"])

    def test_excluded_appears_in_stream_output(self):
        import io

        from beacon.tracking import main

        steps = self._walk()
        lines = "".join(
            json.dumps({"t": float(i), "rssi": (
                dict(step["rssi"]) if i < 20
                else {k: v for k, v in step["rssi"].items()
                      if k != "D"})}) + "\n"
            for i, step in enumerate(steps[:32]))
        out = io.StringIO()
        self.assertEqual(main(argv=[], input_stream=io.StringIO(lines),
                              output_stream=out), 0)
        records = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(records), 32)
        # Before the dark window, no exclusion is recorded…
        self.assertEqual(records[25]["position"]["excluded"], [])
        # …after it, the stream line names the dark beacon.
        self.assertEqual(records[31]["position"]["excluded"],
                         [{"id": "D",
                           "reason": "no reading for 12.0s "
                           "(dark after 10.0s)"}])

    def test_reset_forgets_excluded_beacons(self):
        tracker, _, _ = self._track_with_dead_beacon()
        self.assertEqual([b.id for b in tracker.excluded], ["D"])
        tracker.reset()
        self.assertEqual(tracker.excluded, ())
        self.assertIsNone(tracker.last_position)


class TestLiveTrackReport(unittest.TestCase):
    def _track(self, steps, beacons=BEACONS):
        tracker = LiveTracker(PositioningEngine(beacons))
        return [tracker.feed(step["rssi"], float(i))
                for i, step in enumerate(steps)]

    def test_noiseless_walk_scores_near_zero_error(self):
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10), BEACONS,
                              noise_db=0.0, seed=7)
        report = evaluate_live_track(self._track(steps),
                                     [step["true"] for step in steps])
        self.assertEqual(report.steps, 40)
        self.assertEqual(report.located, 40)
        self.assertEqual(report.coverage, 1.0)
        # Noiseless RSSI, but the Kalman filter still lags the walk
        # around the corners, so the live track is not exact even
        # with perfect readings — mean 0.31 m, worst step 1.02 m.
        self.assertLess(report.mean_error_m, 1.0)
        self.assertLess(report.max_error_m, 1.5)
        self.assertIsNone(report.floor_accuracy)

    def test_noisy_walk_mean_error_matches_live_track(self):
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10), BEACONS,
                              noise_db=2.0, seed=7)
        report = evaluate_live_track(self._track(steps),
                                     [step["true"] for step in steps])
        self.assertEqual(report.coverage, 1.0)
        # The live track's mean error on the default ward walk is
        # 2.16 m; the report must agree with the track it scores.
        self.assertAlmostEqual(report.mean_error_m, 2.16, delta=0.05)
        self.assertLessEqual(report.median_error_m, report.p90_error_m)
        self.assertLessEqual(report.p90_error_m, report.max_error_m)
        self.assertLess(report.mean_error_m, 4.0)

    def test_beacon_dropout_shows_in_error_not_coverage(self):
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10), BEACONS,
                              noise_db=2.0, seed=7)
        tracker = LiveTracker(PositioningEngine(BEACONS))
        positions = []
        for i, step in enumerate(steps):
            rssi = dict(step["rssi"])
            if i >= 20:
                del rssi["D"]  # beacon D dies halfway through the walk
            positions.append(tracker.feed(rssi, float(i)))
        report = evaluate_live_track(
            positions, [step["true"] for step in steps])
        # Three beacons still locate, so coverage stays full — the
        # dropout shows up as a worse mean error instead.
        self.assertEqual(report.coverage, 1.0)
        self.assertGreater(report.mean_error_m, 2.16)

    def test_starved_steps_lower_coverage(self):
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10), BEACONS,
                              noise_db=2.0, seed=7)
        tracker = LiveTracker(PositioningEngine(BEACONS))
        positions = []
        for i, step in enumerate(steps):
            # Only two beacons heard for the second half: the track
            # starves and those steps produce no position at all —
            # except the first five, where C and D are still inside
            # the 5 s stale window and keep the track alive.
            rssi = (step["rssi"] if i < 20
                    else {"A": step["rssi"]["A"], "B": step["rssi"]["B"]})
            positions.append(tracker.feed(rssi, float(i)))
        report = evaluate_live_track(
            positions, [step["true"] for step in steps])
        self.assertEqual(report.steps, 40)
        self.assertEqual(report.located, 25)
        self.assertEqual(report.coverage, 0.625)

    def test_two_floor_walk_scores_floor_accuracy(self):
        beacons = [Beacon("A", 0.0, 0.0, floor=0),
                   Beacon("B", 20.0, 0.0, floor=0),
                   Beacon("C", 20.0, 12.0, floor=0),
                   Beacon("D", 0.0, 12.0, floor=0),
                   Beacon("E", 0.0, 0.0, floor=1),
                   Beacon("F", 20.0, 0.0, floor=1),
                   Beacon("G", 20.0, 12.0, floor=1),
                   Beacon("H", 0.0, 12.0, floor=1)]
        steps = simulate_multifloor_walk(
            two_floor_walk(20.0, 12.0, 10), beacons,
            noise_db=2.0, seed=7)
        report = evaluate_live_track(self._track(steps, beacons),
                                     [step["true"] for step in steps])
        self.assertEqual(report.floor_accuracy, 1.0)
        self.assertLess(report.mean_error_m, 4.0)

    def test_report_serialises_to_json(self):
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10), BEACONS,
                              noise_db=2.0, seed=7)
        report = evaluate_live_track(self._track(steps),
                                     [step["true"] for step in steps])
        record = json.loads(json.dumps(report.to_dict()))
        self.assertEqual(record["steps"], 40)
        self.assertEqual(record["located"], 40)
        self.assertEqual(record["coverage"], 1.0)
        self.assertIsNone(record["floor_accuracy"])
        self.assertAlmostEqual(record["mean_error_m"],
                               report.mean_error_m)

    def test_bad_inputs_raise(self):
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10), BEACONS,
                              noise_db=2.0, seed=7)
        truth = [step["true"] for step in steps]
        positions = self._track(steps)
        with self.assertRaises(ValueError):
            evaluate_live_track([], [])
        with self.assertRaises(ValueError):
            evaluate_live_track(positions, truth[:-1])
        with self.assertRaises(ValueError):
            evaluate_live_track(
                positions, truth[:-1] + [(0.0, 0.0, 0)])
        with self.assertRaises(ValueError):
            evaluate_live_track([None] * len(truth), truth)


class TestLiveTrackExport(unittest.TestCase):
    def _track(self, die_at=None, seed=7):
        """Feed the default walk through the tracker; beacon D
        dies at step ``die_at`` when given. Returns the recorded
        positions (one per step, None where a step was too thin
        to locate)."""
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10), BEACONS,
                              noise_db=2.0, seed=seed)
        tracker = LiveTracker(PositioningEngine(BEACONS))
        positions = []
        for i, step in enumerate(steps):
            rssi = dict(step["rssi"])
            if die_at is not None and i >= die_at:
                del rssi["D"]
            positions.append(tracker.feed(rssi, float(i)))
        return positions

    def test_row_carries_everything_a_dashboard_needs(self):
        rows = track_rows(self._track())
        self.assertEqual(len(rows), 40)
        row = rows[0]
        self.assertEqual(row["t"], 0.0)
        self.assertEqual(row["floor"], 0)
        self.assertGreater(row["floor_confidence"], 0.0)
        self.assertGreater(row["ellipse_semi_major_m"], 0.0)
        self.assertGreaterEqual(row["ellipse_semi_major_m"],
                                row["ellipse_semi_minor_m"])
        self.assertGreater(row["ellipse_semi_minor_m"], 0.0)
        self.assertGreaterEqual(row["ellipse_angle_deg"], -90.0)
        self.assertLessEqual(row["ellipse_angle_deg"], 90.0)
        self.assertEqual(row["beacons_used"], ["A", "B", "C", "D"])
        self.assertEqual(row["beacons_excluded"], [])
        self.assertEqual(row["coverage"], 1.0)

    def test_json_round_trip_is_lossless(self):
        positions = self._track(die_at=20)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "track.json")
            export_track_json(positions, path)
            loaded = load_track_json(path)
            # The file header says what the track was: 40 steps, all
            # located, coverage 1.0 even though a beacon died.
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        self.assertEqual(loaded, track_rows(positions))
        self.assertEqual(data["format"], "beacon-live-track")
        self.assertEqual(data["steps"], 40)
        self.assertEqual(data["located"], 40)
        self.assertEqual(data["coverage"], 1.0)

    def test_csv_round_trip_matches_json_rows(self):
        positions = self._track(die_at=20)
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "track.json")
            csv_path = os.path.join(tmp, "track.csv")
            export_track_json(positions, json_path)
            export_track_csv(positions, csv_path)
            self.assertEqual(load_track_csv(csv_path),
                             load_track_json(json_path))
            with open(csv_path, encoding="utf-8") as handle:
                header = handle.readline().strip()
        self.assertEqual(header, "t,x,y,floor,floor_confidence,"
                                 "ellipse_semi_major_m,"
                                 "ellipse_semi_minor_m,"
                                 "ellipse_angle_deg,beacons_used,"
                                 "beacons_excluded,coverage")

    def test_dropout_shows_in_beacons_used_and_excluded(self):
        rows = track_rows(self._track(die_at=20))
        # Before D dies, the rows run on all four beacons.
        early = rows[15]
        self.assertEqual(early["beacons_used"], ["A", "B", "C", "D"])
        self.assertEqual(early["beacons_excluded"], [])
        # D is last heard at step 19 and ages out of the merged
        # set after the 5 s stale window, so from step 25 the rows
        # run on three beacons — but D is not called dark until
        # the 10 s window passes, so no exclusion is recorded yet.
        mid = rows[25]
        self.assertEqual(mid["beacons_used"], ["A", "B", "C"])
        self.assertEqual(mid["beacons_excluded"], [])
        # From step 30 D is recorded excluded, with why.
        late = rows[35]
        self.assertEqual(late["beacons_used"], ["A", "B", "C"])
        self.assertEqual(len(late["beacons_excluded"]), 1)
        self.assertEqual(late["beacons_excluded"][0]["id"], "D")
        self.assertIn("no reading for", late["beacons_excluded"][0]["reason"])

    def test_starved_steps_are_counted_not_hidden(self):
        # Two beacons only: nothing locates, so there are no rows
        # but the coverage still says so.
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10), BEACONS,
                              noise_db=2.0, seed=7)
        tracker = LiveTracker(PositioningEngine(BEACONS))
        positions = [tracker.feed({"A": step["rssi"]["A"],
                                   "B": step["rssi"]["B"]}, float(i))
                      for i, step in enumerate(steps)]
        self.assertEqual(track_rows(positions), [])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "track.json")
            export_track_json(positions, path)
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        self.assertEqual(data["steps"], 40)
        self.assertEqual(data["located"], 0)
        self.assertEqual(data["coverage"], 0.0)

    def test_row_without_an_ellipse_round_trips(self):
        # A step whose geometry was degenerate carries no ellipse;
        # the export keeps the ellipse fields empty, not made up.
        position = LivePosition(t=0.0, x=10.0, y=6.0, floor=0,
                                floor_confidence=1.0, uncertainty=None,
                                heard=("A", "B", "C"), excluded=())
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "track.json")
            csv_path = os.path.join(tmp, "track.csv")
            export_track_json([position], json_path)
            export_track_csv([position], csv_path)
            rows = load_track_json(json_path)
            self.assertEqual(load_track_csv(csv_path), rows)
        self.assertIsNone(rows[0]["ellipse_semi_major_m"])
        self.assertIsNone(rows[0]["ellipse_semi_minor_m"])
        self.assertIsNone(rows[0]["ellipse_angle_deg"])

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            track_rows([])
        with self.assertRaises(FileNotFoundError):
            load_track_json("/nonexistent/track.json")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "track.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"format": "not-a-track", "version": 1,
                           "rows": []}, handle)
            with self.assertRaises(ValueError):
                load_track_json(path)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"format": "beacon-live-track", "version": 99,
                           "rows": []}, handle)
            with self.assertRaises(ValueError):
                load_track_json(path)
            csv_path = os.path.join(tmp, "track.csv")
            with open(csv_path, "w", encoding="utf-8") as handle:
                handle.write("t,x\n0.0,1.0\n")
            with self.assertRaises(ValueError):
                load_track_csv(csv_path)

    def test_cli_demo_exports_json_and_csv(self):
        import io

        from beacon.tracking import main

        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "track.json")
            csv_path = os.path.join(tmp, "track.csv")
            out = io.StringIO()
            self.assertEqual(main(argv=["--demo",
                                        "--export-json", json_path,
                                        "--export-csv", csv_path],
                                  output_stream=out), 0)
            self.assertEqual(len(load_track_json(json_path)), 40)
            self.assertEqual(load_track_csv(csv_path),
                             load_track_json(json_path))


class TestWardMapOverlay(unittest.TestCase):
    def _example_rows(self):
        here = os.path.dirname(os.path.abspath(__file__))
        return load_track_json(
            os.path.join(here, "..", "examples", "live-track-export.json"))

    def _row(self, x, y, semi_major, floor=0, t=0.0):
        return {
            "t": t, "x": x, "y": y, "floor": floor,
            "floor_confidence": 1.0,
            "ellipse_semi_major_m": semi_major,
            "ellipse_semi_minor_m": (semi_major or 1.0) / 2.0,
            "ellipse_angle_deg": 30.0,
            "beacons_used": ["A", "B", "C", "D"],
            "beacons_excluded": [], "coverage": 1.0,
        }

    def test_example_export_renders_all_points_and_beacons(self):
        rows = self._example_rows()
        svg = render_map_overlay(rows, BEACONS)
        self.assertIn("<svg", svg)
        self.assertIn("</svg>", svg)
        # One track point per located row, in order.
        self.assertEqual(svg.count('data-index="'), len(rows))
        self.assertEqual(svg.count("<ellipse"), len(rows))
        # Every beacon is labelled on the plan.
        for beacon in BEACONS:
            self.assertIn(f'data-beacon="{beacon.id}"', svg)
            self.assertIn(f">{beacon.id}</text>", svg)
        # Start and end markers, and the legend.
        self.assertIn(">S</text>", svg)
        self.assertIn(">E</text>", svg)
        self.assertIn("Legend", svg)
        self.assertIn("floor 0", svg)
        self.assertIn("40 points", svg)

    def test_csv_export_renders_the_same_track(self):
        here = os.path.dirname(os.path.abspath(__file__))
        csv_rows = load_track_csv(
            os.path.join(here, "..", "examples", "live-track-export.csv"))
        json_svg = render_map_overlay(self._example_rows(), BEACONS)
        csv_svg = render_map_overlay(csv_rows, BEACONS)
        self.assertEqual(json_svg, csv_svg)

    def test_confidence_colour_follows_ellipse_size(self):
        rows = [
            self._row(2.0, 2.0, 1.0, t=0.0),
            self._row(5.0, 5.0, 3.0, t=1.0),
            self._row(8.0, 8.0, 8.0, t=2.0),
            self._row(11.0, 9.0, None, t=3.0),
        ]
        svg = render_map_overlay(rows, BEACONS)
        # Tight, loose, very loose, and no-ellipse each get their
        # own colour on the point markers.
        for colour in ("#2e7d32", "#e6a817", "#c0392b", "#757575"):
            self.assertIn(f'fill="{colour}"', svg)
        # The point with no ellipse has a marker but no ellipse.
        self.assertEqual(svg.count("<ellipse"), 3)
        self.assertEqual(svg.count('data-index="'), 4)

    def test_multi_floor_track_plots_one_floor_only(self):
        rows = [self._row(2.0, 2.0, 1.0, floor=0, t=0.0),
                self._row(3.0, 3.0, 1.0, floor=0, t=1.0),
                self._row(15.0, 9.0, 1.0, floor=1, t=2.0)]
        # Default floor is the most common one (floor 0).
        svg = render_map_overlay(rows, BEACONS)
        self.assertEqual(svg.count('data-index="'), 2)
        self.assertIn("floor 0", svg)
        svg = render_map_overlay(rows, BEACONS, floor=1)
        self.assertEqual(svg.count('data-index="'), 1)
        self.assertIn("floor 1", svg)

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            render_map_overlay([], BEACONS)
        with self.assertRaises(ValueError):
            render_map_overlay([self._row(2.0, 2.0, 1.0)], [])
        with self.assertRaises(ValueError):
            render_map_overlay([self._row(2.0, 2.0, 1.0, floor=0)],
                               BEACONS, floor=1)
        with self.assertRaises(ValueError):
            load_track_for_overlay("track.txt")
        with self.assertRaises(FileNotFoundError):
            load_track_for_overlay("missing.json")

    def test_write_map_overlay_round_trips_through_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "map.svg")
            svg = write_map_overlay(
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "examples", "live-track-export.json"),
                BEACONS, out)
            with open(out, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), svg)
            self.assertEqual(svg.count('data-index="'), 40)

    def test_coverage_heatmap_grids_the_ward(self):
        from beacon import coverage_heatmap, heatmap_summary

        cells = coverage_heatmap(BEACONS)
        # 20 x 12 m ward at 2 m spacing, 1 m off the walls: 10 x 6.
        self.assertEqual(len(cells), 60)
        self.assertTrue(all(cell.semi_major is not None for cell in cells))
        summary = heatmap_summary(cells)
        self.assertEqual(summary.n_cells, 60)
        self.assertEqual(summary.degenerate_cells, 0)
        # The corner layout is thin everywhere, so the mean sits
        # in the loose band and the worst cell is worse than it.
        self.assertGreater(summary.mean_semi_major, 2.0)
        self.assertGreater(summary.worst_semi_major,
                           summary.mean_semi_major)
        self.assertIsNotNone(summary.worst_point)

    def test_coverage_heatmap_rejects_bad_inputs(self):
        from beacon import coverage_heatmap, heatmap_summary

        with self.assertRaises(ValueError):
            coverage_heatmap(BEACONS[:2])
        with self.assertRaises(ValueError):
            coverage_heatmap(BEACONS, spacing=0)
        with self.assertRaises(ValueError):
            heatmap_summary([])

    def test_heatmap_layer_renders_on_the_overlay(self):
        rows = self._example_rows()
        svg = render_map_overlay(rows, BEACONS, heatmap=True)
        # One shaded cell per grid point, under the track.
        self.assertEqual(svg.count("data-heatmap-cell="), 60)
        self.assertIn("Coverage heatmap:", svg)
        self.assertIn("mean 95% semi-major", svg)
        # The track still renders on top of the heatmap.
        self.assertEqual(svg.count('data-index="'), len(rows))
        # Off by default: no heatmap cells, no summary line.
        plain = render_map_overlay(rows, BEACONS)
        self.assertNotIn("data-heatmap-cell=", plain)
        self.assertNotIn("Coverage heatmap:", plain)

    def test_cli_map_heatmap_flag_writes_svg(self):
        from beacon.cli import main

        here = os.path.dirname(os.path.abspath(__file__))
        track = os.path.join(here, "..", "examples",
                             "live-track-export.json")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "map.svg")
            self.assertEqual(main(argv=["--map-overlay", track,
                                       "--map-output", out,
                                       "--map-heatmap"]), 0)
            with open(out, encoding="utf-8") as handle:
                svg = handle.read()
            self.assertIn("<svg", svg)
            self.assertEqual(svg.count("data-heatmap-cell="), 60)
            self.assertIn("Coverage heatmap:", svg)

    def test_cli_map_overlay_flag_writes_svg(self):
        from beacon.cli import main

        here = os.path.dirname(os.path.abspath(__file__))
        track = os.path.join(here, "..", "examples",
                             "live-track-export.json")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "map.svg")
            self.assertEqual(main(argv=["--map-overlay", track,
                                       "--map-output", out]), 0)
            with open(out, encoding="utf-8") as handle:
                svg = handle.read()
            self.assertIn("<svg", svg)
            self.assertEqual(svg.count('data-index="'), 40)


if __name__ == "__main__":
    unittest.main()
