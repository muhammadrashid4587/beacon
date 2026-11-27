"""Tests for beacon: trilateration, RSSI model, engine, simulator."""

import json
import math
import os
import random
import statistics
import tempfile
import unittest

from beacon import (
    BatteryModel,
    Beacon,
    BeaconHealth,
    FingerprintMap,
    FloorEstimate,
    FloorLayoutVerdict,
    KalmanFilter,
    LayoutValidation,
    LayoutScore,
    LivePosition,
    LiveTracker,
    ExcludedBeacon,
    MultiDeviceTracker,
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
    DriftClassification,
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
    multi_track_rows,
    placement_uncertainty,
    plan_deployment,
    PositionUncertainty,
    render_map_overlay,
    render_track_replay,
    rssi_to_distance,
    score_layout,
    simulate_multifloor_walk,
    simulate_walk,
    Survey,
    track_rows,
    TrackReport,
    track_to_dict,
    trilaterate,
    two_floor_walk,
    validate_floor_layouts,
    write_map_overlay,
    write_track_replay,
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

    def test_multifloor_position_accepted(self):
        # The monitor is floor-aware: a (x, y, floor) position is
        # accepted, and a healthy beacon heard through the slab
        # reads ~0 residual once the floor model accounts for the
        # vertical gap and the slab attenuation — on the floor the
        # reading was taken on. The beacon's own floor has no
        # readings, so it has no status there: the windows are per
        # floor, and a cross-floor reading never counts as an
        # own-floor one. A malformed position is still rejected.
        import math

        from beacon.rssi import distance_to_rssi

        monitor = DriftMonitor(BEACONS)
        distance = math.hypot(5.0, 5.0, 3.0)
        expected = distance_to_rssi(distance, -59.0, 2.0) - 15.0
        monitor.observe((5.0, 5.0, 1), {"A": expected})
        self.assertIsNone(monitor.status("A"))
        self.assertIsNone(monitor.alarm("A"))
        self.assertAlmostEqual(
            monitor.status("A", floor=1).mean_residual_db, 0.0,
            places=6)
        with self.assertRaises(ValueError):
            monitor.observe((5.0, 5.0, 0, 0), {"A": -60.0})


class TestDriftClassification(unittest.TestCase):
    """DriftMonitor.classify: sag vs move vs noise vs uncertain.

    Every scenario runs the rectangular ward loop twice (80 steps,
    2 dB noise) so the monitor can match each recent reading
    against the same spot a loop earlier — that revisit delta is
    what separates a draining battery from a moved beacon. The
    expected verdicts below are the measured ones across seeds 1,
    2, 3, 7, 11, including the two cases the classifier gets
    wrong, which are asserted as such rather than hidden.
    """

    def _two_loops(self, seed, **kwargs):
        """Two loops of the default walk, fed to a fresh monitor."""
        loop = rectangular_walk(20.0, 12.0, 10)
        steps = simulate_walk(loop + loop, BEACONS,
                              noise_db=2.0, seed=seed, **kwargs)
        monitor = DriftMonitor(BEACONS)
        for step in steps:
            monitor.feed(step)
        return monitor

    def _two_loops_with(self, seed, beacons, **kwargs):
        """Two loops simulated with ``beacons`` but monitored
        against the calibrated ``BEACONS`` — the moved-beacon
        setup: the simulator knows where the beacon really is,
        the monitor only knows where it should be."""
        loop = rectangular_walk(20.0, 12.0, 10)
        steps = simulate_walk(loop + loop, beacons,
                              noise_db=2.0, seed=seed, **kwargs)
        monitor = DriftMonitor(BEACONS)
        for step in steps:
            monitor.feed(step)
        return monitor

    def test_healthy_walk_classifies_healthy(self):
        for seed in (1, 2, 3, 7, 11):
            monitor = self._two_loops(seed)
            for beacon in BEACONS:
                result = monitor.classify(beacon.id)
                self.assertEqual(result.verdict, "healthy", seed)
                self.assertEqual(result.samples, 20)
                self.assertEqual(result.revisit_pairs, 20)

    def test_slow_battery_sag_classifies_as_battery_sag(self):
        # D drains at 0.3 per hour over 120 s steps: level 1.0 down
        # to 0.20 across the two loops, still heard throughout.
        # The v0.25.0 alarm misses this (mean residual only
        # -2.2 to -4.7 dB, under the 5 dB threshold); the revisit
        # delta catches it on every seed.
        for seed in (1, 2, 3, 7, 11):
            monitor = self._two_loops(
                seed, batteries={"D": BatteryModel(
                    drain_rate_per_hour=0.3)}, step_seconds=120.0)
            result = monitor.classify("D")
            self.assertEqual(result.verdict, "battery-sag", seed)
            self.assertLess(result.mean_residual_db, -1.5)
            self.assertIsNotNone(result.revisit_delta_db)
            self.assertLess(result.revisit_delta_db, -1.0)
            for other in ("A", "B", "C"):
                self.assertEqual(monitor.classify(other).verdict,
                                 "healthy", (seed, other))

    def test_moved_beacon_classifies_as_moved(self):
        # D displaced from (0, 12) to (5, 7): stable at the same
        # spot, wrong by a position-dependent amount.
        moved = [Beacon(b.id, b.x, b.y) for b in BEACONS]
        moved[3] = Beacon("D", 5.0, 7.0)
        for seed in (1, 2, 3, 7):
            monitor = self._two_loops_with(seed, moved)
            result = monitor.classify("D")
            self.assertEqual(result.verdict, "moved", seed)
            self.assertGreater(result.mean_residual_db, 1.5)
            self.assertGreater(result.residual_std_db, 2.5)
            self.assertLess(abs(result.revisit_delta_db), 1.0)
        # Seed 11 used to be the honest miss: the noise itself
        # drifts down over the walk (short-window revisit delta
        # -1.84 dB), so the short window cannot rule a decline out
        # and the moved call does not fire there. Over the longer
        # history the delta settles to -1.00 dB over 41 pairs —
        # inside the same 1.0 dB gate by 0.004 dB, a knife-edge,
        # not a comfortable margin — and the spread over the same
        # history is 2.90 dB around a +3.97 dB mean, so the moved
        # call fires from the long window instead.
        monitor = self._two_loops_with(11, moved)
        result = monitor.classify("D")
        self.assertEqual(result.verdict, "moved")
        self.assertIn("longer", result.reason)

    def test_noisy_beacon_classifies_as_noisy(self):
        for seed in (1, 2, 3, 7, 11):
            loop = rectangular_walk(20.0, 12.0, 10)
            steps = simulate_walk(loop + loop, BEACONS,
                                  noise_db=2.0, seed=seed)
            rng = random.Random(seed + 1000)
            for step in steps:
                step["rssi"]["D"] += rng.gauss(0.0, 5.0)
            monitor = DriftMonitor(BEACONS)
            for step in steps:
                monitor.feed(step)
            result = monitor.classify("D")
            self.assertEqual(result.verdict, "noisy", seed)
            self.assertGreater(result.residual_std_db, 4.0)

    def test_constant_offset_is_uncertain_not_sag(self):
        # A steady -6 dB offset alarms (mean beyond 5 dB) but is
        # steady at revisited spots with no position pattern — a
        # stabilised sag and a miscalibration look identical.
        # Seed 11 used to be the false positive: the noise drifts
        # down -1.84 dB over the walk, which the revisit test reads
        # as a real decline. The sag call now also requires the
        # residual level to have fallen across the walk — the last
        # quarter of the history at least 1.0 dB below the first —
        # and the offset's level only falls 0.75 dB, so it stays
        # uncertain on all five seeds.
        for seed in (1, 2, 3, 7, 11):
            loop = rectangular_walk(20.0, 12.0, 10)
            steps = simulate_walk(loop + loop, BEACONS,
                                  noise_db=2.0, seed=seed)
            for step in steps:
                step["rssi"]["D"] -= 6.0
            monitor = DriftMonitor(BEACONS)
            for step in steps:
                monitor.feed(step)
            result = monitor.classify("D")
            self.assertEqual(result.verdict, "uncertain", seed)
            self.assertIsNotNone(monitor.alarm("D"))
            self.assertIsNotNone(result.level_drop_db)
            self.assertGreater(result.level_drop_db, -1.0, seed)

    def test_level_drop_separates_sag_from_offset(self):
        # The discriminator itself, pinned on the numbers: a real
        # sag's level falls from the first quarter to the last on
        # every seed (the drain is a process), while a steady
        # offset's level stays flat (it is a level, weak from the
        # first reading) — even on seed 11, where the revisit
        # delta fakes a decline. The sag verdicts carry the drop
        # in the classification, so the evidence is visible.
        for seed in (1, 2, 3, 7, 11):
            monitor = self._two_loops(
                seed, batteries={"D": BatteryModel(
                    drain_rate_per_hour=0.3)}, step_seconds=120.0)
            result = monitor.classify("D")
            self.assertEqual(result.verdict, "battery-sag", seed)
            self.assertIsNotNone(result.level_drop_db)
            self.assertLess(result.level_drop_db, -1.0, seed)
            self.assertIn("fell", result.reason, seed)
        loop = rectangular_walk(20.0, 12.0, 10)
        steps = simulate_walk(loop + loop, BEACONS,
                              noise_db=2.0, seed=11)
        for step in steps:
            step["rssi"]["D"] -= 6.0
        monitor = DriftMonitor(BEACONS)
        for step in steps:
            monitor.feed(step)
        result = monitor.classify("D")
        self.assertEqual(result.verdict, "uncertain")
        self.assertAlmostEqual(result.revisit_delta_db, -1.84,
                               places=1)
        self.assertGreater(result.level_drop_db, -1.0)

    def test_small_move_is_at_the_detection_floor(self):
        # A 1.4 m move (D to (1, 11)) leaves a residual pattern
        # about the size of the 2 dB RSSI noise: on the canonical
        # seeds the classifier never invents a cause for it —
        # healthy or uncertain only — but across seeds 1-100 it
        # reads moved on 18, so the detection floor is
        # statistical, not a hard line (see the hundred-seed
        # sweep below).
        moved = [Beacon(b.id, b.x, b.y) for b in BEACONS]
        moved[3] = Beacon("D", 1.0, 11.0)
        for seed in (1, 2, 3, 7, 11):
            monitor = self._two_loops_with(seed, moved)
            verdict = monitor.classify("D").verdict
            self.assertIn(verdict, ("healthy", "uncertain"), seed)

    def test_no_revisit_fallback_catches_fast_sag(self):
        # One loop only, no revisits: the README fast-sag scenario
        # (1.2 per hour, 120 s steps, D dark from step 24). The
        # within-window decline carries the verdict on 4/5 seeds.
        for seed in (1, 2, 3, 7):
            steps = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                                  BEACONS, noise_db=2.0, seed=seed,
                                  batteries={"D": BatteryModel(
                                      drain_rate_per_hour=1.2)},
                                  step_seconds=120.0)
            monitor = DriftMonitor(BEACONS)
            for step in steps:
                monitor.feed(step)
            self.assertEqual(monitor.classify("D").verdict,
                             "battery-sag", seed)
        # Seed 11's decline is buried in the noise on a single
        # pass (slope -0.05 dB per reading) — no signature fires.
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                              BEACONS, noise_db=2.0, seed=11,
                              batteries={"D": BatteryModel(
                                  drain_rate_per_hour=1.2)},
                              step_seconds=120.0)
        monitor = DriftMonitor(BEACONS)
        for step in steps:
            monitor.feed(step)
        self.assertIn(monitor.classify("D").verdict,
                      ("healthy", "uncertain"))

    def test_no_revisit_sag_fallback_needs_level_drop(self):
        # One loop only, no revisits: a steady -6 dB offset reads
        # weak from the first reading and stays flat — a level,
        # not a process — but its within-window slope can still
        # fluke below the trend gate on a single pass. Seed 13
        # slopes -0.202 dB per reading, past the -0.15 gate, and
        # read "battery-sag" before the fix; its level only falls
        # 0.45 dB from the first quarter to the last, so the
        # level-drop gate vetoes it now. The other canonical
        # seeds never fired the trend gate, but the gate is what
        # keeps the offset uncertain on every seed, not luck.
        for seed in (1, 2, 3, 7, 11, 13):
            loop = rectangular_walk(20.0, 12.0, 10)
            steps = simulate_walk(loop, BEACONS, noise_db=2.0,
                                  seed=seed)
            for step in steps:
                step["rssi"]["D"] -= 6.0
            monitor = DriftMonitor(BEACONS)
            for step in steps:
                monitor.feed(step)
            result = monitor.classify("D")
            self.assertEqual(result.verdict, "uncertain", seed)
            self.assertIsNotNone(monitor.alarm("D"), seed)
            self.assertIsNotNone(result.level_drop_db, seed)
        # The pin against the pre-fix behaviour: seed 13's slope
        # really does clear the trend gate, so without the
        # level-drop gate this verdict would be "battery-sag".
        loop = rectangular_walk(20.0, 12.0, 10)
        steps = simulate_walk(loop, BEACONS, noise_db=2.0, seed=13)
        for step in steps:
            step["rssi"]["D"] -= 6.0
        monitor = DriftMonitor(BEACONS)
        for step in steps:
            monitor.feed(step)
        result = monitor.classify("D")
        self.assertLessEqual(result.time_slope_db_per_step, -0.15)
        self.assertGreater(result.level_drop_db, -1.0)
        self.assertEqual(result.verdict, "uncertain")

    def test_no_revisit_sag_fallback_drop_gate_passes_real_sag(self):
        # The gate must not veto a real fast sag: on a single
        # pass the 1.2-per-hour drain's level falls 2.47 to
        # 4.67 dB from the first quarter to the last on seeds 1,
        # 2, 3 and 7 — well past the -1.0 dB gate — so the
        # fallback still fires, and the reason cites the drop it
        # rests on. Seed 3 is the thin one: its drop is only 1.52
        # standard errors from zero, 0.02 past the gate, so a
        # slightly unluckier noise draw could still veto it —
        # the same knife-edge the revisit path carries on seed 11.
        for seed in (1, 2, 3, 7):
            steps = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                                  BEACONS, noise_db=2.0, seed=seed,
                                  batteries={"D": BatteryModel(
                                      drain_rate_per_hour=1.2)},
                                  step_seconds=120.0)
            monitor = DriftMonitor(BEACONS)
            for step in steps:
                monitor.feed(step)
            result = monitor.classify("D")
            self.assertEqual(result.verdict, "battery-sag", seed)
            self.assertIsNotNone(result.level_drop_db, seed)
            self.assertLess(result.level_drop_db, -1.0, seed)
            self.assertIn("level fell", result.reason, seed)

    def test_single_loop_positive_offset_is_uncertain(self):
        # One loop only, no revisits: a steady +6 dB offset reads
        # louder from the first reading, so the no-revisit moved
        # fallback is the only moved path that can fire — and it
        # used to fire on a noise fluke. Seed 31's 20-reading
        # window spreads 2.58 dB, 0.08 dB over the moved gate,
        # around a +6 dB mean, and read "moved" before the fix;
        # its 40-reading history spreads only 2.23 dB, under the
        # 2.34 dB the longer history must clear (2 dB noise plus
        # 1.5 standard errors over 40 readings), so the fallback
        # now needs that agreement and the offset reads
        # "uncertain" — the same guard v0.64.0 gave the revisit
        # path, which the fallback never had.
        for seed in (1, 2, 3, 7, 11, 31):
            loop = rectangular_walk(20.0, 12.0, 10)
            steps = simulate_walk(loop, BEACONS, noise_db=2.0,
                                  seed=seed)
            for step in steps:
                step["rssi"]["D"] += 6.0
            monitor = DriftMonitor(BEACONS)
            for step in steps:
                monitor.feed(step)
            result = monitor.classify("D")
            self.assertEqual(result.verdict, "uncertain", seed)
            self.assertGreater(result.mean_residual_db, 5.0, seed)
            self.assertIsNotNone(monitor.alarm("D"))
            for other in ("A", "B", "C"):
                self.assertEqual(monitor.classify(other).verdict,
                                 "healthy", (seed, other))
        # The pin against the pre-fix behaviour: seed 31's short
        # window really does clear the moved gate, so without the
        # longer-history agreement this verdict would be "moved".
        loop = rectangular_walk(20.0, 12.0, 10)
        steps = simulate_walk(loop, BEACONS, noise_db=2.0, seed=31)
        for step in steps:
            step["rssi"]["D"] += 6.0
        monitor = DriftMonitor(BEACONS)
        for step in steps:
            monitor.feed(step)
        result = monitor.classify("D")
        self.assertGreater(result.residual_std_db, 2.5)
        self.assertEqual(result.verdict, "uncertain")

    def test_single_loop_moved_beacon_still_moved(self):
        # The guard must not veto a real move: on one loop D
        # displaced from (0, 12) to (5, 7) reads louder on
        # average with a position-dependent spread, and the
        # longer 40-reading history agrees on every seed — its
        # spread is 2.53 to 2.92 dB against the 2.34 dB the
        # agreement needs, so the fallback still fires, and the
        # reason cites the longer history it rests on.
        moved = [Beacon(b.id, b.x, b.y) for b in BEACONS]
        moved[3] = Beacon("D", 5.0, 7.0)
        for seed in (1, 2, 3, 7, 11):
            loop = rectangular_walk(20.0, 12.0, 10)
            steps = simulate_walk(loop, moved, noise_db=2.0,
                                  seed=seed)
            monitor = DriftMonitor(BEACONS)
            for step in steps:
                monitor.feed(step)
            result = monitor.classify("D")
            self.assertEqual(result.verdict, "moved", seed)
            self.assertGreater(result.mean_residual_db, 1.5, seed)
            self.assertGreater(result.residual_std_db, 2.5, seed)
            self.assertIn("longer", result.reason, seed)
            for other in ("A", "B", "C"):
                self.assertEqual(monitor.classify(other).verdict,
                                 "healthy", (seed, other))

    def test_long_history_trend_carries_fallback_sag(self):
        # The fallback's trend gate used to read the 20-reading
        # window alone. On seeds 6 and 24 the short window's
        # slope (-0.097 and -0.062 dB per reading) does not clear
        # the -0.15 gate, but the decline shows across the longer
        # 24-reading history (-0.169 and -0.177) — the beacon goes
        # dark at step 24, so the long window is all the readings
        # there are — and the drop clears its gate too, so the
        # fallback fires from the longer history and the reason
        # says so.
        for seed in (6, 24):
            steps = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                                  BEACONS, noise_db=2.0, seed=seed,
                                  batteries={"D": BatteryModel(
                                      drain_rate_per_hour=1.2)},
                                  step_seconds=120.0)
            monitor = DriftMonitor(BEACONS)
            for step in steps:
                monitor.feed(step)
            result = monitor.classify("D")
            self.assertEqual(result.verdict, "battery-sag", seed)
            self.assertGreater(result.time_slope_db_per_step, -0.15,
                               seed)
            self.assertIn("longer", result.reason, seed)
            self.assertIn("level fell", result.reason, seed)

    def test_long_sag_beats_noisy_on_slow_drain(self):
        # The noisy check used to run before the long-history sag
        # check, so a slow drain whose window spread crossed 4 dB
        # read "noisy" even though the long history proved the
        # decline. Seeds 10, 25, 40 and 42 are the ones the
        # reorder rescues: their long revisit deltas (-1.11 to
        # -1.82 dB over 41 pairs) and level drops (-1.93 to
        # -2.69 dB) clear the sag gates, and their long means sit
        # far enough below zero — so they read battery-sag now,
        # from the longer history.
        for seed in (10, 25, 40, 42):
            monitor = self._two_loops(
                seed, batteries={"D": BatteryModel(
                    drain_rate_per_hour=0.3)}, step_seconds=120.0)
            result = monitor.classify("D")
            self.assertEqual(result.verdict, "battery-sag", seed)
            self.assertIn("longer", result.reason, seed)

    def test_noisy_beacon_with_fluked_decline_stays_healthy(self):
        # The reorder's guard: seed 48's noisy beacon flukes a
        # hard decline over the long history — revisit delta
        # -4.17 dB over 41 pairs (1.27 dB standard error), level
        # down -4.35 dB at -3.29 standard errors — but its long
        # mean is only -0.16 dB, near zero, so the long-mean gate
        # (1.5 standard errors, 0.98 dB) blocks the sag route and
        # it does not read battery-sag. Its window spread
        # (3.83 dB) also sits under the noisy gate, so it reads
        # healthy — the honest miss for a noisy beacon on one
        # short window.
        loop = rectangular_walk(20.0, 12.0, 10)
        steps = simulate_walk(loop + loop, BEACONS, noise_db=2.0,
                              seed=48)
        rng = random.Random(48 + 1000)
        for step in steps:
            step["rssi"]["D"] += rng.gauss(0.0, 5.0)
        monitor = DriftMonitor(BEACONS)
        for step in steps:
            monitor.feed(step)
        result = monitor.classify("D")
        self.assertEqual(result.verdict, "healthy")
        self.assertLess(result.residual_std_db, 4.0)

    def test_hundred_seed_sweep_pins_verdict_counts(self):
        # The v0.67.0 measurement: every classification scenario
        # swept over seeds 1-100, pinning the exact tallies. The
        # canonical five seeds flatter the classifier — the
        # single-loop fast sag reads battery-sag on 4/5 of them
        # but only 50/100, the slow sag 38/50 on seeds 1-50 but
        # 84/100, and the 1.4 m move reads moved on 18/100 — so
        # a future change that shifts these counts must update
        # this test and the README deliberately, not silently.
        moved = [Beacon(b.id, b.x, b.y) for b in BEACONS]
        moved[3] = Beacon("D", 5.0, 7.0)
        small_moved = [Beacon(b.id, b.x, b.y) for b in BEACONS]
        small_moved[3] = Beacon("D", 1.0, 11.0)

        def tally(build):
            counts = {}
            for seed in range(1, 101):
                verdict = build(seed).classify("D").verdict
                counts[verdict] = counts.get(verdict, 0) + 1
            return counts

        def one_loop(seed, beacons=None, **kwargs):
            loop = rectangular_walk(20.0, 12.0, 10)
            steps = simulate_walk(loop, beacons or BEACONS,
                                  noise_db=2.0, seed=seed, **kwargs)
            monitor = DriftMonitor(BEACONS)
            for step in steps:
                monitor.feed(step)
            return monitor

        def offset(seed, delta_db, loops=2):
            loop = rectangular_walk(20.0, 12.0, 10)
            path = loop + loop if loops == 2 else loop
            steps = simulate_walk(path, BEACONS, noise_db=2.0,
                                  seed=seed)
            for step in steps:
                step["rssi"]["D"] += delta_db
            monitor = DriftMonitor(BEACONS)
            for step in steps:
                monitor.feed(step)
            return monitor

        def noisy(seed):
            loop = rectangular_walk(20.0, 12.0, 10)
            steps = simulate_walk(loop + loop, BEACONS,
                                  noise_db=2.0, seed=seed)
            rng = random.Random(seed + 1000)
            for step in steps:
                step["rssi"]["D"] += rng.gauss(0.0, 5.0)
            monitor = DriftMonitor(BEACONS)
            for step in steps:
                monitor.feed(step)
            return monitor

        slow_sag = {"D": BatteryModel(drain_rate_per_hour=0.3)}
        fast_sag = {"D": BatteryModel(drain_rate_per_hour=1.2)}
        self.assertEqual(
            tally(lambda s: self._two_loops(s)), {"healthy": 100})
        self.assertEqual(
            tally(lambda s: self._two_loops(
                s, batteries=slow_sag, step_seconds=120.0)),
            {"battery-sag": 84, "noisy": 11, "healthy": 5})
        self.assertEqual(
            tally(lambda s: self._two_loops_with(s, moved)),
            {"moved": 98, "uncertain": 2})
        self.assertEqual(
            tally(lambda s: self._two_loops_with(s, small_moved)),
            {"uncertain": 64, "healthy": 18, "moved": 18})
        self.assertEqual(tally(noisy),
                         {"noisy": 91, "healthy": 8, "moved": 1})
        self.assertEqual(
            tally(lambda s: offset(s, -6.0)),
            {"uncertain": 99, "moved": 1})
        self.assertEqual(
            tally(lambda s: one_loop(
                s, batteries=fast_sag, step_seconds=120.0)),
            {"battery-sag": 50, "healthy": 29, "noisy": 20,
             "uncertain": 1})
        self.assertEqual(tally(lambda s: one_loop(s)),
                         {"healthy": 100})
        self.assertEqual(
            tally(lambda s: one_loop(s, beacons=moved)),
            {"moved": 91, "uncertain": 8, "noisy": 1})
        self.assertEqual(
            tally(lambda s: offset(s, -6.0, loops=1)),
            {"uncertain": 98, "battery-sag": 2})
        self.assertEqual(
            tally(lambda s: offset(s, 6.0, loops=1)),
            {"uncertain": 100})

    def test_other_beacons_stay_healthy_across_hundred_seeds(self):
        # Whatever D does on any seed, the other three beacons
        # stay healthy — the classifier never blames the wrong
        # beacon for D's fault, on any of the 100 seeds, in any
        # of the two-loop scenarios.
        moved = [Beacon(b.id, b.x, b.y) for b in BEACONS]
        moved[3] = Beacon("D", 5.0, 7.0)
        slow_sag = {"D": BatteryModel(drain_rate_per_hour=0.3)}
        scenarios = [
            lambda s: self._two_loops(s),
            lambda s: self._two_loops(
                s, batteries=slow_sag, step_seconds=120.0),
            lambda s: self._two_loops_with(s, moved),
        ]
        for build in scenarios:
            for seed in range(1, 101):
                monitor = build(seed)
                for other in ("A", "B", "C"):
                    self.assertEqual(
                        monitor.classify(other).verdict, "healthy",
                        (seed, other))

    def test_fixed_position_decline_classifies_as_sag(self):
        # No walk at all: a reference tag at one fixed spot hears
        # D fade 0.3 dB per reading. Revisits are exact (same spot
        # every reading), so the delta is unambiguous.
        monitor = DriftMonitor(BEACONS)
        for i in range(20):
            rssi = {}
            for beacon in BEACONS:
                distance = math.hypot(10.0 - beacon.x, 6.0 - beacon.y)
                rssi[beacon.id] = distance_to_rssi(distance)
            rssi["D"] -= 0.3 * i
            monitor.observe((10.0, 6.0), rssi)
        result = monitor.classify("D")
        self.assertEqual(result.verdict, "battery-sag")
        # Each window reading pairs with the earliest reading at
        # the same spot (10+ readings back): readings 11..20 pair
        # with reading 1, so the deltas run -3.0 to -5.7 dB.
        self.assertEqual(result.revisit_pairs, 10)
        self.assertAlmostEqual(result.revisit_delta_db, -4.35, delta=0.01)
        self.assertEqual(monitor.classify("A").verdict, "healthy")

    def test_unheard_beacon_is_uncertain(self):
        monitor = DriftMonitor(BEACONS)
        result = monitor.classify("A")
        self.assertEqual(result.verdict, "uncertain")
        self.assertEqual(result.samples, 0)
        self.assertIsNone(result.revisit_delta_db)
        self.assertEqual(result.revisit_pairs, 0)

    def test_too_few_readings_is_uncertain(self):
        monitor = DriftMonitor(BEACONS)
        steps = simulate_walk([(10.0, 6.0)] * 3, BEACONS,
                              noise_db=0.0, seed=7)
        for step in steps:
            monitor.feed(step)
        result = monitor.classify("A")
        self.assertEqual(result.verdict, "uncertain")
        self.assertEqual(result.samples, 3)

    def test_unknown_beacon_classify_raises(self):
        monitor = DriftMonitor(BEACONS)
        with self.assertRaises(ValueError):
            monitor.classify("Z")

    def test_bad_parameters_raise(self):
        with self.assertRaises(ValueError):
            DriftMonitor(BEACONS, noise_db=0)
        with self.assertRaises(ValueError):
            DriftMonitor(BEACONS, history=10)
        with self.assertRaises(ValueError):
            DriftMonitor(BEACONS, revisit_gap=0)
        with self.assertRaises(ValueError):
            DriftMonitor(BEACONS, revisit_radius_m=0)

    def test_classifications_returns_all_beacons_in_order(self):
        monitor = self._two_loops(7)
        results = monitor.classifications()
        self.assertEqual([r.id for r in results], ["A", "B", "C", "D"])
        self.assertTrue(all(isinstance(r, DriftClassification)
                            for r in results))
        self.assertTrue(all(r.verdict == "healthy" for r in results))

    def test_cli_drift_classify_flags(self):
        import io
        from contextlib import redirect_stdout
        from beacon.cli import main

        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["--drift-classify", "--seed", "7"]), 0)
        self.assertIn("healthy", out.getvalue())

        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["--drift-classify", "--seed", "7",
                                   "--battery-degrade", "D"]), 0)
        self.assertIn("battery-sag", out.getvalue())

        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["--drift-classify", "--seed", "7",
                                   "--move-beacon", "D:5,-5"]), 0)
        self.assertIn("moved", out.getvalue())

        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["--drift-classify", "--seed", "7",
                                   "--battery-degrade", "D",
                                   "--loops", "3"]), 0)
        self.assertIn("battery-sag", out.getvalue())

        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["--drift-classify", "--seed", "7",
                                   "--loops", "1"]), 0)
        self.assertIn("healthy", out.getvalue())

        with self.assertRaises(SystemExit):
            main(["--battery-degrade", "D"])
        with self.assertRaises(SystemExit):
            main(["--move-beacon", "D:5,-5"])
        self.assertEqual(main(["--drift-classify", "--move-beacon",
                               "D:bad"]), 2)


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
                                 "beacons_excluded,coverage,battery,"
                                 "forecasts,forecast_alerts,"
                                 "drift_alarms,drift_alerts,"
                                 "drift_classifications,device")

    def _planned_track(self):
        """Feed the default walk through a tracker with a
        deployment plan: measured levels 0.9, 0.6, 0.4, 0.2, all
        draining at 0.05/hr, so D is already past its 3 dB loss
        milestone and every beacon alerts inside 24 hours."""
        from beacon import plan_deployment

        plan = plan_deployment(
            {b.id: BatteryModel(drain_rate_per_hour=0.05)
             for b in BEACONS},
            levels={"A": 0.9, "B": 0.6, "C": 0.4, "D": 0.2})
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                              BEACONS, noise_db=2.0, seed=7)
        tracker = LiveTracker(PositioningEngine(BEACONS), plan=plan)
        return [tracker.feed(step["rssi"], float(i))
                for i, step in enumerate(steps)]

    def test_forecast_flags_ride_on_export_rows_and_round_trip(self):
        positions = self._planned_track()
        rows = track_rows(positions)
        self.assertEqual(len(rows), 40)
        by_id = {flag["beacon"]: flag
                 for flag in rows[0]["forecasts"]}
        self.assertEqual(set(by_id), {"A", "B", "C", "D"})
        self.assertEqual(by_id["D"]["milestone"], "3 dB loss")
        self.assertEqual(by_id["D"]["hours"], 0.0)
        self.assertEqual(by_id["D"]["status"], "degrading")
        self.assertAlmostEqual(by_id["C"]["hours"], 1.3, places=1)
        self.assertAlmostEqual(by_id["B"]["hours"], 5.3, places=1)
        self.assertAlmostEqual(by_id["A"]["hours"], 11.3, places=1)
        # Every beacon is due inside the 24-hour window.
        self.assertEqual(len(rows[0]["forecast_alerts"]), 4)
        self.assertIn("D: 3 dB loss now (degrading)",
                      rows[0]["forecast_alerts"])
        # The flags count down with the track, not frozen at the
        # plan: the last row still carries all four, D still due.
        last = {flag["beacon"]: flag
                for flag in rows[-1]["forecasts"]}
        self.assertEqual(last["D"]["milestone"], "3 dB loss")
        self.assertEqual(last["D"]["hours"], 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "track.json")
            csv_path = os.path.join(tmp, "track.csv")
            export_track_json(positions, json_path)
            export_track_csv(positions, csv_path)
            self.assertEqual(load_track_json(json_path), rows)
            self.assertEqual(load_track_csv(csv_path), rows)

    def test_track_without_plan_exports_empty_forecasts(self):
        rows = track_rows(self._track())
        self.assertEqual(rows[0]["forecasts"], [])
        self.assertEqual(rows[0]["forecast_alerts"], [])

    def _drifted_track(self, offset=12.0):
        """Feed the default walk through a tracker with a drift
        monitor, with beacon A reading ``offset`` dB too loud the
        whole walk, so A alarms from step 4 (once the monitor has
        its 5 readings) at a rolling mean near +8 dB — attenuated
        from +12 dB because the estimate is pulled towards A."""
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                              BEACONS, noise_db=2.0, seed=7)
        tracker = LiveTracker(PositioningEngine(BEACONS),
                              drift_monitor=DriftMonitor(BEACONS))
        positions = []
        for i, step in enumerate(steps):
            rssi = dict(step["rssi"])
            rssi["A"] = rssi["A"] + offset
            positions.append(tracker.feed(rssi, float(i)))
        return positions

    def test_drift_alarms_ride_on_export_rows_and_round_trip(self):
        positions = self._drifted_track()
        rows = track_rows(positions)
        self.assertEqual(len(rows), 40)
        # The monitor needs 5 readings before it can alarm, so
        # the first rows carry no alarms.
        self.assertEqual(rows[0]["drift_alarms"], [])
        self.assertEqual(rows[0]["drift_alerts"], [])
        self.assertEqual(rows[3]["drift_alarms"], [])
        # From step 4 A alarms, on the right beacon, at a mean
        # attenuated from +12 dB by the pull on the estimate.
        alarm = next(a for a in rows[5]["drift_alarms"]
                     if a["id"] == "A")
        self.assertGreaterEqual(alarm["mean_residual_db"], 7.0)
        self.assertLessEqual(alarm["mean_residual_db"], 9.5)
        self.assertGreaterEqual(alarm["samples"], 5)
        self.assertTrue(any(a.startswith("A: RSSI +")
                            and a.endswith("dB off calibration")
                            for a in rows[5]["drift_alerts"]))
        # The alarm is still on A at the end of the walk, and
        # only on A.
        last = rows[-1]["drift_alarms"]
        self.assertEqual([a["id"] for a in last], ["A"])
        self.assertGreaterEqual(last[0]["mean_residual_db"], 7.0)
        self.assertLessEqual(last[0]["mean_residual_db"], 9.5)
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "track.json")
            csv_path = os.path.join(tmp, "track.csv")
            export_track_json(positions, json_path)
            export_track_csv(positions, csv_path)
            self.assertEqual(load_track_json(json_path), rows)
            self.assertEqual(load_track_csv(csv_path), rows)

    def test_track_without_monitor_exports_empty_drift_alarms(self):
        rows = track_rows(self._track())
        self.assertEqual(rows[0]["drift_alarms"], [])
        self.assertEqual(rows[0]["drift_alerts"], [])
        self.assertEqual(rows[-1]["drift_alarms"], [])
        self.assertEqual(rows[-1]["drift_alerts"], [])

    def test_drift_classifications_ride_on_export_rows_and_round_trip(self):
        positions = self._drifted_track()
        rows = track_rows(positions)
        self.assertEqual(len(rows), 40)
        # Every row carries one classification per beacon, in
        # beacon order, even before the monitor has enough
        # readings — those early verdicts are "uncertain", not
        # missing.
        self.assertEqual([c["id"] for c in
                          rows[0]["drift_classifications"]],
                         ["A", "B", "C", "D"])
        self.assertEqual({c["verdict"] for c in
                          rows[0]["drift_classifications"]},
                         {"uncertain"})
        # From step 4 the monitor has its 5 readings: A
        # classifies "moved" at a mean near +9 dB — attenuated
        # from +12 dB by the pull on the estimate — while the
        # other three stay healthy.
        by_id = {c["id"]: c
                 for c in rows[4]["drift_classifications"]}
        self.assertEqual(by_id["A"]["verdict"], "moved")
        self.assertGreaterEqual(by_id["A"]["mean_residual_db"], 8.0)
        self.assertLessEqual(by_id["A"]["mean_residual_db"], 10.0)
        self.assertEqual(by_id["A"]["samples"], 5)
        self.assertEqual(by_id["B"]["verdict"], "healthy")
        self.assertEqual(by_id["C"]["verdict"], "healthy")
        self.assertEqual(by_id["D"]["verdict"], "healthy")
        # The last row still carries all four, and the revisit
        # delta is null — this walk revisits too few spots to
        # measure one — so the null round-trips through both
        # formats.
        last = {c["id"]: c
                for c in rows[-1]["drift_classifications"]}
        self.assertEqual(last["A"]["verdict"], "uncertain")
        self.assertGreaterEqual(last["A"]["mean_residual_db"], 7.0)
        self.assertLessEqual(last["A"]["mean_residual_db"], 9.5)
        self.assertIsNone(last["A"]["revisit_delta_db"])
        self.assertLess(last["A"]["revisit_pairs"], 5)
        # C inherits a false "moved" verdict on the last row:
        # the estimate dragged towards loud A makes C's
        # residuals spread around a shifted mean — the
        # self-referenced monitor's verdict is a hint, not a
        # diagnosis.
        self.assertEqual(last["C"]["verdict"], "moved")
        # The live position's own dict carries them too, for the
        # stream.
        self.assertEqual(
            positions[-1].to_dict()["drift_classifications"],
            rows[-1]["drift_classifications"])
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "track.json")
            csv_path = os.path.join(tmp, "track.csv")
            export_track_json(positions, json_path)
            export_track_csv(positions, csv_path)
            self.assertEqual(load_track_json(json_path), rows)
            self.assertEqual(load_track_csv(csv_path), rows)

    def test_track_without_monitor_exports_empty_drift_classifications(self):
        rows = track_rows(self._track())
        self.assertEqual(rows[0]["drift_classifications"], [])
        self.assertEqual(rows[-1]["drift_classifications"], [])
        # A healthy monitored track classifies every beacon
        # healthy by the end of the walk, not uncertain.
        healthy = track_rows(self._drifted_track(offset=0.0))
        self.assertEqual({c["verdict"] for c in
                          healthy[-1]["drift_classifications"]},
                         {"healthy"})

    def test_old_csv_without_drift_classification_column_still_loads(self):
        # A sixteen-column export (drift alarms, no
        # classifications) loads with empty drift
        # classifications, like the pre-drift example file loads
        # with empty drift alarms.
        import csv

        rows = track_rows(self._drifted_track())
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "old.csv")
            with open(csv_path, "w", encoding="utf-8",
                      newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["t", "x", "y", "floor",
                                "floor_confidence",
                                "ellipse_semi_major_m",
                                "ellipse_semi_minor_m",
                                "ellipse_angle_deg", "beacons_used",
                                "beacons_excluded", "coverage",
                                "battery", "forecasts",
                                "forecast_alerts", "drift_alarms",
                                "drift_alerts"])
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        "t": repr(row["t"]), "x": repr(row["x"]),
                        "y": repr(row["y"]),
                        "floor": str(row["floor"]),
                        "floor_confidence":
                            repr(row["floor_confidence"]),
                        "ellipse_semi_major_m":
                            repr(row["ellipse_semi_major_m"]),
                        "ellipse_semi_minor_m":
                            repr(row["ellipse_semi_minor_m"]),
                        "ellipse_angle_deg":
                            repr(row["ellipse_angle_deg"]),
                        "beacons_used": ";".join(row["beacons_used"]),
                        "beacons_excluded":
                            json.dumps(row["beacons_excluded"]),
                        "coverage": repr(row["coverage"]),
                        "battery": ";".join(
                            f"{k}={v!r}" for k, v in
                            sorted(row["battery"].items())),
                        "forecasts": json.dumps(row["forecasts"]),
                        "forecast_alerts":
                            ";".join(row["forecast_alerts"]),
                        "drift_alarms":
                            json.dumps(row["drift_alarms"]),
                        "drift_alerts":
                            ";".join(row["drift_alerts"]),
                    })
            loaded = load_track_csv(csv_path)
        self.assertEqual(len(loaded), 40)
        self.assertEqual(loaded[0]["drift_classifications"], [])
        self.assertEqual(loaded[-1]["drift_classifications"], [])
        # The alarms column still parses — A alarms on the last
        # row.
        self.assertEqual([a["id"] for a in
                          loaded[-1]["drift_alarms"]], ["A"])

    def test_old_json_without_drift_classification_key_still_loads(self):
        from beacon.export import track_from_dict, track_to_dict

        data = track_to_dict(self._drifted_track())
        for row in data["rows"]:
            del row["drift_classifications"]
        rows = track_from_dict(data)
        self.assertEqual(len(rows), 40)
        self.assertEqual(rows[0]["drift_classifications"], [])
        self.assertEqual(rows[-1]["drift_classifications"], [])
        self.assertEqual([a["id"] for a in
                          rows[-1]["drift_alarms"]], ["A"])

    def test_multi_device_classifications_are_per_device(self):
        # tag-1's walk has A reading 12 dB too loud; tag-2's is
        # healthy. Each device's rows carry only its own
        # monitor's verdicts — tag-2 never inherits tag-1's
        # drift.
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                              BEACONS, noise_db=2.0, seed=7)
        tracker = MultiDeviceTracker(
            BEACONS, drift_monitor=DriftMonitor(BEACONS))
        per_device = {"tag-1": [], "tag-2": []}
        for i, step in enumerate(steps):
            for device_id in ("tag-1", "tag-2"):
                rssi = dict(step["rssi"])
                if device_id == "tag-1":
                    rssi["A"] = rssi["A"] + 12.0
                per_device[device_id].append(
                    tracker.feed(device_id, rssi, float(i)))
        rows_1 = track_rows(per_device["tag-1"])
        rows_2 = track_rows(per_device["tag-2"])
        self.assertEqual(len(rows_1), 40)
        self.assertEqual(len(rows_2), 40)
        last_1 = {c["id"]: c for c in
                  rows_1[-1]["drift_classifications"]}
        self.assertEqual(last_1["A"]["verdict"], "uncertain")
        self.assertGreaterEqual(last_1["A"]["mean_residual_db"], 7.0)
        self.assertEqual({c["verdict"] for c in
                          rows_2[-1]["drift_classifications"]},
                         {"healthy"})
        # tag-2's accessor agrees with its rows.
        self.assertEqual(
            [c.verdict for c in
             tracker.drift_classifications("tag-2")],
            ["healthy"] * 4)

    def test_old_csv_without_drift_columns_still_loads(self):
        # A fourteen-column export (forecasts, no drift alarms)
        # loads with empty drift alarms and alerts, like the
        # pre-forecast example file loads with empty forecasts.
        import csv

        rows = track_rows(self._drifted_track())
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "old.csv")
            with open(csv_path, "w", encoding="utf-8",
                      newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["t", "x", "y", "floor",
                                "floor_confidence",
                                "ellipse_semi_major_m",
                                "ellipse_semi_minor_m",
                                "ellipse_angle_deg", "beacons_used",
                                "beacons_excluded", "coverage",
                                "battery", "forecasts",
                                "forecast_alerts"])
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        "t": repr(row["t"]), "x": repr(row["x"]),
                        "y": repr(row["y"]),
                        "floor": str(row["floor"]),
                        "floor_confidence":
                            repr(row["floor_confidence"]),
                        "ellipse_semi_major_m":
                            repr(row["ellipse_semi_major_m"]),
                        "ellipse_semi_minor_m":
                            repr(row["ellipse_semi_minor_m"]),
                        "ellipse_angle_deg":
                            repr(row["ellipse_angle_deg"]),
                        "beacons_used": ";".join(row["beacons_used"]),
                        "beacons_excluded":
                            json.dumps(row["beacons_excluded"]),
                        "coverage": repr(row["coverage"]),
                        "battery": ";".join(
                            f"{k}={v!r}" for k, v in
                            sorted(row["battery"].items())),
                        "forecasts": json.dumps(row["forecasts"]),
                        "forecast_alerts":
                            ";".join(row["forecast_alerts"]),
                    })
            loaded = load_track_csv(csv_path)
        self.assertEqual(len(loaded), 40)
        self.assertEqual(loaded[0]["drift_alarms"], [])
        self.assertEqual(loaded[0]["drift_alerts"], [])
        # The alarm itself is gone with the columns, but the
        # forecasts column still parses — empty here, since the
        # drifted track was recorded without a plan.
        self.assertEqual(loaded[0]["forecasts"], [])
        self.assertEqual(loaded[0]["forecast_alerts"], [])

    def test_old_json_without_drift_keys_still_loads(self):
        from beacon.export import track_from_dict, track_to_dict

        data = track_to_dict(self._drifted_track())
        for row in data["rows"]:
            del row["drift_alarms"]
            del row["drift_alerts"]
        rows = track_from_dict(data)
        self.assertEqual(len(rows), 40)
        self.assertEqual(rows[0]["drift_alarms"], [])
        self.assertEqual(rows[0]["drift_alerts"], [])
        self.assertEqual(rows[-1]["drift_alarms"], [])
        self.assertEqual(rows[-1]["drift_alerts"], [])

    def test_never_draining_forecast_round_trips_through_inf(self):
        # A beacon that never drains writes null horizons and
        # reads them back as infinity, in both formats.
        from beacon import ForecastFlag

        position = LivePosition(
            t=0.0, x=10.0, y=6.0, floor=0, floor_confidence=1.0,
            uncertainty=None, heard=("A", "B", "C"), excluded=(),
            forecasts=(ForecastFlag(
                beacon_id="A", milestone="never", hours=math.inf,
                status="ok", hours_to_dark=math.inf,
                hours_to_rssi_loss_db=math.inf,
                hours_to_extra_noise_db=math.inf),),
            forecast_alerts=())
        rows = track_rows([position])
        self.assertIsNone(rows[0]["forecasts"][0]["hours"])
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "track.json")
            csv_path = os.path.join(tmp, "track.csv")
            export_track_json([position], json_path)
            export_track_csv([position], csv_path)
            loaded = load_track_json(json_path)
            self.assertEqual(load_track_csv(csv_path), loaded)
        flag = loaded[0]["forecasts"][0]
        self.assertEqual(flag["milestone"], "never")
        self.assertTrue(math.isinf(flag["hours"]))
        self.assertTrue(math.isinf(flag["hours_to_dark"]))

    def test_old_csv_without_forecast_columns_still_loads(self):
        # A twelve-column export (battery, no forecasts) loads
        # with empty forecasts and alerts, like the pre-battery
        # example file loads with an empty battery.
        import csv

        rows = track_rows(self._planned_track())
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "old.csv")
            with open(csv_path, "w", encoding="utf-8",
                      newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["t", "x", "y", "floor",
                                "floor_confidence",
                                "ellipse_semi_major_m",
                                "ellipse_semi_minor_m",
                                "ellipse_angle_deg", "beacons_used",
                                "beacons_excluded", "coverage",
                                "battery"])
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        "t": repr(row["t"]), "x": repr(row["x"]),
                        "y": repr(row["y"]),
                        "floor": str(row["floor"]),
                        "floor_confidence":
                            repr(row["floor_confidence"]),
                        "ellipse_semi_major_m":
                            repr(row["ellipse_semi_major_m"]),
                        "ellipse_semi_minor_m":
                            repr(row["ellipse_semi_minor_m"]),
                        "ellipse_angle_deg":
                            repr(row["ellipse_angle_deg"]),
                        "beacons_used": ";".join(row["beacons_used"]),
                        "beacons_excluded":
                            json.dumps(row["beacons_excluded"]),
                        "coverage": repr(row["coverage"]),
                        "battery": ";".join(
                            f"{k}={v!r}" for k, v in
                            sorted(row["battery"].items())),
                    })
            loaded = load_track_csv(csv_path)
        self.assertEqual(len(loaded), 40)
        self.assertEqual(loaded[0]["forecasts"], [])
        self.assertEqual(loaded[0]["forecast_alerts"], [])

    def test_old_json_without_forecast_keys_still_loads(self):
        from beacon.export import track_from_dict, track_to_dict

        data = track_to_dict(self._planned_track())
        for row in data["rows"]:
            del row["forecasts"]
            del row["forecast_alerts"]
        rows = track_from_dict(data)
        self.assertEqual(len(rows), 40)
        self.assertEqual(rows[0]["forecasts"], [])
        self.assertEqual(rows[0]["forecast_alerts"], [])

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


class TestPlannerMapOverlay(unittest.TestCase):
    def _example_rows(self):
        here = os.path.dirname(os.path.abspath(__file__))
        return load_track_json(
            os.path.join(here, "..", "examples", "live-track-export.json"))

    def _plan(self, levels=None, drain_rate=0.05):
        from beacon import plan_deployment

        return plan_deployment(
            {b.id: BatteryModel(drain_rate_per_hour=drain_rate)
             for b in BEACONS},
            levels=levels)

    def test_planner_heatmap_grids_the_ward(self):
        from beacon import planner_heatmap, planner_heatmap_summary

        # Four fresh beacons at 0.05 per hour: the 2 dB noise
        # milestone comes first for every one, at 13.33 h, so
        # every cell's horizon is that — the ward degrades evenly.
        cells = planner_heatmap(BEACONS, self._plan())
        self.assertEqual(len(cells), 60)
        for cell in cells:
            self.assertAlmostEqual(cell.hours_to_degrade, 13.3333,
                                   places=3)
            self.assertEqual(cell.limiting_milestone, "extra_noise")
        summary = planner_heatmap_summary(cells)
        self.assertEqual(summary.n_cells, 60)
        self.assertEqual(summary.never_degrade_cells, 0)
        self.assertAlmostEqual(summary.earliest_hours, 13.3333,
                               places=3)
        self.assertEqual(summary.earliest_milestone, "extra_noise")

    def test_planner_heatmap_uneven_levels(self):
        from beacon import planner_heatmap, planner_heatmap_summary

        # D at 20% is already past 3 dB of loss, so the cells near
        # it degrade now; the far corner holds until C's noise
        # milestone at 1.33 h.
        plan = self._plan(levels={"A": 0.9, "B": 0.6,
                                  "C": 0.4, "D": 0.2})
        cells = planner_heatmap(BEACONS, plan)
        by_point = {(cell.x, cell.y): cell for cell in cells}
        near_d = by_point[(1.0, 11.0)]
        self.assertEqual(near_d.hours_to_degrade, 0.0)
        self.assertEqual(near_d.limiting_beacon, "D")
        self.assertEqual(near_d.limiting_milestone, "rssi_loss")
        far_corner = by_point[(19.0, 1.0)]
        self.assertAlmostEqual(far_corner.hours_to_degrade, 1.3333,
                               places=3)
        self.assertEqual(far_corner.limiting_beacon, "C")
        self.assertEqual(far_corner.limiting_milestone, "extra_noise")
        summary = planner_heatmap_summary(cells)
        self.assertEqual(summary.earliest_hours, 0.0)
        self.assertEqual(summary.earliest_point, (1.0, 1.0))
        self.assertEqual(summary.earliest_beacon, "D")
        self.assertEqual(summary.earliest_milestone, "rssi_loss")
        self.assertEqual(summary.never_degrade_cells, 0)

    def test_planner_heatmap_never_degrades_without_drain(self):
        from beacon import planner_heatmap, planner_heatmap_summary

        cells = planner_heatmap(BEACONS, self._plan(drain_rate=0.0))
        self.assertTrue(all(cell.hours_to_degrade is None
                            for cell in cells))
        self.assertTrue(all(cell.limiting_beacon is None
                            for cell in cells))
        summary = planner_heatmap_summary(cells)
        self.assertIsNone(summary.earliest_hours)
        self.assertIsNone(summary.earliest_point)
        self.assertEqual(summary.never_degrade_cells, 60)

    def test_planner_colour_bands(self):
        from beacon import planner_colour

        self.assertEqual(planner_colour(2.0), "#c0392b")
        self.assertEqual(planner_colour(8.0), "#e6a817")
        self.assertEqual(planner_colour(20.0), "#2e7d32")
        self.assertEqual(planner_colour(None), "#00897b")

    def test_planner_heatmap_rejects_bad_inputs(self):
        from beacon import (planner_heatmap, planner_heatmap_summary)

        with self.assertRaises(ValueError):
            planner_heatmap(BEACONS[:2], self._plan())
        with self.assertRaises(ValueError):
            planner_heatmap(BEACONS, self._plan(), spacing=0)
        # A beacon in the layout with no forecast in the plan is
        # a gap in the plan, not a stable beacon.
        from beacon import plan_deployment

        partial = plan_deployment(
            {b.id: BatteryModel(drain_rate_per_hour=0.05)
             for b in BEACONS[:3]})
        with self.assertRaises(ValueError):
            planner_heatmap(BEACONS, partial)
        with self.assertRaises(ValueError):
            planner_heatmap_summary([])

    def test_overlay_with_plan_annotates_beacons(self):
        rows = self._example_rows()
        # D measured at 4% charge is already dark.
        plan = self._plan(levels={"D": 0.04})
        svg = render_map_overlay(rows, BEACONS, plan=plan)
        # Every beacon carries its forecast horizon; D is dark now
        # and its diamond is red, while the fresh beacons stay blue.
        self.assertEqual(svg.count('data-forecast="'), 4)
        self.assertIn("dark now", svg)
        self.assertIn("2 dB noise in 13.3h", svg)
        self.assertIn("Deployment plan: earliest dark 0.0h (D)", svg)
        self.assertIn("below 3 lit beacons in 19.0h", svg)
        self.assertIn('fill="#c0392b"', svg)
        # Plan only: no planner shading, no planner summary line.
        self.assertNotIn("data-planner-cell=", svg)
        self.assertNotIn("Planner heatmap:", svg)

    def test_planner_heatmap_layer_renders_on_the_overlay(self):
        rows = self._example_rows()
        plan = self._plan(levels={"A": 0.9, "B": 0.6,
                                  "C": 0.4, "D": 0.2})
        svg = render_map_overlay(rows, BEACONS, plan=plan, planner=True)
        # One shaded cell per grid point, with the limiting beacon
        # and milestone on each, and the summary line naming the
        # ward's first degradation.
        self.assertEqual(svg.count("data-planner-cell="), 60)
        self.assertIn('data-limiting-beacon="D"', svg)
        self.assertIn('data-limiting-milestone="rssi_loss"', svg)
        self.assertIn("Planner heatmap: earliest degradation 0.0h "
                      "at (1, 1) m (D, 3 dB loss)", svg)
        self.assertIn("Deployment plan: earliest dark 3.0h (D)", svg)
        # The track still renders on top of the planner shading.
        self.assertEqual(svg.count('data-index="'), len(rows))
        # Off by default: no planner cells, no plan lines.
        plain = render_map_overlay(rows, BEACONS)
        self.assertNotIn("data-planner-cell=", plain)
        self.assertNotIn("Deployment plan:", plain)

    def test_planner_heatmap_needs_a_plan_and_conflicts(self):
        rows = self._example_rows()
        with self.assertRaises(ValueError):
            render_map_overlay(rows, BEACONS, planner=True)
        with self.assertRaises(ValueError):
            render_map_overlay(rows, BEACONS, plan=self._plan(),
                               planner=True, heatmap=True)
        # A beacon in the layout with no forecast in the plan.
        from beacon import plan_deployment

        partial = plan_deployment(
            {b.id: BatteryModel(drain_rate_per_hour=0.05)
             for b in BEACONS[:3]})
        with self.assertRaises(ValueError):
            render_map_overlay(rows, BEACONS, plan=partial)

    def test_cli_map_planner_flag_writes_svg(self):
        from beacon.cli import main

        here = os.path.dirname(os.path.abspath(__file__))
        track = os.path.join(here, "..", "examples",
                             "live-track-export.json")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "map.svg")
            self.assertEqual(main(argv=["--map-overlay", track,
                                       "--map-output", out,
                                       "--map-planner",
                                       "--battery-levels",
                                       "A=0.9,B=0.6,C=0.4,D=0.2"]), 0)
            with open(out, encoding="utf-8") as handle:
                svg = handle.read()
            self.assertIn("<svg", svg)
            self.assertEqual(svg.count("data-planner-cell="), 60)
            self.assertIn("Deployment plan: earliest dark 3.0h (D)",
                          svg)
            self.assertIn("Planner heatmap:", svg)

    def test_cli_map_planner_conflicts(self):
        from beacon.cli import main

        here = os.path.dirname(os.path.abspath(__file__))
        track = os.path.join(here, "..", "examples",
                             "live-track-export.json")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "map.svg")
            self.assertEqual(main(argv=["--map-overlay", track,
                                       "--map-output", out,
                                       "--map-planner",
                                       "--map-heatmap"]), 2)
            self.assertEqual(main(argv=["--map-overlay", track,
                                       "--map-output", out,
                                       "--map-planner",
                                       "--map-replay"]), 2)


class TestDriftMapOverlay(unittest.TestCase):
    def _row(self, x, y, semi_major, floor=0, t=0.0, alarms=(),
             classifications=()):
        return {
            "t": t, "x": x, "y": y, "floor": floor,
            "floor_confidence": 1.0,
            "ellipse_semi_major_m": semi_major,
            "ellipse_semi_minor_m": (semi_major or 1.0) / 2.0,
            "ellipse_angle_deg": 30.0,
            "beacons_used": ["A", "B", "C", "D"],
            "beacons_excluded": [], "coverage": 1.0,
            "drift_alarms": [dict(alarm) for alarm in alarms],
            "drift_classifications": [
                dict(c) for c in classifications],
        }

    def _classification(self, beacon_id, verdict, mean_db):
        return {"id": beacon_id, "verdict": verdict,
                "mean_residual_db": mean_db, "residual_std_db": 1.0,
                "time_slope_db_per_step": 0.0, "time_correlation": 0.0,
                "revisit_delta_db": None, "revisit_pairs": 0,
                "samples": 20, "reason": "test fixture"}

    def _alarming_rows(self):
        # A alarms on the last row only; B and C stay quiet.
        return [
            self._row(2.0, 2.0, 1.0, t=0.0),
            self._row(5.0, 5.0, 1.0, t=1.0,
                      alarms=[{"id": "A", "mean_residual_db": 8.53,
                               "samples": 20}],
                      classifications=[
                          self._classification("A", "uncertain", 8.53),
                          self._classification("B", "healthy", 0.1),
                          self._classification("C", "moved", 2.9),
                          self._classification("D", "healthy", 1.6)]),
        ]

    def test_alarming_beacon_gets_badge_ring_and_label(self):
        svg = render_map_overlay(self._alarming_rows(), BEACONS,
                                 drift=True)
        # Only A is badged; the other three beacons stay plain.
        self.assertEqual(svg.count('data-drift-alarm="'), 1)
        self.assertIn('data-drift-alarm="A"', svg)
        self.assertIn(">!</text>", svg)
        self.assertIn('data-drift-verdict="uncertain"', svg)
        self.assertIn("ALARM +8.5 dB \u00b7 uncertain", svg)
        # The summary line names the alarm state at the end.
        self.assertIn("Drift alarms at end of track: "
                      "A +8.5 dB (uncertain)", svg)
        self.assertIn("drift alarm (red ! badge, end of track)", svg)
        # The track still renders.
        self.assertEqual(svg.count('data-index="'), 2)

    def test_only_the_last_rows_alarms_are_marked(self):
        # A alarmed mid-track and cleared: the overlay shows the
        # end state, so nothing is marked.
        rows = [
            self._row(2.0, 2.0, 1.0, t=0.0,
                      alarms=[{"id": "A", "mean_residual_db": 8.5,
                               "samples": 20}],
                      classifications=[
                          self._classification("A", "uncertain", 8.5)]),
            self._row(5.0, 5.0, 1.0, t=1.0),
        ]
        svg = render_map_overlay(rows, BEACONS, drift=True)
        self.assertNotIn("data-drift-alarm=", svg)
        self.assertNotIn("ALARM", svg)
        self.assertIn("Drift alarms at end of track: none", svg)

    def test_no_alarms_marks_nothing(self):
        rows = [self._row(2.0, 2.0, 1.0, t=0.0),
                self._row(5.0, 5.0, 1.0, t=1.0)]
        svg = render_map_overlay(rows, BEACONS, drift=True)
        self.assertNotIn("data-drift-alarm=", svg)
        self.assertIn("Drift alarms at end of track: none", svg)
        # A track recorded without a monitor carries no drift
        # keys at all; that renders the same as no alarms.
        bare = [{key: value for key, value in row.items()
                 if not key.startswith("drift_")}
                for row in rows]
        self.assertEqual(
            render_map_overlay(bare, BEACONS, drift=True), svg)

    def test_drift_off_by_default(self):
        svg = render_map_overlay(self._alarming_rows(), BEACONS)
        self.assertNotIn("data-drift-alarm=", svg)
        self.assertNotIn("ALARM", svg)
        self.assertNotIn("Drift alarms at end of track:", svg)

    def test_alarm_without_classification_has_no_verdict(self):
        rows = [self._row(2.0, 2.0, 1.0, t=0.0),
                self._row(5.0, 5.0, 1.0, t=1.0,
                          alarms=[{"id": "A",
                                   "mean_residual_db": -6.9,
                                   "samples": 20}])]
        svg = render_map_overlay(rows, BEACONS, drift=True)
        self.assertIn("ALARM -6.9 dB", svg)
        self.assertNotIn("data-drift-verdict=", svg)
        self.assertIn("Drift alarms at end of track: A -6.9 dB", svg)

    def test_drift_marks_the_real_alarm_end_to_end(self):
        # A reads 12 dB too loud on a real walk, streamed through
        # a tracker with a monitor and exported: the overlay
        # badges the beacon the monitor alarmed on.
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                              BEACONS, noise_db=2.0, seed=7)
        for step in steps:
            step["rssi"]["A"] += 12.0
        tracker = LiveTracker(PositioningEngine(BEACONS),
                              drift_monitor=DriftMonitor(BEACONS))
        positions = [tracker.feed(step["rssi"], float(i))
                     for i, step in enumerate(steps)]
        rows = track_rows(positions)
        svg = render_map_overlay(rows, BEACONS, drift=True)
        self.assertEqual(svg.count('data-drift-alarm="'), 1)
        self.assertIn('data-drift-alarm="A"', svg)
        # Seed 7: A alarms at +8.5 dB on 20 readings and the
        # monitor calls it uncertain — the same alarm and verdict
        # the export carries, now on the map.
        self.assertIn("ALARM +8.5 dB \u00b7 uncertain", svg)
        self.assertIn("Drift alarms at end of track: "
                      "A +8.5 dB (uncertain)", svg)

    def test_drift_composes_with_plan(self):
        from beacon import plan_deployment

        plan = plan_deployment(
            {b.id: BatteryModel(drain_rate_per_hour=0.05)
             for b in BEACONS})
        svg = render_map_overlay(self._alarming_rows(), BEACONS,
                                 plan=plan, drift=True)
        # Both annotations ride on the same beacon: the forecast
        # horizon and the drift alarm, on separate lines.
        self.assertEqual(svg.count('data-forecast="'), 4)
        self.assertIn('data-drift-alarm="A"', svg)
        self.assertIn("ALARM +8.5 dB \u00b7 uncertain", svg)
        self.assertIn("2 dB noise in 13.3h", svg)

    def test_cli_map_drift_flag_writes_svg(self):
        from beacon.cli import main

        here = os.path.dirname(os.path.abspath(__file__))
        track = os.path.join(here, "..", "examples",
                             "live-track-export.json")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "map.svg")
            self.assertEqual(main(argv=["--map-overlay", track,
                                       "--map-output", out,
                                       "--map-drift"]), 0)
            with open(out, encoding="utf-8") as handle:
                svg = handle.read()
            self.assertIn("<svg", svg)
            # The example export was recorded without a monitor,
            # so nothing is marked — and the legend says so.
            self.assertNotIn("data-drift-alarm=", svg)
            self.assertIn("Drift alarms at end of track: none", svg)
            # Drift now composes with the replay too: the same
            # flags render the animated replay, still with
            # nothing marked on a track recorded without a
            # monitor.
            out2 = os.path.join(tmp, "replay.svg")
            self.assertEqual(main(argv=["--map-overlay", track,
                                       "--map-output", out2,
                                       "--map-drift",
                                       "--map-replay"]), 0)
            with open(out2, encoding="utf-8") as handle:
                replay = handle.read()
            self.assertIn('data-dot="true"', replay)
            self.assertNotIn("data-drift-alarm=", replay)
            self.assertIn("Drift alarms during replay: none", replay)


class TestLiveTrackReplay(unittest.TestCase):
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

    def test_example_export_replays_all_frames(self):
        rows = self._example_rows()
        svg = render_track_replay(rows, BEACONS)
        self.assertIn("<svg", svg)
        self.assertIn("</svg>", svg)
        # One moving dot, one trail that draws itself, one
        # ellipse and one timestamp label per frame.
        self.assertEqual(svg.count('data-dot="true"'), 1)
        self.assertEqual(svg.count('data-trail="true"'), 1)
        self.assertEqual(svg.count("<ellipse"), len(rows))
        self.assertEqual(svg.count("data-timestamp="), len(rows))
        # The dot animates cx and cy through the frames, the
        # trail animates its dash offset, and the replay loops.
        self.assertIn('<animate attributeName="cx"', svg)
        self.assertIn('<animate attributeName="cy"', svg)
        self.assertIn('<animate attributeName="stroke-dashoffset"', svg)
        self.assertIn('repeatCount="indefinite"', svg)
        # 40 steps at 1 s per step: a 39 s replay.
        self.assertIn('dur="39s"', svg)
        self.assertIn("Replay: 39 s loop", svg)
        self.assertIn("SMIL animation, no JavaScript", svg)
        # Every beacon is labelled, start and end marked.
        for beacon in BEACONS:
            self.assertIn(f'data-beacon="{beacon.id}"', svg)
        self.assertIn(">S</text>", svg)
        self.assertIn(">E</text>", svg)
        self.assertIn("floor 0", svg)
        self.assertIn("40 points", svg)

    def test_csv_export_replays_the_same_track(self):
        here = os.path.dirname(os.path.abspath(__file__))
        csv_rows = load_track_csv(
            os.path.join(here, "..", "examples", "live-track-export.csv"))
        json_svg = render_track_replay(self._example_rows(), BEACONS)
        csv_svg = render_track_replay(csv_rows, BEACONS)
        self.assertEqual(json_svg, csv_svg)

    def test_frame_timing_follows_the_track_timestamps(self):
        rows = [self._row(2.0, 2.0, 1.0, t=0.0),
                self._row(5.0, 5.0, 1.0, t=10.0),
                self._row(8.0, 8.0, 1.0, t=11.0)]
        svg = render_track_replay(rows, BEACONS)
        # The dot's key times are the timestamps' share of the
        # 11 s span, and the replay is 2 s long (2 steps at 1 s).
        self.assertIn('keyTimes="0.0000;0.9091;1.0000"', svg)
        self.assertIn('dur="2s"', svg)
        self.assertIn("Replay: 2 s loop", svg)
        # Each frame's ellipse is visible between the midpoints
        # with its neighbours: frame 1 (t=10) from 0.4545 to
        # 0.9545 of the replay.
        self.assertIn('keyTimes="0;0.4545;0.9545"', svg)
        # The timestamp labels carry the track's own times.
        for t in ("t = 0 s", "t = 10 s", "t = 11 s"):
            self.assertIn(f">{t}<", svg)

    def test_seconds_per_step_and_loop_change_the_replay(self):
        rows = self._example_rows()
        svg = render_track_replay(rows, BEACONS,
                                  seconds_per_step=0.5, loop=False)
        self.assertIn('dur="19.5s"', svg)
        self.assertIn('repeatCount="1"', svg)
        self.assertNotIn('repeatCount="indefinite"', svg)
        self.assertIn("Replay: 19.5 s once", svg)

    def test_single_point_track_is_static(self):
        svg = render_track_replay([self._row(2.0, 2.0, 1.0)], BEACONS)
        self.assertEqual(svg.count('data-dot="true"'), 1)
        self.assertEqual(svg.count("<ellipse"), 1)
        self.assertEqual(svg.count("data-timestamp="), 1)
        # No animation at all: nothing to move between.
        self.assertNotIn("<animate", svg)
        self.assertNotIn('data-trail="true"', svg)
        self.assertIn("t = 0 s", svg)

    def test_frame_without_ellipse_has_no_ellipse(self):
        rows = [self._row(2.0, 2.0, 1.0, t=0.0),
                self._row(5.0, 5.0, None, t=1.0)]
        svg = render_track_replay(rows, BEACONS)
        self.assertEqual(svg.count("<ellipse"), 1)
        self.assertEqual(svg.count("data-timestamp="), 2)
        self.assertEqual(svg.count('data-dot="true"'), 1)

    def test_multi_floor_track_replays_one_floor_only(self):
        rows = [self._row(2.0, 2.0, 1.0, floor=0, t=0.0),
                self._row(3.0, 3.0, 1.0, floor=0, t=1.0),
                self._row(15.0, 9.0, 1.0, floor=1, t=2.0)]
        # Default floor is the most common one (floor 0).
        svg = render_track_replay(rows, BEACONS)
        self.assertEqual(svg.count("data-timestamp="), 2)
        self.assertIn("floor 0", svg)
        svg = render_track_replay(rows, BEACONS, floor=1)
        self.assertEqual(svg.count("data-timestamp="), 1)
        self.assertIn("floor 1", svg)

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            render_track_replay([], BEACONS)
        with self.assertRaises(ValueError):
            render_track_replay([self._row(2.0, 2.0, 1.0)], [])
        with self.assertRaises(ValueError):
            render_track_replay([self._row(2.0, 2.0, 1.0, floor=0)],
                                BEACONS, floor=1)
        with self.assertRaises(ValueError):
            render_track_replay([self._row(2.0, 2.0, 1.0)],
                                BEACONS, seconds_per_step=0)

    def test_heatmap_layer_renders_on_the_replay(self):
        rows = self._example_rows()
        svg = render_track_replay(rows, BEACONS, heatmap=True)
        self.assertEqual(svg.count("data-heatmap-cell="), 60)
        self.assertIn("Coverage heatmap:", svg)
        self.assertEqual(svg.count("data-timestamp="), len(rows))
        plain = render_track_replay(rows, BEACONS)
        self.assertNotIn("data-heatmap-cell=", plain)

    def test_write_track_replay_round_trips_through_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "replay.svg")
            svg = write_track_replay(
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "examples", "live-track-export.json"),
                BEACONS, out)
            with open(out, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), svg)
            self.assertEqual(svg.count("data-timestamp="), 40)

    def test_cli_map_replay_flag_writes_svg(self):
        from beacon.cli import main

        here = os.path.dirname(os.path.abspath(__file__))
        track = os.path.join(here, "..", "examples",
                             "live-track-export.json")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "replay.svg")
            self.assertEqual(main(argv=["--map-overlay", track,
                                       "--map-output", out,
                                       "--map-replay"]), 0)
            with open(out, encoding="utf-8") as handle:
                svg = handle.read()
            self.assertIn("<svg", svg)
            self.assertEqual(svg.count("data-timestamp="), 40)
            self.assertIn('data-dot="true"', svg)
            # Half-second steps halve the replay length.
            out2 = os.path.join(tmp, "replay-fast.svg")
            self.assertEqual(main(argv=["--map-overlay", track,
                                       "--map-output", out2,
                                       "--map-replay",
                                       "--replay-step-seconds", "0.5"]), 0)
            with open(out2, encoding="utf-8") as handle:
                self.assertIn('dur="19.5s"', handle.read())


class TestDriftTrackReplay(unittest.TestCase):
    def _row(self, x, y, semi_major, floor=0, t=0.0, alarms=(),
             classifications=()):
        return {
            "t": t, "x": x, "y": y, "floor": floor,
            "floor_confidence": 1.0,
            "ellipse_semi_major_m": semi_major,
            "ellipse_semi_minor_m": (semi_major or 1.0) / 2.0,
            "ellipse_angle_deg": 30.0,
            "beacons_used": ["A", "B", "C", "D"],
            "beacons_excluded": [], "coverage": 1.0,
            "drift_alarms": [dict(alarm) for alarm in alarms],
            "drift_classifications": [
                dict(c) for c in classifications],
        }

    def _classification(self, beacon_id, verdict, mean_db):
        return {"id": beacon_id, "verdict": verdict,
                "mean_residual_db": mean_db, "residual_std_db": 1.0,
                "time_slope_db_per_step": 0.0, "time_correlation": 0.0,
                "revisit_delta_db": None, "revisit_pairs": 0,
                "samples": 20, "reason": "test fixture"}

    def _alarming_rows(self):
        # A alarms on the last row only; B and C stay quiet.
        return [
            self._row(2.0, 2.0, 1.0, t=0.0),
            self._row(5.0, 5.0, 1.0, t=1.0,
                      alarms=[{"id": "A", "mean_residual_db": 8.53,
                               "samples": 20}],
                      classifications=[
                          self._classification("A", "uncertain", 8.53),
                          self._classification("B", "healthy", 0.1),
                          self._classification("C", "moved", 2.9),
                          self._classification("D", "healthy", 1.6)]),
        ]

    def test_alarming_beacon_gets_badge_ring_and_label(self):
        svg = render_track_replay(self._alarming_rows(), BEACONS,
                                  drift=True)
        # Only A is badged; the other three beacons stay plain.
        # A alarms on the last row only, so one frame carries it.
        self.assertEqual(svg.count('data-drift-alarm="'), 1)
        self.assertIn('data-drift-alarm="A"', svg)
        self.assertIn('data-frame="1"', svg)
        self.assertIn(">!</text>", svg)
        self.assertIn('data-drift-verdict="uncertain"', svg)
        self.assertIn("ALARM +8.5 dB \u00b7 uncertain", svg)
        # The summary line names the alarm history: A alarmed on
        # frame 1 only.
        self.assertIn("Drift alarms during replay: "
                      "A +8.5 dB (uncertain), frame 1", svg)
        self.assertIn("drift alarm (red ! badge, per frame)", svg)
        # The replay still animates: the dot, the trail, both
        # frames' ellipses and timestamp labels.
        self.assertEqual(svg.count('data-dot="true"'), 1)
        self.assertEqual(svg.count('data-trail="true"'), 1)
        self.assertEqual(svg.count("<ellipse"), 2)
        self.assertEqual(svg.count("data-timestamp="), 2)

    def test_cleared_alarm_is_marked_on_its_frame_only(self):
        # A alarmed mid-track and cleared: the replay badges it
        # on the frame it was alarming and nowhere else — the
        # end-of-track marking used to miss this alarm entirely.
        rows = [
            self._row(2.0, 2.0, 1.0, t=0.0,
                      alarms=[{"id": "A", "mean_residual_db": 8.5,
                               "samples": 20}],
                      classifications=[
                          self._classification("A", "uncertain", 8.5)]),
            self._row(5.0, 5.0, 1.0, t=1.0),
        ]
        svg = render_track_replay(rows, BEACONS, drift=True)
        self.assertEqual(svg.count('data-drift-alarm="'), 1)
        self.assertIn('data-drift-alarm="A"', svg)
        self.assertIn('data-drift-alarm-frame="A"', svg)
        self.assertIn("ALARM +8.5 dB \u00b7 uncertain", svg)
        self.assertIn("Drift alarms during replay: "
                      "A +8.5 dB (uncertain), frame 0", svg)
        # The frame-0 marking is visible only while frame 0 is
        # current: it carries a visibility animation, and the
        # frame-1 ellipse carries one too, but no second alarm.
        self.assertIn('data-frame="0"', svg)
        self.assertNotIn('data-frame="1" data-drift-alarm-frame', svg)

    def test_alarm_state_follows_each_frame(self):
        # A alarms on frames 0 and 1, B only on frame 1, and
        # neither on frame 2: each frame marks exactly the
        # beacons alarming on its row, with that row's residual.
        rows = [
            self._row(2.0, 2.0, 1.0, t=0.0,
                      alarms=[{"id": "A", "mean_residual_db": 8.5,
                               "samples": 20}],
                      classifications=[
                          self._classification("A", "uncertain", 8.5)]),
            self._row(5.0, 5.0, 1.0, t=1.0,
                      alarms=[{"id": "A", "mean_residual_db": 9.0,
                               "samples": 20},
                              {"id": "B", "mean_residual_db": -6.9,
                               "samples": 20}],
                      classifications=[
                          self._classification("A", "moved", 9.0),
                          self._classification("B", "uncertain", -6.9)]),
            self._row(8.0, 8.0, 1.0, t=2.0),
        ]
        svg = render_track_replay(rows, BEACONS, drift=True)
        self.assertEqual(svg.count('data-drift-alarm="A"'), 2)
        self.assertEqual(svg.count('data-drift-alarm="B"'), 1)
        self.assertEqual(svg.count('data-drift-alarm="'), 3)
        # Both of A's residuals ride on the replay, each on its
        # own frame, with the verdict that frame's monitor gave.
        self.assertIn("ALARM +8.5 dB \u00b7 uncertain", svg)
        self.assertIn("ALARM +9.0 dB \u00b7 moved", svg)
        self.assertIn("ALARM -6.9 dB \u00b7 uncertain", svg)
        # The summary carries each beacon's frame range and the
        # residual and verdict at its last alarming frame.
        self.assertIn("Drift alarms during replay: "
                      "A +9.0 dB (moved), frames 0–1, "
                      "B -6.9 dB (uncertain), frame 1", svg)

    def test_single_frame_track_marks_its_alarm_without_animation(self):
        rows = [self._row(2.0, 2.0, 1.0, t=0.0,
                          alarms=[{"id": "A",
                                   "mean_residual_db": 8.5,
                                   "samples": 20}],
                          classifications=[
                              self._classification("A", "uncertain",
                                                   8.5)])]
        svg = render_track_replay(rows, BEACONS, drift=True)
        self.assertIn('data-drift-alarm="A"', svg)
        self.assertIn("ALARM +8.5 dB \u00b7 uncertain", svg)
        self.assertIn("Drift alarms during replay: "
                      "A +8.5 dB (uncertain), frame 0", svg)
        # A single frame has no animation at all: no element
        # hides behind a visibility toggle.
        self.assertNotIn('visibility="hidden"', svg)

    def test_no_alarms_marks_nothing(self):
        rows = [self._row(2.0, 2.0, 1.0, t=0.0),
                self._row(5.0, 5.0, 1.0, t=1.0)]
        svg = render_track_replay(rows, BEACONS, drift=True)
        self.assertNotIn("data-drift-alarm=", svg)
        self.assertIn("Drift alarms during replay: none", svg)
        # A track recorded without a monitor carries no drift
        # keys at all; that renders the same as no alarms.
        bare = [{key: value for key, value in row.items()
                 if not key.startswith("drift_")}
                for row in rows]
        self.assertEqual(
            render_track_replay(bare, BEACONS, drift=True), svg)

    def test_drift_off_by_default(self):
        svg = render_track_replay(self._alarming_rows(), BEACONS)
        self.assertNotIn("data-drift-alarm=", svg)
        self.assertNotIn("ALARM", svg)
        self.assertNotIn("Drift alarms during replay:", svg)

    def test_alarm_without_classification_has_no_verdict(self):
        rows = [self._row(2.0, 2.0, 1.0, t=0.0),
                self._row(5.0, 5.0, 1.0, t=1.0,
                          alarms=[{"id": "A",
                                   "mean_residual_db": -6.9,
                                   "samples": 20}])]
        svg = render_track_replay(rows, BEACONS, drift=True)
        self.assertIn("ALARM -6.9 dB", svg)
        self.assertNotIn("data-drift-verdict=", svg)
        self.assertIn("Drift alarms during replay: A -6.9 dB, frame 1",
                      svg)

    def test_drift_marks_the_real_alarm_end_to_end(self):
        # A reads 12 dB too loud on a real walk, streamed through
        # a tracker with a monitor and exported: the replay
        # badges the beacon the monitor alarmed on.
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                              BEACONS, noise_db=2.0, seed=7)
        for step in steps:
            step["rssi"]["A"] += 12.0
        tracker = LiveTracker(PositioningEngine(BEACONS),
                              drift_monitor=DriftMonitor(BEACONS))
        positions = [tracker.feed(step["rssi"], float(i))
                     for i, step in enumerate(steps)]
        rows = track_rows(positions)
        svg = render_track_replay(rows, BEACONS, drift=True)
        # Seed 7: A alarms on frames 4–39, and C flickers into
        # alarm on frames 25–33 and clears — the per-frame
        # marking catches the C flicker the end-of-track badge
        # never showed.
        self.assertEqual(svg.count('data-drift-alarm="A"'), 36)
        self.assertEqual(svg.count('data-drift-alarm="C"'), 9)
        self.assertEqual(svg.count('data-drift-alarm="'), 45)
        self.assertIn('data-drift-alarm="A"', svg)
        self.assertIn('data-drift-alarm="C"', svg)
        # Seed 7: A alarms at +8.5 dB on 20 readings and the
        # monitor calls it uncertain — the same alarm and verdict
        # the export carries, now on the replay — and C's last
        # alarming frame reads +5.1 dB, moved.
        self.assertIn("ALARM +8.5 dB \u00b7 uncertain", svg)
        self.assertIn("ALARM +5.1 dB \u00b7 moved", svg)
        self.assertIn("Drift alarms during replay: "
                      "A +8.5 dB (uncertain), frames 4–39, "
                      "C +5.1 dB (moved), frames 25–33", svg)

    def test_timeline_strip_matches_per_frame_alarm_data(self):
        # Seed 7, A 12 dB too loud: the strip shows exactly the
        # frames the per-frame badges mark — A 4–39, C 25–33 —
        # and the strip's ranges match the per-frame alarm data
        # exactly, not just the first-to-last span in the legend.
        import re

        from beacon.map_overlay import _drift_state_at_row
        from beacon.replay import _alarm_ranges

        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                              BEACONS, noise_db=2.0, seed=7)
        for step in steps:
            step["rssi"]["A"] += 12.0
        tracker = LiveTracker(PositioningEngine(BEACONS),
                              drift_monitor=DriftMonitor(BEACONS))
        positions = [tracker.feed(step["rssi"], float(i))
                     for i, step in enumerate(steps)]
        rows = track_rows(positions)
        svg = render_track_replay(rows, BEACONS, drift=True)
        frame_drift = [_drift_state_at_row(row) for row in rows]
        expected = _alarm_ranges(frame_drift)
        # A's verdict changes mid-alarm — moved on frames 4–6,
        # uncertain from frame 7 — so its run splits where the
        # verdict changes; C stays moved throughout.
        self.assertEqual(expected, {
            "A": [(4, 6, "moved"), (7, 39, "uncertain")],
            "C": [(25, 33, "moved")]})
        found: dict[str, list[tuple[int, int, str | None]]] = {}
        for beacon_id, group in re.findall(
                r'<g data-drift-timeline="([^"]+)">(.*?)</g>',
                svg, re.DOTALL):
            found[beacon_id] = []
            for rect in re.findall(r"<rect [^>]*>", group):
                first, last = re.search(
                    r'data-drift-timeline-range="(\d+)-(\d+)"',
                    rect).groups()
                verdict = re.search(
                    r'data-drift-verdict="([^"]+)"', rect)
                found[beacon_id].append(
                    (int(first), int(last),
                     verdict.group(1) if verdict else None))
        self.assertEqual(found, expected)
        # The segments are coloured by their verdict, not all
        # red: A's early segment is the moved amber, its long
        # segment the uncertain grey, C's the moved amber.
        self.assertIn('fill="#e67e22" '
                      'data-drift-timeline-range="4-6" '
                      'data-drift-verdict="moved"', svg)
        self.assertIn('fill="#7f8c8d" '
                      'data-drift-timeline-range="7-39" '
                      'data-drift-verdict="uncertain"', svg)
        self.assertIn('fill="#e67e22" '
                      'data-drift-timeline-range="25-33" '
                      'data-drift-verdict="moved"', svg)
        self.assertIn("Drift alarm timeline", svg)
        self.assertIn("A \u00b7 frames 4\u201339", svg)
        self.assertIn("C \u00b7 frames 25\u201333", svg)
        # The ranges cover exactly the alarming frames: 36 for
        # A, 9 for C, the same counts the badges carry.
        self.assertEqual(svg.count('data-drift-alarm="A"'), 36)
        self.assertEqual(svg.count('data-drift-alarm="C"'), 9)

    def test_timeline_strip_splits_disjoint_alarms(self):
        # A alarms on frames 0 and 2 but not 1: the strip draws
        # two segments, not one run covering the gap — and B's
        # single-frame alarm is one segment.
        rows = [
            self._row(2.0, 2.0, 1.0, t=0.0,
                      alarms=[{"id": "A", "mean_residual_db": 8.5,
                               "samples": 20}],
                      classifications=[
                          self._classification("A", "uncertain", 8.5)]),
            self._row(5.0, 5.0, 1.0, t=1.0,
                      alarms=[{"id": "B", "mean_residual_db": -6.9,
                               "samples": 20}],
                      classifications=[
                          self._classification("B", "uncertain", -6.9)]),
            self._row(8.0, 8.0, 1.0, t=2.0,
                      alarms=[{"id": "A", "mean_residual_db": 9.0,
                               "samples": 20}],
                      classifications=[
                          self._classification("A", "moved", 9.0)]),
        ]
        svg = render_track_replay(rows, BEACONS, drift=True)
        self.assertEqual(svg.count('data-drift-timeline-range="0-0"'), 1)
        self.assertEqual(svg.count('data-drift-timeline-range="2-2"'), 1)
        self.assertEqual(svg.count('data-drift-timeline-range="1-1"'), 1)
        self.assertEqual(svg.count("data-drift-timeline-range="), 3)
        self.assertIn("A \u00b7 frames 0, 2", svg)
        self.assertIn("B \u00b7 frame 1", svg)

    def test_timeline_strip_segments_coloured_by_verdict(self):
        # Every verdict gets its own colour, a verdict change
        # mid-alarm splits the segment, and an alarm with no
        # classification recorded falls back to the alarm red —
        # with the strip's colours read off the same per-frame
        # verdicts the badges carry, so the two cannot disagree.
        rows = [
            self._row(2.0, 2.0, 1.0, t=0.0,
                      alarms=[{"id": "A", "mean_residual_db": 8.5,
                               "samples": 20},
                              {"id": "D", "mean_residual_db": -7.0,
                               "samples": 20}],
                      classifications=[
                          self._classification("A", "uncertain", 8.5)]),
            self._row(5.0, 5.0, 1.0, t=1.0,
                      alarms=[{"id": "A", "mean_residual_db": 8.6,
                               "samples": 20},
                              {"id": "B", "mean_residual_db": -6.9,
                               "samples": 20}],
                      classifications=[
                          self._classification("A", "uncertain", 8.6),
                          self._classification("B", "battery-sag",
                                               -6.9)]),
            self._row(8.0, 8.0, 1.0, t=2.0,
                      alarms=[{"id": "A", "mean_residual_db": 9.0,
                               "samples": 20},
                              {"id": "C", "mean_residual_db": 7.0,
                               "samples": 20}],
                      classifications=[
                          self._classification("A", "moved", 9.0),
                          self._classification("C", "noisy", 7.0)]),
        ]
        svg = render_track_replay(rows, BEACONS, drift=True)
        self.assertEqual(svg.count("data-drift-timeline-range="), 5)
        # A splits where its verdict changes: uncertain grey on
        # frames 0–1, moved amber on frame 2.
        self.assertIn('fill="#7f8c8d" '
                      'data-drift-timeline-range="0-1" '
                      'data-drift-verdict="uncertain"', svg)
        self.assertIn('fill="#e67e22" '
                      'data-drift-timeline-range="2-2" '
                      'data-drift-verdict="moved"', svg)
        self.assertIn('fill="#2980b9" '
                      'data-drift-timeline-range="1-1" '
                      'data-drift-verdict="battery-sag"', svg)
        self.assertIn('fill="#8e44ad" '
                      'data-drift-timeline-range="2-2" '
                      'data-drift-verdict="noisy"', svg)
        # D has no classification: alarm red, no verdict
        # attribute on its segment — the badges carry verdict
        # attributes too, so check D's strip group itself.
        self.assertIn('fill="#c0392b" '
                      'data-drift-timeline-range="0-0"', svg)
        import re

        d_group = re.search(
            r'<g data-drift-timeline="D">(.*?)</g>', svg, re.DOTALL)
        self.assertIsNotNone(d_group)
        self.assertNotIn("data-drift-verdict=", d_group.group(1))
        # The row labels still give the full frame ranges.
        self.assertIn("A \u00b7 frames 0\u20132", svg)
        self.assertIn("B \u00b7 frame 1", svg)
        self.assertIn("C \u00b7 frame 2", svg)
        self.assertIn("D \u00b7 frame 0", svg)
        # The legend keys the colours actually shown.
        for key in ("timeline: moved", "timeline: battery-sag",
                    "timeline: noisy", "timeline: uncertain",
                    "timeline: alarm, no verdict"):
            self.assertIn(key, svg)

    def test_timeline_strip_empty_when_no_beacon_alarms(self):
        rows = [self._row(2.0, 2.0, 1.0, t=0.0),
                self._row(5.0, 5.0, 1.0, t=1.0)]
        svg = render_track_replay(rows, BEACONS, drift=True)
        # The strip is still there, labelled and empty — the
        # absence of segments reads as no alarm recorded, not
        # a missing strip.
        self.assertIn("Drift alarm timeline", svg)
        self.assertIn(">none</text>", svg)
        self.assertNotIn("data-drift-timeline-range=", svg)
        self.assertNotIn('data-drift-timeline="', svg)
        # A track recorded without a monitor carries no drift
        # keys at all; that renders the same empty strip.
        bare = [{key: value for key, value in row.items()
                 if not key.startswith("drift_")}
                for row in rows]
        self.assertEqual(
            render_track_replay(bare, BEACONS, drift=True), svg)

    def test_timeline_strip_absent_without_drift(self):
        svg = render_track_replay(self._alarming_rows(), BEACONS)
        self.assertNotIn("Drift alarm timeline", svg)
        self.assertNotIn("data-drift-timeline", svg)

    def test_drift_composes_with_heatmap(self):
        here = os.path.dirname(os.path.abspath(__file__))
        rows = load_track_json(
            os.path.join(here, "..", "examples",
                         "live-track-export.json"))
        # The example export was recorded without a monitor, so
        # the heatmap shades and the legend still says none.
        svg = render_track_replay(rows, BEACONS, heatmap=True,
                                  drift=True)
        self.assertEqual(svg.count("data-heatmap-cell="), 60)
        self.assertIn("Coverage heatmap:", svg)
        self.assertIn("Drift alarms during replay: none", svg)
        # And with a real alarm the badge rides on top of the
        # heatmap: both overlays compose.
        svg = render_track_replay(self._alarming_rows(), BEACONS,
                                  heatmap=True, drift=True)
        self.assertEqual(svg.count("data-heatmap-cell="), 60)
        self.assertIn('data-drift-alarm="A"', svg)
        self.assertIn("ALARM +8.5 dB \u00b7 uncertain", svg)

    def test_cli_map_replay_with_drift_flag_writes_svg(self):
        from beacon.cli import main

        here = os.path.dirname(os.path.abspath(__file__))
        track = os.path.join(here, "..", "examples",
                             "live-track-export.json")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "replay.svg")
            self.assertEqual(main(argv=["--map-overlay", track,
                                       "--map-output", out,
                                       "--map-replay",
                                       "--map-drift"]), 0)
            with open(out, encoding="utf-8") as handle:
                svg = handle.read()
            self.assertIn('data-dot="true"', svg)
            self.assertEqual(svg.count("data-timestamp="), 40)
            # The example export was recorded without a monitor,
            # so nothing is marked — and the legend says so.
            self.assertNotIn("data-drift-alarm=", svg)
            self.assertIn("Drift alarms during replay: none", svg)

    def test_replay_canvas_grows_with_the_legend(self):
        # The plain replay's "Replay:" line used to run past the
        # 560 px canvas; the canvas now grows with the legend,
        # so no legend line is ever clipped below it.
        import re

        rows = load_track_json(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "examples", "live-track-export.json"))
        svg = render_track_replay(rows, BEACONS)
        height = int(re.search(r'height="(\d+)"', svg).group(1))
        ys = [float(y) for y in re.findall(r' y="([\d.]+)"', svg)]
        self.assertGreaterEqual(height, max(ys) + 12)
        svg = render_track_replay(self._alarming_rows(), BEACONS,
                                  heatmap=True, drift=True)
        height = int(re.search(r'height="(\d+)"', svg).group(1))
        ys = [float(y) for y in re.findall(r' y="([\d.]+)"', svg)]
        self.assertGreaterEqual(height, max(ys) + 12)


class TestForecastCheckTrackReplay(unittest.TestCase):
    def _row(self, x, y, semi_major, floor=0, t=0.0, alarms=(),
             classifications=(), forecasts=(), forecast_alerts=()):
        return {
            "t": t, "x": x, "y": y, "floor": floor,
            "floor_confidence": 1.0,
            "ellipse_semi_major_m": semi_major,
            "ellipse_semi_minor_m": (semi_major or 1.0) / 2.0,
            "ellipse_angle_deg": 30.0,
            "beacons_used": ["A", "B", "C", "D"],
            "beacons_excluded": [], "coverage": 1.0,
            "forecasts": [dict(f) for f in forecasts],
            "forecast_alerts": list(forecast_alerts),
            "drift_alarms": [dict(alarm) for alarm in alarms],
            "drift_classifications": [
                dict(c) for c in classifications],
        }

    def _forecast(self, beacon_id, milestone, hours, status="ok"):
        return {"beacon": beacon_id, "milestone": milestone,
                "hours": hours, "status": status,
                "hours_to_dark": hours,
                "hours_to_rssi_loss_db": hours,
                "hours_to_extra_noise_db": hours}

    def _alarm(self, beacon_id, mean_db):
        return {"id": beacon_id, "mean_residual_db": mean_db,
                "samples": 20}

    def test_forecast_check_outcomes_from_frame_data(self):
        # The check itself, on the per-frame data: a due forecast
        # confirmed by a later alarm, a hours-away forecast
        # contradicted by an alarm within seconds, and a forecast
        # with no later alarm not observed.
        from beacon.replay import _forecast_check

        checks = _forecast_check(
            [{"A": ("3 dB loss", 0.0, "degrading"),
              "B": ("2 dB noise", 11.3, "ok"),
              "C": ("2 dB noise", 5.0, "ok")},
             {"A": ("3 dB loss", 0.0, "degrading"),
              "B": ("2 dB noise", 11.3, "ok"),
              "C": ("2 dB noise", 5.0, "ok")},
             {"A": ("3 dB loss", 0.0, "degrading"),
              "B": ("2 dB noise", 11.3, "ok"),
              "C": ("2 dB noise", 5.0, "ok")}],
            [{}, {"A": (9.2, "moved")}, {"B": (8.5, "uncertain")}],
            [0.0, 1.0, 2.0])
        self.assertEqual(checks["A"].outcome, "confirmed")
        self.assertEqual(checks["A"].first_alarm_frame, 1)
        self.assertEqual(checks["A"].first_active_frame, 0)
        self.assertEqual(checks["B"].outcome, "contradicted")
        self.assertEqual(checks["B"].first_alarm_frame, 2)
        self.assertEqual(checks["C"].outcome, "not observed")
        self.assertIsNone(checks["C"].first_alarm_frame)

    def test_forecast_live_only_on_last_frame_is_pending(self):
        from beacon.replay import _forecast_check

        checks = _forecast_check(
            [{}, {"D": ("2 dB noise", 1.0, "ok")}],
            [{}, {}], [0.0, 1.0])
        self.assertEqual(checks["D"].outcome, "pending")
        self.assertIsNone(checks["D"].first_alarm_frame)

    def test_due_forecast_confirmed_by_later_alarm(self):
        # A is forecast to cross 3 dB loss now and alarms from
        # frame 1: one F badge per frame it is due, each carrying
        # the forecast and the confirmed outcome.
        rows = [
            self._row(2.0, 2.0, 1.0, t=0.0,
                      forecasts=[self._forecast(
                          "A", "3 dB loss", 0.0, "degrading")],
                      forecast_alerts=[
                          "A: 3 dB loss now (degrading)"]),
            self._row(5.0, 5.0, 1.0, t=1.0,
                      alarms=[self._alarm("A", 9.2)],
                      forecasts=[self._forecast(
                          "A", "3 dB loss", 0.0, "degrading")],
                      forecast_alerts=[
                          "A: 3 dB loss now (degrading)"]),
            self._row(8.0, 8.0, 1.0, t=2.0,
                      alarms=[self._alarm("A", 8.5)],
                      forecasts=[self._forecast(
                          "A", "3 dB loss", 0.0, "degrading")],
                      forecast_alerts=[
                          "A: 3 dB loss now (degrading)"]),
        ]
        svg = render_track_replay(rows, BEACONS, forecast=True)
        self.assertEqual(svg.count('data-forecast="A"'), 3)
        self.assertEqual(
            svg.count('data-forecast-outcome="confirmed"'), 3)
        self.assertIn("F: 3 dB loss now \u00b7 confirmed", svg)
        self.assertIn('data-forecast-alarm-frame="1"', svg)
        self.assertIn("Forecast check \u2014 A: 3 dB loss now, "
                      "confirmed, alarm from frame 1", svg)
        self.assertIn("forecast due (amber F badge, per frame)", svg)
        # The F badge sits top-left, opposite the drift badge:
        # with drift off there is no drift badge at all.
        self.assertNotIn("data-drift-alarm=", svg)

    def test_hours_away_forecast_contradicted_by_early_alarm(self):
        # B is forecast to lie in 11.3 hours but alarms on frame
        # 2, seconds in: the drift came far earlier than the
        # battery model said it would, so the forecast is
        # contradicted, not confirmed.
        rows = [
            self._row(2.0, 2.0, 1.0, t=0.0,
                      forecasts=[self._forecast("B", "2 dB noise", 11.3)],
                      forecast_alerts=["B: 2 dB noise in 11.3h (ok)"]),
            self._row(5.0, 5.0, 1.0, t=1.0,
                      forecasts=[self._forecast("B", "2 dB noise", 11.3)],
                      forecast_alerts=["B: 2 dB noise in 11.3h (ok)"]),
            self._row(8.0, 8.0, 1.0, t=2.0,
                      alarms=[self._alarm("B", 8.5)],
                      forecasts=[self._forecast("B", "2 dB noise", 11.3)],
                      forecast_alerts=["B: 2 dB noise in 11.3h (ok)"]),
        ]
        svg = render_track_replay(rows, BEACONS, forecast=True)
        self.assertEqual(svg.count('data-forecast="B"'), 3)
        self.assertEqual(
            svg.count('data-forecast-outcome="contradicted"'), 3)
        self.assertIn("F: 2 dB noise in 11.3h \u00b7 contradicted", svg)
        self.assertIn("Forecast check \u2014 B: 2 dB noise in 11.3h, "
                      "contradicted, alarm from frame 2", svg)

    def test_forecast_with_no_later_alarm_is_not_observed(self):
        rows = [
            self._row(2.0, 2.0, 1.0, t=0.0,
                      forecasts=[self._forecast("C", "2 dB noise", 5.0)],
                      forecast_alerts=["C: 2 dB noise in 5.0h (ok)"]),
            self._row(5.0, 5.0, 1.0, t=1.0,
                      forecasts=[self._forecast("C", "2 dB noise", 5.0)],
                      forecast_alerts=["C: 2 dB noise in 5.0h (ok)"]),
        ]
        svg = render_track_replay(rows, BEACONS, forecast=True)
        self.assertEqual(svg.count('data-forecast="C"'), 2)
        self.assertEqual(
            svg.count('data-forecast-outcome="not observed"'), 2)
        self.assertNotIn("data-forecast-alarm-frame=", svg)
        self.assertIn("Forecast check \u2014 C: 2 dB noise in 5.0h, "
                      "not observed in this track", svg)

    def test_forecast_outside_the_window_gets_no_marker(self):
        # A plan is recorded, but nothing is due inside the
        # window: no F badge, and the legend says so instead of
        # reporting no forecast at all.
        rows = [
            self._row(2.0, 2.0, 1.0, t=0.0,
                      forecasts=[self._forecast("A", "2 dB noise", 99.0)]),
            self._row(5.0, 5.0, 1.0, t=1.0,
                      forecasts=[self._forecast("A", "2 dB noise", 99.0)]),
        ]
        svg = render_track_replay(rows, BEACONS, forecast=True)
        self.assertNotIn("data-forecast=", svg)
        self.assertIn("Forecast check: no forecast due in this track",
                      svg)

    def test_no_plan_records_no_forecast(self):
        rows = [self._row(2.0, 2.0, 1.0, t=0.0),
                self._row(5.0, 5.0, 1.0, t=1.0)]
        bare = [{key: value for key, value in row.items()
                 if not key.startswith("forecast")}
                for row in rows]
        svg = render_track_replay(bare, BEACONS, forecast=True)
        self.assertNotIn("data-forecast=", svg)
        self.assertIn("Forecast check: no plan recorded", svg)

    def test_forecast_off_by_default(self):
        rows = [
            self._row(2.0, 2.0, 1.0, t=0.0,
                      forecasts=[self._forecast(
                          "A", "3 dB loss", 0.0, "degrading")],
                      forecast_alerts=[
                          "A: 3 dB loss now (degrading)"]),
            self._row(5.0, 5.0, 1.0, t=1.0,
                      forecasts=[self._forecast(
                          "A", "3 dB loss", 0.0, "degrading")],
                      forecast_alerts=[
                          "A: 3 dB loss now (degrading)"]),
        ]
        svg = render_track_replay(rows, BEACONS)
        self.assertNotIn("data-forecast=", svg)
        self.assertNotIn("Forecast check", svg)

    def test_forecast_check_matches_the_badges_on_a_seeded_walk(self):
        # Seed 7, A 12 dB too loud, plan levels A=0.2 (due now)
        # and the rest at 0.9 (2 dB noise in 11.3 h): the check
        # reads the same per-frame rows as the drift badges, so
        # it agrees with them frame for frame — A alarms 4–39
        # and its due forecast is confirmed from frame 4, C
        # flickers 25–33 against an 11.3 h forecast and is
        # contradicted from frame 25, B and D never alarm.
        from beacon import plan_deployment

        plan = plan_deployment(
            {b.id: BatteryModel(drain_rate_per_hour=0.05)
             for b in BEACONS},
            levels={"A": 0.2, "B": 0.9, "C": 0.9, "D": 0.9})
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                              BEACONS, noise_db=2.0, seed=7)
        for step in steps:
            step["rssi"]["A"] += 12.0
        tracker = LiveTracker(PositioningEngine(BEACONS), plan=plan,
                              drift_monitor=DriftMonitor(BEACONS))
        positions = [tracker.feed(step["rssi"], float(i))
                     for i, step in enumerate(steps)]
        rows = track_rows(positions)
        svg = render_track_replay(rows, BEACONS,
                                  drift=True, forecast=True)
        for beacon_id in ("A", "B", "C", "D"):
            self.assertEqual(
                svg.count(f'data-forecast="{beacon_id}"'), 40)
        self.assertEqual(
            svg.count('data-forecast-outcome="confirmed"'), 40)
        self.assertEqual(
            svg.count('data-forecast-outcome="contradicted"'), 40)
        self.assertEqual(
            svg.count('data-forecast-outcome="not observed"'), 80)
        self.assertEqual(svg.count('data-drift-alarm="A"'), 36)
        self.assertEqual(svg.count('data-drift-alarm="C"'), 9)
        self.assertIn("F: 3 dB loss now \u00b7 confirmed", svg)
        self.assertIn("F: 2 dB noise in 11.3h \u00b7 contradicted", svg)
        self.assertIn("F: 2 dB noise in 11.3h \u00b7 not observed", svg)
        self.assertIn("Forecast check \u2014 A: 3 dB loss now, "
                      "confirmed, alarm from frame 4", svg)
        self.assertIn("Forecast check \u2014 B: 2 dB noise in 11.3h, "
                      "not observed in this track", svg)
        self.assertIn("Forecast check \u2014 C: 2 dB noise in 11.3h, "
                      "contradicted, alarm from frame 25", svg)
        self.assertIn("Forecast check \u2014 D: 2 dB noise in 11.3h, "
                      "not observed in this track", svg)

    def test_cli_map_replay_with_forecast_flag_writes_svg(self):
        from beacon.cli import main

        here = os.path.dirname(os.path.abspath(__file__))
        track = os.path.join(here, "..", "examples",
                             "live-track-export.json")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "replay.svg")
            self.assertEqual(main(argv=["--map-overlay", track,
                                       "--map-output", out,
                                       "--map-replay",
                                       "--map-forecast"]), 0)
            with open(out, encoding="utf-8") as handle:
                svg = handle.read()
            self.assertIn('data-dot="true"', svg)
            # The example export was recorded without a plan, so
            # nothing is marked — and the legend says so.
            self.assertNotIn("data-forecast=", svg)
            self.assertIn("Forecast check: no plan recorded", svg)

    def test_cli_map_forecast_without_replay_is_rejected(self):
        from beacon.cli import main

        here = os.path.dirname(os.path.abspath(__file__))
        track = os.path.join(here, "..", "examples",
                             "live-track-export.json")
        self.assertEqual(main(argv=["--map-overlay", track,
                                   "--map-forecast"]), 2)


class TestMultiDeviceTrackReplay(unittest.TestCase):
    def _row(self, x, y, semi_major, floor=0, t=0.0, device=None,
             alarms=(), classifications=(), forecasts=(),
             forecast_alerts=()):
        row = {
            "t": t, "x": x, "y": y, "floor": floor,
            "floor_confidence": 1.0,
            "ellipse_semi_major_m": semi_major,
            "ellipse_semi_minor_m": (semi_major or 1.0) / 2.0,
            "ellipse_angle_deg": 30.0,
            "beacons_used": ["A", "B", "C", "D"],
            "beacons_excluded": [], "coverage": 1.0,
            "forecasts": [dict(f) for f in forecasts],
            "forecast_alerts": list(forecast_alerts),
            "drift_alarms": [dict(alarm) for alarm in alarms],
            "drift_classifications": [
                dict(c) for c in classifications],
        }
        if device is not None:
            row["device"] = device
        return row

    def _forecast(self, beacon_id, milestone, hours, status="ok"):
        return {"beacon": beacon_id, "milestone": milestone,
                "hours": hours, "status": status,
                "hours_to_dark": hours,
                "hours_to_rssi_loss_db": hours,
                "hours_to_extra_noise_db": hours}

    def _alarm(self, beacon_id, mean_db):
        return {"id": beacon_id, "mean_residual_db": mean_db,
                "samples": 20}

    def _classification(self, beacon_id, verdict, mean_db):
        return {"id": beacon_id, "verdict": verdict,
                "mean_residual_db": mean_db, "residual_std_db": 1.0,
                "time_slope_db_per_step": 0.0, "time_correlation": 0.0,
                "revisit_delta_db": None, "revisit_pairs": 0,
                "samples": 20, "reason": "test fixture"}

    def _two_devices(self):
        # Two three-frame tracks on different paths: tag-1 walks
        # one diagonal, tag-2 the other.
        return [
            self._row(2.0, 2.0, 1.0, t=0.0, device="tag-1"),
            self._row(5.0, 5.0, 1.0, t=1.0, device="tag-1"),
            self._row(8.0, 8.0, 1.0, t=2.0, device="tag-1"),
            self._row(18.0, 10.0, 1.0, t=0.0, device="tag-2"),
            self._row(15.0, 7.0, 1.0, t=1.0, device="tag-2"),
            self._row(12.0, 4.0, 1.0, t=2.0, device="tag-2"),
        ]

    def _positions(self):
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                              BEACONS, noise_db=2.0, seed=7)
        tracker = LiveTracker(PositioningEngine(BEACONS))
        return [tracker.feed(step["rssi"], float(i))
                for i, step in enumerate(steps)]

    def test_two_devices_get_their_own_dot_trail_and_series(self):
        rows = self._two_devices()
        svg = render_track_replay(rows, BEACONS)
        # One moving dot, one trail, one ellipse and one timestamp
        # label per device per frame.
        self.assertEqual(svg.count('data-dot="true"'), 2)
        self.assertEqual(svg.count('data-trail="true"'), 2)
        self.assertEqual(svg.count("<ellipse"), 6)
        self.assertEqual(svg.count("data-timestamp="), 6)
        # Each device's dot and trail carry its id, in its own
        # colour: tag-1 keeps the single-device pink, tag-2 blue.
        self.assertIn('data-dot="true" data-device="tag-1"', svg)
        self.assertIn('data-dot="true" data-device="tag-2"', svg)
        self.assertEqual(svg.count('fill="#d81b60" stroke="#ffffff"'), 1)
        self.assertEqual(svg.count('fill="#1565c0" stroke="#ffffff"'), 1)
        # Timestamp labels name their device and stack, one line
        # per device, so they never print over each other.
        self.assertIn(">tag-1 · t = 0 s<", svg)
        self.assertIn(">tag-2 · t = 0 s<", svg)
        # Start and end markers per device.
        self.assertEqual(svg.count(">S</text>"), 2)
        self.assertEqual(svg.count(">E</text>"), 2)
        # The heading counts both devices, and the legend keys them.
        self.assertIn("6 points across 2 devices", svg)
        self.assertIn("tag-1 track", svg)
        self.assertIn("tag-2 track", svg)
        # Each device replays at its own length: 2 s, 3 frames.
        self.assertIn("Replay tag-1: 2 s loop, 3 frames", svg)
        self.assertIn("Replay tag-2: 2 s loop, 3 frames", svg)
        # Interleaved rows group the same way: the replay reads
        # rows per device, not in file order.
        interleaved = [rows[0], rows[3], rows[1],
                       rows[4], rows[2], rows[5]]
        svg2 = render_track_replay(interleaved, BEACONS)
        self.assertEqual(svg2.count('data-dot="true"'), 2)
        self.assertEqual(svg2.count("data-timestamp="), 6)

    def test_single_unnamed_device_renders_as_before(self):
        rows = [self._row(2.0, 2.0, 1.0, t=0.0),
                self._row(5.0, 5.0, 1.0, t=1.0)]
        svg = render_track_replay(rows, BEACONS)
        self.assertNotIn("data-device=", svg)
        self.assertNotIn("across 2 devices", svg)
        self.assertEqual(svg.count('data-dot="true"'), 1)
        self.assertEqual(svg.count('data-trail="true"'), 1)
        self.assertIn("Replay: 1 s loop, 2 frames", svg)
        self.assertIn(">t = 0 s<", svg)

    def test_drift_badges_and_strip_stay_per_device(self):
        rows = [
            self._row(2.0, 2.0, 1.0, t=0.0, device="tag-1"),
            self._row(5.0, 5.0, 1.0, t=1.0, device="tag-1",
                      alarms=[self._alarm("A", 8.5)],
                      classifications=[
                          self._classification("A", "uncertain", 8.5)]),
            self._row(8.0, 8.0, 1.0, t=2.0, device="tag-1",
                      alarms=[self._alarm("A", 9.0)],
                      classifications=[
                          self._classification("A", "moved", 9.0)]),
            self._row(18.0, 10.0, 1.0, t=0.0, device="tag-2"),
            self._row(15.0, 7.0, 1.0, t=1.0, device="tag-2"),
            self._row(12.0, 4.0, 1.0, t=2.0, device="tag-2"),
        ]
        svg = render_track_replay(rows, BEACONS, drift=True)
        # A alarms on tag-1's frames 1 and 2 only; tag-2 stays quiet.
        self.assertEqual(svg.count('data-drift-alarm="A"'), 2)
        self.assertEqual(
            svg.count('data-drift-alarm-frame="A" data-device="tag-1"'),
            2)
        self.assertNotIn(
            'data-drift-alarm-frame="A" data-device="tag-2"', svg)
        # The strip has one row for tag-1's alarm, labelled with both.
        self.assertIn("tag-1 · A · frames 1–2", svg)
        self.assertNotIn("tag-2 · A", svg)
        # The history is one line per device: tag-1 alarmed, tag-2 none.
        self.assertIn("Drift alarms during replay — tag-1: "
                      "A +9.0 dB (moved), frames 1–2", svg)
        self.assertIn("Drift alarms during replay — tag-2: none", svg)

    def test_forecast_checks_are_per_device(self):
        rows = [
            self._row(2.0, 2.0, 1.0, t=0.0, device="tag-1",
                      forecasts=[self._forecast("A", "3 dB loss", 0.0,
                                                "degrading")],
                      forecast_alerts=["A: 3 dB loss now (degrading)"]),
            self._row(5.0, 5.0, 1.0, t=1.0, device="tag-1",
                      forecasts=[self._forecast("A", "3 dB loss", 0.0,
                                                "degrading")],
                      forecast_alerts=["A: 3 dB loss now (degrading)"],
                      alarms=[self._alarm("A", 9.2)],
                      classifications=[
                          self._classification("A", "moved", 9.2)]),
            self._row(18.0, 10.0, 1.0, t=0.0, device="tag-2",
                      forecasts=[self._forecast("A", "3 dB loss", 0.0,
                                                "degrading")],
                      forecast_alerts=["A: 3 dB loss now (degrading)"]),
            self._row(15.0, 7.0, 1.0, t=1.0, device="tag-2",
                      forecasts=[self._forecast("A", "3 dB loss", 0.0,
                                                "degrading")],
                      forecast_alerts=["A: 3 dB loss now (degrading)"]),
        ]
        svg = render_track_replay(rows, BEACONS, drift=True, forecast=True)
        # tag-1's forecast is confirmed by its frame-1 alarm; tag-2
        # carries the same forecast but never alarms, so its check
        # is not observed — one tag's confirmation never speaks for
        # the other.
        self.assertIn("Forecast check — tag-1 · A: 3 dB loss now, "
                      "confirmed, alarm from frame 1", svg)
        self.assertIn("Forecast check — tag-2 · A: 3 dB loss now, "
                      "not observed in this track", svg)
        # The F markers ride on both devices' frames, each with its
        # own check outcome.
        self.assertIn('data-forecast-outcome="confirmed"', svg)
        self.assertIn('data-forecast-outcome="not observed"', svg)

    def test_device_tag_round_trips_through_json_and_csv(self):
        positions = self._positions()
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "tag-1.json")
            csv_path = os.path.join(tmp, "tag-1.csv")
            export_track_json(positions, json_path, device="tag-1")
            export_track_csv(positions, csv_path, device="tag-1")
            json_rows = load_track_json(json_path)
            csv_rows = load_track_csv(csv_path)
        self.assertEqual(json_rows, csv_rows)
        self.assertEqual(len(json_rows), 40)
        self.assertTrue(all(row["device"] == "tag-1"
                            for row in json_rows))
        self.assertEqual(json_rows, track_rows(positions,
                                               device="tag-1"))
        # A file written before the device column still loads, with
        # the device defaulting to None.
        here = os.path.dirname(os.path.abspath(__file__))
        old_rows = load_track_json(
            os.path.join(here, "..", "examples", "live-track-export.json"))
        self.assertTrue(all(row["device"] is None for row in old_rows))
        old_csv = load_track_csv(
            os.path.join(here, "..", "examples", "live-track-export.csv"))
        self.assertEqual(old_csv, old_rows)

    def test_empty_or_non_string_device_is_rejected(self):
        positions = self._positions()
        with self.assertRaises(ValueError):
            track_rows(positions, device="")
        with self.assertRaises(ValueError):
            track_rows(positions, device=7)

    def test_combined_multi_device_file_replays_both_devices(self):
        from beacon.cli import main
        from beacon.export import CSV_FIELDS, _row_to_csv
        import csv

        # Two real tracked walks, exported per device and combined
        # into one file — the shape a ward dashboard writes.
        tracks = {}
        for device_id, walk in (
                ("tag-1", rectangular_walk(20.0, 12.0, 10)),
                ("tag-2", list(reversed(
                    rectangular_walk(20.0, 12.0, 10))))):
            steps = simulate_walk(walk, BEACONS, noise_db=2.0, seed=7)
            tracker = LiveTracker(PositioningEngine(BEACONS))
            tracks[device_id] = [tracker.feed(step["rssi"], float(i))
                                 for i, step in enumerate(steps)]
        rows = multi_track_rows(tracks)
        self.assertEqual(len(rows), 80)
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "both.json")
            csv_path = os.path.join(tmp, "both.csv")
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump({"format": "beacon-live-track", "version": 1,
                           "steps": len(rows), "located": len(rows),
                           "coverage": 1.0, "rows": rows}, handle)
            with open(csv_path, "w", encoding="utf-8",
                      newline="") as handle:
                writer = csv.DictWriter(handle,
                                        fieldnames=list(CSV_FIELDS))
                writer.writeheader()
                for row in rows:
                    writer.writerow(_row_to_csv(row))
            out = os.path.join(tmp, "both.svg")
            self.assertEqual(main(argv=["--map-overlay", json_path,
                                       "--map-output", out,
                                       "--map-replay"]), 0)
            with open(out, encoding="utf-8") as handle:
                json_svg = handle.read()
            out_csv = os.path.join(tmp, "both-csv.svg")
            self.assertEqual(main(argv=["--map-overlay", csv_path,
                                       "--map-output", out_csv,
                                       "--map-replay"]), 0)
            with open(out_csv, encoding="utf-8") as handle:
                csv_svg = handle.read()
        self.assertEqual(json_svg, csv_svg)
        self.assertEqual(json_svg.count('data-dot="true"'), 2)
        self.assertEqual(json_svg.count("data-timestamp="), 80)

    def test_measured_two_device_demo(self):
        # Two simulated walks through one MultiDeviceTracker —
        # tag-1 forward with beacon A reading 12 dB too loud,
        # tag-2 the reverse walk, healthy — with the plan's levels
        # A=0.2, the rest 0.9, so A is due 3 dB loss now and the
        # rest 2 dB noise in 11.3 h.
        plan = plan_deployment(
            {b.id: BatteryModel(drain_rate_per_hour=0.05)
             for b in BEACONS},
            levels={"A": 0.2, "B": 0.9, "C": 0.9, "D": 0.9})
        forward = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                                BEACONS, noise_db=2.0, seed=7)
        reverse = simulate_walk(
            list(reversed(rectangular_walk(20.0, 12.0, 10))),
            BEACONS, noise_db=2.0, seed=11)
        for step in forward:
            step["rssi"]["A"] += 12.0
        tracker = MultiDeviceTracker(BEACONS, plan=plan,
                                     drift_monitor=DriftMonitor(BEACONS))
        positions = {"tag-1": [], "tag-2": []}
        for i, (step_a, step_b) in enumerate(zip(forward, reverse)):
            for device_id, step in (("tag-1", step_a), ("tag-2", step_b)):
                positions[device_id].append(
                    tracker.feed(device_id, step["rssi"], float(i)))
        rows = multi_track_rows(positions)
        svg = render_track_replay(rows, BEACONS, drift=True,
                                  forecast=True)
        self.assertEqual(svg.count('data-dot="true"'), 2)
        self.assertEqual(svg.count("data-timestamp="), 80)
        # tag-1 alarms on A from frame 4 (moved, then uncertain)
        # and C flickers frames 25–33; tag-2 never alarms.
        self.assertEqual(
            svg.count('data-drift-alarm-frame="A" data-device="tag-1"'),
            36)
        self.assertEqual(
            svg.count('data-drift-alarm-frame="C" data-device="tag-1"'),
            9)
        self.assertNotIn(
            'data-drift-alarm-frame="A" data-device="tag-2"', svg)
        self.assertIn("tag-1 · A · frames 4–39", svg)
        self.assertIn("tag-1 · C · frames 25–33", svg)
        self.assertIn("Drift alarms during replay — tag-1: "
                      "A +8.5 dB (uncertain), frames 4–39, "
                      "C +5.1 dB (moved), frames 25–33", svg)
        self.assertIn("Drift alarms during replay — tag-2: none", svg)
        # tag-1's forecast is confirmed for A and contradicted for
        # C; tag-2 carries the same forecasts but alarms on nothing.
        self.assertIn("Forecast check — tag-1 · A: 3 dB loss now, "
                      "confirmed, alarm from frame 4", svg)
        self.assertIn("Forecast check — tag-1 · C: 2 dB noise in 11.3h, "
                      "contradicted, alarm from frame 25", svg)
        self.assertIn("Forecast check — tag-2 · A: 3 dB loss now, "
                      "not observed in this track", svg)
        self.assertIn("Forecast check — tag-2 · B: 2 dB noise in 11.3h, "
                      "not observed in this track", svg)


class TestMultiDeviceTracking(unittest.TestCase):
    def _walks(self):
        forward = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                                BEACONS, noise_db=2.0, seed=7)
        reverse = simulate_walk(
            list(reversed(rectangular_walk(20.0, 12.0, 10))),
            BEACONS, noise_db=2.0, seed=11)
        return forward, reverse

    def _feed_both(self, tracker, forward, reverse):
        """Feed both walks interleaved, one reading per device per
        step, and return each device's live positions."""
        positions_a, positions_b = [], []
        for i, (step_a, step_b) in enumerate(zip(forward, reverse)):
            positions_a.append(tracker.feed("tag-1", step_a["rssi"],
                                            float(i)))
            positions_b.append(tracker.feed("tag-2", step_b["rssi"],
                                            float(i)))
        return positions_a, positions_b

    @staticmethod
    def _mean_error(positions, steps):
        errors = [math.hypot(p.x - s["true"][0], p.y - s["true"][1])
                  for p, s in zip(positions, steps) if p is not None]
        return statistics.fmean(errors)

    def test_two_devices_track_separately(self):
        forward, reverse = self._walks()
        tracker = MultiDeviceTracker(BEACONS)
        positions_a, positions_b = self._feed_both(
            tracker, forward, reverse)
        self.assertEqual(tracker.devices, ("tag-1", "tag-2"))
        # Both tracks stay within the usual accuracy guard.
        self.assertLess(self._mean_error(positions_a, forward), 4.0)
        self.assertLess(self._mean_error(positions_b, reverse), 4.0)
        # And they are genuinely two tracks, not one smeared
        # position: the devices walk opposite directions, so
        # their positions stay metres apart.
        gaps = [math.hypot(a.x - b.x, a.y - b.y)
                for a, b in zip(positions_a, positions_b)
                if a is not None and b is not None]
        self.assertGreater(statistics.fmean(gaps), 5.0)

    def test_one_device_does_not_move_the_other(self):
        forward, reverse = self._walks()
        tracker = MultiDeviceTracker(BEACONS)
        for i, step in enumerate(forward[:10]):
            tracker.feed("tag-1", step["rssi"], float(i))
        before = tracker.last_position("tag-1")
        for i, step in enumerate(reverse):
            tracker.feed("tag-2", step["rssi"], float(i))
        # Feeding tag-2 all forty steps left tag-1 exactly where
        # it was.
        self.assertEqual(tracker.last_position("tag-1"), before)
        self.assertEqual(tracker.last_position("tag-2").floor, 0)

    def test_thin_device_returns_none_while_other_locates(self):
        tracker = MultiDeviceTracker(BEACONS)
        thin = {"A": -62.1, "B": -71.4}
        self.assertIsNone(tracker.feed("tag-1", thin, 0.0))
        self.assertIsNone(tracker.last_position("tag-1"))
        full = {"A": -62.1, "B": -71.4, "C": -68.0, "D": -59.8}
        self.assertIsNotNone(tracker.feed("tag-2", full, 0.0))
        # The dashboard snapshot only carries devices that
        # actually located.
        self.assertEqual(set(tracker.positions()), {"tag-2"})

    def test_per_device_clocks_are_independent(self):
        tracker = MultiDeviceTracker(BEACONS)
        full = {"A": -62.1, "B": -71.4, "C": -68.0, "D": -59.8}
        tracker.feed("tag-1", full, 10.0)
        # tag-2's clock starts at 0 even though tag-1 is already
        # at t=10 — interleaved streams do not share a clock.
        self.assertIsNotNone(tracker.feed("tag-2", full, 0.0))
        # But one device's own clock still may not go backwards.
        with self.assertRaises(ValueError):
            tracker.feed("tag-1", full, 9.0)

    def test_bad_device_ids_raise(self):
        tracker = MultiDeviceTracker(BEACONS)
        full = {"A": -62.1, "B": -71.4, "C": -68.0, "D": -59.8}
        with self.assertRaises(ValueError):
            tracker.feed("", full, 0.0)
        with self.assertRaises(ValueError):
            tracker.feed(7, full, 0.0)
        with self.assertRaises(ValueError):
            MultiDeviceTracker(BEACONS[:2])

    def test_reset_one_device_leaves_the_other(self):
        forward, reverse = self._walks()
        tracker = MultiDeviceTracker(BEACONS)
        self._feed_both(tracker, forward, reverse)
        tracker.reset("tag-1")
        self.assertIsNone(tracker.last_position("tag-1"))
        self.assertIsNotNone(tracker.last_position("tag-2"))
        # Resetting again is harmless, and a reset device starts
        # a fresh track on its next reading.
        tracker.reset("tag-1")
        live = tracker.feed("tag-1", forward[0]["rssi"], 0.0)
        self.assertIsNotNone(live)
        self.assertTrue(tracker.remove("tag-2"))
        self.assertFalse(tracker.remove("tag-2"))
        self.assertEqual(tracker.devices, ("tag-1",))
        self.assertEqual(set(tracker.positions()), {"tag-1"})

    def test_stream_round_trip(self):
        import io

        from beacon.multi_device import (main as multi_main,
                                         parse_device_line)

        lines = [
            '{"device": "tag-1", "t": 0.0, "rssi": {"A": -62.1, '
            '"B": -71.4, "C": -68.0, "D": -59.8}}',
            '{"device": "tag-2", "t": 0.0, "rssi": {"A": -70.0, '
            '"B": -65.0, "C": -60.0, "D": -72.0}}',
            '{"device": "tag-1", "t": 1.0, "rssi": {"A": -63.0, '
            '"B": -70.0, "C": -67.0, "D": -60.5}}',
        ]
        out = io.StringIO()
        self.assertEqual(
            multi_main(argv=[], input_stream=io.StringIO("\n".join(lines)),
                       output_stream=out), 0)
        records = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(records), 3)
        self.assertEqual([r["device"] for r in records],
                         ["tag-1", "tag-2", "tag-1"])
        self.assertTrue(all(r["position"] is not None for r in records))
        # A line without a device is rejected, not silently
        # tracked as nobody's.
        with self.assertRaises(ValueError):
            parse_device_line('{"t": 0.0, "rssi": {"A": -62.1}}')

    def test_demo_tracks_two_devices(self):
        import io

        from beacon.multi_device import run_demo

        out = io.StringIO()
        tracker = run_demo(5.0, 10.0, 3, 2.0, out)
        records = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(records), 80)
        self.assertEqual({r["device"] for r in records},
                         {"tag-1", "tag-2"})
        self.assertEqual(set(tracker.positions()), {"tag-1", "tag-2"})
        # Both demo tracks end near the ward, on the ground floor
        # (estimates are not clamped to the room, so a filtered
        # position may sit just outside its walls).
        for device_id in ("tag-1", "tag-2"):
            last = tracker.last_position(device_id)
            self.assertGreaterEqual(last.x, -5.0)
            self.assertLessEqual(last.x, 25.0)
            self.assertGreaterEqual(last.y, -5.0)
            self.assertLessEqual(last.y, 17.0)
            self.assertEqual(last.floor, 0)


class TestMultiDeviceExport(unittest.TestCase):
    def _feed_two_devices(self, tracker):
        """Two seeded walks through the tracker — tag-1 forward
        (seed 7), tag-2 the reverse walk (seed 11) — and return
        the walks."""
        forward = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                                BEACONS, noise_db=2.0, seed=7)
        reverse = simulate_walk(
            list(reversed(rectangular_walk(20.0, 12.0, 10))),
            BEACONS, noise_db=2.0, seed=11)
        for i, (step_a, step_b) in enumerate(zip(forward, reverse)):
            tracker.feed("tag-1", step_a["rssi"], float(i))
            tracker.feed("tag-2", step_b["rssi"], float(i))
        return forward, reverse

    def test_two_device_export_round_trips(self):
        tracker = MultiDeviceTracker(BEACONS)
        self._feed_two_devices(tracker)
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "both.json")
            csv_path = os.path.join(tmp, "both.csv")
            tracker.export_track_json(json_path)
            tracker.export_track_csv(csv_path)
            json_rows = load_track_json(json_path)
            csv_rows = load_track_csv(csv_path)
            with open(json_path, encoding="utf-8") as handle:
                header = json.load(handle)
        self.assertEqual(json_rows, csv_rows)
        # The file's rows are the same rows the overlay and the
        # replay group by device — the export cannot disagree
        # with them about which rows belong to which device.
        self.assertEqual(json_rows, multi_track_rows(tracker.tracks()))
        self.assertEqual(len(json_rows), 80)
        self.assertEqual([row["device"] for row in json_rows],
                         ["tag-1"] * 40 + ["tag-2"] * 40)
        self.assertTrue(all(row["coverage"] == 1.0
                            for row in json_rows))
        self.assertEqual(header["steps"], 80)
        self.assertEqual(header["located"], 80)
        self.assertEqual(header["coverage"], 1.0)

    def test_exported_file_replays_and_overlays_both_devices(self):
        tracker = MultiDeviceTracker(BEACONS)
        self._feed_two_devices(tracker)
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "both.json")
            csv_path = os.path.join(tmp, "both.csv")
            tracker.export_track_json(json_path)
            tracker.export_track_csv(csv_path)
            json_rows = load_track_json(json_path)
            csv_rows = load_track_csv(csv_path)
        replay_json = render_track_replay(json_rows, BEACONS)
        replay_csv = render_track_replay(csv_rows, BEACONS)
        self.assertEqual(replay_json, replay_csv)
        self.assertEqual(replay_json.count('data-dot="true"'), 2)
        self.assertEqual(replay_json.count("data-timestamp="), 80)
        overlay_json = render_map_overlay(json_rows, BEACONS)
        overlay_csv = render_map_overlay(csv_rows, BEACONS)
        self.assertEqual(overlay_json, overlay_csv)
        # Both series in their own colours — tag-1 pink,
        # tag-2 blue — the same colours the replay uses.
        self.assertIn("#d81b60", overlay_json)
        self.assertIn("#1565c0", overlay_json)

    def test_single_device_export_matches_single_device_file(self):
        tracker = MultiDeviceTracker(BEACONS)
        forward = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                                BEACONS, noise_db=2.0, seed=7)
        positions = [tracker.feed("tag-1", step["rssi"], float(i))
                     for i, step in enumerate(forward)]
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "tag-1.json")
            csv_path = os.path.join(tmp, "tag-1.csv")
            single_path = os.path.join(tmp, "single.json")
            tracker.export_track_json(json_path)
            tracker.export_track_csv(csv_path)
            export_track_json(positions, single_path, device="tag-1")
            with open(json_path, encoding="utf-8") as handle:
                header = json.load(handle)
            with open(single_path, encoding="utf-8") as handle:
                single = json.load(handle)
            csv_rows = load_track_csv(csv_path)
            json_rows = load_track_json(json_path)
        # One device in the tracker exports exactly the
        # single-device file for that device's track.
        self.assertEqual(header, single)
        self.assertEqual(header, track_to_dict(positions,
                                               device="tag-1"))
        self.assertEqual(csv_rows, json_rows)

    def test_thin_step_counts_against_coverage(self):
        tracker = MultiDeviceTracker(BEACONS)
        thin = {"A": -62.1, "B": -71.4}
        self.assertIsNone(tracker.feed("tag-2", thin, 0.0))
        # Feed both walks starting at t=1, so the thin step at
        # t=0 stays the first step of tag-2's recorded track.
        forward = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                                BEACONS, noise_db=2.0, seed=7)
        reverse = simulate_walk(
            list(reversed(rectangular_walk(20.0, 12.0, 10))),
            BEACONS, noise_db=2.0, seed=11)
        for i, (step_a, step_b) in enumerate(zip(forward, reverse)):
            tracker.feed("tag-1", step_a["rssi"], float(i + 1))
            tracker.feed("tag-2", step_b["rssi"], float(i + 1))
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "both.json")
            tracker.export_track_json(json_path)
            json_rows = load_track_json(json_path)
            with open(json_path, encoding="utf-8") as handle:
                header = json.load(handle)
        # The thin step is a step, not a row: 80 rows from 81
        # steps, and tag-2's rows carry its own 40/41 coverage
        # while tag-1's stay at 1.0.
        self.assertEqual(len(json_rows), 80)
        self.assertEqual(header["steps"], 81)
        self.assertEqual(header["located"], 80)
        self.assertEqual(header["coverage"], 80 / 81)
        tag_2_rows = [row for row in json_rows
                      if row["device"] == "tag-2"]
        self.assertEqual(len(tag_2_rows), 40)
        self.assertTrue(all(row["coverage"] == 40 / 41
                            for row in tag_2_rows))
        tag_1_rows = [row for row in json_rows
                      if row["device"] == "tag-1"]
        self.assertTrue(all(row["coverage"] == 1.0
                            for row in tag_1_rows))

    def test_reset_and_remove_drop_the_recorded_track(self):
        tracker = MultiDeviceTracker(BEACONS)
        forward, reverse = self._feed_two_devices(tracker)
        tracker.reset("tag-1")
        self.assertEqual(set(tracker.tracks()), {"tag-2"})
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "both.json")
            tracker.export_track_json(json_path)
            rows = load_track_json(json_path)
        self.assertEqual(len(rows), 40)
        self.assertTrue(all(row["device"] == "tag-2" for row in rows))
        # A reset device records a fresh track on its next
        # reading, and removing the other device drops it from
        # the export too.
        tracker.feed("tag-1", forward[0]["rssi"], 0.0)
        self.assertTrue(tracker.remove("tag-2"))
        self.assertEqual(set(tracker.tracks()), {"tag-1"})
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "tag-1.json")
            tracker.export_track_json(json_path)
            rows = load_track_json(json_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["device"], "tag-1")

    def test_export_with_no_devices_raises(self):
        tracker = MultiDeviceTracker(BEACONS)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                tracker.export_track_json(
                    os.path.join(tmp, "both.json"))
            with self.assertRaises(ValueError):
                tracker.export_track_csv(
                    os.path.join(tmp, "both.csv"))

    def test_demo_cli_exports_combined_file(self):
        import io

        from beacon.multi_device import main as multi_main

        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "both.json")
            csv_path = os.path.join(tmp, "both.csv")
            self.assertEqual(
                multi_main(argv=["--demo", "--export-json", json_path,
                                "--export-csv", csv_path],
                           output_stream=out), 0)
            json_rows = load_track_json(json_path)
            csv_rows = load_track_csv(csv_path)
        self.assertEqual(json_rows, csv_rows)
        self.assertEqual(len(json_rows), 80)
        self.assertEqual({row["device"] for row in json_rows},
                         {"tag-1", "tag-2"})


class TestFloorLayoutValidation(unittest.TestCase):
    def _two_floor_beacons(self):
        ground = [Beacon("A", 0.0, 0.0, floor=0),
                  Beacon("B", 20.0, 0.0, floor=0),
                  Beacon("C", 20.0, 12.0, floor=0),
                  Beacon("D", 0.0, 12.0, floor=0)]
        upper = [Beacon("E", 0.0, 0.0, floor=1),
                 Beacon("F", 20.0, 0.0, floor=1),
                 Beacon("G", 20.0, 12.0, floor=1),
                 Beacon("H", 0.0, 12.0, floor=1)]
        return ground + upper

    def test_corner_layout_validates_ok_on_one_floor(self):
        validation = validate_floor_layouts(BEACONS, 20.0, 12.0)
        self.assertIsInstance(validation, LayoutValidation)
        self.assertTrue(validation.ok)
        self.assertEqual(len(validation.verdicts), 1)
        verdict = validation.verdicts[0]
        self.assertIsInstance(verdict, FloorLayoutVerdict)
        self.assertEqual(verdict.floor, 0)
        self.assertEqual(verdict.n_beacons, 4)
        self.assertEqual(verdict.status, "ok")
        self.assertEqual(verdict.warnings, ())
        self.assertIsNotNone(verdict.score)
        self.assertLess(verdict.score.worst_semi_major, 10.0)
        self.assertEqual(validation.warnings(), [])
        self.assertIs(validation.by_floor(0), verdict)
        self.assertIsNone(validation.by_floor(1))

    def test_two_floors_get_one_verdict_each(self):
        validation = validate_floor_layouts(
            self._two_floor_beacons(), 20.0, 12.0)
        self.assertTrue(validation.ok)
        self.assertEqual([v.floor for v in validation.verdicts], [0, 1])
        self.assertTrue(all(v.status == "ok"
                            for v in validation.verdicts))

    def test_floor_with_two_beacons_cannot_triangulate(self):
        beacons = self._two_floor_beacons()[:4] + [
            Beacon("E", 0.0, 0.0, floor=1),
            Beacon("F", 20.0, 12.0, floor=1)]
        validation = validate_floor_layouts(beacons, 20.0, 12.0)
        self.assertFalse(validation.ok)
        self.assertEqual(validation.by_floor(0).status, "ok")
        upper = validation.by_floor(1)
        self.assertEqual(upper.status, "cannot_triangulate")
        self.assertEqual(upper.n_beacons, 2)
        self.assertIsNone(upper.score)
        self.assertEqual(len(upper.warnings), 1)
        self.assertIn("only 2 beacons", upper.warnings[0])
        self.assertIn("falls back", upper.warnings[0])
        # The failure is in the warnings list, floor-ordered.
        self.assertEqual(validation.warnings(), list(upper.warnings))

    def test_floor_with_no_beacons_at_all_is_not_a_verdict(self):
        # A floor with no beacons at all never appears: only floors
        # that carry beacons are validated, so a missing floor is a
        # deployment gap, not a verdict.
        validation = validate_floor_layouts(BEACONS, 20.0, 12.0)
        self.assertIsNone(validation.by_floor(1))

    def test_coincident_beacons_cannot_triangulate(self):
        beacons = [Beacon("A", 10.0, 6.0),
                   Beacon("B", 10.0, 6.0),
                   Beacon("C", 10.0, 6.0)]
        validation = validate_floor_layouts(beacons, 20.0, 12.0)
        self.assertFalse(validation.ok)
        verdict = validation.verdicts[0]
        self.assertEqual(verdict.status, "cannot_triangulate")
        self.assertIsNone(verdict.score)
        self.assertIn("collinear or coincide", verdict.warnings[0])

    def test_collinear_beacons_leave_degenerate_points(self):
        # Three beacons on one line: points on the line have an
        # undefined ellipse, so the floor is thin, not ok. The
        # 1 m margin puts test points on the beacon line (y = 6);
        # the default 0.5 m margin would step over it.
        beacons = [Beacon("A", 0.0, 6.0),
                   Beacon("B", 10.0, 6.0),
                   Beacon("C", 20.0, 6.0)]
        validation = validate_floor_layouts(beacons, 20.0, 12.0,
                                            margin=1.0)
        verdict = validation.verdicts[0]
        self.assertEqual(verdict.status, "thin")
        self.assertGreater(verdict.score.degenerate_points, 0)
        self.assertIn("undefined", verdict.warnings[0])

    def test_advised_layout_is_thin_at_its_worst_corner(self):
        # The placement advisor's mean-optimal layout trades the
        # worst point for the mean: its worst 95% error is 11.16 m,
        # above the 10 m default limit, so it validates thin — and
        # raising the limit accepts it.
        advice = advise_placement(20.0, 12.0, 4)
        self.assertGreater(advice.score.worst_semi_major, 10.0)
        validation = validate_floor_layouts(
            advice.beacons, 20.0, 12.0)
        self.assertEqual(validation.verdicts[0].status, "thin")
        self.assertFalse(validation.ok)
        self.assertIn("worst 95% error is",
                      validation.verdicts[0].warnings[-1])
        relaxed = validate_floor_layouts(
            advice.beacons, 20.0, 12.0, max_worst_semi_major=12.0)
        self.assertTrue(relaxed.ok)

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            validate_floor_layouts([], 20.0, 12.0)
        with self.assertRaises(ValueError):
            validate_floor_layouts(BEACONS, 0.0, 12.0)
        with self.assertRaises(ValueError):
            validate_floor_layouts(
                BEACONS, 20.0, 12.0, max_worst_semi_major=0.0)

    def test_cli_validate_layout_flag(self):
        from beacon.cli import main

        self.assertEqual(main(["--validate-layout"]), 0)


class TestBatteryDegradation(unittest.TestCase):
    def _draining_walk(self, seed=7, drain_per_hour=1.2):
        """The default ward walk with beacon D draining at
        ``drain_per_hour`` — at 120 s per step the 40-step walk
        spans 80 minutes, so D (cutoff 0.05) goes dark at step
        24. Returns (steps, positions) through a live tracker."""
        steps = simulate_walk(rectangular_walk(20.0, 12.0, 10), BEACONS,
                              noise_db=2.0, seed=seed,
                              batteries={"D": BatteryModel(
                                  drain_rate_per_hour=drain_per_hour)},
                              step_seconds=120.0)
        tracker = LiveTracker(PositioningEngine(BEACONS))
        positions = [tracker.feed(step["rssi"], float(i) * 120.0,
                                  battery=step.get("battery"))
                     for i, step in enumerate(steps)]
        return steps, positions

    def _mean_error(self, positions, steps, lo, hi):
        errors = [math.hypot(p.x - s["true"][0], p.y - s["true"][1])
                  for p, s in zip(positions[lo:hi], steps[lo:hi])
                  if p is not None]
        return statistics.fmean(errors)

    def test_model_drains_linearly_and_floors_at_zero(self):
        model = BatteryModel(drain_rate_per_hour=0.5)
        self.assertEqual(model.level_at(0.0), 1.0)
        self.assertAlmostEqual(model.level_at(3600.0), 0.5)
        self.assertAlmostEqual(model.level_at(7200.0), 0.0)
        # A dead cell stays dead, it does not go negative.
        self.assertEqual(model.level_at(99999.0), 0.0)
        with self.assertRaises(ValueError):
            model.level_at(-1.0)

    def test_loss_grows_with_the_square_of_charge_used(self):
        model = BatteryModel(max_rssi_loss_db=6.0)
        self.assertEqual(model.rssi_loss_db(1.0), 0.0)
        # Half drained: a quarter of the max loss.
        self.assertAlmostEqual(model.rssi_loss_db(0.5), 1.5)
        self.assertEqual(model.rssi_loss_db(0.0), 6.0)
        self.assertLess(model.rssi_loss_db(0.75),
                        model.rssi_loss_db(0.5))
        self.assertLess(model.rssi_loss_db(0.5),
                        model.rssi_loss_db(0.25))

    def test_extra_noise_grows_as_battery_falls(self):
        model = BatteryModel(max_extra_noise_db=3.0)
        self.assertEqual(model.extra_noise_db(1.0), 0.0)
        self.assertAlmostEqual(model.extra_noise_db(0.5), 1.5)
        self.assertEqual(model.extra_noise_db(0.0), 3.0)

    def test_cutoff_makes_beacon_dark(self):
        model = BatteryModel(cutoff=0.05)
        self.assertFalse(model.is_dark(0.06))
        self.assertTrue(model.is_dark(0.05))
        self.assertTrue(model.is_dark(0.0))
        rng = random.Random(1)
        self.assertIsNone(model.apply(-70.0, 0.04, rng))
        self.assertIsNotNone(model.apply(-70.0, 0.06, rng))

    def test_bad_models_raise(self):
        with self.assertRaises(ValueError):
            BatteryModel(initial=0.0)
        with self.assertRaises(ValueError):
            BatteryModel(initial=1.5)
        with self.assertRaises(ValueError):
            BatteryModel(drain_rate_per_hour=-1.0)
        with self.assertRaises(ValueError):
            BatteryModel(cutoff=1.0)
        with self.assertRaises(ValueError):
            BatteryModel(max_rssi_loss_db=-1.0)
        with self.assertRaises(ValueError):
            BatteryModel(max_extra_noise_db=-1.0)

    def test_simulator_weakens_a_draining_beacon_by_the_model(self):
        # No extra noise, so the degradation is exact: the
        # draining beacon reads weaker than the healthy one by
        # exactly the model's loss at that level.
        path = rectangular_walk(20.0, 12.0, 10)
        healthy = simulate_walk(path, BEACONS, noise_db=0.0, seed=7)
        model = BatteryModel(drain_rate_per_hour=1.2,
                             max_extra_noise_db=0.0)
        degraded = simulate_walk(path, BEACONS, noise_db=0.0, seed=7,
                                 batteries={"D": model},
                                 step_seconds=120.0)
        for i, (good, bad) in enumerate(zip(healthy, degraded)):
            level = model.level_at(i * 120.0)
            if model.is_dark(level):
                # Past the cutoff the beacon is silent, not weak.
                self.assertNotIn("D", bad["rssi"])
                continue
            self.assertAlmostEqual(
                bad["rssi"]["D"],
                good["rssi"]["D"] - model.rssi_loss_db(level))
            # The healthy beacons are untouched.
            for beacon_id in ("A", "B", "C"):
                self.assertEqual(bad["rssi"][beacon_id],
                                 good["rssi"][beacon_id])

    def test_dark_beacon_is_not_heard_and_battery_is_reported(self):
        steps, _ = self._draining_walk()
        # Every step reports D's level, falling monotonically.
        levels = [step["battery"]["D"] for step in steps]
        self.assertEqual(levels[0], 1.0)
        self.assertEqual(levels[-1], 0.0)
        self.assertTrue(all(a >= b for a, b in
                            zip(levels, levels[1:])))
        # D is heard through step 23 and silent from step 24,
        # when its level crosses the 0.05 cutoff.
        self.assertIn("D", steps[23]["rssi"])
        self.assertLessEqual(steps[23]["battery"]["D"], 0.08)
        self.assertNotIn("D", steps[24]["rssi"])
        self.assertNotIn("D", steps[39]["rssi"])
        # The other beacons never carry battery state — only D
        # was given a model — so their readings are untouched.
        for step in steps:
            self.assertEqual(list(step["battery"]), ["D"])

    def test_live_track_error_grows_as_beacon_drains(self):
        steps, positions = self._draining_walk()
        report = evaluate_live_track(
            positions, [step["true"] for step in steps])
        baseline_steps, baseline_positions = self._draining_walk(
            drain_per_hour=0.0)
        baseline_report = evaluate_live_track(
            baseline_positions,
            [step["true"] for step in baseline_steps])
        # The walk stays fully located — three beacons still
        # locate — but the error is far worse with D draining:
        # 2.16 m healthy against 5.09 m degraded on seed 7.
        self.assertEqual(report.coverage, 1.0)
        self.assertGreater(report.mean_error_m,
                           baseline_report.mean_error_m)
        self.assertAlmostEqual(baseline_report.mean_error_m, 2.16,
                               places=1)
        self.assertAlmostEqual(report.mean_error_m, 5.09, places=1)
        # The error grows while D is dying, not after it is
        # dead: steps 0-5 (D healthy) average 2.19 m, steps
        # 12-19 (D at 0.52 down to 0.24) average 13.66 m, and
        # once D is dark the three-beacon track recovers to
        # 3.36 m over steps 24-39.
        self.assertAlmostEqual(
            self._mean_error(positions, steps, 0, 6), 2.19, places=1)
        self.assertAlmostEqual(
            self._mean_error(positions, steps, 12, 20), 13.66,
            places=1)
        self.assertLess(self._mean_error(positions, steps, 24, 40),
                        self._mean_error(positions, steps, 12, 20))

    def test_dark_beacon_excluded_with_battery_reason(self):
        steps, positions = self._draining_walk()
        # D's last reading is step 23; the 10 s dark window
        # passes at the next 120 s step, so it is first recorded
        # excluded at step 24 — with the battery named as the
        # cause, not just the silence.
        first = next(p for p in positions if p is not None
                     and p.excluded)
        self.assertEqual(first.t, 24 * 120.0)
        self.assertEqual(len(first.excluded), 1)
        self.assertEqual(first.excluded[0].id, "D")
        self.assertIn("battery exhausted", first.excluded[0].reason)
        self.assertIn("no reading for", first.excluded[0].reason)
        # The battery level rides on every live position.
        self.assertEqual(dict(positions[0].battery), {"D": 1.0})
        self.assertEqual(dict(positions[-1].battery), {"D": 0.0})

    def test_battery_rides_on_export_rows_and_round_trips(self):
        _, positions = self._draining_walk()
        rows = track_rows(positions)
        self.assertEqual(len(rows), 40)
        self.assertEqual(rows[0]["battery"], {"D": 1.0})
        self.assertEqual(rows[-1]["battery"], {"D": 0.0})
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "track.json")
            csv_path = os.path.join(tmp, "track.csv")
            export_track_json(positions, json_path)
            export_track_csv(positions, csv_path)
            self.assertEqual(load_track_json(json_path), rows)
            self.assertEqual(load_track_csv(csv_path), rows)
            with open(csv_path, encoding="utf-8") as handle:
                header = handle.readline().strip()
        self.assertIn("battery", header)
        # A track without batteries exports empty battery dicts.
        healthy = simulate_walk(rectangular_walk(20.0, 12.0, 10),
                                BEACONS, noise_db=2.0, seed=7)
        tracker = LiveTracker(PositioningEngine(BEACONS))
        plain = [tracker.feed(step["rssi"], float(i))
                 for i, step in enumerate(healthy)]
        self.assertEqual(track_rows(plain)[0]["battery"], {})

    def test_old_csv_without_battery_column_still_loads(self):
        rows = load_track_csv("examples/live-track-export.csv")
        self.assertEqual(len(rows), 40)
        self.assertEqual(rows[0]["battery"], {})

    def test_multifloor_walk_degrades_too(self):
        model = BatteryModel(drain_rate_per_hour=1.2)
        steps = simulate_multifloor_walk(
            two_floor_walk(20.0, 12.0, 10), BEACONS, noise_db=2.0,
            seed=7, batteries={"D": model}, step_seconds=120.0)
        self.assertEqual(len(steps), 80)
        self.assertIn("D", steps[0]["rssi"])
        self.assertNotIn("D", steps[24]["rssi"])
        self.assertEqual(steps[0]["battery"]["D"], 1.0)

    def test_demo_battery_flag_drains_beacon_d(self):
        import io
        import json as json_lib

        from beacon.tracking import main

        out = io.StringIO()
        self.assertEqual(main(argv=["--demo",
                                    "--battery-drain-per-hour", "150"],
                              output_stream=out), 0)
        lines = [json_lib.loads(line) for line in
                 out.getvalue().splitlines()]
        self.assertEqual(len(lines), 40)
        drained = [line for line in lines
                   if line["position"] is not None
                   and line["position"]["excluded"]]
        self.assertTrue(drained)
        first = drained[0]
        self.assertEqual(first["position"]["excluded"][0]["id"], "D")
        self.assertIn("battery exhausted",
                      first["position"]["excluded"][0]["reason"])
        self.assertEqual(first["position"]["battery"]["D"], 0.0)


if __name__ == "__main__":
    unittest.main()
