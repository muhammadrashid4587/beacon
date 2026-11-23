"""Tests for the deployment planner (beacon 0.27.0)."""

import io
import json
import math
import unittest
from contextlib import redirect_stdout

from beacon import BatteryModel, forecast_beacon, plan_deployment
from beacon.cli import main


class ForecastBeaconTest(unittest.TestCase):
    def test_fresh_beacon_dark_hours(self):
        # (1.0 - 0.05) / 0.05 = 19 hours to the 0.05 cutoff.
        forecast = forecast_beacon(
            "A", BatteryModel(drain_rate_per_hour=0.05))
        self.assertAlmostEqual(forecast.hours_to_dark, 19.0)
        self.assertEqual(forecast.level, 1.0)
        self.assertEqual(forecast.status, "ok")

    def test_no_drain_never_goes_dark(self):
        forecast = forecast_beacon(
            "A", BatteryModel(drain_rate_per_hour=0.0))
        self.assertTrue(math.isinf(forecast.hours_to_dark))
        self.assertTrue(math.isinf(forecast.hours_to_rssi_loss_db))
        self.assertTrue(math.isinf(forecast.hours_to_extra_noise_db))

    def test_hours_to_rssi_loss_threshold(self):
        # Loss reaches 3 dB when (1 - level)^2 = 3/6, i.e. level
        # 1 - sqrt(0.5) = 0.2929; (1 - 0.2929) / 0.05 = 14.14 h.
        forecast = forecast_beacon(
            "A", BatteryModel(drain_rate_per_hour=0.05))
        self.assertAlmostEqual(forecast.hours_to_rssi_loss_db, 14.1421,
                               places=3)

    def test_hours_to_extra_noise_threshold(self):
        # Noise reaches 2 dB when 3 * (1 - level) = 2, i.e. level
        # 1/3; (1 - 1/3) / 0.05 = 13.33 h.
        forecast = forecast_beacon(
            "A", BatteryModel(drain_rate_per_hour=0.05))
        self.assertAlmostEqual(forecast.hours_to_extra_noise_db, 13.3333,
                               places=3)

    def test_loss_threshold_above_cutoff_loss_never_reached(self):
        # At the 0.05 cutoff the loss is 6 * 0.95^2 = 5.415 dB, so a
        # 6 dB threshold is never reached — the beacon goes dark
        # first instead of reading that weak.
        forecast = forecast_beacon(
            "A", BatteryModel(drain_rate_per_hour=0.05),
            loss_threshold_db=6.0)
        self.assertTrue(math.isinf(forecast.hours_to_rssi_loss_db))
        self.assertAlmostEqual(forecast.hours_to_dark, 19.0)

    def test_already_past_loss_threshold_is_zero_hours(self):
        # At level 0.2 the loss is already 6 * 0.8^2 = 3.84 dB,
        # past the 3 dB threshold, so the horizon is 0, not negative.
        forecast = forecast_beacon(
            "A", BatteryModel(initial=0.2, drain_rate_per_hour=0.05))
        self.assertEqual(forecast.hours_to_rssi_loss_db, 0.0)
        self.assertEqual(forecast.status, "degrading")

    def test_already_dark(self):
        forecast = forecast_beacon(
            "A", BatteryModel(initial=0.5, drain_rate_per_hour=0.05),
            level=0.03)
        self.assertEqual(forecast.hours_to_dark, 0.0)
        self.assertEqual(forecast.status, "dark")

    def test_measured_level_overrides_initial(self):
        # A beacon measured at half charge has half the runway of a
        # fresh one: (0.5 - 0.05) / 0.05 = 9 hours.
        forecast = forecast_beacon(
            "A", BatteryModel(drain_rate_per_hour=0.05), level=0.5)
        self.assertAlmostEqual(forecast.hours_to_dark, 9.0)

    def test_level_above_initial_rejected(self):
        with self.assertRaises(ValueError):
            forecast_beacon(
                "A", BatteryModel(initial=0.5, drain_rate_per_hour=0.05),
                level=0.8)

    def test_negative_threshold_rejected(self):
        with self.assertRaises(ValueError):
            forecast_beacon("A", BatteryModel(),
                            loss_threshold_db=-1.0)
        with self.assertRaises(ValueError):
            forecast_beacon("A", BatteryModel(),
                            noise_threshold_db=-1.0)

    def test_to_dict_never_is_null(self):
        forecast = forecast_beacon(
            "A", BatteryModel(drain_rate_per_hour=0.0))
        data = forecast.to_dict()
        self.assertIsNone(data["hours_to_dark"])
        self.assertIsNone(data["hours_to_rssi_loss_db"])
        # And the dict is real JSON, not Python's Infinity literal.
        self.assertNotIn("Infinity", json.dumps(data))


class PlanDeploymentTest(unittest.TestCase):
    def _models(self):
        return {
            "A": BatteryModel(drain_rate_per_hour=0.05),
            "B": BatteryModel(drain_rate_per_hour=0.10),
            "C": BatteryModel(drain_rate_per_hour=0.05),
            "D": BatteryModel(drain_rate_per_hour=0.20),
        }

    def test_sorted_soonest_dark_first(self):
        plan = plan_deployment(self._models())
        self.assertEqual([f.beacon_id for f in plan.forecasts],
                         ["D", "B", "A", "C"])
        self.assertAlmostEqual(plan.earliest_dark_hours, 4.75)

    def test_hours_until_fewer_than_three(self):
        # Four beacons: below three once the second goes dark —
        # B at (1 - 0.05) / 0.10 = 9.5 hours.
        plan = plan_deployment(self._models())
        self.assertAlmostEqual(
            plan.hours_until_fewer_than(3), 9.5)

    def test_hours_until_fewer_than_one_is_last_dark(self):
        plan = plan_deployment(self._models())
        self.assertAlmostEqual(
            plan.hours_until_fewer_than(1), 19.0)

    def test_already_below_count_is_zero(self):
        plan = plan_deployment(self._models(),
                               levels={"A": 0.03, "B": 0.04})
        self.assertEqual(plan.hours_until_fewer_than(3), 0.0)

    def test_never_dark_never_below_three(self):
        plan = plan_deployment(
            {k: BatteryModel(drain_rate_per_hour=0.0)
             for k in ("A", "B", "C", "D")})
        self.assertTrue(math.isinf(plan.earliest_dark_hours))
        self.assertTrue(
            math.isinf(plan.hours_until_fewer_than(3)))

    def test_unknown_level_id_rejected(self):
        with self.assertRaises(ValueError):
            plan_deployment(self._models(), levels={"Z": 0.5})

    def test_to_dict_round_trips_through_json(self):
        plan = plan_deployment(self._models())
        data = json.loads(json.dumps(plan.to_dict()))
        self.assertEqual(len(data["forecasts"]), 4)
        self.assertAlmostEqual(data["earliest_dark_hours"], 4.75)
        self.assertAlmostEqual(
            data["hours_until_fewer_than_three"], 9.5)


class DeploymentCliTest(unittest.TestCase):
    def test_deployment_plan_cli(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["--deployment-plan"]), 0)
        text = out.getvalue()
        self.assertIn("earliest dark: 19.0 hours", text)
        self.assertIn("below 3 lit beacons: 19.0 hours", text)

    def test_deployment_plan_cli_with_levels(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(
                main(["--deployment-plan",
                      "--battery-levels", "A=0.5,B=0.5,C=0.5,D=0.5"]),
                0)
        text = out.getvalue()
        self.assertIn("earliest dark: 9.0 hours", text)

    def test_deployment_plan_cli_unknown_beacon(self):
        err = io.StringIO()
        with redirect_stdout(io.StringIO()):
            import sys
            old = sys.stderr
            sys.stderr = err
            try:
                code = main(["--deployment-plan",
                             "--battery-levels", "Z=0.5"])
            finally:
                sys.stderr = old
        self.assertEqual(code, 2)
        self.assertIn("unknown beacon", err.getvalue())


if __name__ == "__main__":
    unittest.main()
