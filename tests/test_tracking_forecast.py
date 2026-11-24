"""Tests for planner forecast flags on the live track (beacon 0.30.0)."""

import io
import json
import math
import unittest
from contextlib import redirect_stderr

from beacon import (BatteryModel, Beacon, ForecastFlag, LiveTracker,
                    PositioningEngine, forecast_alerts, forecast_flags,
                    plan_deployment)
from beacon.simulate import rectangular_walk, simulate_walk
from beacon.tracking import (DEFAULT_FORECAST_WINDOW_HOURS, format_output,
                             main)

# Four corner beacons in a 20 x 12 m ward, matching the CLI defaults.
BEACONS = [
    Beacon("A", 0.0, 0.0),
    Beacon("B", 20.0, 0.0),
    Beacon("C", 20.0, 12.0),
    Beacon("D", 0.0, 12.0),
]


def _fresh_plan(drain_rate_per_hour=0.05, levels=None):
    return plan_deployment(
        {b.id: BatteryModel(drain_rate_per_hour=drain_rate_per_hour)
         for b in BEACONS},
        levels=levels)


class ForecastFlagsTest(unittest.TestCase):
    def test_fresh_beacon_flag_is_its_soonest_milestone(self):
        # Four fresh beacons at 0.05/hr: 2 dB of extra noise comes
        # first (13.33 h), before 3 dB of loss (14.14 h) and dark
        # (19.0 h).
        flags = forecast_flags(_fresh_plan())
        self.assertEqual(len(flags), 4)
        for flag in flags:
            self.assertIsInstance(flag, ForecastFlag)
            self.assertEqual(flag.milestone, "2 dB noise")
            self.assertAlmostEqual(flag.hours, 13.3333, places=3)
            self.assertEqual(flag.status, "ok")
            self.assertAlmostEqual(flag.hours_to_dark, 19.0)
            self.assertAlmostEqual(flag.hours_to_rssi_loss_db, 14.1421,
                                   places=3)
            self.assertAlmostEqual(flag.hours_to_extra_noise_db, 13.3333,
                                   places=3)

    def test_degraded_beacon_flag_names_the_milestone_already_past(self):
        # D measured at 20% charge is already past both the 3 dB
        # loss and the 2 dB noise thresholds; the loss wins the
        # tie, and the flag is due now.
        flags = forecast_flags(_fresh_plan(levels={"D": 0.2}))
        by_id = {flag.beacon_id: flag for flag in flags}
        self.assertEqual(by_id["D"].milestone, "3 dB loss")
        self.assertEqual(by_id["D"].hours, 0.0)
        self.assertEqual(by_id["D"].status, "degrading")
        self.assertEqual(by_id["A"].milestone, "2 dB noise")

    def test_never_draining_beacon_flag_is_never(self):
        flags = forecast_flags(_fresh_plan(drain_rate_per_hour=0.0))
        for flag in flags:
            self.assertEqual(flag.milestone, "never")
            self.assertTrue(math.isinf(flag.hours))
            self.assertTrue(math.isinf(flag.hours_to_dark))
            self.assertEqual(flag.status, "ok")
            # JSON-friendly: never-reached horizons are null.
            self.assertIsNone(flag.to_dict()["hours"])
            self.assertIsNone(flag.to_dict()["hours_to_dark"])

    def test_flags_count_down_as_the_track_runs(self):
        flags = forecast_flags(_fresh_plan(), elapsed_s=3600.0)
        for flag in flags:
            self.assertAlmostEqual(flag.hours, 12.3333, places=3)

    def test_milestone_already_past_floors_at_zero(self):
        flags = forecast_flags(_fresh_plan(levels={"D": 0.2}),
                               elapsed_s=3600.0)
        by_id = {flag.beacon_id: flag for flag in flags}
        self.assertEqual(by_id["D"].hours, 0.0)
        self.assertAlmostEqual(by_id["A"].hours, 12.3333, places=3)

    def test_alerts_only_inside_the_window(self):
        flags = forecast_flags(_fresh_plan())
        # 13.3 h is inside the default 24 h window, outside a 1 h one.
        self.assertEqual(len(forecast_alerts(
            flags, DEFAULT_FORECAST_WINDOW_HOURS)), 4)
        self.assertEqual(forecast_alerts(flags, 1.0), ())
        # A beacon that never drains never alerts, whatever the window.
        never = forecast_flags(_fresh_plan(drain_rate_per_hour=0.0))
        self.assertEqual(forecast_alerts(never, 1000.0), ())

    def test_alert_wording(self):
        flags = forecast_flags(_fresh_plan(levels={"D": 0.2}))
        alerts = forecast_alerts(flags, DEFAULT_FORECAST_WINDOW_HOURS)
        self.assertIn("D: 3 dB loss now (degrading)", alerts)
        self.assertIn("A: 2 dB noise in 13.3h (ok)", alerts)


class LiveTrackerForecastTest(unittest.TestCase):
    def _walk(self, steps_per_side=10, seed=7):
        return simulate_walk(rectangular_walk(20.0, 12.0, steps_per_side),
                             BEACONS, noise_db=2.0, seed=seed)

    def test_no_plan_means_no_forecasts(self):
        tracker = LiveTracker(PositioningEngine(BEACONS))
        live = tracker.feed(self._walk()[0]["rssi"], 0.0)
        self.assertEqual(live.forecasts, ())
        self.assertEqual(live.forecast_alerts, ())
        self.assertEqual(tracker.forecasts, ())
        self.assertEqual(tracker.forecast_alerts, ())

    def test_position_carries_forecast_flags_and_alerts(self):
        tracker = LiveTracker(PositioningEngine(BEACONS),
                              plan=_fresh_plan())
        live = tracker.feed(self._walk()[0]["rssi"], 0.0)
        self.assertEqual(len(live.forecasts), 4)
        self.assertEqual([f.beacon_id for f in live.forecasts],
                         ["A", "B", "C", "D"])
        self.assertEqual(len(live.forecast_alerts), 4)
        self.assertIn("A: 2 dB noise in 13.3h (ok)",
                      live.forecast_alerts)
        # The flags are JSON-friendly, for the stream output.
        json.dumps(live.to_dict())
        self.assertEqual(
            live.to_dict()["forecasts"][0]["milestone"], "2 dB noise")
        # The tracker exposes the same flags without a position.
        self.assertEqual(tracker.forecasts, live.forecasts)
        self.assertEqual(tracker.forecast_alerts, live.forecast_alerts)

    def test_narrow_window_keeps_flags_but_drops_alerts(self):
        tracker = LiveTracker(PositioningEngine(BEACONS),
                              plan=_fresh_plan(),
                              forecast_window_hours=1.0)
        live = tracker.feed(self._walk()[0]["rssi"], 0.0)
        self.assertEqual(len(live.forecasts), 4)
        self.assertEqual(live.forecast_alerts, ())

    def test_horizons_shrink_as_the_track_runs(self):
        tracker = LiveTracker(PositioningEngine(BEACONS),
                              plan=_fresh_plan())
        steps = self._walk()
        first = tracker.feed(steps[0]["rssi"], 0.0)
        # One hour later the same beacons are one hour closer.
        later = tracker.feed(steps[1]["rssi"], 3600.0)
        self.assertAlmostEqual(first.forecasts[0].hours, 13.3333,
                               places=3)
        self.assertAlmostEqual(later.forecasts[0].hours, 12.3333,
                               places=3)

    def test_reset_restarts_the_forecast_clock(self):
        tracker = LiveTracker(PositioningEngine(BEACONS),
                              plan=_fresh_plan())
        steps = self._walk()
        tracker.feed(steps[0]["rssi"], 0.0)
        tracker.feed(steps[1]["rssi"], 3600.0)
        tracker.reset()
        live = tracker.feed(steps[0]["rssi"], 7200.0)
        self.assertAlmostEqual(live.forecasts[0].hours, 13.3333,
                               places=3)

    def test_nonpositive_window_is_rejected(self):
        with self.assertRaises(ValueError):
            LiveTracker(PositioningEngine(BEACONS), plan=_fresh_plan(),
                        forecast_window_hours=0.0)

    def test_null_line_still_carries_forecasts(self):
        # A reading too thin to locate has no position, but the
        # dashboard still needs to see a beacon about to go dark.
        tracker = LiveTracker(PositioningEngine(BEACONS),
                              plan=_fresh_plan(levels={"D": 0.2}))
        position = tracker.feed({"A": -62.1, "B": -71.4}, 0.0)
        self.assertIsNone(position)
        line = format_output(0.0, position, tracker.heard,
                             tracker.excluded, tracker.forecasts,
                             tracker.forecast_alerts)
        self.assertIsNone(line["position"])
        self.assertEqual(len(line["forecasts"]), 4)
        self.assertIn("D: 3 dB loss now (degrading)",
                      line["forecast_alerts"])


class TrackingForecastCliTest(unittest.TestCase):
    def test_demo_with_forecast_flags_every_line(self):
        out = io.StringIO()
        self.assertEqual(main(["--demo", "--forecast"], None, out), 0)
        lines = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(lines), 40)
        for line in lines:
            self.assertEqual(len(line["position"]["forecasts"]), 4)
            self.assertEqual(len(line["position"]["forecast_alerts"]), 4)
            self.assertEqual(
                line["position"]["forecasts"][0]["milestone"],
                "2 dB noise")

    def test_demo_without_forecast_has_empty_flags(self):
        out = io.StringIO()
        self.assertEqual(main(["--demo"], None, out), 0)
        line = json.loads(out.getvalue().splitlines()[0])
        self.assertEqual(line["position"]["forecasts"], [])
        self.assertEqual(line["position"]["forecast_alerts"], [])

    def test_demo_forecast_with_battery_levels(self):
        out = io.StringIO()
        self.assertEqual(main(["--demo", "--forecast",
                               "--battery-levels", "A=0.9,B=0.6,C=0.4,D=0.2"],
                              None, out), 0)
        line = json.loads(out.getvalue().splitlines()[0])
        by_id = {f["beacon"]: f for f in line["position"]["forecasts"]}
        self.assertEqual(by_id["D"]["milestone"], "3 dB loss")
        self.assertEqual(by_id["D"]["hours"], 0.0)
        self.assertEqual(by_id["D"]["status"], "degrading")
        self.assertIn("D: 3 dB loss now (degrading)",
                      line["position"]["forecast_alerts"])
        # C is due its 2 dB noise milestone in 1.3 h.
        self.assertAlmostEqual(by_id["C"]["hours"], 1.3333, places=3)

    def test_demo_forecast_with_narrow_window(self):
        out = io.StringIO()
        self.assertEqual(main(["--demo", "--forecast",
                               "--forecast-window-hours", "1.0"],
                              None, out), 0)
        line = json.loads(out.getvalue().splitlines()[0])
        self.assertEqual(len(line["position"]["forecasts"]), 4)
        self.assertEqual(line["position"]["forecast_alerts"], [])

    def test_stream_with_forecast(self):
        readings = "\n".join(
            json.dumps({"t": float(i),
                        "rssi": {"A": -62.1, "B": -71.4,
                                 "C": -68.0, "D": -59.8}})
            for i in range(3))
        out = io.StringIO()
        self.assertEqual(main(["--forecast"], io.StringIO(readings), out),
                         0)
        lines = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(lines), 3)
        self.assertEqual(len(lines[0]["position"]["forecasts"]), 4)
        self.assertEqual(len(lines[0]["position"]["forecast_alerts"]), 4)

    def test_forecast_with_bad_battery_levels_fails(self):
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(main(["--demo", "--forecast",
                                   "--battery-levels", "Z=0.5"],
                                  None, out), 2)
        self.assertIn("unknown beacon", err.getvalue())


if __name__ == "__main__":
    unittest.main()
