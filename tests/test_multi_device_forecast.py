"""Tests for planner forecast flags on multi-device tracks
(beacon 0.31.0)."""

import io
import json
import unittest
from contextlib import redirect_stderr

from beacon import (BatteryModel, Beacon, MultiDeviceTracker,
                    plan_deployment)
from beacon.multi_device import main as multi_main
from beacon.multi_device import run_demo

# Four corner beacons in a 20 x 12 m ward, matching the CLI defaults.
BEACONS = [
    Beacon("A", 0.0, 0.0),
    Beacon("B", 20.0, 0.0),
    Beacon("C", 20.0, 12.0),
    Beacon("D", 0.0, 12.0),
]

FULL = {"A": -62.1, "B": -71.4, "C": -68.0, "D": -59.8}


def _fresh_plan(drain_rate_per_hour=0.05, levels=None):
    return plan_deployment(
        {b.id: BatteryModel(drain_rate_per_hour=drain_rate_per_hour)
         for b in BEACONS},
        levels=levels)


class MultiDeviceForecastTest(unittest.TestCase):
    def test_each_device_position_carries_the_plan_flags(self):
        # Four fresh beacons at 0.05/hr: every flag is the 2 dB
        # noise milestone in 13.33 h, and all four alert inside
        # the 24-hour window.
        tracker = MultiDeviceTracker(BEACONS, plan=_fresh_plan())
        for device_id in ("tag-1", "tag-2"):
            live = tracker.feed(device_id, FULL, 0.0)
            self.assertIsNotNone(live)
            self.assertEqual(len(live.forecasts), 4)
            for flag in live.forecasts:
                self.assertEqual(flag.milestone, "2 dB noise")
                self.assertAlmostEqual(flag.hours, 13.3333, places=3)
                self.assertEqual(flag.status, "ok")
            self.assertEqual(len(live.forecast_alerts), 4)
            self.assertTrue(
                live.forecast_alerts[0].startswith("A: 2 dB noise"))

    def test_degraded_beacon_flags_on_every_device(self):
        # D measured at 20% charge is already past 3 dB of loss:
        # every device's track flags it due now.
        tracker = MultiDeviceTracker(
            BEACONS, plan=_fresh_plan(levels={"D": 0.2}))
        for device_id in ("tag-1", "tag-2"):
            live = tracker.feed(device_id, FULL, 0.0)
            by_id = {flag.beacon_id: flag for flag in live.forecasts}
            self.assertEqual(by_id["D"].milestone, "3 dB loss")
            self.assertEqual(by_id["D"].hours, 0.0)
            self.assertEqual(by_id["D"].status, "degrading")
            self.assertIn("D: 3 dB loss now (degrading)",
                          live.forecast_alerts)

    def test_each_device_counts_down_from_its_own_first_reading(self):
        # tag-1 starts at t=0, tag-2 only at t=3600 s (one hour
        # in): after tag-1 runs five more hours, its flags have
        # counted down 5 hours while tag-2's have counted down
        # only 4 — the plan is taken as made at each device's
        # first reading, not at the first tag's.
        tracker = MultiDeviceTracker(BEACONS, plan=_fresh_plan())
        tracker.feed("tag-1", FULL, 0.0)
        tracker.feed("tag-2", FULL, 3600.0)
        live_a = tracker.feed("tag-1", FULL, 18000.0)
        live_b = tracker.feed("tag-2", FULL, 18000.0)
        self.assertAlmostEqual(live_a.forecasts[0].hours, 8.3333,
                               places=3)
        self.assertAlmostEqual(live_b.forecasts[0].hours, 9.3333,
                               places=3)

    def test_no_plan_means_no_forecasts_on_any_device(self):
        tracker = MultiDeviceTracker(BEACONS)
        for device_id in ("tag-1", "tag-2"):
            live = tracker.feed(device_id, FULL, 0.0)
            self.assertEqual(live.forecasts, ())
            self.assertEqual(live.forecast_alerts, ())

    def test_narrow_window_keeps_flags_but_drops_alerts(self):
        tracker = MultiDeviceTracker(
            BEACONS, plan=_fresh_plan(), forecast_window_hours=1.0)
        live = tracker.feed("tag-1", FULL, 0.0)
        self.assertEqual(len(live.forecasts), 4)
        self.assertEqual(live.forecast_alerts, ())

    def test_reset_restarts_one_devices_forecast_clock(self):
        tracker = MultiDeviceTracker(BEACONS, plan=_fresh_plan())
        tracker.feed("tag-1", FULL, 0.0)
        tracker.feed("tag-2", FULL, 0.0)
        tracker.feed("tag-1", FULL, 18000.0)
        self.assertAlmostEqual(
            tracker.last_position("tag-1").forecasts[0].hours,
            8.3333, places=3)
        # Resetting tag-1 restarts its clock; tag-2 keeps counting
        # down from its own first reading.
        tracker.reset("tag-1")
        live_a = tracker.feed("tag-1", FULL, 18000.0)
        live_b = tracker.feed("tag-2", FULL, 18000.0)
        self.assertAlmostEqual(live_a.forecasts[0].hours, 13.3333,
                               places=3)
        self.assertAlmostEqual(live_b.forecasts[0].hours, 8.3333,
                               places=3)

    def test_nonpositive_window_is_rejected(self):
        with self.assertRaises(ValueError):
            MultiDeviceTracker(BEACONS, plan=_fresh_plan(),
                               forecast_window_hours=0.0)

    def test_null_line_still_carries_forecasts(self):
        # A device too thin to locate still gets the flags on its
        # null line, so a dashboard sees a beacon about to go
        # dark even for a device it cannot place.
        out = io.StringIO()
        lines = [
            '{"device": "tag-1", "t": 0.0, "rssi": {"A": -62.1, '
            '"B": -71.4}}',
        ]
        self.assertEqual(
            multi_main(argv=["--forecast"],
                       input_stream=io.StringIO("\n".join(lines)),
                       output_stream=out), 0)
        record = json.loads(out.getvalue().splitlines()[0])
        self.assertIsNone(record["position"])
        self.assertEqual(len(record["forecasts"]), 4)
        self.assertEqual(record["forecasts"][0]["milestone"],
                         "2 dB noise")
        self.assertEqual(len(record["forecast_alerts"]), 4)

    def test_demo_with_forecast_flags_every_line(self):
        out = io.StringIO()
        tracker = run_demo(5.0, 10.0, 3, 2.0, out,
                           plan=_fresh_plan())
        records = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(records), 80)
        for record in records:
            self.assertIsNotNone(record["position"])
            flags = record["position"]["forecasts"]
            self.assertEqual(len(flags), 4)
            self.assertEqual(flags[0]["milestone"], "2 dB noise")
            # The horizons count down as the demo runs: 13.33 h
            # at t=0, a couple of seconds less by the last step.
            self.assertAlmostEqual(
                flags[0]["hours"],
                13.3333 - record["position"]["t"] / 3600.0, places=3)
            self.assertEqual(len(record["position"]["forecast_alerts"]),
                             4)
        # Both devices carry the same flags.
        for device_id in ("tag-1", "tag-2"):
            self.assertEqual(
                len(tracker.last_position(device_id).forecasts), 4)

    def test_demo_forecast_with_battery_levels(self):
        out = io.StringIO()
        tracker = run_demo(5.0, 10.0, 3, 2.0, out,
                           plan=_fresh_plan(levels={"D": 0.2}))
        records = [json.loads(line) for line in out.getvalue().splitlines()]
        first = records[0]["position"]
        by_id = {flag["beacon"]: flag for flag in first["forecasts"]}
        self.assertEqual(by_id["D"]["milestone"], "3 dB loss")
        self.assertEqual(by_id["D"]["hours"], 0.0)
        self.assertIn("D: 3 dB loss now (degrading)",
                      first["forecast_alerts"])

    def test_demo_without_forecast_has_empty_flags(self):
        out = io.StringIO()
        run_demo(5.0, 10.0, 3, 2.0, out)
        records = [json.loads(line) for line in out.getvalue().splitlines()]
        for record in records:
            self.assertEqual(record["position"]["forecasts"], [])
            self.assertEqual(record["position"]["forecast_alerts"], [])

    def test_stream_with_forecast(self):
        out = io.StringIO()
        lines = [
            '{"device": "tag-1", "t": 0.0, "rssi": {"A": -62.1, '
            '"B": -71.4, "C": -68.0, "D": -59.8}}',
            '{"device": "tag-2", "t": 0.0, "rssi": {"A": -70.0, '
            '"B": -65.0, "C": -60.0, "D": -72.0}}',
        ]
        self.assertEqual(
            multi_main(argv=["--forecast", "--battery-levels",
                             "A=0.9,B=0.6,C=0.4,D=0.2"],
                       input_stream=io.StringIO("\n".join(lines)),
                       output_stream=out), 0)
        records = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(records), 2)
        for record in records:
            by_id = {flag["beacon"]: flag
                     for flag in record["position"]["forecasts"]}
            self.assertEqual(by_id["D"]["milestone"], "3 dB loss")
            self.assertEqual(by_id["D"]["hours"], 0.0)
            self.assertEqual(by_id["C"]["milestone"], "2 dB noise")
            self.assertAlmostEqual(by_id["C"]["hours"], 1.3333,
                                   places=3)

    def test_forecast_with_bad_battery_levels_fails(self):
        out = io.StringIO()
        with redirect_stderr(io.StringIO()):
            self.assertEqual(
                multi_main(argv=["--forecast", "--battery-levels",
                                 "Z=0.5"],
                           input_stream=io.StringIO(""),
                           output_stream=out), 2)


if __name__ == "__main__":
    unittest.main()
