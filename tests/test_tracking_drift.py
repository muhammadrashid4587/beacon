"""Tests for drift flags on the live track (beacon 0.33.0).

``LiveTracker`` takes an optional ``DriftMonitor`` and feeds it each
located step's fresh readings together with the live position just
computed, so the position carries the beacons currently alarming
plus a plain-language alert per alarming beacon.

The reference is the track's own estimate, not ground truth, and
the tests assert the measured consequences of that, including the
ones that are limits rather than features: a beacon reading too
loud alarms on the right beacon, a beacon reading too quiet can
alarm on the wrong one (the estimate is dragged with it), and a
drift under the alarm threshold is missed entirely.
"""

import io
import json
import unittest
from contextlib import redirect_stderr

from beacon import (Beacon, DriftAlarm, DriftMonitor, LiveTracker,
                    PositioningEngine, drift_alerts)
from beacon.simulate import (rectangular_walk, simulate_multifloor_walk,
                             simulate_walk, two_floor_walk)
from beacon.tracking import format_output, main

# Four corner beacons in a 20 x 12 m ward, matching the CLI defaults.
BEACONS = [
    Beacon("A", 0.0, 0.0),
    Beacon("B", 20.0, 0.0),
    Beacon("C", 20.0, 12.0),
    Beacon("D", 0.0, 12.0),
]


def _walk(seed=7, modify=None):
    steps = simulate_walk(rectangular_walk(20.0, 12.0, 10), BEACONS,
                          noise_db=2.0, seed=seed)
    if modify is not None:
        for step in steps:
            modify(step)
    return steps


def _run(steps, monitor=None):
    """Feed a walk through a tracker with a monitor, return both."""
    monitor = monitor if monitor is not None else DriftMonitor(BEACONS)
    tracker = LiveTracker(PositioningEngine(BEACONS),
                          drift_monitor=monitor)
    positions = [tracker.feed(step["rssi"], float(i))
                 for i, step in enumerate(steps)]
    return tracker, positions


class DriftAlertsTest(unittest.TestCase):
    def test_alert_wording(self):
        self.assertEqual(
            drift_alerts((DriftAlarm("A", 8.53, 20),)),
            ("A: RSSI +8.5 dB off calibration",))
        self.assertEqual(
            drift_alerts((DriftAlarm("C", -6.93, 20),)),
            ("C: RSSI -6.9 dB off calibration",))
        self.assertEqual(drift_alerts(()), ())

    def test_alarm_is_json_friendly(self):
        alarm = DriftAlarm("A", 8.53, 20)
        self.assertEqual(alarm.to_dict(), {
            "id": "A", "mean_residual_db": 8.53, "samples": 20})


class LiveTrackerDriftTest(unittest.TestCase):
    def test_no_monitor_means_no_drift_fields(self):
        tracker = LiveTracker(PositioningEngine(BEACONS))
        live = tracker.feed(_walk()[0]["rssi"], 0.0)
        self.assertEqual(live.drift_alarms, ())
        self.assertEqual(live.drift_alerts, ())
        self.assertEqual(tracker.drift_alarms, ())
        self.assertEqual(tracker.drift_alerts, ())
        self.assertEqual(tracker.drift_classifications(), ())
        self.assertIsNone(tracker.drift_monitor)
        # The fields are still in the JSON form, empty.
        self.assertEqual(live.to_dict()["drift_alarms"], [])
        self.assertEqual(live.to_dict()["drift_alerts"], [])

    def test_healthy_walk_raises_no_drift_alarm(self):
        for seed in (1, 2, 3, 7, 11):
            tracker, positions = _run(_walk(seed))
            self.assertEqual(tracker.drift_alarms, (),
                             f"seed {seed} alarmed on a healthy walk")
            for position in positions:
                if position is not None:
                    self.assertEqual(position.drift_alarms, ())

    def test_loud_beacon_alarms_on_the_right_beacon(self):
        # A reads 12 dB too loud: the estimate is pulled towards
        # it, so the rolling mean is attenuated to about +8 dB —
        # still past the 5 dB threshold, and only A alarms.
        for seed in (1, 3, 7):
            tracker, positions = _run(_walk(
                seed, lambda s: s["rssi"].__setitem__(
                    "A", s["rssi"]["A"] + 12.0)))
            alarms = tracker.drift_alarms
            self.assertEqual([a.id for a in alarms], ["A"],
                             f"seed {seed}")
            self.assertGreater(alarms[0].mean_residual_db, 5.0)
            self.assertLess(alarms[0].mean_residual_db, 12.0)
            self.assertEqual(alarms[0].samples, 20)
            last = positions[-1]
            self.assertEqual(last.drift_alarms, alarms)
            self.assertEqual(len(last.drift_alerts), 1)
            self.assertTrue(
                last.drift_alerts[0].startswith("A: RSSI +"))
            self.assertIn("off calibration", last.drift_alerts[0])
            # The alarm is JSON-friendly, for the stream output.
            json.dumps(last.to_dict())
            self.assertEqual(
                last.to_dict()["drift_alarms"][0]["id"], "A")
            # The tracker exposes the same alarms without a
            # position in hand.
            self.assertEqual(tracker.drift_alarms, last.drift_alarms)
            self.assertEqual(tracker.drift_alerts, last.drift_alerts)

    def test_loud_beacon_below_threshold_is_missed(self):
        # A reads 6 dB too loud — past the drift in the world, but
        # the estimate is pulled towards it, so the rolling mean
        # stays under the 5 dB threshold and nothing alarms. The
        # self-referenced monitor is less sensitive than the
        # monitor fed a reference position.
        tracker, _ = _run(_walk(
            7, lambda s: s["rssi"].__setitem__(
                "A", s["rssi"]["A"] + 6.0)))
        self.assertEqual(tracker.drift_alarms, ())

    def test_quiet_beacon_can_alarm_on_the_wrong_beacon(self):
        # C reads 8 dB too quiet — the battery-sag direction — and
        # the estimate diverges towards it, so the monitor alarms
        # on D, not on C. This is the documented limit of feeding
        # the monitor the track's own estimate: the alarm says the
        # track is inconsistent, but the beacon it names is the
        # prime suspect, not a proven culprit. A monitor fed a
        # reference position alarms on C here.
        for seed in (1, 3, 7):
            tracker, _ = _run(_walk(
                seed, lambda s: s["rssi"].__setitem__(
                    "C", s["rssi"]["C"] - 8.0)))
            alarms = tracker.drift_alarms
            self.assertTrue(alarms, f"seed {seed} raised no alarm")
            self.assertNotIn("C", [a.id for a in alarms],
                             f"seed {seed}")

    def test_alarm_appears_only_after_enough_readings(self):
        # A beacon needs 5 readings in its window before it can
        # alarm, so the first steps of a drifted walk are quiet.
        tracker, positions = _run(_walk(
            7, lambda s: s["rssi"].__setitem__(
                "A", s["rssi"]["A"] + 12.0)))
        self.assertEqual(positions[0].drift_alarms, ())
        self.assertEqual(positions[3].drift_alarms, ())
        self.assertEqual([a.id for a in positions[-1].drift_alarms],
                         ["A"])

    def test_reset_clears_the_monitor(self):
        tracker, _ = _run(_walk(
            7, lambda s: s["rssi"].__setitem__(
                "A", s["rssi"]["A"] + 12.0)))
        self.assertEqual([a.id for a in tracker.drift_alarms], ["A"])
        tracker.reset()
        self.assertEqual(tracker.drift_alarms, ())
        self.assertEqual(tracker.drift_classifications()[0].samples, 0)
        # A fresh healthy walk after the reset stays quiet.
        steps = _walk(7)
        for i, step in enumerate(steps):
            tracker.feed(step["rssi"], float(i))
        self.assertEqual(tracker.drift_alarms, ())

    def test_monitor_reset_forgets_everything(self):
        monitor = DriftMonitor(BEACONS)
        for step in _walk(7):
            monitor.feed(step)
        self.assertIsNotNone(monitor.status("A"))
        monitor.reset()
        self.assertIsNone(monitor.status("A"))
        self.assertEqual(monitor.alarms(), [])

    def test_null_line_still_carries_drift_alarms(self):
        # A reading too thin to locate has no position, but the
        # dashboard still needs to see the beacon going bad. The
        # reading arrives long after the walk, so the other
        # beacons have aged out of the merged set and only two
        # fresh beacons remain.
        tracker, _ = _run(_walk(
            7, lambda s: s["rssi"].__setitem__(
                "A", s["rssi"]["A"] + 12.0)))
        self.assertEqual([a.id for a in tracker.drift_alarms], ["A"])
        position = tracker.feed({"A": -62.1, "B": -71.4}, 100.0)
        self.assertIsNone(position)
        line = format_output(100.0, position, tracker.heard,
                             tracker.excluded,
                             drift_alarms=tracker.drift_alarms,
                             drift_alerts=tracker.drift_alerts)
        self.assertIsNone(line["position"])
        self.assertEqual([a["id"] for a in line["drift_alarms"]], ["A"])
        self.assertTrue(line["drift_alerts"][0].startswith("A: RSSI +"))

    def test_multifloor_walk_feeds_only_the_current_floor(self):
        # A healthy two-floor walk must not alarm: a beacon heard
        # through the slab would read as drift, so only the
        # current floor's beacons reach the monitor.
        beacons = [Beacon("A", 0, 0, floor=0), Beacon("B", 20, 0, floor=0),
                   Beacon("C", 20, 12, floor=0),
                   Beacon("D", 0, 12, floor=0),
                   Beacon("E", 0, 0, floor=1), Beacon("F", 20, 0, floor=1),
                   Beacon("G", 20, 12, floor=1),
                   Beacon("H", 0, 12, floor=1)]
        steps = simulate_multifloor_walk(
            two_floor_walk(20, 12, 10), beacons, noise_db=2.0, seed=7)
        monitor = DriftMonitor(beacons)
        tracker = LiveTracker(PositioningEngine(beacons),
                              drift_monitor=monitor)
        for i, step in enumerate(steps):
            tracker.feed(step["rssi"], float(i))
        self.assertEqual(tracker.drift_alarms, ())
        # Both floors were actually monitored, not skipped.
        self.assertIsNotNone(monitor.status("A"))
        self.assertIsNotNone(monitor.status("E"))


class TrackingDriftCliTest(unittest.TestCase):
    def test_demo_with_drift_has_empty_alarms(self):
        out = io.StringIO()
        self.assertEqual(main(["--demo", "--drift"], None, out), 0)
        lines = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(lines), 40)
        for line in lines:
            self.assertEqual(line["position"]["drift_alarms"], [])
            self.assertEqual(line["position"]["drift_alerts"], [])

    def test_demo_without_drift_has_empty_fields(self):
        out = io.StringIO()
        self.assertEqual(main(["--demo"], None, out), 0)
        line = json.loads(out.getvalue().splitlines()[0])
        self.assertEqual(line["position"]["drift_alarms"], [])
        self.assertEqual(line["position"]["drift_alerts"], [])

    def test_demo_drift_beacon_alarms_on_that_beacon(self):
        out = io.StringIO()
        self.assertEqual(main(["--demo", "--drift",
                               "--drift-beacon", "A:12"], None, out), 0)
        lines = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(lines), 40)
        # The first steps are quiet — the monitor needs 5 readings
        # before it can alarm — and the last step alarms on A.
        self.assertEqual(lines[0]["position"]["drift_alarms"], [])
        last = lines[-1]["position"]
        self.assertEqual([a["id"] for a in last["drift_alarms"]], ["A"])
        self.assertGreater(last["drift_alarms"][0]["mean_residual_db"],
                           5.0)
        self.assertEqual(len(last["drift_alerts"]), 1)
        self.assertTrue(last["drift_alerts"][0].startswith("A: RSSI +"))

    def test_stream_with_drift(self):
        # A simulated walk with A 12 dB too loud, streamed as
        # JSON lines: the last line alarms on A, and only A.
        steps = _walk(7, lambda s: s["rssi"].__setitem__(
            "A", s["rssi"]["A"] + 12.0))
        readings = "\n".join(
            json.dumps({"t": float(i), "rssi": step["rssi"]})
            for i, step in enumerate(steps))
        out = io.StringIO()
        self.assertEqual(main(["--drift"], io.StringIO(readings), out), 0)
        lines = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(lines), 40)
        self.assertEqual(lines[0]["position"]["drift_alarms"], [])
        last = lines[-1]["position"]
        self.assertEqual([a["id"] for a in last["drift_alarms"]], ["A"])
        self.assertTrue(last["drift_alerts"][0].startswith("A: RSSI +"))

    def test_drift_beacon_needs_drift(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--demo", "--drift-beacon", "A:12"],
                     None, io.StringIO())

    def test_drift_beacon_unknown_beacon_fails(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--demo", "--drift", "--drift-beacon", "Z:12"],
                     None, io.StringIO())

    def test_drift_beacon_bad_spec_fails(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--demo", "--drift", "--drift-beacon", "A12"],
                     None, io.StringIO())


if __name__ == "__main__":
    unittest.main()
