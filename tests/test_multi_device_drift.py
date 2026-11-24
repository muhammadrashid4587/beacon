"""Tests for drift flags on multi-device tracks (beacon 0.34.0).

``MultiDeviceTracker`` takes an optional ``DriftMonitor`` and gives
each device's tracker its own fresh monitor with the same tuning,
so every device's live position carries the beacons currently
alarming on its own track, with a plain-language alert per alarming
beacon. One monitor per device, not one shared across them: a
shared window would mix two devices' readings, so one tag's
healthy readings would dilute another tag's alarm, and resetting
one device would clear the monitor for every device.

The reference is each track's own estimate, not ground truth, and
the tests assert the measured consequences of that, including the
ones that are limits rather than features: a beacon reading too
loud alarms on the right beacon, a beacon reading too quiet can
alarm on the wrong one (the estimate is dragged with it).
"""

import io
import json
import unittest
from contextlib import redirect_stderr

from beacon import Beacon, DriftMonitor, MultiDeviceTracker
from beacon.multi_device import format_device_output, main as multi_main
from beacon.multi_device import run_demo
from beacon.simulate import rectangular_walk, simulate_walk

# Four corner beacons in a 20 x 12 m ward, matching the CLI defaults.
BEACONS = [
    Beacon("A", 0.0, 0.0),
    Beacon("B", 20.0, 0.0),
    Beacon("C", 20.0, 12.0),
    Beacon("D", 0.0, 12.0),
]

FULL = {"A": -62.1, "B": -71.4, "C": -68.0, "D": -59.8}


def _walk(seed=7, modify=None):
    steps = simulate_walk(rectangular_walk(20.0, 12.0, 10), BEACONS,
                          noise_db=2.0, seed=seed)
    if modify is not None:
        for step in steps:
            modify(step)
    return steps


def _feed_walk(tracker, device_id, steps):
    """Feed a walk through one device's track, return its positions."""
    return [tracker.feed(device_id, step["rssi"], float(i))
            for i, step in enumerate(steps)]


class MultiDeviceDriftTest(unittest.TestCase):
    def test_no_monitor_means_no_drift_fields(self):
        tracker = MultiDeviceTracker(BEACONS)
        live = tracker.feed("tag-1", FULL, 0.0)
        self.assertEqual(live.drift_alarms, ())
        self.assertEqual(live.drift_alerts, ())
        self.assertEqual(tracker.drift_alarms("tag-1"), ())
        self.assertEqual(tracker.drift_alerts("tag-1"), ())
        self.assertEqual(tracker.drift_classifications("tag-1"), ())
        # The fields are still in the JSON form, empty.
        self.assertEqual(live.to_dict()["drift_alarms"], [])
        self.assertEqual(live.to_dict()["drift_alerts"], [])
        # A device never seen has no alarms either.
        self.assertEqual(tracker.drift_alarms("tag-9"), ())

    def test_healthy_walk_raises_no_alarm_on_any_device(self):
        tracker = MultiDeviceTracker(
            BEACONS, drift_monitor=DriftMonitor(BEACONS))
        for device_id, seed in (("tag-1", 7), ("tag-2", 11)):
            positions = _feed_walk(tracker, device_id, _walk(seed))
            self.assertEqual(tracker.drift_alarms(device_id), (),
                             f"{device_id} alarmed on a healthy walk")
            for position in positions:
                if position is not None:
                    self.assertEqual(position.drift_alarms, ())

    def test_loud_beacon_alarms_on_the_right_beacon_only_there(self):
        # A reads 12 dB too loud on tag-1's walk only: tag-1
        # alarms on A at a rolling mean of about +8 dB — attenuated
        # because the estimate is pulled towards the loud beacon —
        # while tag-2, walking a healthy ward, stays quiet. The
        # monitors are separate: tag-2's healthy readings never
        # dilute tag-1's alarm.
        for seed in (1, 3, 7):
            tracker = MultiDeviceTracker(
                BEACONS, drift_monitor=DriftMonitor(BEACONS))
            positions = _feed_walk(tracker, "tag-1", _walk(
                seed, lambda s: s["rssi"].__setitem__(
                    "A", s["rssi"]["A"] + 12.0)))
            _feed_walk(tracker, "tag-2", _walk(11))
            alarms = tracker.drift_alarms("tag-1")
            self.assertEqual([a.id for a in alarms], ["A"],
                             f"seed {seed}")
            self.assertGreater(alarms[0].mean_residual_db, 5.0)
            self.assertLess(alarms[0].mean_residual_db, 12.0)
            self.assertEqual(alarms[0].samples, 20)
            self.assertEqual(tracker.drift_alarms("tag-2"), (),
                             f"seed {seed}")
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

    def test_loud_beacon_on_both_devices_alarms_on_both(self):
        # The same beacon going bad is visible from either tag:
        # each device's own monitor alarms on it.
        tracker = MultiDeviceTracker(
            BEACONS, drift_monitor=DriftMonitor(BEACONS))
        for device_id, seed in (("tag-1", 7), ("tag-2", 11)):
            _feed_walk(tracker, device_id, _walk(
                seed, lambda s: s["rssi"].__setitem__(
                    "A", s["rssi"]["A"] + 12.0)))
            self.assertEqual(
                [a.id for a in tracker.drift_alarms(device_id)], ["A"],
                device_id)

    def test_quiet_beacon_can_alarm_on_the_wrong_beacon(self):
        # C reads 8 dB too quiet — the battery-sag direction — on
        # tag-1's walk, and the estimate diverges towards it, so
        # the monitor alarms on D, not on C: the same documented
        # limit as the single-device track. Tag-2 stays quiet.
        for seed in (1, 3, 7):
            tracker = MultiDeviceTracker(
                BEACONS, drift_monitor=DriftMonitor(BEACONS))
            _feed_walk(tracker, "tag-1", _walk(
                seed, lambda s: s["rssi"].__setitem__(
                    "C", s["rssi"]["C"] - 8.0)))
            _feed_walk(tracker, "tag-2", _walk(11))
            alarms = tracker.drift_alarms("tag-1")
            self.assertTrue(alarms, f"seed {seed} raised no alarm")
            self.assertNotIn("C", [a.id for a in alarms],
                             f"seed {seed}")
            self.assertEqual(tracker.drift_alarms("tag-2"), ())

    def test_reset_clears_only_that_devices_monitor(self):
        # Both devices alarm on A; resetting tag-1 clears only
        # tag-1's monitor — tag-2 keeps alarming. A shared monitor
        # would have cleared both.
        tracker = MultiDeviceTracker(
            BEACONS, drift_monitor=DriftMonitor(BEACONS))
        for device_id, seed in (("tag-1", 7), ("tag-2", 11)):
            _feed_walk(tracker, device_id, _walk(
                seed, lambda s: s["rssi"].__setitem__(
                    "A", s["rssi"]["A"] + 12.0)))
        self.assertEqual([a.id for a in tracker.drift_alarms("tag-1")],
                         ["A"])
        self.assertEqual([a.id for a in tracker.drift_alarms("tag-2")],
                         ["A"])
        tracker.reset("tag-1")
        self.assertEqual(tracker.drift_alarms("tag-1"), ())
        self.assertEqual([a.id for a in tracker.drift_alarms("tag-2")],
                         ["A"])
        # A fresh healthy walk on tag-1 after the reset stays quiet.
        _feed_walk(tracker, "tag-1", _walk(7))
        self.assertEqual(tracker.drift_alarms("tag-1"), ())

    def test_template_monitor_is_not_fed(self):
        # The monitor passed in is a template for tuning, not the
        # monitor the devices feed: it stays empty however much
        # the devices walk, and each device gets its own window.
        template = DriftMonitor(BEACONS)
        tracker = MultiDeviceTracker(BEACONS, drift_monitor=template)
        _feed_walk(tracker, "tag-1", _walk(
            7, lambda s: s["rssi"].__setitem__(
                "A", s["rssi"]["A"] + 12.0)))
        self.assertEqual(template.alarms(), [])
        self.assertIsNone(template.status("A"))
        self.assertEqual([a.id for a in tracker.drift_alarms("tag-1")],
                         ["A"])

    def test_fresh_keeps_the_template_tuning(self):
        template = DriftMonitor(BEACONS, window=10,
                                drift_threshold_db=3.0, min_samples=3)
        tracker = MultiDeviceTracker(BEACONS, drift_monitor=template)
        tracker.feed("tag-1", FULL, 0.0)
        monitor = tracker.tracker("tag-1").drift_monitor
        self.assertIsNotNone(monitor)
        self.assertEqual(monitor._window, 10)
        self.assertEqual(monitor._threshold, 3.0)
        self.assertEqual(monitor._min_samples, 3)
        self.assertIsNot(monitor, template)

    def test_null_line_still_carries_drift_alarms(self):
        # A reading too thin to locate has no position, but the
        # dashboard still needs to see the beacon going bad on
        # that device. The reading arrives long after the walk,
        # so the other beacons have aged out of the merged set
        # and only two fresh beacons remain.
        tracker = MultiDeviceTracker(
            BEACONS, drift_monitor=DriftMonitor(BEACONS))
        _feed_walk(tracker, "tag-1", _walk(
            7, lambda s: s["rssi"].__setitem__(
                "A", s["rssi"]["A"] + 12.0)))
        self.assertEqual([a.id for a in tracker.drift_alarms("tag-1")],
                         ["A"])
        position = tracker.feed("tag-1", {"A": -62.1, "B": -71.4},
                                100.0)
        self.assertIsNone(position)
        device_tracker = tracker.tracker("tag-1")
        line = format_device_output(
            "tag-1", 100.0, position, device_tracker.heard,
            device_tracker.excluded,
            drift_alarms=device_tracker.drift_alarms,
            drift_alerts=device_tracker.drift_alerts)
        self.assertIsNone(line["position"])
        self.assertEqual([a["id"] for a in line["drift_alarms"]], ["A"])
        self.assertTrue(line["drift_alerts"][0].startswith("A: RSSI +"))


class MultiDeviceDriftCliTest(unittest.TestCase):
    def test_demo_with_drift_has_empty_alarms(self):
        out = io.StringIO()
        self.assertEqual(multi_main(["--demo", "--drift"], None, out), 0)
        lines = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(lines), 80)
        for line in lines:
            self.assertEqual(line["position"]["drift_alarms"], [])
            self.assertEqual(line["position"]["drift_alerts"], [])

    def test_demo_without_drift_has_empty_fields(self):
        out = io.StringIO()
        self.assertEqual(multi_main(["--demo"], None, out), 0)
        line = json.loads(out.getvalue().splitlines()[0])
        self.assertEqual(line["position"]["drift_alarms"], [])
        self.assertEqual(line["position"]["drift_alerts"], [])

    def test_demo_drift_beacon_alarms_on_tag_1_only(self):
        out = io.StringIO()
        self.assertEqual(multi_main(["--demo", "--drift",
                                     "--drift-beacon", "A:12"],
                                    None, out), 0)
        lines = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(lines), 80)
        tag1 = [line for line in lines if line["device"] == "tag-1"]
        tag2 = [line for line in lines if line["device"] == "tag-2"]
        # The first steps are quiet — the monitor needs 5 readings
        # before it can alarm — and tag-1's last step alarms on A.
        self.assertEqual(tag1[0]["position"]["drift_alarms"], [])
        last = tag1[-1]["position"]
        self.assertEqual([a["id"] for a in last["drift_alarms"]], ["A"])
        self.assertGreater(last["drift_alarms"][0]["mean_residual_db"],
                           5.0)
        self.assertEqual(len(last["drift_alerts"]), 1)
        self.assertTrue(last["drift_alerts"][0].startswith("A: RSSI +"))
        # Tag-2 walked a healthy ward: it never alarms.
        for line in tag2:
            self.assertEqual(line["position"]["drift_alarms"], [])
            self.assertEqual(line["position"]["drift_alerts"], [])

    def test_stream_with_drift(self):
        # A simulated walk with A 12 dB too loud, streamed as JSON
        # lines for tag-1, then a healthy walk for tag-2: only
        # tag-1's last line alarms, on A.
        steps = _walk(7, lambda s: s["rssi"].__setitem__(
            "A", s["rssi"]["A"] + 12.0))
        readings = "\n".join(
            json.dumps({"device": "tag-1", "t": float(i),
                        "rssi": step["rssi"]})
            for i, step in enumerate(steps))
        readings += "\n" + "\n".join(
            json.dumps({"device": "tag-2", "t": float(i),
                        "rssi": step["rssi"]})
            for i, step in enumerate(_walk(11)))
        out = io.StringIO()
        self.assertEqual(multi_main(["--drift"], io.StringIO(readings),
                                    out), 0)
        lines = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(lines), 80)
        tag1 = [line for line in lines if line["device"] == "tag-1"]
        tag2 = [line for line in lines if line["device"] == "tag-2"]
        self.assertEqual(tag1[0]["position"]["drift_alarms"], [])
        last = tag1[-1]["position"]
        self.assertEqual([a["id"] for a in last["drift_alarms"]], ["A"])
        self.assertTrue(last["drift_alerts"][0].startswith("A: RSSI +"))
        for line in tag2:
            self.assertEqual(line["position"]["drift_alarms"], [])

    def test_drift_beacon_needs_drift(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                multi_main(["--demo", "--drift-beacon", "A:12"],
                           None, io.StringIO())

    def test_drift_beacon_unknown_beacon_fails(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                multi_main(["--demo", "--drift", "--drift-beacon", "Z:12"],
                           None, io.StringIO())

    def test_drift_beacon_bad_spec_fails(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                multi_main(["--demo", "--drift", "--drift-beacon", "A12"],
                           None, io.StringIO())


if __name__ == "__main__":
    unittest.main()
