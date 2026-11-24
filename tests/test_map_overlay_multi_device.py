"""Per-device series on the ward-map overlay.

v0.44.0: a multi-device track renders one series per device on
the static overlay — each device's own dots, trail and ellipses
in its own colour, its own start and end markers, and drift
alarms read off each device's own rows — using the same grouping
and the same colours as the replay, so the two cannot disagree.
"""

import unittest

from beacon.beacon import Beacon
from beacon.map_overlay import (
    DEVICE_COLOURS,
    TRACK_COLOUR,
    _group_rows_by_device,
    render_map_overlay,
)
from beacon.replay import render_track_replay

BEACONS = [
    Beacon("A", 0.0, 0.0),
    Beacon("B", 20.0, 0.0),
    Beacon("C", 20.0, 12.0),
    Beacon("D", 0.0, 12.0),
]


class TestMultiDeviceWardMapOverlay(unittest.TestCase):
    def _row(self, x, y, semi_major, floor=0, t=0.0, device=None,
             alarms=(), classifications=()):
        row = {
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
        if device is not None:
            row["device"] = device
        return row

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

    def test_two_devices_get_their_own_series(self):
        rows = self._two_devices()
        svg = render_map_overlay(rows, BEACONS)
        # One trail, one start and one end marker per device,
        # one ellipse and one dot per row — and no zigzag path
        # joining the two devices.
        self.assertEqual(svg.count('data-trail="true"'), 2)
        self.assertEqual(svg.count(">S</text>"), 2)
        self.assertEqual(svg.count(">E</text>"), 2)
        self.assertEqual(svg.count("<ellipse"), 6)
        self.assertEqual(svg.count('data-index="'), 6)
        # Each device's trail and markers carry its id, in its
        # own colour: tag-1 the replay's pink, tag-2 blue.
        self.assertIn('data-trail="true" data-device="tag-1"', svg)
        self.assertIn('data-trail="true" data-device="tag-2"', svg)
        self.assertIn(
            f'stroke="{DEVICE_COLOURS[0]}" stroke-width="2" '
            f'data-trail="true" data-device="tag-1"', svg)
        self.assertIn(
            f'stroke="{DEVICE_COLOURS[1]}" stroke-width="2" '
            f'data-trail="true" data-device="tag-2"', svg)
        self.assertEqual(svg.count('data-device="tag-1"'), 9)
        self.assertEqual(svg.count('data-device="tag-2"'), 9)
        # The heading counts both devices, and the legend keys
        # each device's colour.
        self.assertIn("6 points across 2 devices", svg)
        self.assertIn("tag-1 track", svg)
        self.assertIn("tag-2 track", svg)

    def test_single_unnamed_device_renders_as_before(self):
        rows = [self._row(2.0, 2.0, 1.0, t=0.0),
                self._row(5.0, 5.0, 1.0, t=1.0)]
        svg = render_map_overlay(rows, BEACONS)
        self.assertNotIn("data-device=", svg)
        self.assertNotIn("data-trail=", svg)
        self.assertNotIn("across 2 devices", svg)
        self.assertEqual(svg.count("<polyline"), 1)
        self.assertIn(f'stroke="{TRACK_COLOUR}"', svg)
        self.assertEqual(svg.count(">S</text>"), 1)
        self.assertEqual(svg.count(">E</text>"), 1)
        self.assertIn("2 points, coverage 100%", svg)

    def test_interleaved_rows_group_per_device(self):
        rows = self._two_devices()
        interleaved = [rows[0], rows[3], rows[1],
                       rows[4], rows[2], rows[5]]
        svg = render_map_overlay(interleaved, BEACONS)
        # The grouping reads the device tag, not the file order:
        # still two trails, still one series per device.
        self.assertEqual(svg.count('data-trail="true"'), 2)
        self.assertEqual(svg.count("<ellipse"), 6)
        self.assertEqual(svg.count('data-device="tag-1"'), 9)
        self.assertEqual(svg.count('data-device="tag-2"'), 9)
        # Each device's trail follows only its own points: the
        # tag-1 trail starts at tag-1's first point, not at a
        # point tag-2 walked.
        self.assertIn('stroke="#d81b60" stroke-width="2" '
                      'data-trail="true" data-device="tag-1"', svg)

    def test_drift_badges_stay_per_device(self):
        # tag-1 alarms on A at its last row; tag-2 stays healthy.
        rows = [
            self._row(2.0, 2.0, 1.0, t=0.0, device="tag-1"),
            self._row(5.0, 5.0, 1.0, t=1.0, device="tag-1",
                      alarms=[self._alarm("A", 8.5)],
                      classifications=[
                          self._classification("A", "uncertain", 8.5)]),
            self._row(18.0, 10.0, 1.0, t=0.0, device="tag-2"),
            self._row(15.0, 7.0, 1.0, t=1.0, device="tag-2"),
        ]
        svg = render_map_overlay(rows, BEACONS, drift=True)
        # Only A is badged, once — and the label and the legend
        # name tag-1's alarm, not tag-2's silence.
        self.assertEqual(svg.count('data-drift-alarm="'), 1)
        self.assertIn('data-drift-alarm="A"', svg)
        self.assertIn("tag-1 \u00b7 ALARM +8.5 dB \u00b7 uncertain", svg)
        self.assertIn("Drift alarms at end of track: "
                      "tag-1: A +8.5 dB (uncertain); tag-2: none", svg)

    def test_cleared_alarm_on_one_device_does_not_badge(self):
        # tag-1 alarmed mid-track and cleared; tag-2 alarms at
        # its end — only tag-2's alarm is marked.
        rows = [
            self._row(2.0, 2.0, 1.0, t=0.0, device="tag-1",
                      alarms=[self._alarm("A", 8.5)],
                      classifications=[
                          self._classification("A", "uncertain", 8.5)]),
            self._row(5.0, 5.0, 1.0, t=1.0, device="tag-1"),
            self._row(18.0, 10.0, 1.0, t=0.0, device="tag-2"),
            self._row(15.0, 7.0, 1.0, t=1.0, device="tag-2",
                      alarms=[self._alarm("B", 6.2)],
                      classifications=[
                          self._classification("B", "moved", 6.2)]),
        ]
        svg = render_map_overlay(rows, BEACONS, drift=True)
        self.assertEqual(svg.count('data-drift-alarm="'), 1)
        self.assertIn('data-drift-alarm="B"', svg)
        self.assertIn("tag-2 \u00b7 ALARM +6.2 dB \u00b7 moved", svg)
        self.assertIn("Drift alarms at end of track: "
                      "tag-1: none; tag-2: B +6.2 dB (moved)", svg)

    def test_overlay_and_replay_agree_on_device_colours(self):
        rows = self._two_devices()
        overlay = render_map_overlay(rows, BEACONS)
        replay = render_track_replay(rows, BEACONS)
        # The same device is the same colour on the overlay and
        # in the replay: tag-1's trail stroke is tag-1's dot
        # fill, tag-2's likewise.
        for device, colour in (("tag-1", DEVICE_COLOURS[0]),
                               ("tag-2", DEVICE_COLOURS[1])):
            self.assertIn(
                f'stroke="{colour}" stroke-width="2" '
                f'data-trail="true" data-device="{device}"', overlay)
            self.assertIn(
                f'fill="{colour}" stroke="#ffffff"', replay)
            self.assertIn(f'data-dot="true" data-device="{device}"',
                          replay)

    def test_group_rows_by_device(self):
        rows = self._two_devices()
        grouped = _group_rows_by_device(rows)
        self.assertEqual([device for device, _ in grouped],
                         ["tag-1", "tag-2"])
        self.assertEqual([len(device_rows) for _, device_rows in grouped],
                         [3, 3])
        # A row with no device is the single unnamed device.
        single = [self._row(2.0, 2.0, 1.0, t=0.0),
                  self._row(5.0, 5.0, 1.0, t=1.0)]
        self.assertEqual(_group_rows_by_device(single),
                         [(None, single)])


if __name__ == "__main__":
    unittest.main()
