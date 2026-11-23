"""Tests for the battery-swap replacement schedule (beacon 0.29.0)."""

import io
import json
import math
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta

from beacon import (BatteryModel, ReplacementSchedule, SwapEntry,
                    plan_deployment, replacement_schedule)
from beacon.cli import main


def _default_plan(levels=None, drain_rate=0.05):
    return plan_deployment(
        {beacon_id: BatteryModel(drain_rate_per_hour=drain_rate)
         for beacon_id in ("A", "B", "C", "D")},
        levels=levels)


class ReplacementScheduleTest(unittest.TestCase):
    def test_fresh_ward_swaps_on_first_noise(self):
        # Every fresh beacon's first milestone is the 2 dB noise
        # one at (1 - 1/3) / 0.05 = 13.33 h — the noise arrives
        # before the 3 dB loss (14.14 h) and the dark (19 h).
        schedule = replacement_schedule(_default_plan())
        self.assertEqual(len(schedule.entries), 4)
        for entry in schedule.entries:
            self.assertAlmostEqual(entry.swap_by_hours, 13.3333,
                                   places=3)
            self.assertEqual(entry.reason, "2 dB noise")
            self.assertEqual(entry.status, "ok")
        # The ward drops below three lit beacons when the second
        # beacon goes dark — all four go dark at 19 h.
        self.assertAlmostEqual(schedule.ward_deadline_hours, 19.0)

    def test_measured_levels_order_soonest_first(self):
        # Levels 0.9, 0.6, 0.4, 0.2: D is already past both
        # thresholds (swap now), C's noise comes at 1.33 h, B's at
        # 5.33 h, A's at 11.33 h.
        schedule = replacement_schedule(_default_plan(
            levels={"A": 0.9, "B": 0.6, "C": 0.4, "D": 0.2}))
        self.assertEqual(
            [entry.beacon_id for entry in schedule.entries],
            ["D", "C", "B", "A"])
        first = schedule.entries[0]
        self.assertEqual(first.swap_by_hours, 0.0)
        # D is past both the 3 dB loss and the 2 dB noise at 0 h;
        # the loss wins the tie.
        self.assertEqual(first.reason, "3 dB loss")
        self.assertEqual(first.status, "degrading")
        self.assertAlmostEqual(schedule.entries[1].swap_by_hours,
                               1.3333, places=3)
        self.assertEqual(schedule.entries[1].reason, "2 dB noise")
        self.assertAlmostEqual(schedule.entries[2].swap_by_hours,
                               5.3333, places=3)
        self.assertAlmostEqual(schedule.entries[3].swap_by_hours,
                               11.3333, places=3)

    def test_dark_beacon_is_due_now_for_going_dark(self):
        # A beacon already below its cutoff is dark with 0 hours
        # to dark — and the dark is its soonest milestone, so the
        # reason is dark, not the loss or noise it is also past.
        plan = plan_deployment(
            {"A": BatteryModel(drain_rate_per_hour=0.05)},
            levels={"A": 0.04})
        schedule = replacement_schedule(plan)
        entry = schedule.entries[0]
        self.assertEqual(entry.swap_by_hours, 0.0)
        self.assertEqual(entry.reason, "dark")
        self.assertEqual(entry.status, "dark")

    def test_no_drain_beacon_never_needs_a_swap(self):
        plan = plan_deployment(
            {"A": BatteryModel(drain_rate_per_hour=0.0),
             "B": BatteryModel(drain_rate_per_hour=0.05)})
        schedule = replacement_schedule(plan)
        self.assertEqual(
            [entry.beacon_id for entry in schedule.entries],
            ["B", "A"])
        never = schedule.entries[1]
        self.assertTrue(math.isinf(never.swap_by_hours))
        self.assertEqual(never.reason, "never")
        self.assertIsNone(never.due_at())
        # The ward has only two beacons, so it is already below
        # three lit beacons — the deadline is now, not never.
        self.assertEqual(schedule.ward_deadline_hours, 0.0)

    def test_lead_time_pulls_swaps_earlier(self):
        schedule = replacement_schedule(_default_plan(),
                                        lead_time_hours=2.0)
        for entry in schedule.entries:
            self.assertAlmostEqual(entry.swap_by_hours, 11.3333,
                                   places=3)
        # A lead longer than the milestone floors the swap at now.
        schedule = replacement_schedule(_default_plan(),
                                        lead_time_hours=24.0)
        for entry in schedule.entries:
            self.assertEqual(entry.swap_by_hours, 0.0)

    def test_negative_lead_time_rejected(self):
        with self.assertRaises(ValueError):
            replacement_schedule(_default_plan(), lead_time_hours=-1.0)

    def test_due_at_is_now_plus_swap_by_hours(self):
        now = datetime(2026, 11, 23, 12, 0, 0)
        schedule = replacement_schedule(_default_plan())
        due = schedule.entries[0].due_at(now)
        self.assertEqual(due, now + timedelta(hours=13.333333333333334))

    def test_to_dict_nulls_never(self):
        plan = plan_deployment(
            {"A": BatteryModel(drain_rate_per_hour=0.0)})
        schedule = replacement_schedule(plan)
        data = schedule.to_dict()
        self.assertIsNone(data["entries"][0]["swap_by_hours"])
        self.assertIsNone(data["entries"][0]["hours_to_dark"])
        self.assertEqual(data["entries"][0]["reason"], "never")
        # A one-beacon ward is already below three lit beacons,
        # so its deadline is 0.0, not never.
        self.assertEqual(data["ward_deadline_hours"], 0.0)
        # And the dict round-trips through JSON.
        json.loads(json.dumps(data))


class ReplacementScheduleCliTest(unittest.TestCase):
    def test_cli_lists_swaps_soonest_first(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--replacement-schedule",
                         "--battery-levels", "A=0.9,B=0.6,C=0.4,D=0.2"])
        self.assertEqual(code, 0)
        lines = output.getvalue().splitlines()
        # Header, then D, C, B, A, then the ward deadline line.
        self.assertIn("beacon", lines[0])
        self.assertEqual([line.split()[0] for line in lines[1:5]],
                         ["D", "C", "B", "A"])
        self.assertIn("D", lines[1])
        self.assertIn("now", lines[1])
        self.assertIn("3 dB loss", lines[1])
        self.assertIn("ward drops below 3 lit beacons", lines[-1])

    def test_cli_fresh_ward_shows_datetimes(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--replacement-schedule"])
        self.assertEqual(code, 0)
        text = output.getvalue()
        self.assertIn("2 dB noise", text)
        self.assertIn("in 13.3h", text)

    def test_cli_bad_levels_exits_two(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--replacement-schedule",
                         "--battery-levels", "Z=0.5"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
