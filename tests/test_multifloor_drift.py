"""Drift classification on multi-floor walks (beacon 0.58.0).

The drift monitor used to reject multi-floor positions: a beacon
heard through a floor slab reads ~15 dB weaker than the path-loss
model predicts, so the slab looked like drift. The monitor is now
floor-aware — it predicts RSSI with the 3D distance (3 m of
vertical separation per floor) minus the slab attenuation, the same
model the simulator uses — so a healthy beacon on another floor
reads as healthy, and the classifier works on a two-floor walk.

The window matters as much as the floor model: the classifier's
evidence is the rolling window, and the default 20-reading window
only ever sees one floor's half-loop, where the slab compresses
the moved signature below its threshold. The classification window
has to span a full cycle of the walk — 80 readings for the
two-floor ward walked twice — and then the verdicts match the
single-floor table.
"""

import io
import random
import unittest
from contextlib import redirect_stdout

from beacon import Beacon, DriftMonitor
from beacon.battery import BatteryModel
from beacon.cli import main
from beacon.simulate import simulate_multifloor_walk, two_floor_walk

SEEDS = (1, 2, 3, 7, 11)

BEACONS = [Beacon("A", 0, 0, floor=0), Beacon("B", 20, 0, floor=0),
           Beacon("C", 20, 12, floor=0), Beacon("D", 0, 12, floor=0),
           Beacon("E", 0, 0, floor=1), Beacon("F", 20, 0, floor=1),
           Beacon("G", 20, 12, floor=1), Beacon("H", 0, 12, floor=1)]

# One full two-floor cycle: 4 sides x 10 steps per side x 2 floors.
CYCLE_WINDOW = 80


def _walk(seed, scenario):
    """Two loops of the two-floor ward, with D drifted per scenario."""
    walk = two_floor_walk(20, 12, 10)
    sim = list(BEACONS)
    batteries = None
    step_seconds = 1.0
    if scenario == "sag":
        batteries = {"D": BatteryModel(drain_rate_per_hour=0.3)}
        step_seconds = 60.0
    elif scenario == "move":
        sim = [Beacon(b.id, b.x + 5, b.y - 5, floor=b.floor)
               if b.id == "D" else b for b in BEACONS]
    steps = simulate_multifloor_walk(walk + walk, sim, noise_db=2.0,
                                     seed=seed, batteries=batteries,
                                     step_seconds=step_seconds)
    rng = random.Random(seed)
    for step in steps:
        if scenario == "noisy" and "D" in step["rssi"]:
            step["rssi"]["D"] += rng.gauss(0.0, 5.0)
        if scenario == "offset" and "D" in step["rssi"]:
            step["rssi"]["D"] -= 6.0
    return steps


def _classify(seed, scenario, window=CYCLE_WINDOW):
    monitor = DriftMonitor(BEACONS, window=window)
    for step in _walk(seed, scenario):
        monitor.feed(step)
    return monitor


class MultifloorDriftMonitorTest(unittest.TestCase):
    def test_cross_floor_reading_is_not_drift(self):
        # A noiseless two-floor walk: every beacon's residual is ~0
        # on both floors, so nothing alarms and everything is
        # healthy — the slab no longer reads as drift.
        walk = two_floor_walk(20, 12, 10)
        steps = simulate_multifloor_walk(walk + walk, BEACONS,
                                         noise_db=0.0, seed=7)
        monitor = DriftMonitor(BEACONS)
        for step in steps:
            monitor.feed(step)
        self.assertEqual(monitor.alarms(), [])
        for result in monitor.classifications():
            self.assertEqual(result.verdict, "healthy")
            self.assertAlmostEqual(result.mean_residual_db, 0.0, places=6)

    def test_observe_accepts_two_and_three_tuples(self):
        # A two-tuple is floor 0, as before; a three-tuple names its
        # floor. Anything else is rejected rather than misread.
        monitor = DriftMonitor(BEACONS)
        monitor.observe((2.0, 2.0), {"A": -62.0})
        self.assertIsNotNone(monitor.status("A"))
        monitor.observe((2.0, 2.0, 1), {"E": -62.0})
        self.assertIsNotNone(monitor.status("E"))
        with self.assertRaises(ValueError):
            monitor.observe((2.0, 2.0, 1, 0), {"A": -62.0})
        with self.assertRaises(ValueError):
            monitor.observe((2.0,), {"A": -62.0})

    def test_floor_model_parameters_are_validated(self):
        with self.assertRaises(ValueError):
            DriftMonitor(BEACONS, floor_height_m=0.0)
        with self.assertRaises(ValueError):
            DriftMonitor(BEACONS, floor_attenuation_db=-1.0)

    def test_fresh_keeps_the_floor_model(self):
        monitor = DriftMonitor(BEACONS, floor_height_m=4.0,
                               floor_attenuation_db=10.0)
        fresh = monitor.fresh()
        self.assertEqual(fresh._floor_height_m, 4.0)
        self.assertEqual(fresh._floor_attenuation_db, 10.0)

    def test_revisit_pairs_do_not_cross_floors(self):
        # Ten readings at (2, 2) downstairs, then ten at the same
        # (x, y) upstairs: the same spot on another floor is not a
        # revisit, so no pairs match and no delta is measured.
        monitor = DriftMonitor(BEACONS)
        for _ in range(10):
            monitor.observe((2.0, 2.0, 0), {"A": -62.0})
        for _ in range(10):
            monitor.observe((2.0, 2.0, 1), {"A": -62.0})
        result = monitor.classify("A")
        self.assertEqual(result.revisit_pairs, 0)
        self.assertIsNone(result.revisit_delta_db)

    def test_healthy_multifloor_walk_is_healthy_on_every_seed(self):
        for seed in SEEDS:
            monitor = _classify(seed, "healthy")
            self.assertEqual(monitor.alarms(), [],
                             f"seed {seed} alarmed on a healthy walk")
            for result in monitor.classifications():
                self.assertEqual(result.verdict, "healthy",
                                 f"seed {seed} beacon {result.id}")

    def test_draining_beacon_is_battery_sag_on_every_seed(self):
        # The slow sag never reaches the 5 dB alarm threshold, so
        # the alarm misses it — the classifier catches it through
        # the revisits, as on the single-floor walk.
        for seed in SEEDS:
            monitor = _classify(seed, "sag")
            self.assertEqual(monitor.classify("D").verdict,
                             "battery-sag", f"seed {seed}")
            self.assertEqual(monitor.alarms(), [])
            others = [c for c in monitor.classifications()
                      if c.id != "D"]
            self.assertTrue(all(c.verdict == "healthy" for c in others),
                            f"seed {seed}")

    def test_moved_beacon_is_moved_on_four_of_five_seeds(self):
        # Seed 2 is the honest miss: the window's spread is 2.32 dB,
        # under the 2.5 dB moved threshold, so it reads uncertain —
        # the same kind of miss as seed 11 on the single-floor walk.
        expected = {1: "moved", 2: "uncertain", 3: "moved",
                    7: "moved", 11: "moved"}
        for seed in SEEDS:
            monitor = _classify(seed, "move")
            self.assertEqual(monitor.classify("D").verdict,
                             expected[seed], f"seed {seed}")
            others = [c for c in monitor.classifications()
                      if c.id != "D"]
            self.assertTrue(all(c.verdict == "healthy" for c in others),
                            f"seed {seed}")

    def test_noisy_beacon_is_noisy_on_every_seed(self):
        for seed in SEEDS:
            monitor = _classify(seed, "noisy")
            self.assertEqual(monitor.classify("D").verdict, "noisy",
                             f"seed {seed}")
            others = [c for c in monitor.classifications()
                      if c.id != "D"]
            self.assertTrue(all(c.verdict == "healthy" for c in others),
                            f"seed {seed}")

    def test_steady_offset_is_uncertain_on_every_seed(self):
        # A steady -6 dB offset alarms (the mean is beyond 5 dB) but
        # no signature fits, so the classifier says uncertain
        # instead of guessing — on all five seeds, where the
        # single-floor walk misreads one seed as a sag.
        for seed in SEEDS:
            monitor = _classify(seed, "offset")
            self.assertEqual(monitor.classify("D").verdict,
                             "uncertain", f"seed {seed}")
            self.assertEqual([a.id for a in monitor.alarms()], ["D"],
                             f"seed {seed}")
            others = [c for c in monitor.classifications()
                      if c.id != "D"]
            self.assertTrue(all(c.verdict == "healthy" for c in others),
                            f"seed {seed}")

    def test_default_window_misses_the_move(self):
        # The failure the longer window fixes: with the default
        # 20-reading window the last window only ever sees one
        # floor's half-loop, the slab compresses the moved
        # signature, and the beacon reads uncertain on 4 of 5 seeds.
        expected = {1: "uncertain", 2: "uncertain", 3: "uncertain",
                    7: "moved", 11: "uncertain"}
        for seed in SEEDS:
            monitor = _classify(seed, "move", window=20)
            self.assertEqual(monitor.classify("D").verdict,
                             expected[seed], f"seed {seed}")

    def test_moved_verdict_depends_on_which_floor_the_window_covers(self):
        # Halfway through the walk — at the end of the floor-0 pass,
        # with the window on D's own floor — the moved beacon reads
        # "moved" on 4 of 5 seeds; by the end of the walk the window
        # has moved upstairs and the same beacon reads "uncertain"
        # on 4 of 5. The verdict follows the window, not the beacon.
        expected_own_floor = {1: "moved", 2: "moved", 3: "uncertain",
                              7: "moved", 11: "moved"}
        for seed in SEEDS:
            monitor = DriftMonitor(BEACONS)
            for step in _walk(seed, "move")[:120]:
                monitor.feed(step)
            self.assertEqual(monitor.classify("D").verdict,
                             expected_own_floor[seed], f"seed {seed}")


class MultifloorDriftCliTest(unittest.TestCase):
    def test_cli_multifloor_healthy(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["--drift-classify", "--multifloor",
                                   "--seed", "7"]), 0)
        text = out.getvalue()
        self.assertIn("healthy", text)
        # All eight beacons are classified, none alarming.
        self.assertEqual(text.count("healthy"), 8)

    def test_cli_multifloor_battery_sag(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["--drift-classify", "--multifloor",
                                   "--seed", "7", "--battery-degrade",
                                   "D"]), 0)
        self.assertIn("battery-sag", out.getvalue())

    def test_cli_multifloor_moved(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["--drift-classify", "--multifloor",
                                   "--seed", "7", "--move-beacon",
                                   "D:5,-5"]), 0)
        self.assertIn("moved", out.getvalue())

    def test_cli_multifloor_needs_drift_classify(self):
        with self.assertRaises(SystemExit):
            main(["--multifloor"])
        self.assertEqual(main(["--drift-classify", "--multifloor",
                               "--battery-degrade", "Z"]), 2)


if __name__ == "__main__":
    unittest.main()
