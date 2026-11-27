"""Drift classification on multi-floor walks (beacon 0.59.0).

The drift monitor used to reject multi-floor positions: a beacon
heard through a floor slab reads ~15 dB weaker than the path-loss
model predicts, so the slab looked like drift. The monitor is now
floor-aware — it predicts RSSI with the 3D distance (3 m of
vertical separation per floor) minus the slab attenuation, the same
model the simulator uses — so a healthy beacon on another floor
reads as healthy, and the classifier works on a two-floor walk.

The window is per floor, not one window per beacon: each beacon
keeps one rolling window per floor it was heard on, and the
classification reads the beacon's own floor, so the verdict
always rests on the beacon's own geometry instead of whichever
floor a single mixed window happens to cover. Classified halfway
through the walk and at the end, the moved beacon reads the same
on every seed — the own-floor window does not change while the
device is upstairs. The classification carries the floor it was
taken on.
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


def _walk(seed, scenario, loops=2):
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
    steps = simulate_multifloor_walk(walk * loops, sim, noise_db=2.0,
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

    def test_slow_sag_is_caught_on_two_of_five_seeds(self):
        # The long-history fix for the v0.59.0 miss: the own-floor
        # window averages the fresh first loop with the drained
        # second loop, so its mean stays at -0.60 to -0.79 dB and
        # the short-window sag gate (-1.5 dB) never fires — but the
        # decline is still there across the full per-floor history,
        # and the classifier consults that long window for the sag
        # only. Seeds 1 and 11 have a long-history revisit delta
        # of -1.50 dB, so they read battery-sag; seeds 2, 3 and 7
        # lack the signal in the revisit pairs (-0.87, -0.19 and
        # -0.48 dB — the noise buries the ~1.5 dB inter-loop
        # decline), so the classifier stays honest and says healthy
        # rather than inventing a decline. No alarm either: the mean
        # never reaches the 5 dB alarm threshold. The short-window
        # mean is above the gate on every seed, so only the long
        # route can produce the battery-sag verdicts.
        expected = {1: "battery-sag", 2: "healthy", 3: "healthy",
                    7: "healthy", 11: "battery-sag"}
        for seed in SEEDS:
            monitor = _classify(seed, "sag")
            result = monitor.classify("D")
            self.assertEqual(result.verdict, expected[seed],
                             f"seed {seed}")
            self.assertGreater(result.mean_residual_db, -1.5,
                               f"seed {seed}")
            self.assertEqual(monitor.alarms(), [])
            others = [c for c in monitor.classifications()
                      if c.id != "D"]
            self.assertTrue(all(c.verdict == "healthy" for c in others),
                            f"seed {seed}")

    def test_three_loops_catch_the_slow_sag_on_every_seed(self):
        # Longer history still: a third loop, and the window fills
        # with drained readings until its mean crosses the sag gate
        # on its own — the fix the v0.59.0 notes pointed at. With
        # three loops the slow sag reads battery-sag on 5/5 seeds,
        # and the other seven beacons stay healthy.
        for seed in SEEDS:
            monitor = DriftMonitor(BEACONS, window=CYCLE_WINDOW)
            for step in _walk(seed, "sag", loops=3):
                monitor.feed(step)
            self.assertEqual(monitor.classify("D").verdict,
                             "battery-sag", f"seed {seed}")
            others = [c for c in monitor.classifications()
                      if c.id != "D"]
            self.assertTrue(all(c.verdict == "healthy" for c in others),
                            f"seed {seed}")

    def test_three_loops_catch_the_moved_beacon_on_every_seed(self):
        # The seed-2 near-miss the v0.61.0 notes left: on the
        # tripled two-floor walk the short window's spread is 2.48
        # dB, 0.02 dB under the moved threshold, so the short
        # window calls it uncertain — but the long window holds
        # 120 readings, and over them the spread settles to
        # 2.57 dB around a +4.29 dB mean with a stable revisit
        # delta of +0.38 dB over 81 pairs, so the moved call fires
        # from the longer history instead. Same thresholds, more
        # readings — no gate was moved. The other seven beacons
        # stay healthy, and seed 2's verdict says which evidence
        # it rests on.
        for seed in SEEDS:
            monitor = DriftMonitor(BEACONS, window=CYCLE_WINDOW)
            for step in _walk(seed, "move", loops=3):
                monitor.feed(step)
            result = monitor.classify("D")
            self.assertEqual(result.verdict, "moved", f"seed {seed}")
            others = [c for c in monitor.classifications()
                      if c.id != "D"]
            self.assertTrue(all(c.verdict == "healthy" for c in others),
                            f"seed {seed}")
        monitor = DriftMonitor(BEACONS, window=CYCLE_WINDOW)
        for step in _walk(2, "move", loops=3):
            monitor.feed(step)
        self.assertIn("longer", monitor.classify("D").reason)

    def test_long_window_is_validated(self):
        with self.assertRaises(ValueError):
            DriftMonitor(BEACONS, long_window=0)
        with self.assertRaises(ValueError):
            DriftMonitor(BEACONS, window=20, long_window=10)
        with self.assertRaises(ValueError):
            DriftMonitor(BEACONS, history=100, long_window=200)

    def test_fresh_keeps_the_long_window(self):
        monitor = DriftMonitor(BEACONS, window=20, long_window=120)
        fresh = monitor.fresh()
        self.assertEqual(fresh._long_window, 120)
        self.assertEqual(fresh._window, 20)

    def test_long_window_default_is_the_full_history(self):
        monitor = DriftMonitor(BEACONS, window=20)
        self.assertEqual(monitor._long_window, 500)

    def test_moved_beacon_is_moved_on_three_of_five_seeds(self):
        # Seeds 1 and 2 are the honest misses: the own-floor
        # window's spread is 2.47 and 2.50 dB, just under the 2.5 dB
        # moved threshold — the same kind of miss as seed 2 with
        # the old mixed window, and seed 11 on the single-floor
        # walk.
        expected = {1: "uncertain", 2: "uncertain", 3: "moved",
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

    def test_long_route_sag_needs_a_level_drop_too(self):
        # The same discriminator on the long route: the two-floor
        # slow sag is caught from the long history on seeds 1 and
        # 11, and the verdict rests on the level having fallen
        # across that history too — a steady offset fakes a long
        # revisit decline the same way it fakes a short one, so
        # the drop gate guards this route as well. The offset's
        # level stays flat on every seed, and its verdict stays
        # uncertain.
        for seed in (1, 11):
            monitor = _classify(seed, "sag")
            result = monitor.classify("D")
            self.assertEqual(result.verdict, "battery-sag",
                             f"seed {seed}")
            self.assertIsNotNone(result.level_drop_db)
            self.assertLess(result.level_drop_db, -1.0,
                            f"seed {seed}")
        for seed in SEEDS:
            monitor = _classify(seed, "offset")
            result = monitor.classify("D")
            self.assertEqual(result.verdict, "uncertain",
                             f"seed {seed}")
            # The drop gate is not what excludes the offset here:
            # its long revisit delta never clears the sag gate on
            # the two-floor walk. Seed 11's level does fall 1.18 dB
            # across the walk — a noise excursion spanning the
            # whole walk — so the drop alone would not exclude it;
            # the verdict stays uncertain because the delta fails.
            if seed != 11:
                self.assertIsNotNone(result.level_drop_db)
                self.assertGreater(result.level_drop_db, -1.0,
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

    def test_default_per_floor_window_catches_the_move(self):
        # The failure the per-floor window fixes: with the old mixed
        # 20-reading window the last window only ever saw one
        # floor's half-loop, the slab compressed the moved
        # signature, and the beacon read uncertain on 4 of 5 seeds.
        # The window is per floor now, so the default window holds
        # the last 20 readings on the beacon's own floor, and the
        # moved beacon reads "moved" on 4 of 5 seeds from the short
        # window alone — seed 3 is the short-window miss (its
        # revisit delta is +1.20 dB, so the moved call, which needs
        # a stable revisit, does not fire there) — and on 5/5 with
        # the longer history consulted too: over 41 pairs the
        # delta settles to +0.80 dB and the spread over the same
        # history is 2.91 dB, so the moved call fires from the long
        # window.
        expected = {1: "moved", 2: "moved", 3: "moved",
                    7: "moved", 11: "moved"}
        for seed in SEEDS:
            monitor = _classify(seed, "move", window=20)
            self.assertEqual(monitor.classify("D").verdict,
                             expected[seed], f"seed {seed}")

    def test_moved_verdict_is_the_same_mid_walk_and_at_the_end(self):
        # The verdict used to follow the window: classified halfway
        # through the walk — at the end of the floor-0 pass, with
        # the window on D's own floor — the moved beacon read
        # "moved" on 4 of 5 seeds, but by the end of the walk the
        # window had moved upstairs and it read "uncertain" on 4
        # of 5. The window is per floor now, so the own-floor
        # window does not change while the device is upstairs, and
        # the verdict at the end of the walk is the verdict taken
        # mid-walk, on every seed — at the default window and at
        # the demo window alike.
        expected = {
            20: {1: "moved", 2: "moved", 3: "moved",
                 7: "moved", 11: "moved"},
            CYCLE_WINDOW: {1: "uncertain", 2: "uncertain",
                           3: "moved", 7: "moved", 11: "moved"},
        }
        for window in (20, CYCLE_WINDOW):
            for seed in SEEDS:
                mid = DriftMonitor(BEACONS, window=window)
                for step in _walk(seed, "move")[:120]:
                    mid.feed(step)
                end = _classify(seed, "move", window=window)
                self.assertEqual(mid.classify("D").verdict,
                                 end.classify("D").verdict,
                                 f"window {window} seed {seed}")
                self.assertEqual(end.classify("D").verdict,
                                 expected[window][seed],
                                 f"window {window} seed {seed}")

    def test_windows_are_kept_per_floor(self):
        # D reads 10 dB too loud upstairs and clean downstairs: the
        # own-floor verdict is healthy and the cross-floor window
        # says otherwise — the two floors never mix into one
        # window, and the classification says which floor it was
        # taken on.
        import math

        from beacon.rssi import distance_to_rssi

        monitor = DriftMonitor(BEACONS)
        downstairs = distance_to_rssi(math.hypot(2.0, 10.0), -59.0,
                                      2.0)
        upstairs = distance_to_rssi(math.hypot(2.0, 10.0, 3.0), -59.0,
                                    2.0) - 15.0
        for _ in range(10):
            monitor.observe((2.0, 2.0, 0), {"D": downstairs})
        for _ in range(10):
            monitor.observe((2.0, 2.0, 1), {"D": upstairs + 10.0})
        own = monitor.classify("D")
        self.assertEqual(own.floor, 0)
        self.assertEqual(own.verdict, "healthy")
        self.assertAlmostEqual(own.mean_residual_db, 0.0, places=6)
        other = monitor.classify("D", floor=1)
        self.assertEqual(other.floor, 1)
        self.assertAlmostEqual(other.mean_residual_db, 10.0, places=6)
        self.assertEqual(monitor.alarm("D"), None)
        self.assertIsNotNone(monitor.alarm("D", floor=1))

    def test_classification_carries_its_floor(self):
        # Every beacon's classification names the floor its verdict
        # rests on — its own floor by default.
        monitor = DriftMonitor(BEACONS)
        for step in _walk(7, "healthy"):
            monitor.feed(step)
        for result in monitor.classifications():
            beacon = next(b for b in BEACONS if b.id == result.id)
            self.assertEqual(result.floor, beacon.floor,
                             f"beacon {result.id}")

    def test_unheard_on_its_own_floor_is_uncertain(self):
        # A beacon heard only through the slab — never on its own
        # floor — has no own-floor window to classify from, so it
        # is uncertain, not healthy, and the reason names the floor.
        monitor = DriftMonitor(BEACONS)
        for _ in range(10):
            monitor.observe((2.0, 2.0, 1), {"D": -62.0})
        result = monitor.classify("D")
        self.assertEqual(result.verdict, "uncertain")
        self.assertEqual(result.floor, 0)
        self.assertIn("floor 0", result.reason)
        self.assertIsNone(monitor.status("D"))


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
        # The slow sag is missed on the two-floor walk: the
        # per-floor window averages the fresh first loop with the
        # drained second loop, so D's mean stays above the sag
        # gate — all eight beacons read healthy, and no verdict
        # claims a sag.
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["--drift-classify", "--multifloor",
                                   "--seed", "7", "--battery-degrade",
                                   "D"]), 0)
        text = out.getvalue()
        self.assertNotIn("battery-sag", text)
        self.assertEqual(text.count("healthy"), 8)

    def test_cli_multifloor_moved(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["--drift-classify", "--multifloor",
                                   "--seed", "7", "--move-beacon",
                                   "D:5,-5"]), 0)
        self.assertIn("moved", out.getvalue())

    def test_cli_three_loops_catch_the_slow_sag(self):
        # The fix the v0.60.0 notes pointed at, from the command
        # line: a third loop fills the long window with drained
        # readings until its mean crosses the sag gate on its own,
        # so the slow sag reads battery-sag on 5/5 seeds and the
        # other seven beacons stay healthy. The default two loops
        # still miss it on three seeds (see
        # test_cli_multifloor_battery_sag).
        for seed in SEEDS:
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(main(["--drift-classify", "--multifloor",
                                       "--seed", str(seed),
                                       "--battery-degrade", "D",
                                       "--loops", "3"]), 0)
            text = out.getvalue()
            self.assertIn("battery-sag", text, f"seed {seed}")
            self.assertEqual(text.count("healthy"), 7, f"seed {seed}")

    def test_cli_three_loops_healthy(self):
        for seed in SEEDS:
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(main(["--drift-classify", "--multifloor",
                                       "--seed", str(seed),
                                       "--loops", "3"]), 0)
            self.assertEqual(out.getvalue().count("healthy"), 8,
                             f"seed {seed}")

    def test_cli_loops_validation(self):
        # Zero loops is rejected, and --loops needs --drift-classify
        # like the other drift flags.
        self.assertEqual(main(["--drift-classify", "--seed", "7",
                               "--loops", "0"]), 2)
        with self.assertRaises(SystemExit):
            main(["--loops", "3"])

    def test_cli_multifloor_needs_drift_classify(self):
        with self.assertRaises(SystemExit):
            main(["--multifloor"])
        self.assertEqual(main(["--drift-classify", "--multifloor",
                               "--battery-degrade", "Z"]), 2)


if __name__ == "__main__":
    unittest.main()
