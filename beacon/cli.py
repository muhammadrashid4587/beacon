"""CLI: run a simulated walk and print estimated vs true positions.

Default scenario: a 20 x 12 m hospital ward with four beacons, one in
each corner, and a device walking a rectangular loop inside it.

Usage:
    python -m beacon.cli [--steps 40] [--noise 2.0] [--seed 7]
"""

import argparse
import math
import statistics

from .beacon import Beacon
from .engine import PositioningEngine
from .simulate import rectangular_walk, simulate_walk

WARD_WIDTH_M = 20.0
WARD_HEIGHT_M = 12.0


def default_beacons() -> list[Beacon]:
    """Four beacons, one in each corner of the ward."""
    return [
        Beacon("A", 0.0, 0.0),
        Beacon("B", WARD_WIDTH_M, 0.0),
        Beacon("C", WARD_WIDTH_M, WARD_HEIGHT_M),
        Beacon("D", 0.0, WARD_HEIGHT_M),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate a BLE indoor-positioning walk.")
    parser.add_argument("--steps", type=int, default=10,
                        help="steps per side of the rectangular walk")
    parser.add_argument("--noise", type=float, default=2.0,
                        help="RSSI noise stddev in dB")
    parser.add_argument("--seed", type=int, default=7,
                        help="random seed (omit for a fresh run)")
    args = parser.parse_args()

    path = rectangular_walk(WARD_WIDTH_M, WARD_HEIGHT_M, args.steps)
    steps = simulate_walk(path, default_beacons(), noise_db=args.noise,
                          seed=args.seed)
    engine = PositioningEngine(default_beacons())

    print(f"{'step':>4} {'true':>14} {'estimated':>14} {'error m':>7}")
    errors = []
    for i, step in enumerate(steps):
        true = step["true"]
        est = engine.estimate(step["rssi"])
        err = math.hypot(est[0] - true[0], est[1] - true[1])
        errors.append(err)
        print(f"{i:>4} ({true[0]:5.1f},{true[1]:5.1f}) "
              f"({est[0]:5.1f},{est[1]:5.1f}) {err:7.2f}")

    errors.sort()
    p95 = errors[int(0.95 * (len(errors) - 1))]
    print(f"\nsteps: {len(errors)}  mean error: {statistics.mean(errors):.2f} m  "
          f"p95: {p95:.2f} m  max: {errors[-1]:.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
