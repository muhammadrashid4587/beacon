"""CLI: run a simulated walk and print estimated vs true positions.

Default scenario: a 20 x 12 m hospital ward with four beacons, one in
each corner, and a device walking a rectangular loop inside it.

Usage:
    python -m beacon.cli [--steps 40] [--noise 2.0] [--seed 7]
    python -m beacon.cli --map-overlay track.json [--map-output map.svg]
    python -m beacon.cli --map-overlay track.json --map-replay
        [--map-output replay.svg] [--replay-step-seconds 0.5]
"""

import argparse
import math
import statistics
import sys

from .beacon import Beacon
from .engine import PositioningEngine
from .map_overlay import load_track_for_overlay, render_map_overlay
from .replay import render_track_replay
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Simulate a BLE indoor-positioning walk.")
    parser.add_argument("--steps", type=int, default=10,
                        help="steps per side of the rectangular walk")
    parser.add_argument("--noise", type=float, default=2.0,
                        help="RSSI noise stddev in dB")
    parser.add_argument("--seed", type=int, default=7,
                        help="random seed (omit for a fresh run)")
    parser.add_argument("--filter", choices=("kalman", "moving_average"),
                        default="kalman",
                        help="position filter (default: kalman; "
                        "moving_average is the pre-0.5.0 default)")
    parser.add_argument("--map-overlay", default=None, metavar="PATH",
                        help="render a ward-map SVG overlay from an "
                        "exported live track (.json or .csv) instead "
                        "of simulating a walk")
    parser.add_argument("--map-output", default=None, metavar="PATH",
                        help="write the map overlay SVG here "
                        "(default: print it to stdout)")
    parser.add_argument("--map-floor", type=int, default=None,
                        help="floor to plot on the overlay (default: "
                        "the track's most common floor)")
    parser.add_argument("--map-heatmap", action="store_true",
                        help="shade the overlay with a per-cell "
                        "coverage heatmap (positioning quality from "
                        "the beacon layout)")
    parser.add_argument("--map-replay", action="store_true",
                        help="render the track as an animated replay "
                        "SVG (SMIL, no JavaScript) instead of the "
                        "static overlay")
    parser.add_argument("--replay-step-seconds", type=float, default=1.0,
                        help="replay time per step in seconds "
                        "(default: 1.0)")
    args = parser.parse_args(argv)

    if args.map_overlay is not None:
        rows = load_track_for_overlay(args.map_overlay)
        if args.map_replay:
            svg = render_track_replay(
                rows, default_beacons(),
                width_m=WARD_WIDTH_M, height_m=WARD_HEIGHT_M,
                floor=args.map_floor, heatmap=args.map_heatmap,
                seconds_per_step=args.replay_step_seconds)
        else:
            svg = render_map_overlay(rows, default_beacons(),
                                     width_m=WARD_WIDTH_M,
                                     height_m=WARD_HEIGHT_M,
                                     floor=args.map_floor,
                                     heatmap=args.map_heatmap)
        if args.map_output is not None:
            with open(args.map_output, "w", encoding="utf-8") as handle:
                handle.write(svg)
        else:
            sys.stdout.write(svg)
        return 0

    path = rectangular_walk(WARD_WIDTH_M, WARD_HEIGHT_M, args.steps)
    steps = simulate_walk(path, default_beacons(), noise_db=args.noise,
                          seed=args.seed)
    engine = PositioningEngine(default_beacons(), filter=args.filter)

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
