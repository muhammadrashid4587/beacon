"""CLI: run a simulated walk and print estimated vs true positions.

Default scenario: a 20 x 12 m hospital ward with four beacons, one in
each corner, and a device walking a rectangular loop inside it.

Usage:
    python -m beacon.cli [--steps 40] [--noise 2.0] [--seed 7]
    python -m beacon.cli --map-overlay track.json [--map-output map.svg]
    python -m beacon.cli --map-overlay track.json --map-replay
        [--map-output replay.svg] [--replay-step-seconds 0.5]
    python -m beacon.cli --map-overlay track.json --map-planner
        [--map-output map.svg] [--battery-levels A=0.9,B=0.6]
        [--drain-rate 0.05]
    python -m beacon.cli --map-overlay track.json --map-drift
        [--map-output map.svg]
    python -m beacon.cli --map-overlay track.json
        --map-smoothed track-smoothed.json [--map-output map.svg]
    python -m beacon.cli --drift-classify [--seed 7]
        [--battery-degrade D] [--move-beacon D:5,-5]
    python -m beacon.cli --deployment-plan
        [--battery-levels A=0.9,B=0.6] [--drain-rate 0.05]
    python -m beacon.cli --replacement-schedule
        [--battery-levels A=0.9,B=0.6] [--drain-rate 0.05]
        [--swap-lead-hours 2]
"""

import argparse
import math
import statistics
import sys
from datetime import datetime

from .battery import BatteryModel
from .beacon import Beacon
from .deployment import (DEFAULT_LOSS_THRESHOLD_DB,
                         DEFAULT_NOISE_THRESHOLD_DB, DeploymentPlan,
                         plan_deployment)
from .drift import DriftMonitor
from .engine import PositioningEngine
from .map_overlay import load_track_for_overlay, render_map_overlay
from .replay import render_track_replay
from .schedule import replacement_schedule
from .simulate import rectangular_walk, simulate_walk
from .validation import validate_floor_layouts

WARD_WIDTH_M = 20.0
WARD_HEIGHT_M = 12.0

# The drift-classification demo drains at this rate (120 s steps, two
# loops): slow enough that the beacon survives both loops, fast
# enough that the decline is visible — level 1.0 down to 0.20.
DRIFT_BATTERY_DRAIN_PER_HOUR = 0.3
DRIFT_STEP_SECONDS = 120.0


def default_beacons() -> list[Beacon]:
    """Four beacons, one in each corner of the ward."""
    return [
        Beacon("A", 0.0, 0.0),
        Beacon("B", WARD_WIDTH_M, 0.0),
        Beacon("C", WARD_WIDTH_M, WARD_HEIGHT_M),
        Beacon("D", 0.0, WARD_HEIGHT_M),
    ]


def _parse_move_beacon(spec: str) -> tuple[str, float, float]:
    """Parse a --move-beacon ID:DX,DY spec, or raise ValueError."""
    beacon_id, displacement = spec.split(":", 1)
    dx_text, dy_text = displacement.split(",", 1)
    return beacon_id, float(dx_text), float(dy_text)


def run_drift_classify(args: argparse.Namespace) -> int:
    """Walk the loop twice and classify each beacon's drift.

    The second loop is the point: the monitor matches each recent
    reading against the same spot a loop earlier, so a beacon that
    reads weaker at the same place over time is a draining battery,
    while a beacon whose residual is stable at the same spot but
    wrong by a position-dependent amount has moved. ``--battery-
    degrade`` drains one beacon and ``--move-beacon`` displaces one,
    so the classifier can be shown a real sag and a real move.
    """
    beacons = default_beacons()
    known_ids = {b.id for b in beacons}
    sim_beacons = list(beacons)
    batteries = None
    step_seconds = 1.0
    if args.battery_degrade is not None:
        if args.battery_degrade not in known_ids:
            print(f"unknown beacon: {args.battery_degrade!r}",
                  file=sys.stderr)
            return 2
        batteries = {args.battery_degrade: BatteryModel(
            drain_rate_per_hour=DRIFT_BATTERY_DRAIN_PER_HOUR)}
        step_seconds = DRIFT_STEP_SECONDS
    if args.move_beacon is not None:
        try:
            beacon_id, dx, dy = _parse_move_beacon(args.move_beacon)
        except ValueError:
            print("--move-beacon takes ID:DX,DY, e.g. D:5,-5",
                  file=sys.stderr)
            return 2
        if beacon_id not in known_ids:
            print(f"unknown beacon: {beacon_id!r}", file=sys.stderr)
            return 2
        sim_beacons = [
            Beacon(b.id, b.x + dx, b.y + dy) if b.id == beacon_id else b
            for b in beacons]

    loop = rectangular_walk(WARD_WIDTH_M, WARD_HEIGHT_M, args.steps)
    steps = simulate_walk(loop + loop, sim_beacons,
                          noise_db=args.noise, seed=args.seed,
                          batteries=batteries, step_seconds=step_seconds)
    monitor = DriftMonitor(beacons)
    for step in steps:
        monitor.feed(step)

    print(f"{'beacon':>6} {'verdict':>11} {'mean dB':>8} {'std dB':>7} "
          f"{'revisit dB':>10} {'pairs':>5}")
    for result in monitor.classifications():
        revisit = (f"{result.revisit_delta_db:+.2f}"
                   if result.revisit_delta_db is not None else "—")
        print(f"{result.id:>6} {result.verdict:>11} "
              f"{result.mean_residual_db:+8.2f} "
              f"{result.residual_std_db:7.2f} "
              f"{revisit:>10} {result.revisit_pairs:5d}")
        print(f"       {result.reason}")
    return 0


def _parse_battery_levels(spec: str) -> dict[str, float]:
    """Parse a --battery-levels ID=LEVEL,... spec, or raise ValueError."""
    levels = {}
    for part in spec.split(","):
        beacon_id, level_text = part.split("=", 1)
        levels[beacon_id] = float(level_text)
    return levels


def _deployment_plan(args: argparse.Namespace) -> DeploymentPlan:
    """Build the deployment plan for the default ward.

    Every beacon drains at ``--drain-rate`` per hour from its
    ``--battery-levels`` charge (1.0, fresh, unless listed).
    Raises ValueError for a bad levels spec, an unknown beacon,
    or a level the battery model rejects.
    """
    beacons = default_beacons()
    known_ids = {b.id for b in beacons}
    try:
        levels = _parse_battery_levels(args.battery_levels) \
            if args.battery_levels is not None else {}
    except ValueError:
        raise ValueError(
            "--battery-levels takes ID=LEVEL,... e.g. A=0.9,D=0.2")
    unknown = sorted(set(levels) - known_ids)
    if unknown:
        raise ValueError(f"unknown beacon: {unknown[0]!r}")
    return plan_deployment(
        {b.id: BatteryModel(drain_rate_per_hour=args.drain_rate)
         for b in beacons},
        levels=levels)


def run_deployment_plan(args: argparse.Namespace) -> int:
    """Forecast when each default beacon goes dark.

    Every beacon drains at ``--drain-rate`` per hour from its
    ``--battery-levels`` charge (1.0, fresh, unless listed), and the
    table prints the hours until each one goes dark, until its RSSI
    loss crosses 3 dB, and until its extra noise crosses 2 dB —
    plus when the ward drops below three lit beacons and can no
    longer triangulate at all.
    """
    try:
        plan = _deployment_plan(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    def _hours(value: float) -> str:
        return "never" if math.isinf(value) else f"{value:.1f}"

    def _hours_label(value: float) -> str:
        return "never" if math.isinf(value) else f"{value:.1f} hours"

    print(f"{'beacon':>6} {'level':>5} {'dark in':>8} "
          f"{'3 dB loss in':>12} {'2 dB noise in':>13} {'status':>9}")
    for forecast in plan.forecasts:
        print(f"{forecast.beacon_id:>6} {forecast.level:5.2f} "
              f"{_hours(forecast.hours_to_dark):>8} "
              f"{_hours(forecast.hours_to_rssi_loss_db):>12} "
              f"{_hours(forecast.hours_to_extra_noise_db):>13} "
              f"{forecast.status:>9}")
    print(f"earliest dark: {_hours_label(plan.earliest_dark_hours)}")
    below_three = plan.hours_until_fewer_than(3)
    print(f"below 3 lit beacons: {_hours_label(below_three)}")
    return 0


def run_replacement_schedule(args: argparse.Namespace) -> int:
    """Print the battery-swap work order for the default ward.

    Each beacon is due a new cell by its soonest forecast
    milestone — dark, 3 dB of RSSI loss, or 2 dB of extra noise —
    pulled earlier by ``--swap-lead-hours`` so the cells can be
    ordered ahead of the failure. The swap-by time is printed as
    hours from now and as a clock time, soonest first, with the
    ward's hard deadline — when it drops below three lit beacons
    — at the bottom.
    """
    try:
        plan = _deployment_plan(args)
        schedule = replacement_schedule(
            plan, lead_time_hours=args.swap_lead_hours)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    now = datetime.now()
    print(f"{'beacon':>6} {'swap by':>30} {'reason':>9} {'status':>9}")
    for entry in schedule.entries:
        due = entry.due_at(now)
        if due is None:
            when = "never"
        elif entry.swap_by_hours == 0.0:
            when = "now"
        else:
            when = (f"in {entry.swap_by_hours:.1f}h "
                    f"({due:%Y-%m-%d %H:%M})")
        print(f"{entry.beacon_id:>6} {when:>30} {entry.reason:>9} "
              f"{entry.status:>9}")
    deadline = schedule.ward_deadline_hours
    print("ward drops below 3 lit beacons: "
          f"{'never' if math.isinf(deadline) else f'in {deadline:.1f} hours'}")
    return 0


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
    parser.add_argument("--map-planner", action="store_true",
                        help="make the overlay planner-aware: annotate "
                        "each beacon with its forecast horizon from "
                        "the deployment plan and shade the cells by "
                        "when their coverage first degrades — needs "
                        "--map-overlay, conflicts with --map-heatmap "
                        "and --map-replay; the plan uses "
                        "--battery-levels and --drain-rate")
    parser.add_argument("--map-drift", action="store_true",
                        help="mark the beacons alarming at the end of "
                        "the track on the overlay, or at each frame's "
                        "row on the replay: a "
                        "red ! badge and the rolling mean residual "
                        "with the drift classification verdict — "
                        "needs --map-overlay")
    parser.add_argument("--map-forecast", action="store_true",
                        help="on the replay, mark the beacons whose "
                        "planner forecast is due on each frame's row "
                        "with an amber F badge, and check the "
                        "forecast against the drift that actually "
                        "followed — needs --map-overlay --map-replay")
    parser.add_argument("--map-replay", action="store_true",
                        help="render the track as an animated replay "
                        "SVG (SMIL, no JavaScript) instead of the "
                        "static overlay")
    parser.add_argument("--map-smoothed", default=None, metavar="PATH",
                        help="render the smoothed track from a smoothed "
                        "export (.json or .csv) as a dashed series "
                        "alongside the raw track on the overlay or "
                        "the replay — needs --map-overlay; the rows "
                        "come from the smoothed export file, so the "
                        "overlay and the replay draw exactly what "
                        "the export says")
    parser.add_argument("--replay-step-seconds", type=float, default=1.0,
                        help="replay time per step in seconds "
                        "(default: 1.0)")
    parser.add_argument("--validate-layout", action="store_true",
                        help="validate the default beacon layout "
                        "floor by floor (can every floor "
                        "triangulate?) instead of simulating a walk; "
                        "exits non-zero when a floor is thin or "
                        "cannot triangulate")
    parser.add_argument("--drift-classify", action="store_true",
                        help="walk the loop twice and classify each "
                        "beacon's drift instead of simulating a "
                        "walk — battery-sag, moved, noisy, "
                        "uncertain or healthy (the second loop is "
                        "what separates a sag from a move: the same "
                        "spot, read again later)")
    parser.add_argument("--battery-degrade", default=None, metavar="ID",
                        help="with --drift-classify, drain this "
                        "beacon's battery during the walk (0.3 per "
                        "hour, 120 s steps) so the classifier has a "
                        "real sag to find")
    parser.add_argument("--move-beacon", default=None,
                        metavar="ID:DX,DY",
                        help="with --drift-classify, simulate this "
                        "beacon displaced by (DX, DY) metres from "
                        "its calibrated position, e.g. D:5,-5, so "
                        "the classifier has a real move to find")
    parser.add_argument("--deployment-plan", action="store_true",
                        help="forecast when each default beacon goes "
                        "dark from the battery model instead of "
                        "simulating a walk — hours until dark, "
                        "until 3 dB of RSSI loss, and until 2 dB of "
                        "extra noise, plus when the ward drops "
                        "below three lit beacons")
    parser.add_argument("--replacement-schedule", action="store_true",
                        help="print the battery-swap work order for "
                        "the default ward instead of simulating a "
                        "walk — each beacon's swap-by time, soonest "
                        "first, with the forecast milestone that "
                        "drives it; uses --battery-levels, "
                        "--drain-rate and --swap-lead-hours")
    parser.add_argument("--swap-lead-hours", type=float, default=0.0,
                        help="with --replacement-schedule, pull every "
                        "swap earlier by this many hours so the "
                        "cells are ordered ahead of the failure "
                        "(default: 0)")
    parser.add_argument("--battery-levels", default=None,
                        metavar="ID=LEVEL,...",
                        help="with --deployment-plan or --map-planner, "
                        "each beacon's current charge as a fraction "
                        "of full, e.g. A=0.9,D=0.2 (default: 1.0, "
                        "fresh, for every beacon)")
    parser.add_argument("--drain-rate", type=float, default=0.05,
                        help="with --deployment-plan or --map-planner, "
                        "the fraction of a full charge every beacon "
                        "burns per hour (default: 0.05)")
    args = parser.parse_args(argv)

    if args.deployment_plan:
        return run_deployment_plan(args)

    if args.replacement_schedule:
        return run_replacement_schedule(args)

    if args.drift_classify:
        return run_drift_classify(args)

    if args.battery_degrade is not None or args.move_beacon is not None:
        parser.error("--battery-degrade and --move-beacon need "
                     "--drift-classify")

    if args.validate_layout:
        validation = validate_floor_layouts(
            default_beacons(), WARD_WIDTH_M, WARD_HEIGHT_M)
        for verdict in validation.verdicts:
            if verdict.score is None:
                print(f"floor {verdict.floor}: "
                      f"{verdict.status} "
                      f"({verdict.n_beacons} beacons)")
            else:
                print(f"floor {verdict.floor}: {verdict.status} "
                      f"({verdict.n_beacons} beacons, mean 95% error "
                      f"{verdict.score.mean_semi_major:.2f} m, worst "
                      f"{verdict.score.worst_semi_major:.2f} m)")
            for warning in verdict.warnings:
                print(f"  warning: {warning}")
        return 0 if validation.ok else 1

    if args.map_overlay is not None:
        if args.map_planner and args.map_replay:
            print("--map-planner works with the static overlay, not "
                  "with --map-replay", file=sys.stderr)
            return 2
        if args.map_forecast and not args.map_replay:
            print("--map-forecast works with --map-replay, not the "
                  "static overlay (the overlay carries the forecast "
                  "with --map-planner)", file=sys.stderr)
            return 2
        if args.map_planner and args.map_heatmap:
            print("--map-planner and --map-heatmap shade the same "
                  "cells in different colours; pick one",
                  file=sys.stderr)
            return 2
        plan = None
        if args.map_planner:
            try:
                plan = _deployment_plan(args)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 2
        rows = load_track_for_overlay(args.map_overlay)
        smoothed_rows = (load_track_for_overlay(args.map_smoothed)
                         if args.map_smoothed is not None else None)
        if args.map_replay:
            svg = render_track_replay(
                rows, default_beacons(),
                width_m=WARD_WIDTH_M, height_m=WARD_HEIGHT_M,
                floor=args.map_floor, heatmap=args.map_heatmap,
                seconds_per_step=args.replay_step_seconds,
                drift=args.map_drift, forecast=args.map_forecast,
                smoothed_rows=smoothed_rows)
        else:
            svg = render_map_overlay(rows, default_beacons(),
                                     width_m=WARD_WIDTH_M,
                                     height_m=WARD_HEIGHT_M,
                                     floor=args.map_floor,
                                     heatmap=args.map_heatmap,
                                     plan=plan,
                                     planner=args.map_planner,
                                     drift=args.map_drift,
                                     smoothed_rows=smoothed_rows)
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
