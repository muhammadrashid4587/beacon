"""Multi-device tracking: track several devices at once on one beacon layout.

A ward has more than one tracked device — a patient tag, a nurse's
badge, a pump on a trolley — and one ``LiveTracker`` tracks one
device: its Kalman filter state, its floor vote, and its dark-beacon
bookkeeping all belong to that device's track. Feeding two devices
through one tracker would smear them into a single position
somewhere between them. ``MultiDeviceTracker`` keeps one
``LiveTracker`` per device id, each with its own
``PositioningEngine`` (so each device gets its own filter and
floor state) over the same shared beacon layout, and routes each
reading to the right track:

- Feed a reading with the device it came from — ``feed(device_id,
  rssi, t)`` — and it returns that device's live position, exactly
  as ``LiveTracker.feed`` would: the smoothed (x, y), the floor,
  and the 95% confidence ellipse for that step's geometry.
- Devices are created lazily, on their first reading, so a
  deployment never has to declare its tags up front — a new tag
  showing up mid-shift just starts tracking.
- Each device's clock is its own: timestamps only have to be
  non-decreasing *per device*, so readings from different devices
  can interleave in any order, and one device going quiet never
  stalls another's track — the quiet device's last position just
  stays where it was.
- ``last_position(device_id)`` is the most recent live position
  for one device, ``positions()`` is every device's last position
  at once, for a dashboard snapshot. ``reset(device_id)`` forgets
  one device's track (a tag handed to a new patient), and
  ``remove(device_id)`` drops the device entirely.

The stream format is JSON lines, one reading per line — the same
shape a BLE scanner would emit, with the device attached:

    {"device": "tag-1", "t": 0.0, "rssi": {"A": -62.1, "B": -71.4,
                                           "C": -68.0, "D": -59.8}}

Run it as a module: ``python -m beacon.multi_device`` reads those
lines from stdin and prints one JSON line per reading with the
device and its live position attached (``position`` is null on a
reading too thin to locate). ``--demo`` runs two simulated walks
through the tracker instead, so the mode can be tried with no
scanner at all. With a deployment plan attached (``--forecast``),
every device's live position also carries the planner's forecast
flags — one per beacon, the soonest milestone with its hours
remaining — counted down from each device's own first reading,
so a tag that shows up mid-shift warns on its own clock.
"""

import argparse
import json
import sys

from .beacon import Beacon
from .deployment import DeploymentPlan
from .engine import PositioningEngine
from .simulate import rectangular_walk, simulate_walk
from .tracking import (DEFAULT_DARK_AFTER_S, DEFAULT_FORECAST_WINDOW_HOURS,
                       DEFAULT_READMIT_AFTER, DEFAULT_SIGMA_RSSI_DB,
                       DEFAULT_STALE_AFTER_S, ExcludedBeacon, LivePosition,
                       LiveTracker, _deployment_plan, format_output,
                       parse_line)

# The default ward walk, walked in reverse, so the demo's second
# device follows a different path from the first.
DEMO_DEVICES = ("tag-1", "tag-2")


class MultiDeviceTracker:
    """Track several devices at once on one beacon layout.

    ``beacons`` is the shared layout every device localises
    against. Each device gets its own ``PositioningEngine`` and
    ``LiveTracker`` — created lazily on its first reading — so a
    device's Kalman filter, floor vote, and dark-beacon state are
    never shared with another device. ``stale_after``,
    ``dark_after``, ``readmit_after`` and ``sigma_rssi_db`` are
    passed through to each device's tracker unchanged (see
    ``LiveTracker`` for what they mean). ``filter`` and
    ``floor_switch_confidence`` are passed through to each
    device's engine. ``plan`` is an optional ``DeploymentPlan``,
    passed through to each device's tracker, so every device's
    live position carries the planner's forecast flags — one per
    beacon, the soonest milestone with its hours remaining — with
    the plan taken as made at each device's own first reading, so
    a tag that shows up mid-shift counts the horizons down from
    when it started, not from when the first tag did.
    """

    def __init__(self, beacons: list[Beacon],
                 stale_after: float = DEFAULT_STALE_AFTER_S,
                 dark_after: float = DEFAULT_DARK_AFTER_S,
                 readmit_after: int = DEFAULT_READMIT_AFTER,
                 sigma_rssi_db: float = DEFAULT_SIGMA_RSSI_DB,
                 filter: str = "kalman",
                 floor_switch_confidence: float = 0.6,
                 plan: DeploymentPlan | None = None,
                 forecast_window_hours: float
                 = DEFAULT_FORECAST_WINDOW_HOURS):
        if len(beacons) < 3:
            raise ValueError("positioning needs at least 3 beacons")
        if forecast_window_hours <= 0:
            raise ValueError("forecast_window_hours must be positive")
        self.beacons = list(beacons)
        self._stale_after = stale_after
        self._dark_after = dark_after
        self._readmit_after = readmit_after
        self._sigma_rssi_db = sigma_rssi_db
        self._filter = filter
        self._floor_switch_confidence = floor_switch_confidence
        self._plan = plan
        self._forecast_window_hours = forecast_window_hours
        # Device id -> its own tracker (and engine, inside it).
        self._trackers: dict[str, LiveTracker] = {}

    @property
    def devices(self) -> tuple[str, ...]:
        """The device ids seen so far, in first-seen order."""
        return tuple(self._trackers)

    def tracker(self, device_id: str) -> LiveTracker | None:
        """The tracker for one device, or None before its first
        reading."""
        return self._trackers.get(device_id)

    def _tracker_for(self, device_id: str) -> LiveTracker:
        """The tracker for one device, creating it on first use."""
        tracker = self._trackers.get(device_id)
        if tracker is None:
            tracker = LiveTracker(
                PositioningEngine(
                    self.beacons, filter=self._filter,
                    floor_switch_confidence=self._floor_switch_confidence),
                stale_after=self._stale_after,
                dark_after=self._dark_after,
                readmit_after=self._readmit_after,
                sigma_rssi_db=self._sigma_rssi_db,
                plan=self._plan,
                forecast_window_hours=self._forecast_window_hours)
            self._trackers[device_id] = tracker
        return tracker

    def feed(self, device_id: str, rssi: dict[str, float],
             t: float) -> LivePosition | None:
        """Feed one reading for one device and return its live
        position.

        ``device_id`` names the device the reading came from — a
        non-empty string; a reading for a device never seen before
        starts that device's track. ``rssi`` maps beacon id to RSSI
        in dBm (a partial scan is fine), ``t`` is the reading's
        timestamp in seconds, non-decreasing per device. Unknown
        beacon ids are ignored rather than failing the stream.

        Returns the device's live position, or None when fewer
        than three fresh beacons remain for that device — the same
        rule ``LiveTracker.feed`` applies, per device.
        """
        if not isinstance(device_id, str) or not device_id:
            raise ValueError("device_id must be a non-empty string")
        return self._tracker_for(device_id).feed(rssi, t)

    def last_position(self, device_id: str) -> LivePosition | None:
        """The most recent live position for one device, or None
        before its first locatable reading (or for a device never
        seen)."""
        tracker = self._trackers.get(device_id)
        return None if tracker is None else tracker.last_position

    def positions(self) -> dict[str, LivePosition]:
        """Every device's most recent live position, keyed by
        device id — the dashboard snapshot. A device with no
        locatable reading yet is left out rather than reported
        with a made-up position."""
        return {device_id: tracker.last_position
                for device_id, tracker in self._trackers.items()
                if tracker.last_position is not None}

    def reset(self, device_id: str | None = None) -> None:
        """Forget one device's track (``reset("tag-1")`` — a tag
        handed to a new patient starts fresh), or every device's
        track when no id is given."""
        if device_id is None:
            for tracker in self._trackers.values():
                tracker.reset()
            return
        tracker = self._trackers.get(device_id)
        if tracker is not None:
            tracker.reset()

    def remove(self, device_id: str) -> bool:
        """Drop a device entirely, so it stops appearing in
        ``devices`` and ``positions``. Returns True when the
        device was being tracked, False when it was never seen."""
        return self._trackers.pop(device_id, None) is not None


def parse_device_line(line: str) -> tuple[str, float, dict[str, float]]:
    """Parse one multi-device stream line: ``{"device": ...,
    "t": ..., "rssi": {...}}``.

    Returns ``(device_id, t, rssi)``. Raises ValueError on a line
    that is not JSON, or that lacks a non-empty string ``device``,
    a numeric ``t``, or a ``rssi`` object.
    """
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not a JSON line: {line.strip()!r}") from exc
    if not isinstance(record, dict):
        raise ValueError(
            f"stream line must be a JSON object: {line.strip()!r}")
    device_id = record.get("device")
    if not isinstance(device_id, str) or not device_id:
        raise ValueError(
            f"stream line needs a non-empty string 'device': "
            f"{line.strip()!r}")
    t, rssi = parse_line(line)
    return device_id, t, rssi


def format_device_output(device_id: str, t: float,
                         position: LivePosition | None,
                         heard: tuple[str, ...],
                         excluded: tuple[ExcludedBeacon, ...] = (),
                         forecasts: tuple = (),
                         forecast_alerts: tuple[str, ...] = ()) -> dict:
    """One output line per input line: the device, its live
    position, or null when the reading was too thin to locate,
    with the fresh beacons listed either way. When the tracker
    has a deployment plan, a null line still carries the forecast
    flags and alerts, so a dashboard sees a beacon about to go
    dark even on a step that could not be located; a located
    position carries them in its own dict."""
    output = format_output(t, position, heard, excluded,
                           forecasts, forecast_alerts)
    output["device"] = device_id
    return output


def default_beacons() -> list[Beacon]:
    """Four beacons, one in each corner of the default 20 x 12 m ward."""
    return [
        Beacon("A", 0.0, 0.0),
        Beacon("B", 20.0, 0.0),
        Beacon("C", 20.0, 12.0),
        Beacon("D", 0.0, 12.0),
    ]


def run_demo(stale_after: float, dark_after: float,
             readmit_after: int, sigma_rssi_db: float,
             output_stream,
             plan: DeploymentPlan | None = None,
             forecast_window_hours: float
             = DEFAULT_FORECAST_WINDOW_HOURS) -> MultiDeviceTracker:
    """Feed two simulated ward walks through the tracker at 1 Hz —
    one forward, one in reverse — and print one JSON line per
    reading. With ``plan`` given, each device's live position
    also carries the plan's forecast flags per beacon, with an
    alert for every beacon whose soonest milestone falls inside
    ``forecast_window_hours``. Returns the tracker, so the caller
    can inspect the finished tracks."""
    beacons = default_beacons()
    forward = simulate_walk(rectangular_walk(20.0, 12.0, 10), beacons,
                            noise_db=2.0, seed=7)
    reverse = simulate_walk(
        list(reversed(rectangular_walk(20.0, 12.0, 10))), beacons,
        noise_db=2.0, seed=11)
    tracker = MultiDeviceTracker(beacons, stale_after=stale_after,
                                 dark_after=dark_after,
                                 readmit_after=readmit_after,
                                 sigma_rssi_db=sigma_rssi_db,
                                 plan=plan,
                                 forecast_window_hours
                                 =forecast_window_hours)
    for i, (step_a, step_b) in enumerate(zip(forward, reverse)):
        for device_id, step in (("tag-1", step_a), ("tag-2", step_b)):
            t = float(i)
            position = tracker.feed(device_id, step["rssi"], t)
            device_tracker = tracker.tracker(device_id)
            print(json.dumps(format_device_output(
                device_id, t, position, device_tracker.heard,
                device_tracker.excluded, device_tracker.forecasts,
                device_tracker.forecast_alerts)), file=output_stream)
    return tracker


def main(argv=None, input_stream=None, output_stream=None) -> int:
    """Stream entry point: read JSON-line readings from stdin, print
    one JSON line per reading with the device and its live position
    attached.

    ``--demo`` runs two simulated walks instead of reading stdin.
    ``--stale-after`` (seconds), ``--dark-after`` (seconds),
    ``--readmit-after`` (readings) and ``--sigma-rssi`` (dB) tune
    each device's tracker. ``--forecast`` attaches the deployment
    plan to every device's tracker, so each live position carries
    the planner's forecast flags per beacon, with an alert for
    every beacon whose soonest milestone falls inside
    ``--forecast-window-hours``; ``--drain-rate`` and
    ``--battery-levels`` build that plan for the default ward.
    ``input_stream`` / ``output_stream`` exist so tests can drive
    the stream without touching sys.stdin / sys.stdout.
    """
    parser = argparse.ArgumentParser(
        description="Multi-device BLE indoor-positioning tracker: "
        "read RSSI readings as JSON lines and print a live position "
        "per reading, per device.")
    parser.add_argument("--demo", action="store_true",
                        help="run two simulated walks instead of "
                        "reading stdin")
    parser.add_argument("--stale-after", type=float,
                        default=DEFAULT_STALE_AFTER_S,
                        help="seconds before a reading ages out "
                        f"(default: {DEFAULT_STALE_AFTER_S})")
    parser.add_argument("--dark-after", type=float,
                        default=DEFAULT_DARK_AFTER_S,
                        help="seconds unheard before a beacon counts "
                        "as dark and is recorded as excluded "
                        f"(default: {DEFAULT_DARK_AFTER_S})")
    parser.add_argument("--readmit-after", type=int,
                        default=DEFAULT_READMIT_AFTER,
                        help="consecutive readings hearing a dark "
                        "beacon again before it rejoins the estimate "
                        f"(default: {DEFAULT_READMIT_AFTER})")
    parser.add_argument("--sigma-rssi", type=float,
                        default=DEFAULT_SIGMA_RSSI_DB,
                        help="assumed RSSI noise in dB for the confidence "
                        f"ellipse (default: {DEFAULT_SIGMA_RSSI_DB})")
    parser.add_argument("--forecast", action="store_true",
                        help="attach the deployment plan to every "
                        "device's tracker, so each live position "
                        "carries the planner's forecast flags per "
                        "beacon")
    parser.add_argument("--forecast-window-hours", type=float,
                        default=DEFAULT_FORECAST_WINDOW_HOURS,
                        help="hours ahead a forecast milestone must fall "
                        "to alert on the live track "
                        f"(default: {DEFAULT_FORECAST_WINDOW_HOURS})")
    parser.add_argument("--drain-rate", type=float, default=0.05,
                        help="with --forecast: every default beacon's "
                        "battery drain per hour "
                        "(default: 0.05)")
    parser.add_argument("--battery-levels", default=None, metavar="SPEC",
                        help="with --forecast: measured charges, "
                        "e.g. A=0.9,B=0.6,C=0.4,D=0.2 "
                        "(default: all fresh)")
    args = parser.parse_args(argv)
    if args.stale_after <= 0:
        parser.error("--stale-after must be positive")
    if args.dark_after <= 0:
        parser.error("--dark-after must be positive")
    if args.dark_after < args.stale_after:
        parser.error("--dark-after must be at least --stale-after")
    if args.readmit_after < 1:
        parser.error("--readmit-after must be at least 1")
    if args.sigma_rssi <= 0:
        parser.error("--sigma-rssi must be positive")
    if args.forecast_window_hours <= 0:
        parser.error("--forecast-window-hours must be positive")
    if args.drain_rate < 0:
        parser.error("--drain-rate must not be negative")
    plan = None
    if args.forecast:
        try:
            plan = _deployment_plan(args)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    tracker = MultiDeviceTracker(default_beacons(),
                                 stale_after=args.stale_after,
                                 dark_after=args.dark_after,
                                 readmit_after=args.readmit_after,
                                 sigma_rssi_db=args.sigma_rssi,
                                 plan=plan,
                                 forecast_window_hours
                                 =args.forecast_window_hours)
    sink = output_stream if output_stream is not None else sys.stdout
    if args.demo:
        run_demo(args.stale_after, args.dark_after,
                 args.readmit_after, args.sigma_rssi, sink,
                 plan=plan,
                 forecast_window_hours=args.forecast_window_hours)
        return 0

    source = input_stream if input_stream is not None else sys.stdin
    for line in source:
        if not line.strip():
            continue
        device_id, t, rssi = parse_device_line(line)
        position = tracker.feed(device_id, rssi, t)
        device_tracker = tracker.tracker(device_id)
        print(json.dumps(format_device_output(
            device_id, t, position, device_tracker.heard,
            device_tracker.excluded, device_tracker.forecasts,
            device_tracker.forecast_alerts)), file=sink)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
