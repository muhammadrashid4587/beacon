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
so a tag that shows up mid-shift warns on its own clock. With a
drift monitor attached (``--drift``), every device's live position
also carries the beacons currently alarming on drift — each
device monitored separately, so one tag's healthy readings never
dilute another tag's alarm and resetting one tag never clears
another's window.
"""

import argparse
import json
import sys

from .beacon import Beacon
from .deployment import DeploymentPlan
from .drift import DriftAlarm, DriftClassification, DriftMonitor
from .engine import PositioningEngine
from .export import (export_multi_track_csv, export_multi_track_json,
                     export_smoothed_multi_track_csv,
                     export_smoothed_multi_track_json)
from .simulate import rectangular_walk, simulate_walk
from .tracking import (DEFAULT_DARK_AFTER_S, DEFAULT_FORECAST_WINDOW_HOURS,
                       DEFAULT_READMIT_AFTER, DEFAULT_SIGMA_RSSI_DB,
                       DEFAULT_STALE_AFTER_S, ExcludedBeacon, LivePosition,
                       LiveTracker, _deployment_plan,
                       evaluate_multi_device_track,
                       evaluate_smoothed_multi_device_track,
                       format_multi_device_report,
                       format_output, parse_line)

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
    ``drift_monitor`` is an optional ``DriftMonitor`` (built from
    the same beacons as the engine): when one is given, each
    device's tracker gets its own fresh monitor with the same
    tuning — ``DriftMonitor.fresh()`` — so every device's live
    position carries the beacons currently alarming on its own
    track, with a plain-language alert per alarming beacon, and
    every beacon's drift classification from its own monitor —
    so one tag's verdict never comes from another tag's
    readings. One monitor per device, not one shared across
    them: a shared window would mix two devices' readings, so
    one tag's healthy readings would dilute another tag's alarm,
    and resetting one device would clear the monitor for every
    device. The alarms are the same consistency check as on a
    single-device track —
    each beacon checked against the position the consensus of
    the beacons produced — with the same honest limit: a beacon
    reading too quiet can drag the estimate with it and alarm on
    the wrong beacon. The tracker records every device's track
    as it runs — one position (or None, for a reading too thin
    to locate) per reading — and ``export_track_json`` /
    ``export_track_csv`` write all of them as one combined file,
    the per-device exports with their rows concatenated and each
    row carrying its device, so a ward dashboard can replay the
    whole shift from one file; the rows are the same
    ``multi_track_rows`` rows the overlay and the replay group
    by device, so the file cannot disagree with them.
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
                 = DEFAULT_FORECAST_WINDOW_HOURS,
                 drift_monitor: DriftMonitor | None = None):
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
        self._drift_monitor = drift_monitor
        # Device id -> its own tracker (and engine, inside it).
        self._trackers: dict[str, LiveTracker] = {}
        # Device id -> its recorded track: one LivePosition (or
        # None, for a reading too thin to locate) per reading fed
        # for that device, in feed order — the track the combined
        # export writes. A device keeps its recorded track until
        # it is reset or removed.
        self._tracks: dict[str, list[LivePosition | None]] = {}

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
                forecast_window_hours=self._forecast_window_hours,
                drift_monitor=(None if self._drift_monitor is None
                               else self._drift_monitor.fresh()))
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
        position = self._tracker_for(device_id).feed(rssi, t)
        # Recorded even when None: a step too thin to locate is
        # still a step, and the export counts it in the device's
        # coverage instead of pretending the track was complete.
        self._tracks.setdefault(device_id, []).append(position)
        return position

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

    def tracks(self) -> dict[str, list[LivePosition | None]]:
        """Every device's recorded track — one ``LivePosition``
        (or None, for a reading too thin to locate) per reading,
        in feed order — keyed by device id, in first-seen order.
        A device whose track was reset has no entry until its
        next reading. The lists are copies, so a caller cannot
        rewrite the track the tracker is still recording."""
        return {device_id: list(positions)
                for device_id, positions in self._tracks.items()}

    def export_track_json(self, path: str) -> None:
        """Write every device's recorded track to ``path`` as one
        combined JSON file — the per-device exports with their
        rows concatenated, each row carrying its device, so a
        dashboard can replay the whole shift from one file. The
        rows are the same :func:`beacon.export.multi_track_rows`
        rows the overlay and the replay group by device, so the
        exported file cannot disagree with them about which rows
        belong to which device. Raises ValueError when no device
        has a track yet."""
        export_multi_track_json(self._tracks, path)

    def export_track_csv(self, path: str) -> None:
        """Write every device's recorded track to ``path`` as one
        combined CSV file — the same rows as
        :meth:`export_track_json`, in the fixed CSV column order,
        so a dashboard can take either format and replay the
        whole shift from it. Raises ValueError when no device
        has a track yet."""
        export_multi_track_csv(self._tracks, path)

    def export_smoothed_track_json(self, path: str,
                                   method: str = "rts",
                                   window: int = 5) -> None:
        """Write every device's recorded track to ``path`` as one
        combined JSON file, with every device's positions
        smoothed — the smoothed export alongside the raw one, so
        a dashboard can replay the whole shift's steadier paths
        from one file. Each device's rows are smoothed
        separately, so one tag's walk never pulls another tag's
        positions. Raises ValueError when no device has a track
        yet."""
        export_smoothed_multi_track_json(self._tracks, path,
                                         method=method, window=window)

    def export_smoothed_track_csv(self, path: str,
                                  method: str = "rts",
                                  window: int = 5) -> None:
        """Write every device's recorded track to ``path`` as one
        combined CSV file, with every device's positions smoothed
        — the same rows as
        :meth:`export_smoothed_track_json`, in the fixed CSV
        column order, so a dashboard can take either format.
        Raises ValueError when no device has a track yet."""
        export_smoothed_multi_track_csv(self._tracks, path,
                                        method=method, window=window)

    def drift_alarms(self, device_id: str) -> tuple[DriftAlarm, ...]:
        """The beacons currently alarming on one device's drift
        monitor, in beacon order. Empty before the device's first
        located reading, or when the tracker has no monitor."""
        tracker = self._trackers.get(device_id)
        return () if tracker is None else tracker.drift_alarms

    def drift_alerts(self, device_id: str) -> tuple[str, ...]:
        """The plain-language warnings for the beacons currently
        alarming on one device's drift monitor. Empty when
        nothing alarms, or when the tracker has no monitor."""
        tracker = self._trackers.get(device_id)
        return () if tracker is None else tracker.drift_alerts

    def drift_classifications(
            self, device_id: str) -> tuple[DriftClassification, ...]:
        """Every beacon's drift classification from one device's
        monitor, in beacon order — the verdict with the evidence
        behind it. Empty when the tracker has no monitor. The
        verdicts are a hint, not a diagnosis: with the track's
        own estimate as the reference, a beacon reading too loud
        can read as moved and a healthy beacon can inherit a
        verdict from a neighbour dragging the estimate."""
        tracker = self._trackers.get(device_id)
        return () if tracker is None else tracker.drift_classifications()

    def reset(self, device_id: str | None = None) -> None:
        """Forget one device's track (``reset("tag-1")`` — a tag
        handed to a new patient starts fresh), or every device's
        track when no id is given. The recorded track goes with
        the live state, so a reset device exports nothing until
        its next reading."""
        if device_id is None:
            for tracker in self._trackers.values():
                tracker.reset()
            self._tracks.clear()
            return
        tracker = self._trackers.get(device_id)
        if tracker is not None:
            tracker.reset()
            self._tracks.pop(device_id, None)

    def remove(self, device_id: str) -> bool:
        """Drop a device entirely, so it stops appearing in
        ``devices`` and ``positions``. Returns True when the
        device was being tracked, False when it was never seen.
        Its recorded track goes with it, so it stops appearing
        in the combined export too."""
        self._tracks.pop(device_id, None)
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
                         forecast_alerts: tuple[str, ...] = (),
                         drift_alarms: tuple[DriftAlarm, ...] = (),
                         drift_alerts: tuple[str, ...] = ()) -> dict:
    """One output line per input line: the device, its live
    position, or null when the reading was too thin to locate,
    with the fresh beacons listed either way. When the tracker
    has a deployment plan, a null line still carries the forecast
    flags and alerts, so a dashboard sees a beacon about to go
    dark even on a step that could not be located; a located
    position carries them in its own dict. The same holds for
    the drift monitor: a null line still carries the beacons
    currently alarming on that device's monitor, so a dashboard
    sees a beacon going bad even on a step that could not be
    located."""
    output = format_output(t, position, heard, excluded,
                           forecasts, forecast_alerts,
                           drift_alarms, drift_alerts)
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
             = DEFAULT_FORECAST_WINDOW_HOURS,
             drift_monitor: DriftMonitor | None = None,
             drift_beacon: tuple[str, float] | None = None,
             return_truths: bool = False
             ) -> MultiDeviceTracker | tuple[
                 MultiDeviceTracker,
                 dict[str, list[tuple[float, float]]]]:
    """Feed two simulated ward walks through the tracker at 1 Hz —
    one forward, one in reverse — and print one JSON line per
    reading. With ``plan`` given, each device's live position
    also carries the plan's forecast flags per beacon, with an
    alert for every beacon whose soonest milestone falls inside
    ``forecast_window_hours``. With ``drift_monitor`` given, each
    device's live position also carries the beacons currently
    alarming on its own monitor; ``drift_beacon`` is a ``(id, dB)``
    pair that offsets one beacon's RSSI by that many dB on
    tag-1's walk only, so the demo shows one device alarming
    while the other stays quiet. Returns the tracker, so the
    caller can inspect the finished tracks. With
    ``return_truths`` given, returns ``(tracker, truths)`` —
    the ground truth per device, one ``(x, y)`` per step, in
    feed order — so the caller can score the demo tracks with
    ``evaluate_multi_device_track``."""
    beacons = default_beacons()
    forward = simulate_walk(rectangular_walk(20.0, 12.0, 10), beacons,
                            noise_db=2.0, seed=7)
    reverse = simulate_walk(
        list(reversed(rectangular_walk(20.0, 12.0, 10))), beacons,
        noise_db=2.0, seed=11)
    if drift_beacon is not None:
        beacon_id, offset_db = drift_beacon
        for step in forward:
            if beacon_id in step["rssi"]:
                step["rssi"][beacon_id] += offset_db
    tracker = MultiDeviceTracker(beacons, stale_after=stale_after,
                                 dark_after=dark_after,
                                 readmit_after=readmit_after,
                                 sigma_rssi_db=sigma_rssi_db,
                                 plan=plan,
                                 forecast_window_hours
                                 =forecast_window_hours,
                                 drift_monitor=drift_monitor)
    for i, (step_a, step_b) in enumerate(zip(forward, reverse)):
        for device_id, step in (("tag-1", step_a), ("tag-2", step_b)):
            t = float(i)
            position = tracker.feed(device_id, step["rssi"], t)
            device_tracker = tracker.tracker(device_id)
            print(json.dumps(format_device_output(
                device_id, t, position, device_tracker.heard,
                device_tracker.excluded, device_tracker.forecasts,
                device_tracker.forecast_alerts,
                device_tracker.drift_alarms,
                device_tracker.drift_alerts)), file=output_stream)
    if return_truths:
        truths = {
            "tag-1": [step["true"] for step in forward],
            "tag-2": [step["true"] for step in reverse],
        }
        return tracker, truths
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
    ``--drift`` attaches a drift monitor to every device's tracker
    (one fresh monitor per device, same tuning), so each live
    position carries the beacons currently alarming on drift, with
    an alert per alarming beacon; ``--drift-beacon`` (demo only)
    offsets one beacon's RSSI on tag-1's walk so the demo shows
    one device alarming while the other stays quiet.
    ``--export-json`` / ``--export-csv`` save every device's
    recorded track as one combined file at the end of the run
    (demo or stream), so a ward dashboard can replay the whole
    shift from it — with no devices tracked the flags are
    quietly skipped, like an empty stream.
    ``--export-smoothed-json`` / ``--export-smoothed-csv`` save
    the same combined file with every device's positions
    smoothed (Rauch-Tung-Striebel, per device), alongside the
    raw export, so a dashboard can replay the steadier paths.
    ``--evaluate`` (demo only) scores the two demo walks
    against their ground truth and prints the multi-device
    track accuracy report — one line per device plus the
    overall line over every device's located steps pooled
    together, in the same fields as the single-device demo
    summary — and ``--evaluate-json PATH`` writes the same
    report as JSON. ``--evaluate-smoothed`` scores the same
    walks again after smoothing each device's track
    (Rauch-Tung-Striebel, the same smoothing the smoothed
    export and the overlay/replay's smoothed trails use) and
    prints that report under a "Smoothed track" heading, and
    ``--evaluate-smoothed-json PATH`` writes it as JSON with a
    ``smoothed`` flag and the smoothing method — so the
    improvement the map shows is the improvement the report
    measures, from the same scoring. A stream has no ground
    truth to score against, so all four flags need ``--demo``.
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
    parser.add_argument("--drift", action="store_true",
                        help="attach a drift monitor to every "
                        "device's tracker (one fresh monitor per "
                        "device), so each live position carries the "
                        "beacons currently alarming on drift")
    parser.add_argument("--drift-beacon", default=None, metavar="SPEC",
                        help="demo only: offset one beacon's RSSI by "
                        "this many dB on tag-1's walk, e.g. A:12 — "
                        "so the demo shows one device alarming "
                        "while the other stays quiet "
                        "(needs --drift)")
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
    parser.add_argument("--export-json", default=None, metavar="PATH",
                        help="save every device's recorded track as one "
                        "combined JSON file to PATH at the end of the "
                        "run (demo or stream)")
    parser.add_argument("--export-csv", default=None, metavar="PATH",
                        help="save every device's recorded track as one "
                        "combined CSV file to PATH at the end of the "
                        "run (demo or stream)")
    parser.add_argument("--export-smoothed-json", default=None,
                        metavar="PATH",
                        help="save every device's recorded track as one "
                        "combined JSON file to PATH at the end of the "
                        "run, with every device's positions smoothed "
                        "(Rauch-Tung-Striebel, per device) — the "
                        "smoothed export alongside the raw one")
    parser.add_argument("--export-smoothed-csv", default=None,
                        metavar="PATH",
                        help="save every device's recorded track as one "
                        "combined CSV file to PATH at the end of the "
                        "run, with every device's positions smoothed "
                        "(Rauch-Tung-Striebel, per device)")
    parser.add_argument("--evaluate", action="store_true",
                        help="demo only: score the two demo walks "
                        "against their ground truth and print the "
                        "multi-device track accuracy report — one "
                        "line per device plus the overall line")
    parser.add_argument("--evaluate-json", default=None, metavar="PATH",
                        help="demo only: write the multi-device track "
                        "accuracy report as JSON to PATH")
    parser.add_argument("--evaluate-smoothed", action="store_true",
                        help="demo only: score the two demo walks again "
                        "after smoothing each device's track "
                        "(Rauch-Tung-Striebel) and print that report "
                        "under a 'Smoothed track' heading — the same "
                        "scoring as --evaluate, over the smoothed "
                        "positions the overlay and the replay draw")
    parser.add_argument("--evaluate-smoothed-json", default=None,
                        metavar="PATH",
                        help="demo only: write the smoothed-track "
                        "accuracy report as JSON to PATH, with a "
                        "smoothed flag and the smoothing method")
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
    if args.evaluate and not args.demo:
        parser.error("--evaluate needs --demo: a stream has no "
                     "ground truth to score against")
    if args.evaluate_json is not None and not args.demo:
        parser.error("--evaluate-json needs --demo: a stream has "
                     "no ground truth to score against")
    if args.evaluate_smoothed and not args.demo:
        parser.error("--evaluate-smoothed needs --demo: a stream "
                     "has no ground truth to score against")
    if args.evaluate_smoothed_json is not None and not args.demo:
        parser.error("--evaluate-smoothed-json needs --demo: a "
                     "stream has no ground truth to score against")
    plan = None
    if args.forecast:
        try:
            plan = _deployment_plan(args)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    drift_monitor = None
    drift_beacon = None
    if args.drift:
        drift_monitor = DriftMonitor(default_beacons())
    if args.drift_beacon is not None:
        if not args.drift:
            parser.error("--drift-beacon needs --drift")
        try:
            beacon_id, offset_text = args.drift_beacon.split(":", 1)
            offset_db = float(offset_text)
        except ValueError:
            parser.error("--drift-beacon takes ID:DB, e.g. A:12")
        if beacon_id not in {b.id for b in default_beacons()}:
            parser.error(f"unknown beacon: {beacon_id!r}")
        drift_beacon = (beacon_id, offset_db)

    tracker = MultiDeviceTracker(default_beacons(),
                                 stale_after=args.stale_after,
                                 dark_after=args.dark_after,
                                 readmit_after=args.readmit_after,
                                 sigma_rssi_db=args.sigma_rssi,
                                 plan=plan,
                                 forecast_window_hours
                                 =args.forecast_window_hours,
                                 drift_monitor=drift_monitor)
    sink = output_stream if output_stream is not None else sys.stdout
    if args.demo:
        want_report = (args.evaluate or args.evaluate_json is not None
                       or args.evaluate_smoothed
                       or args.evaluate_smoothed_json is not None)
        if want_report:
            tracker, truths = run_demo(
                args.stale_after, args.dark_after,
                args.readmit_after, args.sigma_rssi, sink,
                plan=plan,
                forecast_window_hours=args.forecast_window_hours,
                drift_monitor=drift_monitor,
                drift_beacon=drift_beacon,
                return_truths=True)
            if args.evaluate or args.evaluate_json is not None:
                report = evaluate_multi_device_track(tracker.tracks(),
                                                     truths)
                if args.evaluate:
                    print(file=sink)
                    print(format_multi_device_report(report), file=sink)
                if args.evaluate_json is not None:
                    with open(args.evaluate_json, "w",
                              encoding="utf-8") as handle:
                        json.dump(report.to_dict(), handle, indent=2)
                        handle.write("\n")
            if (args.evaluate_smoothed
                    or args.evaluate_smoothed_json is not None):
                smoothed_report = evaluate_smoothed_multi_device_track(
                    tracker.tracks(), truths)
                if args.evaluate_smoothed:
                    print(file=sink)
                    print("Smoothed track (Rauch-Tung-Striebel):",
                          file=sink)
                    print(format_multi_device_report(smoothed_report),
                          file=sink)
                if args.evaluate_smoothed_json is not None:
                    smoothed_dict = smoothed_report.to_dict()
                    smoothed_dict["smoothed"] = True
                    smoothed_dict["smoothing_method"] = "rts"
                    with open(args.evaluate_smoothed_json, "w",
                              encoding="utf-8") as handle:
                        json.dump(smoothed_dict, handle, indent=2)
                        handle.write("\n")
        else:
            tracker = run_demo(args.stale_after, args.dark_after,
                               args.readmit_after, args.sigma_rssi, sink,
                               plan=plan,
                               forecast_window_hours
                               =args.forecast_window_hours,
                               drift_monitor=drift_monitor,
                               drift_beacon=drift_beacon)
    else:
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
                device_tracker.forecast_alerts,
                device_tracker.drift_alarms,
                device_tracker.drift_alerts)), file=sink)
    # A run that tracked nothing has no combined file to write,
    # so the flags are quietly skipped rather than failing a run
    # that streamed nothing.
    if tracker.devices:
        if args.export_json is not None:
            tracker.export_track_json(args.export_json)
        if args.export_csv is not None:
            tracker.export_track_csv(args.export_csv)
        if args.export_smoothed_json is not None:
            tracker.export_smoothed_track_json(args.export_smoothed_json)
        if args.export_smoothed_csv is not None:
            tracker.export_smoothed_track_csv(args.export_smoothed_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
