"""Live tracking mode: consume a stream of RSSI readings in real time.

The engine so far answers one-shot questions — one observation set,
one position. A ward dashboard needs the opposite: a device that
keeps reporting, and a position that keeps updating as readings
arrive. ``LiveTracker`` wraps a ``PositioningEngine`` and turns the
observation stream into a live track:

- Feed it each reading as it arrives — ``feed(rssi, t)`` — and it
  emits a ``LivePosition`` per reading: the smoothed (x, y) from
  the engine's filter, the floor, and the 95% confidence ellipse
  for that step's geometry.
- Readings are merged across beacons, so a reading that only hears
  two beacons still updates the track from the beacons heard a
  moment ago — BLE scans rarely hear every beacon on every scan.
- A reading older than ``stale_after`` seconds (5 s by default)
  ages out of the merged set, so a beacon that has gone quiet
  stops dragging the estimate towards where the device was when
  it was last heard.
- When fewer than three fresh beacons remain, trilateration is
  impossible and ``feed`` returns ``None`` instead of a made-up
  position; the last good position stays available as
  ``last_position`` until a fresh reading revives the track.
- A beacon unheard for longer than ``dark_after`` (10 s by
  default) counts as dark: it is already out of the estimate by
  then, and the fallback is recorded on the position as an
  ``ExcludedBeacon`` — which beacon went quiet and why — so the
  track degrades gracefully and a dashboard can see it doing so.
  A beacon heard again is not trusted back at once: it must be
  heard on ``readmit_after`` consecutive readings (3 by default)
  before it rejoins, so one stray scan cannot flap a dying
  beacon back into the estimate.
- With a ``DeploymentPlan`` attached, each live position also
  carries the planner's forecast flags — one per beacon, the
  soonest of its dark, 3 dB loss and 2 dB noise milestones with
  the hours remaining — and a plain-language alert for every
  beacon whose milestone falls inside the forecast window (24
  hours by default), so the track warns about a beacon about to
  start lying before it does.
- With a ``DriftMonitor`` attached, each live position also
  carries the monitor's drift alarms — the beacons whose rolling
  mean RSSI residual, measured against the track's own live
  estimate, has crossed the 5 dB threshold — and a plain-language
  alert per alarming beacon, so a beacon going bad mid-shift is
  caught while the ward is running. The reference is the track's
  own estimate, not ground truth, so the alarm names the beacon
  that disagrees with the consensus of the others; a beacon
  reading too quiet can drag the estimate with it and alarm on
  the wrong beacon — see the README for the measured limits.

The stream format is JSON lines, one reading per line — the same
shape a BLE scanner would emit, and easy to pipe from anything:

    {"t": 0.0, "rssi": {"A": -62.1, "B": -71.4, "C": -68.0, "D": -59.8}}

Run it as a module: ``python -m beacon.tracking`` reads those lines
from stdin and prints one JSON line per reading with the live
position attached (``position`` is null on readings too thin to
locate). ``--demo`` runs a simulated walk through the tracker
instead, so the mode can be tried with no scanner at all.
"""

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass

from .battery import BatteryModel
from .beacon import Beacon
from .deployment import DeploymentPlan, plan_deployment
from .drift import DriftAlarm, DriftClassification, DriftMonitor
from .engine import PositioningEngine
from .schedule import (REASON_DARK, REASON_LOSS, REASON_NEVER, REASON_NOISE)
from .simulate import rectangular_walk, simulate_walk
from .uncertainty import PositionUncertainty

# A reading older than this many seconds no longer describes where
# the device is, so it leaves the merged observation set.
DEFAULT_STALE_AFTER_S = 5.0

# A beacon unheard for this long counts as dark: it has gone quiet
# mid-track, so the estimate is built without it and the fallback is
# recorded on the position. Longer than the stale window, so a dark
# beacon has already aged out of the merged set before it is called
# dark — the two windows never disagree about whether a beacon is
# in the estimate.
DEFAULT_DARK_AFTER_S = 10.0

# A dark beacon heard again must be heard on this many consecutive
# readings before it rejoins the estimate. One stray scan — a
# beacon flickering at the edge of range, or a single reflected
# packet — must not flap a dying beacon back in and out of the
# solve; three in a row says it is really back.
DEFAULT_READMIT_AFTER = 3

# Assumed RSSI noise for the confidence ellipse, matching the
# simulator's default noise and the engine's default.
DEFAULT_SIGMA_RSSI_DB = 2.0

# A beacon whose reported battery level is at or below this is
# treated as dead from a drained cell, not just unheard — the
# same level as ``BatteryModel``'s default cutoff, so the
# exclusion reason can name the battery as the cause.
BATTERY_EXHAUSTED_LEVEL = 0.05

# How far ahead a forecast milestone must fall to raise an alert
# on the live track, in hours. A beacon due a new cell within
# the next day is worth flagging while the ward is running; one
# due next week is a planning matter, not a live-track warning.
DEFAULT_FORECAST_WINDOW_HOURS = 24.0


@dataclass(frozen=True)
class ExcludedBeacon:
    """A beacon left out of a live position, and why.

    ``id`` is the beacon, ``reason`` says what happened to it —
    for a beacon that went dark mid-track, how long it has been
    unheard and the dark window it crossed, e.g.
    ``"no reading for 12.0s (dark after 10.0s)"``.
    """

    id: str
    reason: str

    def to_dict(self) -> dict:
        """JSON-friendly form, for the stream output and dashboards."""
        return {"id": self.id, "reason": self.reason}


@dataclass(frozen=True)
class ForecastFlag:
    """One beacon's deployment forecast, as seen on the live track.

    ``milestone`` names the soonest of the beacon's three forecast
    milestones — ``dark``, ``3 dB loss`` or ``2 dB noise`` — the
    same tie-breaking as the replacement schedule (going dark
    beats reading weak, reading weak beats reading noisy), or
    ``never`` when the beacon does not drain. ``hours`` is the
    time remaining until that milestone (``math.inf`` when never,
    0.0 when already past it), and the three horizons ride along
    so a dashboard can show why the date is what it is.
    ``status`` is the forecast's own status (``ok``, ``degrading``
    or ``dark``).
    """

    beacon_id: str
    milestone: str
    hours: float
    status: str
    hours_to_dark: float
    hours_to_rssi_loss_db: float
    hours_to_extra_noise_db: float

    def to_dict(self) -> dict:
        """JSON-friendly form; never-reached horizons are null."""
        def _hours(value: float) -> float | None:
            return None if math.isinf(value) else value

        return {
            "beacon": self.beacon_id,
            "milestone": self.milestone,
            "hours": _hours(self.hours),
            "status": self.status,
            "hours_to_dark": _hours(self.hours_to_dark),
            "hours_to_rssi_loss_db": _hours(self.hours_to_rssi_loss_db),
            "hours_to_extra_noise_db": _hours(self.hours_to_extra_noise_db),
        }


def forecast_flags(plan: DeploymentPlan,
                   elapsed_s: float = 0.0) -> tuple[ForecastFlag, ...]:
    """One forecast flag per beacon in a deployment plan.

    ``elapsed_s`` is how long the track has been running since the
    plan was made, in seconds — the horizons shrink as the track
    runs, so a milestone that passes mid-track reads 0.0 hours
    remaining rather than going negative. A beacon that never
    drains keeps ``math.inf`` horizons and the ``never``
    milestone.
    """
    elapsed_h = elapsed_s / 3600.0

    def _remaining(value: float) -> float:
        return math.inf if math.isinf(value) else max(0.0, value - elapsed_h)

    flags = []
    for forecast in plan.forecasts:
        milestones = (
            (forecast.hours_to_dark, REASON_DARK),
            (forecast.hours_to_rssi_loss_db, REASON_LOSS),
            (forecast.hours_to_extra_noise_db, REASON_NOISE),
        )
        soonest, milestone = min(milestones, key=lambda m: m[0])
        if math.isinf(soonest):
            milestone = REASON_NEVER
        flags.append(ForecastFlag(
            beacon_id=forecast.beacon_id,
            milestone=milestone,
            hours=_remaining(soonest),
            status=forecast.status,
            hours_to_dark=_remaining(forecast.hours_to_dark),
            hours_to_rssi_loss_db=_remaining(
                forecast.hours_to_rssi_loss_db),
            hours_to_extra_noise_db=_remaining(
                forecast.hours_to_extra_noise_db),
        ))
    return tuple(flags)


def forecast_alerts(flags: tuple[ForecastFlag, ...],
                    window_hours: float) -> tuple[str, ...]:
    """The plain-language warnings for flags due inside a window.

    A flag alerts when its soonest milestone falls within
    ``window_hours`` — ``"D: 2 dB noise in 13.3h (ok)"``, or
    ``"D: 3 dB loss now (degrading)"`` when the milestone is
    already past. A beacon that never drains never alerts.
    """
    alerts = []
    for flag in flags:
        if math.isinf(flag.hours) or flag.hours > window_hours:
            continue
        when = "now" if flag.hours == 0.0 else f"in {flag.hours:.1f}h"
        alerts.append(f"{flag.beacon_id}: {flag.milestone} {when} "
                      f"({flag.status})")
    return tuple(alerts)


def drift_alerts(alarms: tuple[DriftAlarm, ...]) -> tuple[str, ...]:
    """The plain-language warnings for beacons currently alarming.

    One line per alarm — ``"A: RSSI +8.5 dB off calibration"`` —
    stating only what the alarm establishes (the rolling mean
    residual over the monitor's window). The drift classification
    (battery-sag, moved, noisy) is deliberately not in the alert:
    with the track's own estimate as the reference the verdict is
    a hint, not a diagnosis — a beacon reading too loud can read
    as moved, and a healthy beacon can inherit a verdict from a
    neighbour dragging the estimate — so the alert names the
    suspect and the size of the disagreement, and the
    classification is left to ``DriftMonitor.classify`` for
    inspection.
    """
    return tuple(
        f"{alarm.id}: RSSI {alarm.mean_residual_db:+.1f} dB off "
        f"calibration"
        for alarm in alarms)


@dataclass(frozen=True)
class LivePosition:
    """One live position estimate, emitted per reading.

    ``t`` is the timestamp of the reading that produced it (seconds,
    same clock as the feed). ``x`` / ``y`` are the filter-smoothed
    metres on ``floor`` (``floor_confidence`` is the floor vote's
    confidence). ``uncertainty`` is the 95% confidence ellipse for
    this step's raw geometry — None when the geometry is degenerate
    (the filtered position is still valid). ``heard`` is the fresh
    beacons the estimate was built from, in id order. ``excluded``
    is the beacons left out because they went dark mid-track, with
    the reason each was excluded — empty when no beacon is dark,
    so a dashboard can tell a three-beacon estimate from a
    four-beacon one and why. ``battery`` is the battery level
    reported for each beacon at this step (0 to 1), as
    ``(id, level)`` pairs in id order — empty when the readings
    carried no battery state, so a track without batteries is
    unaffected. ``forecasts`` is one ``ForecastFlag`` per beacon
    in the deployment plan the tracker was given — the soonest
    forecast milestone per beacon with its hours remaining — and
    ``forecast_alerts`` the plain-language warnings for the
    beacons whose milestone falls inside the forecast window;
    both are empty when the tracker has no plan, so a track
    without a planner is unaffected. ``drift_alarms`` is the
    beacons currently alarming on the drift monitor the tracker
    was given — each with its rolling mean RSSI residual in dB
    and how many readings are in its window — and
    ``drift_alerts`` the plain-language warning per alarming
    beacon; both are empty when the tracker has no monitor, so a
    track without drift monitoring is unaffected.
    ``drift_classifications`` is every beacon's drift
    classification from that monitor at this step — the verdict
    (healthy, battery-sag, moved, noisy, uncertain) with the
    numbers it rests on — so a dashboard can see not just that a
    beacon is alarming but what the monitor thinks is wrong with
    it; empty when the tracker has no monitor. The verdict is a
    hint, not a diagnosis — the monitor's reference is the
    track's own estimate, not ground truth — so it rides on the
    position for inspection, never inside the alert text.
    """

    t: float
    x: float
    y: float
    floor: int
    floor_confidence: float
    uncertainty: PositionUncertainty | None
    heard: tuple[str, ...]
    excluded: tuple[ExcludedBeacon, ...] = ()
    battery: tuple[tuple[str, float], ...] = ()
    forecasts: tuple[ForecastFlag, ...] = ()
    forecast_alerts: tuple[str, ...] = ()
    drift_alarms: tuple[DriftAlarm, ...] = ()
    drift_alerts: tuple[str, ...] = ()
    drift_classifications: tuple[DriftClassification, ...] = ()

    def to_dict(self) -> dict:
        """JSON-friendly form, for the stream output and dashboards."""
        return {
            "t": self.t,
            "x": self.x,
            "y": self.y,
            "floor": self.floor,
            "floor_confidence": self.floor_confidence,
            "uncertainty": (
                None if self.uncertainty is None else {
                    "semi_major": self.uncertainty.semi_major,
                    "semi_minor": self.uncertainty.semi_minor,
                    "angle_deg": self.uncertainty.angle_deg,
                }),
            "heard": list(self.heard),
            "excluded": [b.to_dict() for b in self.excluded],
            "battery": {beacon_id: level
                        for beacon_id, level in self.battery},
            "forecasts": [flag.to_dict() for flag in self.forecasts],
            "forecast_alerts": list(self.forecast_alerts),
            "drift_alarms": [alarm.to_dict()
                             for alarm in self.drift_alarms],
            "drift_alerts": list(self.drift_alerts),
            "drift_classifications": [c.to_dict() for c in
                                      self.drift_classifications],
        }


class LiveTracker:
    """Turn a stream of RSSI readings into a live position track.

    ``engine`` is the ``PositioningEngine`` doing the localisation;
    one tracker tracks one device. ``stale_after`` is how long a
    reading stays in the merged observation set, in seconds.
    ``dark_after`` is how long a beacon may go unheard before it
    counts as dark: past that window it is excluded from the
    estimate and recorded on the position as an ``ExcludedBeacon``
    (with why), so a beacon dying mid-track degrades the track
    instead of breaking it — and a dashboard can see which beacon
    went quiet. ``dark_after`` must be at least ``stale_after``,
    since a beacon still fresh enough to localise from cannot be
    called dark. ``readmit_after`` is how many consecutive readings
    must hear a dark beacon again before it rejoins the estimate:
    a beacon that flickers back for one scan stays excluded, so a
    dying beacon cannot flap in and out of the solve.
    ``sigma_rssi_db`` is the assumed RSSI noise for the confidence
    ellipse (default 2.0 dB, matching the simulator).
    ``plan`` is an optional ``DeploymentPlan``: when one is given,
    each live position carries one forecast flag per beacon — the
    soonest of its dark, 3 dB loss and 2 dB noise milestones, with
    the hours remaining — and a plain-language alert for every
    beacon whose milestone falls inside ``forecast_window_hours``
    (24 hours by default), so a dashboard watching the track sees
    a beacon about to start lying before it does. The plan is
    taken as made at the first reading's timestamp, so the
    horizons shrink as the track runs.
    ``drift_monitor`` is an optional ``DriftMonitor`` (built from
    the same beacons as the engine): when one is given, every
    located step feeds it the step's fresh readings together with
    the live position just computed, and the position carries the
    beacons currently alarming plus a plain-language alert per
    alarming beacon — and every beacon's drift classification at
    that step, so the position also says what the monitor thinks
    is wrong with each beacon, not just which ones are alarming.
    The reference is the track's own estimate,
    not ground truth — each beacon is checked against the position
    the consensus of the beacons produced — so a beacon reading
    too loud alarms on the right beacon, while a beacon reading
    too quiet drags the estimate with it and can alarm on the
    wrong one; the monitor is a consistency check on the live
    track, and the reliable per-beacon verdict still comes from
    feeding it a reference position (a reference tag at a known
    spot) directly. Only the beacons on the track's current floor
    are fed to the monitor, since a beacon heard through a slab
    would read as drift.
    """

    def __init__(self, engine: PositioningEngine,
                 stale_after: float = DEFAULT_STALE_AFTER_S,
                 dark_after: float = DEFAULT_DARK_AFTER_S,
                 readmit_after: int = DEFAULT_READMIT_AFTER,
                 sigma_rssi_db: float = DEFAULT_SIGMA_RSSI_DB,
                 plan: DeploymentPlan | None = None,
                 forecast_window_hours: float
                 = DEFAULT_FORECAST_WINDOW_HOURS,
                 drift_monitor: DriftMonitor | None = None):
        if stale_after <= 0:
            raise ValueError("stale_after must be positive")
        if dark_after <= 0:
            raise ValueError("dark_after must be positive")
        if dark_after < stale_after:
            raise ValueError(
                f"dark_after ({dark_after}) must be at least "
                f"stale_after ({stale_after}): a beacon still fresh "
                "enough to localise from cannot count as dark")
        if readmit_after < 1:
            raise ValueError("readmit_after must be at least 1")
        if sigma_rssi_db <= 0:
            raise ValueError("sigma_rssi_db must be positive")
        if forecast_window_hours <= 0:
            raise ValueError("forecast_window_hours must be positive")
        self.engine = engine
        self._stale_after = stale_after
        self._dark_after = dark_after
        self._readmit_after = readmit_after
        self._sigma_rssi_db = sigma_rssi_db
        self._plan = plan
        self._forecast_window_hours = forecast_window_hours
        self._drift_monitor = drift_monitor
        # Timestamp of the first reading, when the plan is taken as
        # made — the forecast horizons count down from here.
        self._plan_t0: float | None = None
        # Beacon id -> (rssi, t) of its most recent reading.
        self._latest: dict[str, tuple[float, float]] = {}
        # Beacon id -> t of its most recent reading, kept even
        # after the reading ages out, so a beacon that goes quiet
        # can be recognised as dark rather than never-heard.
        self._seen: dict[str, float] = {}
        # Beacon id -> last reported battery level (0 to 1), from
        # the most recent reading that carried battery state.
        # Empty when the readings carry none, so a track without
        # batteries is unaffected.
        self._battery: dict[str, float] = {}
        # Beacon id -> (consecutive readings that heard it since it
        # went dark, whether it has been heard at all since going
        # dark). A dark beacon leaves this dict — rejoins the
        # estimate — only once the streak reaches ``readmit_after``;
        # a missed reading resets the streak to 0 but keeps the
        # beacon dark, so only consecutive hearings bring it back.
        self._dark: dict[str, tuple[int, bool]] = {}
        self._last_t: float | None = None
        self._last: LivePosition | None = None

    @property
    def last_position(self) -> LivePosition | None:
        """The most recent live position, or None before the first
        locatable reading."""
        return self._last

    @property
    def heard(self) -> tuple[str, ...]:
        """The beacons currently fresh enough to localise from,
        in id order."""
        return tuple(sorted(self._latest))

    def _excluded_at(self, t: float) -> tuple[ExcludedBeacon, ...]:
        """The beacons currently dark at time ``t``, in id order.

        A beacon stays dark until it has been heard on
        ``readmit_after`` consecutive readings: one heard again
        but still short of that count is excluded with a reason
        saying how many fresh readings it still needs, so a
        dashboard can see it coming back rather than vanishing
        from the list on a single stray scan. A beacon never
        heard is not excluded — there is nothing to fall back
        from, it simply was never part of the track. When the
        readings carry battery state and a dark beacon's last
        reported level is at or below the exhausted level, the
        reason names the battery as the cause ("battery exhausted
        at 0.04 — ...") instead of only the silence, so a
        dashboard can tell a beacon that died from a drained cell
        from one that was shielded or unplugged."""
        excluded = []
        for beacon_id, (heard_again, ever_heard) in sorted(
                self._dark.items()):
            if ever_heard:
                excluded.append(ExcludedBeacon(
                    id=beacon_id,
                    reason=(f"heard again, {heard_again} of "
                            f"{self._readmit_after} fresh readings "
                            "to re-admit")))
            else:
                reason = (f"no reading for "
                          f"{t - self._seen[beacon_id]:.1f}s "
                          f"(dark after {self._dark_after:.1f}s)")
                level = self._battery.get(beacon_id)
                if (level is not None
                        and level <= BATTERY_EXHAUSTED_LEVEL):
                    reason = (f"battery exhausted at {level:.2f} — "
                              f"{reason}")
                excluded.append(ExcludedBeacon(
                    id=beacon_id, reason=reason))
        return tuple(excluded)

    def _update_dark(self, heard_now: set[str], t: float) -> None:
        """Maintain the dark set after one reading at time ``t``.

        ``heard_now`` is the beacons heard in this reading. A dark
        beacon heard again counts one step towards re-admission; a
        reading that misses it breaks the streak, so only
        consecutive hearings bring it back. A beacon heard before
        and now silent past ``dark_after`` goes dark. Runs before
        the position is computed, so the exclusion on this step's
        position already reflects the reading that produced it."""
        for beacon_id in list(self._dark):
            streak, _ = self._dark[beacon_id]
            if beacon_id in heard_now:
                streak += 1
                if streak >= self._readmit_after:
                    del self._dark[beacon_id]
                else:
                    self._dark[beacon_id] = (streak, True)
            else:
                # Missed again: the streak resets, but whether the
                # beacon has been heard at all since going dark is
                # unchanged — a beacon never heard back keeps the
                # silence reason, not the re-admission one.
                self._dark[beacon_id] = (0, self._dark[beacon_id][1])
        for beacon_id, last_seen in self._seen.items():
            if (beacon_id not in self._dark
                    and t - last_seen > self._dark_after):
                self._dark[beacon_id] = (0, False)

    @property
    def excluded(self) -> tuple[ExcludedBeacon, ...]:
        """The beacons currently dark — heard before, unheard for
        longer than ``dark_after`` — in id order. Empty before the
        first reading."""
        if self._last_t is None:
            return ()
        return self._excluded_at(self._last_t)

    def _forecasts_at(self, t: float) -> tuple[ForecastFlag, ...]:
        """The plan's forecast flags at time ``t`` — one per beacon,
        with the hours remaining counted down from the first
        reading. Empty when the tracker has no plan."""
        if self._plan is None or self._plan_t0 is None:
            return ()
        return forecast_flags(self._plan, t - self._plan_t0)

    @property
    def forecasts(self) -> tuple[ForecastFlag, ...]:
        """The plan's forecast flags at the latest reading — one
        per beacon, soonest milestone first in plan order. Empty
        before the first reading, or when the tracker has no plan."""
        if self._last_t is None:
            return ()
        return self._forecasts_at(self._last_t)

    @property
    def forecast_alerts(self) -> tuple[str, ...]:
        """The plain-language warnings for beacons whose soonest
        forecast milestone falls inside the forecast window, at
        the latest reading. Empty before the first reading, or
        when the tracker has no plan."""
        return forecast_alerts(self.forecasts,
                               self._forecast_window_hours)

    @property
    def drift_monitor(self) -> DriftMonitor | None:
        """The drift monitor attached to this tracker, or None."""
        return self._drift_monitor

    @property
    def drift_alarms(self) -> tuple[DriftAlarm, ...]:
        """The beacons currently alarming on the drift monitor, in
        beacon order. Empty before the first located reading, or
        when the tracker has no monitor."""
        if self._drift_monitor is None:
            return ()
        return tuple(self._drift_monitor.alarms())

    @property
    def drift_alerts(self) -> tuple[str, ...]:
        """The plain-language warnings for the beacons currently
        alarming on the drift monitor. Empty when nothing alarms,
        or when the tracker has no monitor."""
        return drift_alerts(self.drift_alarms)

    def drift_classifications(self) -> tuple[DriftClassification, ...]:
        """Every beacon's drift classification from the attached
        monitor, in beacon order — the verdict (healthy,
        battery-sag, moved, noisy, uncertain) with the evidence
        behind it. Empty when the tracker has no monitor. The
        verdicts are a hint, not a diagnosis: with the track's own
        estimate as the reference, a beacon reading too loud can
        read as moved and a healthy beacon can inherit a verdict
        from a neighbour dragging the estimate."""
        if self._drift_monitor is None:
            return ()
        return tuple(self._drift_monitor.classifications())

    def feed(self, rssi: dict[str, float], t: float,
             battery: dict[str, float] | None = None) -> LivePosition | None:
        """Feed one reading and return the live position it gives.

        ``rssi`` maps beacon id to RSSI in dBm (a partial scan is
        fine — only the beacons heard need be present), ``t`` is the
        reading's timestamp in seconds. ``battery`` maps beacon id
        to its reported battery level (0 to 1) at this reading,
        when the readings carry battery state — the levels are
        attached to the live position, and a beacon that goes dark
        with an exhausted level is excluded with the battery named
        as the cause. Unknown beacon ids are ignored rather than
        failing the stream. Timestamps must not go backwards: a
        reading older than the previous one is a replayed or
        misordered scan, not a live one. When the tracker was
        given a deployment plan, the position also carries one
        forecast flag per beacon — the soonest milestone with its
        hours remaining — and an alert for every beacon whose
        milestone falls inside the forecast window. When the
        tracker was given a drift monitor, the position also
        carries the beacons currently alarming on it — each
        beacon's rolling mean RSSI residual measured against the
        live position just computed — with a plain-language alert
        per alarming beacon.

        Returns the live position, or None when fewer than three
        fresh beacons remain — trilateration needs three, and a
        made-up position is worse than no position.
        """
        if self._last_t is not None and t < self._last_t:
            raise ValueError(
                f"timestamps must not go backwards: {t} after {self._last_t}")
        self._last_t = t
        if self._plan_t0 is None:
            # The plan counts down from the first reading.
            self._plan_t0 = t
        if battery is not None:
            for beacon_id, level in battery.items():
                if beacon_id in self.engine.beacons:
                    self._battery[beacon_id] = level
        heard_now = set()
        for beacon_id, value in rssi.items():
            if beacon_id not in self.engine.beacons:
                continue
            heard_now.add(beacon_id)
            self._latest[beacon_id] = (value, t)
            self._seen[beacon_id] = t
        # A dark beacon heard again counts towards re-admission,
        # a beacon silent past the dark window goes dark — before
        # the position is computed, so this step's exclusion list
        # already reflects the reading that produced it.
        self._update_dark(heard_now, t)
        # Age out readings that no longer describe where the device
        # is: a beacon heard a while ago must not anchor the track.
        self._latest = {
            beacon_id: reading
            for beacon_id, reading in self._latest.items()
            if t - reading[1] <= self._stale_after}
        fresh = {beacon_id: reading[0]
                 for beacon_id, reading in self._latest.items()}
        if len(fresh) < 3:
            return None
        position = self.engine.estimate_with_floor(fresh)
        # The ellipse describes this step's raw geometry, from the
        # same beacon subset the engine localised on: the winning
        # floor's beacons when at least three were heard there, else
        # every heard beacon (the engine's own fallback).
        on_floor = [b for b in self.engine.beacons.values()
                    if b.floor == position.floor and b.id in fresh]
        subset = (on_floor if len(on_floor) >= 3
                  else [self.engine.beacons[i] for i in fresh])
        try:
            _, _, uncertainty = self.engine.estimate_with_uncertainty(
                {b.id: fresh[b.id] for b in subset},
                sigma_rssi_db=self._sigma_rssi_db)
        except ValueError:
            # Degenerate geometry (collinear beacons, or the estimate
            # sitting on a beacon): the filtered position is still
            # valid, it just has no ellipse this step.
            uncertainty = None
        forecasts = self._forecasts_at(t)
        alarms = self._drift_alarms_at(position, heard_now, rssi)
        live = LivePosition(
            t=t, x=position.x, y=position.y, floor=position.floor,
            floor_confidence=position.floor_confidence,
            uncertainty=uncertainty, heard=tuple(sorted(fresh)),
            excluded=self._excluded_at(t),
            battery=tuple(sorted(self._battery.items())),
            forecasts=forecasts,
            forecast_alerts=forecast_alerts(
                forecasts, self._forecast_window_hours),
            drift_alarms=alarms,
            drift_alerts=drift_alerts(alarms),
            drift_classifications=self.drift_classifications())
        self._last = live
        return live

    def _drift_alarms_at(self, position, heard_now: set[str],
                         rssi: dict[str, float]
                         ) -> tuple[DriftAlarm, ...]:
        """Feed this step to the drift monitor and return the
        beacons alarming now. Empty when the tracker has no
        monitor.

        The monitor gets the live position just computed and the
        beacons heard in this reading that sit on the track's
        current floor — a beacon heard through a slab reads as
        drift, so cross-floor beacons stay out. A step too thin
        to locate never reaches here, so the monitor is only fed
        positions the tracker actually produced.
        """
        if self._drift_monitor is None:
            return ()
        self._drift_monitor.observe(
            (position.x, position.y),
            {beacon_id: rssi[beacon_id] for beacon_id in heard_now
             if self.engine.beacons[beacon_id].floor
             == position.floor})
        return tuple(self._drift_monitor.alarms())

    def reset(self) -> None:
        """Forget the track (e.g. when tracking a new device)."""
        self._latest.clear()
        self._seen.clear()
        self._dark.clear()
        self._battery.clear()
        self._last_t = None
        self._last = None
        self._plan_t0 = None
        if self._drift_monitor is not None:
            self._drift_monitor.reset()
        self.engine.reset()


@dataclass(frozen=True)
class TrackReport:
    """How a live track scored against ground truth.

    ``steps`` is how many ground-truth steps were scored,
    ``located`` how many of them produced a live position, and
    ``coverage`` is ``located / steps`` — the share of the walk the
    tracker actually located, so a track starved by dead beacons
    shows up as low coverage rather than silently short. The error
    figures are the Euclidean distance between the live position
    and the true position, over the located steps only: mean,
    median, 90th percentile, and worst step. ``floor_accuracy`` is
    the share of located steps whose floor was right, or None when
    the ground truth carries no floors.
    """

    steps: int
    located: int
    coverage: float
    mean_error_m: float
    median_error_m: float
    p90_error_m: float
    max_error_m: float
    floor_accuracy: float | None

    def to_dict(self) -> dict:
        """JSON-friendly form, for dashboards and regression logs."""
        return {
            "steps": self.steps,
            "located": self.located,
            "coverage": self.coverage,
            "mean_error_m": self.mean_error_m,
            "median_error_m": self.median_error_m,
            "p90_error_m": self.p90_error_m,
            "max_error_m": self.max_error_m,
            "floor_accuracy": self.floor_accuracy,
        }


def evaluate_live_track(
        positions: list[LivePosition | None],
        truth: list[tuple[float, float] | tuple[float, float, int]]
        ) -> TrackReport:
    """Score a live track against ground truth, step by step.

    ``positions`` is the track — one ``LivePosition`` (or None, for
    a step too thin to locate) per step, in order — and ``truth``
    is the matching ground truth, one ``(x, y)`` per step, or
    ``(x, y, floor)`` when the walk changes floor. The two lists
    must be the same length: feed a recorded walk through a
    ``LiveTracker`` step by step and collect the positions it
    returns, in the same order as the walk's steps.

    Raises ValueError on an empty walk, on mismatched lengths, on
    truth entries that mix two- and three-tuples, or when no step
    was located at all — with no located steps there is no error
    to summarise, and a report of zeroes would hide the failure.
    """
    if not truth:
        raise ValueError("cannot score an empty walk")
    if len(positions) != len(truth):
        raise ValueError(
            f"positions ({len(positions)}) and truth ({len(truth)}) "
            "must have the same length")
    has_floor = [len(point) == 3 for point in truth]
    if any(has_floor) and not all(has_floor):
        raise ValueError(
            "truth must be all (x, y) or all (x, y, floor), not mixed")
    errors: list[float] = []
    floors_right = 0
    located = 0
    for position, point in zip(positions, truth):
        if position is None:
            continue
        located += 1
        errors.append(math.hypot(position.x - point[0],
                                 position.y - point[1]))
        if len(point) == 3 and position.floor == point[2]:
            floors_right += 1
    if not errors:
        raise ValueError("no step was located, nothing to score")
    errors.sort()
    p90_index = max(0, math.ceil(0.9 * len(errors)) - 1)
    return TrackReport(
        steps=len(truth),
        located=located,
        coverage=located / len(truth),
        mean_error_m=statistics.fmean(errors),
        median_error_m=statistics.median(errors),
        p90_error_m=errors[p90_index],
        max_error_m=errors[-1],
        floor_accuracy=(floors_right / located
                        if all(has_floor) else None),
    )


def parse_line(line: str) -> tuple[float, dict[str, float]]:
    """Parse one stream line: ``{"t": ..., "rssi": {...}}``.

    Returns ``(t, rssi)``. Raises ValueError on a line that is not
    JSON, or that lacks a numeric ``t`` or a ``rssi`` object.
    """
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not a JSON line: {line.strip()!r}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"stream line must be a JSON object: {line.strip()!r}")
    t = record.get("t")
    rssi = record.get("rssi")
    if not isinstance(t, (int, float)) or isinstance(t, bool):
        raise ValueError(f"stream line needs a numeric 't': {line.strip()!r}")
    if not isinstance(rssi, dict) or not rssi:
        raise ValueError(f"stream line needs a 'rssi' object: {line.strip()!r}")
    return float(t), {str(k): float(v) for k, v in rssi.items()}


def format_output(t: float, position: LivePosition | None,
                  heard: tuple[str, ...],
                  excluded: tuple[ExcludedBeacon, ...] = (),
                  forecasts: tuple[ForecastFlag, ...] = (),
                  forecast_alerts: tuple[str, ...] = (),
                  drift_alarms: tuple[DriftAlarm, ...] = (),
                  drift_alerts: tuple[str, ...] = ()) -> dict:
    """One output line per input line: the live position, or null
    when the reading was too thin to locate, with the fresh beacons
    listed either way so a dashboard can see the track starving —
    and the dark beacons excluded from the estimate, so it can see
    why the track is running on fewer beacons than it started
    with. When the tracker has a deployment plan, a null line
    still carries the forecast flags and alerts, so a dashboard
    sees a beacon about to go dark even on a step that could not
    be located; a located position carries them in its own dict.
    The same holds for the drift monitor: a null line still
    carries the beacons currently alarming, so a dashboard sees a
    beacon going bad even on a step that could not be located.
    """
    if position is None:
        line = {"t": t, "position": None, "heard": list(heard),
                "excluded": [b.to_dict() for b in excluded]}
        if forecasts:
            line["forecasts"] = [flag.to_dict() for flag in forecasts]
            line["forecast_alerts"] = list(forecast_alerts)
        if drift_alarms:
            line["drift_alarms"] = [alarm.to_dict()
                                    for alarm in drift_alarms]
            line["drift_alerts"] = list(drift_alerts)
        return line
    return {"t": t, "position": position.to_dict(),
            "heard": list(heard)}


def default_beacons() -> list[Beacon]:
    """Four beacons, one in each corner of the default 20 x 12 m ward."""
    return [
        Beacon("A", 0.0, 0.0),
        Beacon("B", 20.0, 0.0),
        Beacon("C", 20.0, 12.0),
        Beacon("D", 0.0, 12.0),
    ]


def run_demo(stale_after: float, dark_after: float,
             readmit_after: int,
             sigma_rssi_db: float, output_stream,
             battery_drain_per_hour: float = 0.0,
             plan: DeploymentPlan | None = None,
             forecast_window_hours: float
             = DEFAULT_FORECAST_WINDOW_HOURS,
             drift_monitor: DriftMonitor | None = None,
             drift_beacon: tuple[str, float] | None = None
             ) -> list[LivePosition | None]:
    """Feed a simulated ward walk through the tracker at 1 Hz and
    print one JSON line per step — the mode with no scanner needed.
    With ``battery_drain_per_hour`` above zero, beacon D drains
    at that rate from a full cell, so the walk shows a beacon
    degrading and then going dark mid-track — its battery level
    rides on each live position. With ``plan`` given, each live
    position also carries the plan's forecast flags per beacon,
    with an alert for every beacon whose soonest milestone falls
    inside ``forecast_window_hours``. With ``drift_monitor``
    given, each live position also carries the beacons currently
    alarming on it; ``drift_beacon`` is a ``(id, dB)`` pair that
    offsets one beacon's RSSI by that many dB for the whole demo
    walk, so the demo can show a beacon going bad mid-track.
    Returns the recorded track, so the caller can export it."""
    beacons = default_beacons()
    batteries = None
    if battery_drain_per_hour > 0:
        batteries = {"D": BatteryModel(
            drain_rate_per_hour=battery_drain_per_hour)}
    steps = simulate_walk(rectangular_walk(20.0, 12.0, 10), beacons,
                          noise_db=2.0, seed=7, batteries=batteries)
    if drift_beacon is not None:
        beacon_id, offset_db = drift_beacon
        for step in steps:
            if beacon_id in step["rssi"]:
                step["rssi"][beacon_id] += offset_db
    tracker = LiveTracker(PositioningEngine(beacons),
                          stale_after=stale_after,
                          dark_after=dark_after,
                          readmit_after=readmit_after,
                          sigma_rssi_db=sigma_rssi_db,
                          plan=plan,
                          forecast_window_hours=forecast_window_hours,
                          drift_monitor=drift_monitor)
    positions: list[LivePosition | None] = []
    for i, step in enumerate(steps):
        t = float(i)
        position = tracker.feed(step["rssi"], t,
                                battery=step.get("battery"))
        positions.append(position)
        print(json.dumps(format_output(t, position, tracker.heard,
                                       tracker.excluded,
                                       tracker.forecasts,
                                       tracker.forecast_alerts,
                                       tracker.drift_alarms,
                                       tracker.drift_alerts)),
              file=output_stream)
    return positions


def _parse_battery_levels(spec: str) -> dict[str, float]:
    """Parse ``--battery-levels``: ``"A=0.9,B=0.6"`` to a dict."""
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


def main(argv=None, input_stream=None, output_stream=None) -> int:
    """Stream entry point: read JSON-line readings from stdin, print
    one JSON line per reading with the live position attached.

    ``--demo`` runs a simulated walk instead of reading stdin.
    ``--stale-after`` (seconds), ``--dark-after`` (seconds),
    ``--readmit-after`` (readings) and ``--sigma-rssi`` (dB) tune
    the tracker. ``--forecast`` attaches the deployment plan to
    the tracker, so each live position carries the planner's
    forecast flags per beacon, with an alert for every beacon
    whose soonest milestone falls inside
    ``--forecast-window-hours``; ``--drain-rate`` and
    ``--battery-levels`` build that plan for the default ward.
    ``--drift`` attaches a drift monitor to the tracker, so each
    live position carries the beacons currently alarming on
    drift, with an alert per alarming beacon; ``--drift-beacon``
    (demo only) offsets one beacon's RSSI so the demo shows a
    beacon going bad mid-track.
    ``--export-json`` / ``--export-csv`` save the recorded track
    (one row per located position) to a file a ward dashboard can
    consume, in either mode. ``input_stream`` / ``output_stream``
    exist so tests can drive the stream without touching
    sys.stdin / sys.stdout.
    """
    parser = argparse.ArgumentParser(
        description="Live BLE indoor-positioning tracker: read RSSI "
        "readings as JSON lines and print a live position per reading.")
    parser.add_argument("--demo", action="store_true",
                        help="run a simulated walk instead of reading stdin")
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
    parser.add_argument("--battery-drain-per-hour", type=float, default=0.0,
                        help="demo only: drain beacon D's battery at "
                        "this fraction of a full charge per hour, so "
                        "the walk shows a beacon degrading and going "
                        "dark mid-track (default: 0, no battery drain)")
    parser.add_argument("--forecast", action="store_true",
                        help="attach the deployment plan to the tracker, "
                        "so each live position carries the planner's "
                        "forecast flags per beacon")
    parser.add_argument("--drift", action="store_true",
                        help="attach a drift monitor to the tracker, "
                        "so each live position carries the beacons "
                        "currently alarming on drift")
    parser.add_argument("--drift-beacon", default=None, metavar="SPEC",
                        help="demo only: offset one beacon's RSSI by "
                        "this many dB for the whole demo walk, "
                        "e.g. A:12 — so the demo shows a beacon "
                        "going bad mid-track (needs --drift)")
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
                        help="save the recorded track as JSON to PATH "
                        "(one row per located position)")
    parser.add_argument("--export-csv", default=None, metavar="PATH",
                        help="save the recorded track as CSV to PATH "
                        "(one row per located position)")
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
    if args.battery_drain_per_hour < 0:
        parser.error("--battery-drain-per-hour must not be negative")
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

    if args.demo:
        positions = run_demo(args.stale_after, args.dark_after,
                             args.readmit_after, args.sigma_rssi,
                             output_stream or sys.stdout,
                             battery_drain_per_hour
                             =args.battery_drain_per_hour,
                             plan=plan,
                             forecast_window_hours
                             =args.forecast_window_hours,
                             drift_monitor=drift_monitor,
                             drift_beacon=drift_beacon)
        _export_track(positions, args.export_json, args.export_csv)
        return 0

    tracker = LiveTracker(PositioningEngine(default_beacons()),
                          stale_after=args.stale_after,
                          dark_after=args.dark_after,
                          readmit_after=args.readmit_after,
                          sigma_rssi_db=args.sigma_rssi,
                          plan=plan,
                          forecast_window_hours
                          =args.forecast_window_hours,
                          drift_monitor=drift_monitor)
    source = input_stream if input_stream is not None else sys.stdin
    sink = output_stream if output_stream is not None else sys.stdout
    positions = []
    for line in source:
        if not line.strip():
            continue
        t, rssi = parse_line(line)
        position = tracker.feed(rssi, t)
        positions.append(position)
        print(json.dumps(format_output(t, position, tracker.heard,
                                       tracker.excluded,
                                       tracker.forecasts,
                                       tracker.forecast_alerts,
                                       tracker.drift_alarms,
                                       tracker.drift_alerts)),
              file=sink)
    _export_track(positions, args.export_json, args.export_csv)
    return 0


def _export_track(positions: list[LivePosition | None],
                  json_path: str | None, csv_path: str | None) -> None:
    """Save a recorded track to the requested export files, if
    any. A track with no steps (an empty stream) has nothing to
    export, so the flags are quietly skipped rather than
    failing a run that streamed nothing. The import is local:
    ``beacon.export`` imports this module for ``LivePosition``,
    so a top-level import here would be circular."""
    from .export import export_track_csv, export_track_json

    if not positions:
        return
    if json_path is not None:
        export_track_json(positions, json_path)
    if csv_path is not None:
        export_track_csv(positions, csv_path)


if __name__ == "__main__":
    raise SystemExit(main())
