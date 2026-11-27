"""Live track export: save a recorded track for a ward dashboard.

A live track is only useful while the dashboard is watching it.
A ward that wants the track afterwards — to replay a shift, to
audit where a device went, to feed a dashboard that was not
running at the time — needs the track as a file. This module
exports a recorded track (the ``LivePosition`` values a
``LiveTracker`` emitted, one per step, ``None`` where a step was
too thin to locate) as JSON or CSV, one row per located position:

    timestamp, x, y, floor, confidence ellipse (semi-major,
    semi-minor, angle), beacons used, beacons excluded, coverage,
    battery, planner forecast flags and alerts, drift alarms and
    alerts, and the device the track belongs to

The coverage on every row is the track's overall located share
(located steps / all steps), so a dashboard can read it off any
row; the file header carries the same figure plus the step counts,
so a track starved by dead beacons shows up as low coverage
rather than silently short. Steps that produced no position are
not rows — there is no position to write — but they are counted
in the coverage, so the file never pretends a starved track was
complete. The forecast flags are the planner's per-beacon
forecast as it stood at that step — the soonest milestone with
its hours remaining — with the plain-language alerts for the
beacons due inside the forecast window, so an exported track is
self-describing about upcoming battery milestones instead of
needing the plan alongside it; a track recorded without a plan
exports empty flags and alerts, so a track without a planner is
unaffected. The drift alarms are the beacons the drift monitor
was alarming on at that step — each with its rolling mean RSSI
residual in dB and how many readings are in its window — with
the plain-language alert per alarming beacon, so an exported
track is self-describing about a beacon going bad mid-shift
too; a track recorded without a monitor exports empty alarms
and alerts. The drift classifications are every beacon's
classification from that monitor at that step — the verdict
(healthy, battery-sag, moved, noisy, uncertain) with the
numbers it rests on — so the export also says what the monitor
thinks is wrong with each beacon, not just which ones are
alarming; a track recorded without a monitor exports empty
classifications. The alarms say only what the monitor establishes —
the monitor's reference is the track's own estimate, not ground
truth, so a beacon 8 dB too quiet can alarm on the wrong beacon
— and the export carries them as recorded, without claiming a
ground-truth attribution; the same holds for the
classifications, which are a hint from the same self-referenced
monitor, not a diagnosis.

Both formats round-trip: ``load_track_json`` / ``load_track_csv``
read back exactly the rows ``track_rows`` produced, so an export
can be verified by loading it again. JSON floats round-trip
exactly in Python; CSV numbers are written with ``repr`` and
parsed back as floats, so they do too. The beacons used are
semicolon-joined in CSV (``"A;B;C"``) and the excluded beacons
are a JSON string in the CSV field, since an exclusion carries a
free-text reason.

Pure standard library, no dependencies.

A recorded track can also be exported smoothed: the live filter
is causal, so the exported positions lag behind turns and keep
some jitter, and for a recorded walk the future is already
known. ``smooth_track_rows`` re-runs each device's located
positions through the Kalman filter and the Rauch-Tung-Striebel
backward recursion (the same smoothing ``PositioningEngine.
smooth_walk`` performs on a walk), revising each past position
with the observations that came after it, and the smoothed rows
export exactly like the raw ones — same file shape, same
columns, same loaders — so a dashboard can take either file.
The smoothed file's header says it is smoothed, and which
method was used. Only the (x, y) is revised: the timestamp, the
floor, the confidence ellipse, the beacons used and excluded,
the battery, the forecasts and the drift state are the recorded
step's own, carried unchanged — the ellipse is the live
estimate's, not the smoothed position's, so it still reads as
wide as the live geometry was. The smoothed rows also carry
their own ellipse — ``smoothed_ellipse_semi_major_m``,
``smoothed_ellipse_semi_minor_m`` and
``smoothed_ellipse_angle_deg`` — the 95% ellipse from the
smoother's own posterior covariance at that step (see
:func:`beacon.uncertainty.ellipse_from_covariance`), so the
overlay and the replay can draw how much the smoothing trusts
each revised position instead of reusing the live ellipse at
the new point: the smoother has seen the observations that
came after, so its ellipse is tighter than the live one. A
device's series is smoothed separately from every other
device's, so one tag's walk never pulls another tag's
positions, and a series that changes floor is split at the
change — the live filter resets on a floor change, so the
smoother does too — instead of smoothing across two planes.
The moving-average method is available as an opt-out
(``method="moving_average"``), the same filter the engine
offers; it has no covariance, so its smoothed rows carry no
smoothed ellipse — and on a track that is already live-filtered
it adds lag and measures worse than the raw track, so the
Rauch-Tung-Striebel method is the default. A smoothed ellipse
is None where the smoother's covariance was degenerate at
that step, so no ellipse is drawn rather than a made-up one.
"""

import csv
import json
import math

from .filter import (MEASURED_MEASUREMENT_NOISE_M, KalmanFilter,
                     MovingAverageFilter)
from .tracking import LivePosition
from .uncertainty import (PositionUncertainty, ellipse_from_covariance,
                          ellipse_rms_sigma)

FORMAT = "beacon-live-track"
FORMAT_VERSION = 1

# CSV column order — fixed, so a dashboard can rely on it.
# ``battery`` was added in v0.25.0; ``forecasts`` and
# ``forecast_alerts`` in v0.32.0; ``drift_alarms`` and
# ``drift_alerts`` in v0.35.0; ``drift_classifications`` in
# v0.36.0; ``device`` in v0.43.0. Files written before that
# (seventeen columns, no device) still load — see
# ``load_track_csv`` — with the device defaulting to None, and
# files before v0.36.0 (sixteen columns, no drift
# classifications) still load the same way, with the
# classifications defaulting to empty, and files before
# v0.35.0 (fourteen columns, no drift alarms) still load the
# same way.
CSV_FIELDS = (
    "t",
    "x",
    "y",
    "floor",
    "floor_confidence",
    "ellipse_semi_major_m",
    "ellipse_semi_minor_m",
    "ellipse_angle_deg",
    "beacons_used",
    "beacons_excluded",
    "coverage",
    "battery",
    "forecasts",
    "forecast_alerts",
    "drift_alarms",
    "drift_alerts",
    "drift_classifications",
    "device",
)

# The smoothed export's columns: the raw columns plus the
# smoothed ellipse per step (added in v0.53.0). Only the
# smoothed CSV writers use these — a raw export keeps the
# header it always had, so its bytes do not change — and
# ``load_track_csv`` accepts both headers. A smoothed file
# written before v0.53.0 has the raw header and loads with no
# smoothed ellipse, the same way a raw file does.
SMOOTHED_CSV_FIELDS = CSV_FIELDS + (
    "smoothed_ellipse_semi_major_m",
    "smoothed_ellipse_semi_minor_m",
    "smoothed_ellipse_angle_deg",
)


def track_rows(positions: list[LivePosition | None],
               device: str | None = None) -> list[dict]:
    """One row per located position in a recorded track.

    ``positions`` is the track — one ``LivePosition`` (or None,
    for a step too thin to locate) per step, in order. Each row
    carries the timestamp, the (x, y) in metres, the floor and
    its confidence, the 95% confidence ellipse (None fields when
    the step's geometry was degenerate), the beacons the estimate
    was built from, the beacons excluded as dark with their
    reasons, the battery level reported for each beacon at that
    step (an empty dict when the readings carried no battery
    state), the planner's forecast flags for that step — one per
    beacon, the soonest milestone with its hours remaining (an
    empty list when the track was recorded without a plan) — and
    the plain-language alerts for the beacons due inside the
    forecast window, the drift alarms for that step — the beacons
    the drift monitor was alarming on, each with its rolling mean
    RSSI residual in dB and how many readings are in its window
    (an empty list when the track was recorded without a monitor)
    — and the plain-language alert per alarming beacon, the drift
    classifications for that step — every beacon's verdict from
    that monitor with the numbers it rests on (an empty list when
    the track was recorded without a monitor) — and the track's
    overall coverage — the located share of the walk, the same on
    every row.

    ``device`` names the device the track belongs to — a patient
    tag, a nurse's badge — so rows from several devices can be
    combined into one multi-device track (see
    :func:`multi_track_rows`) and replayed together. It is None
    for a single unnamed device, so a one-device export is
    unaffected.

    Raises ValueError on an empty track: with no steps there is
    no coverage to report. A track with steps but nothing located
    returns no rows and coverage 0.0 — the starvation is visible
    in the header, not hidden by an error.
    """
    if not positions:
        raise ValueError("cannot export an empty track")
    if device is not None and (not isinstance(device, str)
                               or not device):
        raise ValueError("device must be a non-empty string")
    located = sum(1 for position in positions if position is not None)
    coverage = located / len(positions)
    rows = []
    for position in positions:
        if position is None:
            continue
        ellipse = position.uncertainty
        rows.append({
            "t": position.t,
            "x": position.x,
            "y": position.y,
            "floor": position.floor,
            "floor_confidence": position.floor_confidence,
            "ellipse_semi_major_m": (
                None if ellipse is None else ellipse.semi_major),
            "ellipse_semi_minor_m": (
                None if ellipse is None else ellipse.semi_minor),
            "ellipse_angle_deg": (
                None if ellipse is None else ellipse.angle_deg),
            "beacons_used": list(position.heard),
            "beacons_excluded": [b.to_dict() for b in position.excluded],
            "coverage": coverage,
            "battery": {beacon_id: level
                        for beacon_id, level in position.battery},
            "forecasts": [flag.to_dict() for flag in position.forecasts],
            "forecast_alerts": list(position.forecast_alerts),
            "drift_alarms": [alarm.to_dict()
                             for alarm in position.drift_alarms],
            "drift_alerts": list(position.drift_alerts),
            "drift_classifications": [c.to_dict() for c in
                                      position.drift_classifications],
            "device": device,
        })
    return rows


def multi_track_rows(
        tracks: dict[str, list[LivePosition | None]]) -> list[dict]:
    """One combined row list for several devices' tracks.

    ``tracks`` maps each device id to its recorded track — one
    ``LivePosition`` (or None) per step, in order, as
    :func:`track_rows` takes. The rows come back tagged with
    their device, in device order, so a ward dashboard can
    replay a whole shift — every tag at once — from one row
    list (see :func:`beacon.replay.render_track_replay`). Each
    device's rows keep that device's own coverage, so a tag
    starved by dead beacons still shows up as low coverage on
    its own rows rather than being averaged away.
    """
    rows = []
    for device_id, positions in tracks.items():
        rows.extend(track_rows(positions, device=device_id))
    return rows


def multi_track_to_dict(
        tracks: dict[str, list[LivePosition | None]]) -> dict:
    """Serialise several devices' recorded tracks to one combined
    JSON-ready dict.

    ``tracks`` maps each device id to its recorded track, as
    :func:`multi_track_rows` takes. The header aggregates the
    devices — ``steps`` is the total steps across devices
    (including steps too thin to locate), ``located`` the total
    located rows, ``coverage`` the located share overall — while
    each row keeps its own device's coverage, so a starved tag
    still shows up as low coverage on its own rows. The dict
    loads back with :func:`track_from_dict` exactly like a
    single-device export, so the overlay and the replay render
    it the same way they render rows combined by hand.

    Raises ValueError when no device has a track: with no steps
    there is no coverage to report.
    """
    if not tracks:
        raise ValueError("cannot export an empty track")
    rows = multi_track_rows(tracks)
    steps = sum(len(positions) for positions in tracks.values())
    located = len(rows)
    return {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "steps": steps,
        "located": located,
        "coverage": located / steps if steps else 0.0,
        "rows": rows,
    }


def _step_noise_m(row: dict) -> float | None:
    """One row's per-step smoother noise, in metres, from the
    step's own live confidence ellipse — the RMS sigma of the
    ellipse (see :func:`beacon.uncertainty.ellipse_rms_sigma`),
    so a step located by a weak geometry is smoothed on a
    larger noise than a step pinned down from all sides. None
    when the step carries no ellipse (degenerate geometry), so
    the filter falls back to the measured noise for that step.
    """
    major = row.get("ellipse_semi_major_m")
    minor = row.get("ellipse_semi_minor_m")
    angle = row.get("ellipse_angle_deg")
    if major is None or minor is None or angle is None:
        return None
    return ellipse_rms_sigma(
        PositionUncertainty(major, minor, angle))


def _smooth_series(
        positions: list[tuple[float, float]], method: str,
        window: int,
        measurement_noise_m: float | None = None,
        noises_m: list[float | None] | None = None
        ) -> list[tuple[float, float, PositionUncertainty | None]]:
    """One device's (x, y) series, smoothed in order, with the
    smoothed ellipse per step.

    ``"rts"`` feeds the series through a fresh Kalman filter
    and returns the Rauch-Tung-Striebel smoothed positions with
    the 95% ellipse from each step's smoothed posterior
    covariance — each past position revised with the
    observations that came after it, and its ellipse tightened
    by the same revision. A step whose smoothed covariance is
    degenerate carries None instead of a made-up ellipse.
    ``"moving_average"`` runs the moving-average filter over the
    series instead and carries no ellipse at any step, since a
    moving average has no covariance. The caller splits the
    series at floor changes, so a run never spans two planes.

    ``measurement_noise_m`` is the standard deviation, in
    metres, of the position estimates the smoother is fed — the
    filter's measurement-noise variance is its square. When it
    is None (the default) the noise is per step instead of one
    isotropic number for the whole walk: each step is fed with
    the RMS sigma of its own live confidence ellipse, from
    ``noises_m`` (one sigma per position, in order), so the
    filter trusts geometrically strong steps more than weak
    ones — a step with no ellipse in ``noises_m`` falls back to
    the measured ``MEASURED_MEASUREMENT_NOISE_M`` (2 m) for
    that step. A float selects one scalar noise for every step
    instead: 1.0 reproduces the pre-0.54.0 numbers exactly and
    2.0 the v0.54.0/v0.55.0 calibrated ones. The parameter is
    ignored by the moving-average method, which has no noise
    model. Raises ValueError on a non-positive scalar noise
    for the ``"rts"`` method.
    """
    if method == "rts":
        if measurement_noise_m is not None and measurement_noise_m <= 0.0:
            raise ValueError("measurement_noise_m must be positive")
        kalman = KalmanFilter(
            measurement_noise=(measurement_noise_m ** 2
                               if measurement_noise_m is not None
                               else MEASURED_MEASUREMENT_NOISE_M ** 2))
        if measurement_noise_m is None:
            steps = zip(positions,
                        noises_m if noises_m is not None
                        else [None] * len(positions))
            for position, noise_m in steps:
                if noise_m is None:
                    kalman.update(position)
                else:
                    kalman.update(position,
                                  measurement_noise=noise_m ** 2)
        else:
            for position in positions:
                kalman.update(position)
        smoothed = []
        for x, y, cov in kalman.smooth_with_covariance():
            try:
                ellipse = ellipse_from_covariance(cov)
            except ValueError:
                ellipse = None
            smoothed.append((x, y, ellipse))
        return smoothed
    # method == "moving_average": validated by smooth_track_rows.
    average = MovingAverageFilter(window=window)
    return [(x, y, None)
            for x, y in (average.update(position)
                         for position in positions)]


def smooth_track_rows(rows: list[dict], method: str = "rts",
                      window: int = 5,
                      measurement_noise_m: float | None = None
                      ) -> list[dict]:
    """The rows of a recorded track with their positions smoothed.

    ``rows`` is a row list as :func:`track_rows` produces — or as
    :func:`track_from_dict` / :func:`load_track_json` loads back —
    one row per located position, in order. Each row comes back
    with its (x, y) replaced by the smoothed position and
    everything else copied unchanged: the timestamp, the floor,
    the confidence ellipse, the beacons used and excluded, the
    battery, the forecasts, the drift state and the coverage are
    the recorded step's own. Each row also gains the smoothed
    ellipse — ``smoothed_ellipse_semi_major_m``,
    ``smoothed_ellipse_semi_minor_m`` and
    ``smoothed_ellipse_angle_deg`` — the 95% ellipse from the
    smoother's own posterior covariance at that step, so the
    overlay and the replay can draw how much the smoothing
    trusts each revised position; the moving-average method
    carries None there, since it has no covariance, and a step
    whose smoothed covariance was degenerate carries None too.
    The input rows are not modified.

    ``method`` is ``"rts"`` (the default) or ``"moving_average"``:
    the Rauch-Tung-Striebel recursion over a fresh Kalman filter,
    or the moving average over the last ``window`` positions (the
    same opt-out the engine offers). On a track that is already
    live-filtered the moving average adds lag and measures worse,
    so it is the opt-out, not the default. ``measurement_noise_m``
    is the standard deviation, in metres, of the position
    estimates the smoother is fed — the filter's measurement-noise
    variance is its square. When it is None (the default) the
    noise is per step instead of one isotropic number for the
    whole walk: each step is smoothed on the RMS sigma of its
    own live confidence ellipse (see :func:`_step_noise_m`), so
    the filter trusts a step as much as that step's own beacon
    geometry earned — a step with no ellipse falls back to the
    measured ``MEASURED_MEASUREMENT_NOISE_M`` (2 m). Pass a
    float to use one scalar noise for every step instead: 1.0
    reproduces the pre-0.54.0 behaviour exactly, 2.0 the
    v0.54.0/v0.55.0 calibrated one. It is ignored by the
    moving-average method, which has no noise model. Rows from
    several devices are smoothed per device — each device's series
    separately, in that device's own order — so one tag's walk
    never pulls another tag's positions, and the rows come back
    in their original order. A device's series that changes floor
    is split at the change and each run smoothed on its own, the
    way the live filter resets on a floor change.

    Raises ValueError on an empty row list — with no located
    steps there is nothing to smooth — on an unknown method, on
    a moving-average window below 1, or on a non-positive
    ``measurement_noise_m`` for the ``"rts"`` method.
    """
    if not rows:
        raise ValueError("cannot smooth an empty track")
    if method not in ("rts", "moving_average"):
        raise ValueError(
            f"unknown smoothing method: {method!r} (expected 'rts' "
            "or 'moving_average')")
    if method == "moving_average" and window < 1:
        raise ValueError("window must be at least 1")
    smoothed = [dict(row) for row in rows]
    # Row indices per device, in order — None is the unnamed
    # single device, so a one-device track is unaffected.
    by_device: dict[str | None, list[int]] = {}
    for index, row in enumerate(rows):
        by_device.setdefault(row.get("device"), []).append(index)
    for indices in by_device.values():
        # Contiguous runs of one floor, smoothed separately.
        start = 0
        while start < len(indices):
            end = start + 1
            while (end < len(indices)
                   and rows[indices[end]]["floor"]
                   == rows[indices[start]]["floor"]):
                end += 1
            run = indices[start:end]
            positions = [(rows[i]["x"], rows[i]["y"]) for i in run]
            noises = ([_step_noise_m(rows[i]) for i in run]
                      if measurement_noise_m is None else None)
            for i, (x, y, ellipse) in zip(
                    run, _smooth_series(positions, method, window,
                                        measurement_noise_m, noises)):
                smoothed[i]["x"] = x
                smoothed[i]["y"] = y
                smoothed[i]["smoothed_ellipse_semi_major_m"] = (
                    None if ellipse is None else ellipse.semi_major)
                smoothed[i]["smoothed_ellipse_semi_minor_m"] = (
                    None if ellipse is None else ellipse.semi_minor)
                smoothed[i]["smoothed_ellipse_angle_deg"] = (
                    None if ellipse is None else ellipse.angle_deg)
            start = end
    return smoothed


def smoothed_track_to_dict(
        positions: list[LivePosition | None],
        device: str | None = None,
        method: str = "rts", window: int = 5,
        measurement_noise_m: float | None = None
        ) -> dict:
    """Serialise a recorded track to plain JSON-ready data, with
    the positions smoothed (see :func:`smooth_track_rows`).

    The same shape as :func:`track_to_dict` — same format and
    version, same steps, located and coverage, since smoothing
    revises positions without adding or dropping rows — with a
    ``smoothed`` flag and the ``smoothing_method`` in the header,
    so the file says it is smoothed instead of looking like a
    raw export, and with the smoothed ellipse on each row (see
    :func:`smooth_track_rows`). A track with steps but nothing
    located has no rows to smooth, so it exports empty, like the
    raw export. ``measurement_noise_m`` is the smoother's
    measurement-noise standard deviation in metres (see
    :func:`smooth_track_rows`).
    """
    data = track_to_dict(positions, device=device)
    if data["rows"]:
        data["rows"] = smooth_track_rows(data["rows"], method=method,
                                         window=window,
                                         measurement_noise_m=
                                         measurement_noise_m)
    data["smoothed"] = True
    data["smoothing_method"] = method
    return data


def smoothed_multi_track_to_dict(
        tracks: dict[str, list[LivePosition | None]],
        method: str = "rts", window: int = 5,
        measurement_noise_m: float | None = None
        ) -> dict:
    """Serialise several devices' recorded tracks to one combined
    JSON-ready dict, with every device's positions smoothed.

    The same shape as :func:`multi_track_to_dict` — the header
    aggregates the devices the same way — with a ``smoothed``
    flag and the ``smoothing_method`` in the header. Each
    device's rows are smoothed separately (see
    :func:`smooth_track_rows`), so one tag's walk never pulls
    another tag's positions. The dict loads back with
    :func:`track_from_dict` exactly like a raw combined export,
    since the smoothed rows carry the same fields plus the
    smoothed ellipse. ``measurement_noise_m`` is the smoother's
    measurement-noise standard deviation in metres (see
    :func:`smooth_track_rows`).
    """
    if not tracks:
        raise ValueError("cannot export an empty track")
    data = multi_track_to_dict(tracks)
    if data["rows"]:
        data["rows"] = smooth_track_rows(data["rows"], method=method,
                                         window=window,
                                         measurement_noise_m=
                                         measurement_noise_m)
    data["smoothed"] = True
    data["smoothing_method"] = method
    return data


def track_to_dict(positions: list[LivePosition | None],
                  device: str | None = None) -> dict:
    """Serialise a recorded track to plain JSON-ready data."""
    rows = track_rows(positions, device=device)
    return {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "steps": len(positions),
        "located": len(rows),
        "coverage": rows[0]["coverage"] if rows else 0.0,
        "rows": rows,
    }


def _forecast_hours(value) -> float:
    """One forecast horizon back from its JSON form: null means
    the beacon never reaches that milestone."""
    return math.inf if value is None else float(value)


def _forecasts_from_dict(raw) -> list[dict]:
    """The row's forecast flags back from their JSON form.

    Missing (a file written before v0.32.0) means the track was
    recorded without a plan, so the flags default to empty rather
    than failing a file that predates them.
    """
    if raw is None:
        return []
    return [
        {
            "beacon": str(flag["beacon"]),
            "milestone": str(flag["milestone"]),
            "hours": _forecast_hours(flag.get("hours")),
            "status": str(flag["status"]),
            "hours_to_dark": _forecast_hours(
                flag.get("hours_to_dark")),
            "hours_to_rssi_loss_db": _forecast_hours(
                flag.get("hours_to_rssi_loss_db")),
            "hours_to_extra_noise_db": _forecast_hours(
                flag.get("hours_to_extra_noise_db")),
        }
        for flag in raw
    ]


def _drift_alarms_from_dict(raw) -> list[dict]:
    """The row's drift alarms back from their JSON form.

    Missing (a file written before v0.35.0) means the track was
    recorded without a drift monitor, so the alarms default to
    empty rather than failing a file that predates them.
    """
    if raw is None:
        return []
    return [
        {
            "id": str(alarm["id"]),
            "mean_residual_db": float(alarm["mean_residual_db"]),
            "samples": int(alarm["samples"]),
        }
        for alarm in raw
    ]


def _drift_classifications_from_dict(raw) -> list[dict]:
    """The row's drift classifications back from their JSON form.

    Missing (a file written before v0.36.0) means the track was
    recorded without a drift monitor, so the classifications
    default to empty rather than failing a file that predates
    them. ``revisit_delta_db`` is null when the device never
    revisited a spot, so no delta was measured.
    """
    if raw is None:
        return []
    return [
        {
            "id": str(c["id"]),
            "verdict": str(c["verdict"]),
            "mean_residual_db": float(c["mean_residual_db"]),
            "residual_std_db": float(c["residual_std_db"]),
            "time_slope_db_per_step": float(
                c["time_slope_db_per_step"]),
            "time_correlation": float(c["time_correlation"]),
            "revisit_delta_db": (
                None if c.get("revisit_delta_db") is None
                else float(c["revisit_delta_db"])),
            "revisit_pairs": int(c["revisit_pairs"]),
            "samples": int(c["samples"]),
            "reason": str(c["reason"]),
        }
        for c in raw
    ]


def _smoothed_ellipse_from_dict(row: dict) -> dict:
    """The row's smoothed ellipse back from its JSON form.

    A raw export has no smoothed ellipse fields at all, so the
    row loads without them — exactly as :func:`track_rows`
    produced it — while a smoothed export (see
    :func:`smooth_track_rows`) carries the three fields, None
    where the smoother had no ellipse for that step, and loads
    them back. The keys are only present when the file had
    them, so a raw file and a smoothed file stay
    distinguishable after loading.
    """
    if "smoothed_ellipse_semi_major_m" not in row:
        return {}
    return {
        "smoothed_ellipse_semi_major_m": (
            None if row["smoothed_ellipse_semi_major_m"] is None
            else float(row["smoothed_ellipse_semi_major_m"])),
        "smoothed_ellipse_semi_minor_m": (
            None if row["smoothed_ellipse_semi_minor_m"] is None
            else float(row["smoothed_ellipse_semi_minor_m"])),
        "smoothed_ellipse_angle_deg": (
            None if row["smoothed_ellipse_angle_deg"] is None
            else float(row["smoothed_ellipse_angle_deg"])),
    }


def track_from_dict(data: dict) -> list[dict]:
    """Rebuild the rows from ``track_to_dict`` output. Lossless.

    Returns the row dicts, in order. Rejects data that is not a
    beacon live-track file, and versions it does not understand,
    instead of loading garbage.
    """
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("not a beacon live-track file")
    if data.get("version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported live-track format version: "
            f"{data.get('version')!r} (expected {FORMAT_VERSION})")
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise ValueError("live-track file needs a 'rows' list")
    loaded = []
    for row in rows:
        loaded_row = {
            "t": float(row["t"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "floor": int(row["floor"]),
            "floor_confidence": float(row["floor_confidence"]),
            "ellipse_semi_major_m": (
                None if row["ellipse_semi_major_m"] is None
                else float(row["ellipse_semi_major_m"])),
            "ellipse_semi_minor_m": (
                None if row["ellipse_semi_minor_m"] is None
                else float(row["ellipse_semi_minor_m"])),
            "ellipse_angle_deg": (
                None if row["ellipse_angle_deg"] is None
                else float(row["ellipse_angle_deg"])),
            "beacons_used": list(row["beacons_used"]),
            "beacons_excluded": [
                {"id": str(b["id"]), "reason": str(b["reason"])}
                for b in row["beacons_excluded"]
            ],
            "coverage": float(row["coverage"]),
            "battery": {
                str(beacon_id): float(level)
                for beacon_id, level in row.get("battery", {}).items()
            },
            "forecasts": _forecasts_from_dict(row.get("forecasts")),
            "forecast_alerts": [str(alert) for alert in
                                row.get("forecast_alerts", [])],
            "drift_alarms": _drift_alarms_from_dict(
                row.get("drift_alarms")),
            "drift_alerts": [str(alert) for alert in
                             row.get("drift_alerts", [])],
            "drift_classifications":
                _drift_classifications_from_dict(
                    row.get("drift_classifications")),
            "device": (None if row.get("device") is None
                       else str(row["device"])),
        }
        # The smoothed ellipse, only when the file carried it —
        # a raw export loads without the keys, exactly as
        # track_rows produced it.
        loaded_row.update(_smoothed_ellipse_from_dict(row))
        loaded.append(loaded_row)
    return loaded


def export_track_json(positions: list[LivePosition | None],
                      path: str,
                      device: str | None = None) -> None:
    """Write a recorded track to ``path`` as JSON.

    ``device`` names the device the track belongs to, so the
    file's rows carry it (see :func:`track_rows`)."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(track_to_dict(positions, device=device), handle,
                  indent=2)
        handle.write("\n")


def export_multi_track_json(
        tracks: dict[str, list[LivePosition | None]], path: str) -> None:
    """Write several devices' recorded tracks to ``path`` as one
    combined JSON file, with the header from
    :func:`multi_track_to_dict`. The file loads back with
    :func:`load_track_json`, so a combined export can be
    verified by loading it again — and the overlay and the
    replay read it exactly like the per-device exports it was
    combined from, because the rows are the same
    :func:`multi_track_rows` rows they group by device."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(multi_track_to_dict(tracks), handle, indent=2)
        handle.write("\n")


def export_smoothed_track_json(
        positions: list[LivePosition | None], path: str,
        device: str | None = None,
        method: str = "rts", window: int = 5,
        measurement_noise_m: float | None = None
        ) -> None:
    """Write a recorded track to ``path`` as JSON, with the
    positions smoothed (see :func:`smoothed_track_to_dict`) —
    the smoothed export alongside the raw one, so a dashboard
    can replay the steadier path instead of the jittery live
    one. ``device`` names the device the track belongs to, so
    the file's rows carry it. The file loads back with
    :func:`load_track_json`, since the smoothed rows carry the
    raw fields plus the smoothed ellipse. ``measurement_noise_m``
    is the smoother's measurement-noise standard deviation in
    metres (see :func:`smooth_track_rows`)."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(smoothed_track_to_dict(positions, device=device,
                                         method=method, window=window,
                                         measurement_noise_m=
                                         measurement_noise_m),
                  handle, indent=2)
        handle.write("\n")


def export_smoothed_multi_track_json(
        tracks: dict[str, list[LivePosition | None]], path: str,
        method: str = "rts", window: int = 5,
        measurement_noise_m: float | None = None
        ) -> None:
    """Write several devices' recorded tracks to ``path`` as one
    combined JSON file, with every device's positions smoothed
    (see :func:`smoothed_multi_track_to_dict`) — the smoothed
    export alongside the raw combined one, so a dashboard can
    replay the whole shift's steadier paths from one file. The
    file loads back with :func:`load_track_json` to the same
    rows the raw combined export carries, with the (x, y)
    revised per device and the smoothed ellipse on each row.
    ``measurement_noise_m`` is the smoother's measurement-noise
    standard deviation in metres (see :func:`smooth_track_rows`)."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(smoothed_multi_track_to_dict(tracks, method=method,
                                               window=window,
                                               measurement_noise_m=
                                               measurement_noise_m),
                  handle, indent=2)
        handle.write("\n")


def export_smoothed_track_csv(
        positions: list[LivePosition | None], path: str,
        device: str | None = None,
        method: str = "rts", window: int = 5,
        measurement_noise_m: float | None = None
        ) -> None:
    """Write a recorded track to ``path`` as CSV, with the
    positions smoothed — the same rows as
    :func:`export_smoothed_track_json`, in the smoothed CSV
    column order (see ``SMOOTHED_CSV_FIELDS``: the raw columns
    plus the smoothed ellipse per step), so a dashboard can
    take either format. ``device`` names the device the track
    belongs to. The file loads back with
    :func:`load_track_csv`. ``measurement_noise_m`` is the
    smoother's measurement-noise standard deviation in metres
    (see :func:`smooth_track_rows`)."""
    data = smoothed_track_to_dict(positions, device=device,
                                  method=method, window=window,
                                  measurement_noise_m=measurement_noise_m)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle,
                                fieldnames=list(SMOOTHED_CSV_FIELDS))
        writer.writeheader()
        for row in data["rows"]:
            writer.writerow(_row_to_csv(row, smoothed=True))


def export_smoothed_multi_track_csv(
        tracks: dict[str, list[LivePosition | None]], path: str,
        method: str = "rts", window: int = 5,
        measurement_noise_m: float | None = None
        ) -> None:
    """Write several devices' recorded tracks to ``path`` as one
    combined CSV file, with every device's positions smoothed —
    the same rows as
    :func:`export_smoothed_multi_track_json`, in the smoothed
    CSV column order, so a dashboard can take either format and
    replay the whole shift's steadier paths from it. The file
    loads back with :func:`load_track_csv`.
    ``measurement_noise_m`` is the smoother's measurement-noise
    standard deviation in metres (see :func:`smooth_track_rows`)."""
    data = smoothed_multi_track_to_dict(tracks, method=method,
                                        window=window,
                                        measurement_noise_m=
                                        measurement_noise_m)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle,
                                fieldnames=list(SMOOTHED_CSV_FIELDS))
        writer.writeheader()
        for row in data["rows"]:
            writer.writerow(_row_to_csv(row, smoothed=True))


def load_track_json(path: str) -> list[dict]:
    """Read back the rows from a file written by
    ``export_track_json``."""
    with open(path, encoding="utf-8") as handle:
        return track_from_dict(json.load(handle))


def _row_to_csv(row: dict, smoothed: bool = False) -> dict:
    """One row dict as flat CSV strings, for writing.

    The smoothed ellipse fields are read with ``.get`` and
    only written when ``smoothed`` is true: a raw row has no
    smoothed ellipse, so the raw writers keep the header they
    always had, and the smoothed writers use the wider header
    (see ``SMOOTHED_CSV_FIELDS``).
    """
    fields = {
        "t": repr(row["t"]),
        "x": repr(row["x"]),
        "y": repr(row["y"]),
        "floor": str(row["floor"]),
        "floor_confidence": repr(row["floor_confidence"]),
        "ellipse_semi_major_m": (
            "" if row["ellipse_semi_major_m"] is None
            else repr(row["ellipse_semi_major_m"])),
        "ellipse_semi_minor_m": (
            "" if row["ellipse_semi_minor_m"] is None
            else repr(row["ellipse_semi_minor_m"])),
        "ellipse_angle_deg": (
            "" if row["ellipse_angle_deg"] is None
            else repr(row["ellipse_angle_deg"])),
        "beacons_used": ";".join(row["beacons_used"]),
        "beacons_excluded": json.dumps(row["beacons_excluded"]),
        "coverage": repr(row["coverage"]),
        "battery": ";".join(
            f"{beacon_id}={level!r}"
            for beacon_id, level in sorted(row["battery"].items())),
        "forecasts": json.dumps(row["forecasts"]),
        "forecast_alerts": ";".join(row["forecast_alerts"]),
        "drift_alarms": json.dumps(row["drift_alarms"]),
        "drift_alerts": ";".join(row["drift_alerts"]),
        "drift_classifications": json.dumps(
            row["drift_classifications"]),
        "device": ("" if row.get("device") is None
                   else str(row["device"])),
    }
    if smoothed:
        # The smoothed ellipse, written only for the smoothed
        # writers' wider header — a raw row has no smoothed
        # ellipse, so a raw file never carries these columns.
        fields["smoothed_ellipse_semi_major_m"] = (
            "" if row.get("smoothed_ellipse_semi_major_m") is None
            else repr(row["smoothed_ellipse_semi_major_m"]))
        fields["smoothed_ellipse_semi_minor_m"] = (
            "" if row.get("smoothed_ellipse_semi_minor_m") is None
            else repr(row["smoothed_ellipse_semi_minor_m"]))
        fields["smoothed_ellipse_angle_deg"] = (
            "" if row.get("smoothed_ellipse_angle_deg") is None
            else repr(row["smoothed_ellipse_angle_deg"]))
    return fields


def _forecasts_from_csv(raw_forecasts) -> list[dict]:
    """Parse the CSV forecasts column (a JSON list of flag dicts)
    back to row form. Empty when the column is empty — or when
    the file predates the forecasts columns entirely, so old
    exports still load."""
    if not raw_forecasts:
        return []
    return _forecasts_from_dict(json.loads(raw_forecasts))


def _alerts_from_csv(raw_alerts) -> list[str]:
    """Parse the CSV forecast-alerts column (semicolon-joined
    plain-language warnings) back to a list. Empty when the
    column is empty — or when the file predates the forecasts
    columns entirely."""
    if not raw_alerts:
        return []
    return raw_alerts.split(";")


def _drift_alarms_from_csv(raw_alarms) -> list[dict]:
    """Parse the CSV drift-alarms column (a JSON list of alarm
    dicts) back to row form. Empty when the column is empty — or
    when the file predates the drift columns entirely, so old
    exports still load."""
    if not raw_alarms:
        return []
    return _drift_alarms_from_dict(json.loads(raw_alarms))


def _drift_alerts_from_csv(raw_alerts) -> list[str]:
    """Parse the CSV drift-alerts column (semicolon-joined
    plain-language warnings) back to a list. Empty when the
    column is empty — or when the file predates the drift columns
    entirely."""
    if not raw_alerts:
        return []
    return raw_alerts.split(";")


def _drift_classifications_from_csv(raw) -> list[dict]:
    """Parse the CSV drift-classifications column (a JSON list of
    classification dicts) back to row form. Empty when the column
    is empty — or when the file predates the column entirely, so
    old exports still load."""
    if not raw:
        return []
    return _drift_classifications_from_dict(json.loads(raw))


def _row_from_csv(raw: dict, has_smoothed_ellipse: bool = False
                  ) -> dict:
    """One CSV record back as a row dict, for reading.

    ``has_smoothed_ellipse`` is true only when the file's
    header carried the smoothed ellipse columns (see
    ``SMOOTHED_CSV_FIELDS``): then the row loads with them,
    None where the cell was empty, and a raw file — or a
    smoothed file written before v0.53.0 — loads without the
    keys, exactly as :func:`track_rows` produced it.
    """
    row = {
        "t": float(raw["t"]),
        "x": float(raw["x"]),
        "y": float(raw["y"]),
        "floor": int(raw["floor"]),
        "floor_confidence": float(raw["floor_confidence"]),
        "ellipse_semi_major_m": (
            None if raw["ellipse_semi_major_m"] == ""
            else float(raw["ellipse_semi_major_m"])),
        "ellipse_semi_minor_m": (
            None if raw["ellipse_semi_minor_m"] == ""
            else float(raw["ellipse_semi_minor_m"])),
        "ellipse_angle_deg": (
            None if raw["ellipse_angle_deg"] == ""
            else float(raw["ellipse_angle_deg"])),
        "beacons_used": (
            [] if raw["beacons_used"] == ""
            else raw["beacons_used"].split(";")),
        "beacons_excluded": [
            {"id": str(b["id"]), "reason": str(b["reason"])}
            for b in json.loads(raw["beacons_excluded"])
        ],
        "coverage": float(raw["coverage"]),
        "battery": _battery_from_csv(raw.get("battery", "")),
        "forecasts": _forecasts_from_csv(raw.get("forecasts")),
        "forecast_alerts": _alerts_from_csv(
            raw.get("forecast_alerts")),
        "drift_alarms": _drift_alarms_from_csv(
            raw.get("drift_alarms")),
        "drift_alerts": _drift_alerts_from_csv(
            raw.get("drift_alerts")),
        "drift_classifications": _drift_classifications_from_csv(
            raw.get("drift_classifications")),
        "device": raw.get("device") or None,
    }
    if has_smoothed_ellipse:
        row["smoothed_ellipse_semi_major_m"] = (
            None if raw.get("smoothed_ellipse_semi_major_m")
            in (None, "")
            else float(raw["smoothed_ellipse_semi_major_m"]))
        row["smoothed_ellipse_semi_minor_m"] = (
            None if raw.get("smoothed_ellipse_semi_minor_m")
            in (None, "")
            else float(raw["smoothed_ellipse_semi_minor_m"]))
        row["smoothed_ellipse_angle_deg"] = (
            None if raw.get("smoothed_ellipse_angle_deg")
            in (None, "")
            else float(raw["smoothed_ellipse_angle_deg"]))
    return row


def _battery_from_csv(raw_battery: str) -> dict:
    """Parse the CSV battery column (``"A=0.87;B=0.91"``) back to
    a dict. Empty when the column is empty — or when the file
    predates the battery column entirely, so old exports still
    load."""
    if not raw_battery:
        return {}
    levels = {}
    for pair in raw_battery.split(";"):
        beacon_id, _, level = pair.partition("=")
        levels[beacon_id] = float(level)
    return levels


def export_track_csv(positions: list[LivePosition | None],
                     path: str,
                     device: str | None = None) -> None:
    """Write a recorded track to ``path`` as CSV, one row per
    located position, with the fixed column order in
    ``CSV_FIELDS``. ``device`` names the device the track
    belongs to, so the file's rows carry it (see
    :func:`track_rows`)."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in track_rows(positions, device=device):
            writer.writerow(_row_to_csv(row))


def export_multi_track_csv(
        tracks: dict[str, list[LivePosition | None]], path: str) -> None:
    """Write several devices' recorded tracks to ``path`` as one
    combined CSV file, one row per located position, with the
    fixed column order in ``CSV_FIELDS`` and the device in the
    last column. The file loads back with
    :func:`load_track_csv` to the same rows the JSON export
    carries, so a dashboard can take either format and replay
    the whole shift from it."""
    if not tracks:
        raise ValueError("cannot export an empty track")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in multi_track_rows(tracks):
            writer.writerow(_row_to_csv(row))


def load_track_csv(path: str) -> list[dict]:
    """Read back the rows from a file written by
    ``export_track_csv``.

    Files written before v0.43.0 have no device column — their
    seventeen-column header is accepted too, and those rows load
    with the device defaulting to None, so old exports keep
    working; files written before v0.36.0 have no
    drift-classifications column — their sixteen-column header
    is accepted too, and those rows load with empty drift
    classifications, so old exports keep working; files written
    before v0.35.0 have no drift-alarms columns either, and
    files written before v0.32.0 have no forecasts columns
    either, and files written before v0.25.0 have no battery
    column either, and load with empty drift alarms, forecasts
    and battery the same way. A smoothed CSV (see
    ``SMOOTHED_CSV_FIELDS``) carries the three smoothed ellipse
    columns after the device column, and loads with them; a
    smoothed CSV written before v0.53.0 has the raw header and
    loads with no smoothed ellipse, the same way a raw file
    does.
    """
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == list(SMOOTHED_CSV_FIELDS):
            # A smoothed export: the rows carry the smoothed
            # ellipse, so it loads back with them.
            return [_row_from_csv(raw, has_smoothed_ellipse=True)
                    for raw in reader]
        if reader.fieldnames == list(CSV_FIELDS):
            pass
        elif reader.fieldnames == list(CSV_FIELDS[:-1]):
            # Pre-device export: no device column. DictReader
            # fills the missing field with None, which
            # ``_row_from_csv`` treats as an unnamed device.
            pass
        elif reader.fieldnames == list(CSV_FIELDS[:-2]):
            # Pre-classifications export: no drift_classifications
            # column. DictReader fills the missing field with
            # None, which ``_row_from_csv`` treats as no drift
            # classifications.
            pass
        elif reader.fieldnames == list(CSV_FIELDS[:-4]):
            # Pre-drift export: no drift_alarms or drift_alerts
            # column. DictReader fills the missing fields with
            # None, which ``_row_from_csv`` treats as no drift
            # state.
            pass
        elif reader.fieldnames == list(CSV_FIELDS[:-6]):
            # Pre-forecast export: no forecasts or forecast_alerts
            # column either.
            pass
        elif reader.fieldnames == list(CSV_FIELDS[:-7]):
            # Pre-battery export: no battery column either.
            pass
        else:
            raise ValueError(
                "not a beacon live-track CSV: unexpected columns "
                f"{reader.fieldnames!r}")
        return [_row_from_csv(raw) for raw in reader]
