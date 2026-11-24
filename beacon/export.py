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
    battery, planner forecast flags and alerts

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
unaffected.

Both formats round-trip: ``load_track_json`` / ``load_track_csv``
read back exactly the rows ``track_rows`` produced, so an export
can be verified by loading it again. JSON floats round-trip
exactly in Python; CSV numbers are written with ``repr`` and
parsed back as floats, so they do too. The beacons used are
semicolon-joined in CSV (``"A;B;C"``) and the excluded beacons
are a JSON string in the CSV field, since an exclusion carries a
free-text reason.

Pure standard library, no dependencies.
"""

import csv
import json
import math

from .tracking import LivePosition

FORMAT = "beacon-live-track"
FORMAT_VERSION = 1

# CSV column order — fixed, so a dashboard can rely on it.
# ``battery`` was added in v0.25.0; ``forecasts`` and
# ``forecast_alerts`` in v0.32.0. Files written before that
# (twelve columns, no forecasts) still load — see
# ``load_track_csv`` — with the forecasts defaulting to empty,
# and files before v0.25.0 (eleven columns, no battery either)
# still load the same way.
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
)


def track_rows(positions: list[LivePosition | None]) -> list[dict]:
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
    forecast window, and the track's overall coverage — the
    located share of the walk, the same on every row.

    Raises ValueError on an empty track: with no steps there is
    no coverage to report. A track with steps but nothing located
    returns no rows and coverage 0.0 — the starvation is visible
    in the header, not hidden by an error.
    """
    if not positions:
        raise ValueError("cannot export an empty track")
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
        })
    return rows


def track_to_dict(positions: list[LivePosition | None]) -> dict:
    """Serialise a recorded track to plain JSON-ready data."""
    rows = track_rows(positions)
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
    return [
        {
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
        }
        for row in rows
    ]


def export_track_json(positions: list[LivePosition | None],
                      path: str) -> None:
    """Write a recorded track to ``path`` as JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(track_to_dict(positions), handle, indent=2)
        handle.write("\n")


def load_track_json(path: str) -> list[dict]:
    """Read back the rows from a file written by
    ``export_track_json``."""
    with open(path, encoding="utf-8") as handle:
        return track_from_dict(json.load(handle))


def _row_to_csv(row: dict) -> dict:
    """One row dict as flat CSV strings, for writing."""
    return {
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
    }


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


def _row_from_csv(raw: dict) -> dict:
    """One CSV record back as a row dict, for reading."""
    return {
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
    }


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
                     path: str) -> None:
    """Write a recorded track to ``path`` as CSV, one row per
    located position, with the fixed column order in
    ``CSV_FIELDS``."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in track_rows(positions):
            writer.writerow(_row_to_csv(row))


def load_track_csv(path: str) -> list[dict]:
    """Read back the rows from a file written by
    ``export_track_csv``.

    Files written before v0.32.0 have no forecasts columns —
    their twelve-column header is accepted too, and those rows
    load with empty forecasts and alerts, so old exports keep
    working; files written before v0.25.0 have no battery column
    either, and load with an empty battery dict as well.
    """
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == list(CSV_FIELDS):
            pass
        elif reader.fieldnames == list(CSV_FIELDS[:-2]):
            # Pre-forecast export: no forecasts or forecast_alerts
            # column. DictReader fills the missing fields with
            # None, which ``_row_from_csv`` treats as no forecast
            # state.
            pass
        elif reader.fieldnames == list(CSV_FIELDS[:-3]):
            # Pre-battery export: no battery column either.
            pass
        else:
            raise ValueError(
                "not a beacon live-track CSV: unexpected columns "
                f"{reader.fieldnames!r}")
        return [_row_from_csv(raw) for raw in reader]
