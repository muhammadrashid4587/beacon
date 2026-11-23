"""Live track export: save a recorded track for a ward dashboard.

A live track is only useful while the dashboard is watching it.
A ward that wants the track afterwards — to replay a shift, to
audit where a device went, to feed a dashboard that was not
running at the time — needs the track as a file. This module
exports a recorded track (the ``LivePosition`` values a
``LiveTracker`` emitted, one per step, ``None`` where a step was
too thin to locate) as JSON or CSV, one row per located position:

    timestamp, x, y, floor, confidence ellipse (semi-major,
    semi-minor, angle), beacons used, beacons excluded, coverage

The coverage on every row is the track's overall located share
(located steps / all steps), so a dashboard can read it off any
row; the file header carries the same figure plus the step counts,
so a track starved by dead beacons shows up as low coverage
rather than silently short. Steps that produced no position are
not rows — there is no position to write — but they are counted
in the coverage, so the file never pretends a starved track was
complete.

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

from .tracking import LivePosition

FORMAT = "beacon-live-track"
FORMAT_VERSION = 1

# CSV column order — fixed, so a dashboard can rely on it.
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
)


def track_rows(positions: list[LivePosition | None]) -> list[dict]:
    """One row per located position in a recorded track.

    ``positions`` is the track — one ``LivePosition`` (or None,
    for a step too thin to locate) per step, in order. Each row
    carries the timestamp, the (x, y) in metres, the floor and
    its confidence, the 95% confidence ellipse (None fields when
    the step's geometry was degenerate), the beacons the estimate
    was built from, the beacons excluded as dark with their
    reasons, and the track's overall coverage — the located share
    of the walk, the same on every row.

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
    }


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
    }


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
    ``export_track_csv``."""
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError(
                "not a beacon live-track CSV: unexpected columns "
                f"{reader.fieldnames!r}")
        return [_row_from_csv(raw) for raw in reader]
