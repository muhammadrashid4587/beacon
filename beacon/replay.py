"""Live-track replay: an exported track as an animated SVG.

A static ward-map overlay (see ``beacon.map_overlay``) shows where a
device went; a replay shows *when* it was where. This module renders
an exported live track as a self-contained SVG in which a dot moves
along the track, the path draws itself behind it, the 95%
confidence ellipse for the current step appears with the dot, and a
timestamp label steps through the track's own timestamps — so a ward
dashboard can watch a shift play back instead of reading a table.
With ``drift`` each replay frame marks the beacons alarming at
that frame's row, with the same red ring and "!" badge the
static overlay carries, so a beacon reads as bad while it is
alarming and clears when it recovers — including a beacon that
alarmed mid-track and cleared before the walk ended, which the
end-of-track marking never showed — and a timeline strip under
the replay shows, per beacon, when each alarm started and
ended, each segment coloured by the drift classification
verdict on those frames, so the history — and what the monitor
thought was wrong — reads at a glance without stepping
through the frames.

The animation is SMIL, not JavaScript: the dot's ``cx`` / ``cy``
animate through the projected positions on the track's key times,
the trail is a polyline whose ``stroke-dashoffset`` animates from
its full length to zero, and each frame's ellipse and timestamp
label animate their ``visibility`` (discrete, so a frame is either
on or off) over the same key times. The whole replay loops, so the
file can sit on a dashboard unattended; ``loop=False`` plays it
once. The frame times follow the track's timestamps, so a pause
between readings shows up as a pause in the replay, and the total
length is one ``seconds_per_step`` per step.

Rows on other floors are not plotted on top of each other: the
replay animates one floor, the most common floor in the track by
default, or the floor named explicitly.

Pure standard library, no dependencies.
"""

from xml.sax.saxutils import escape

from .beacon import Beacon
from .map_overlay import (
    BEACON_COLOUR,
    CONFIDENCE_BANDS,
    DRIFT_ALARM_COLOUR,
    HEATMAP_FILL_OPACITY,
    MARGIN,
    NO_ELLIPSE_COLOUR,
    NO_ELLIPSE_LABEL,
    SVG_HEIGHT,
    SVG_WIDTH,
    TRACK_COLOUR,
    _drift_label,
    _drift_state_at_row,
    _most_common_floor,
    _project,
    confidence_colour,
    coverage_heatmap,
    heatmap_summary,
    load_track_for_overlay,
)

DOT_COLOUR = "#d81b60"

# One colour per drift classification verdict, for the timeline
# strip's segments: a segment's colour is the verdict the monitor
# gave on those frames, so the strip reads as a diagnosis at a
# glance, not just a red alarm. An alarm with no classification
# recorded falls back to the alarm red, the same colour the
# badges use, so an unclassified segment still reads as an alarm.
DRIFT_VERDICT_COLOURS = {
    "moved": "#e67e22",
    "battery-sag": "#2980b9",
    "noisy": "#8e44ad",
    "uncertain": "#7f8c8d",
}


def drift_verdict_colour(verdict: str | None) -> str:
    """The timeline-strip colour for one drift verdict.

    ``None`` (no classification recorded for the alarm), a
    ``healthy`` verdict on an alarming beacon, or an unknown
    verdict falls back to the alarm red — an alarm is an alarm
    even when the monitor gave no diagnosis for it.
    """
    if verdict is None:
        return DRIFT_ALARM_COLOUR
    return DRIFT_VERDICT_COLOURS.get(verdict, DRIFT_ALARM_COLOUR)


def _frame_timing(rows: list[dict], seconds_per_step: float
                  ) -> tuple[float, list[float]]:
    """Replay length and each frame's key time (0 to 1).

    The duration is one ``seconds_per_step`` per step, and the key
    times follow the track's timestamps: a frame's key time is its
    timestamp's share of the track's time span, so a pause between
    readings shows up as a pause in the replay. A track whose
    timestamps do not advance (or a single point) steps evenly.
    """
    n = len(rows)
    duration = max(n - 1, 1) * seconds_per_step
    span = rows[-1]["t"] - rows[0]["t"]
    if n == 1 or span <= 0:
        key_times = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
    else:
        key_times = [(row["t"] - rows[0]["t"]) / span for row in rows]
    return duration, key_times


def _frame_boundaries(key_times: list[float]) -> list[float]:
    """Each frame's visibility interval over the replay (0 to 1).

    A frame is a point in time, so its interval runs from the
    midpoint with the previous frame to the midpoint with the
    next — the first starts at 0, the last ends at 1 — so every
    frame is on screen for a real stretch and the switch happens
    halfway between two positions, while the dot is gliding
    between them.
    """
    n = len(key_times)
    boundaries = [0.0]
    for i in range(1, n):
        boundaries.append((key_times[i - 1] + key_times[i]) / 2)
    boundaries.append(1.0)
    return boundaries


def _visibility_animation(start: float, end: float, duration: float,
                          repeat: str) -> str:
    """A SMIL animation toggling ``visibility`` on for one frame.

    The element is hidden, becomes visible at its frame's key time
    and hides again at the next frame's, over the replay's total
    duration — with ``calcMode="discrete"`` so the switch is a
    clean step, and ``repeatCount`` matching the replay so the
    frame comes back on every loop.
    """
    if start <= 0.0 and end >= 1.0:
        return ""
    if start <= 0.0:
        values = "visible;hidden"
        key_times = f"0;{end:.4f}"
    elif end >= 1.0:
        values = "hidden;visible"
        key_times = f"0;{start:.4f}"
    else:
        values = "hidden;visible;hidden"
        key_times = f"0;{start:.4f};{end:.4f}"
    return (f'<animate attributeName="visibility" values="{values}" '
            f'keyTimes="{key_times}" calcMode="discrete" '
            f'dur="{duration:g}s" repeatCount="{repeat}"/>')


def _trail_offsets(points: list[tuple[float, float]]) -> tuple[float, list[float]]:
    """Total polyline length and the dash offset at each point.

    The trail draws itself by animating ``stroke-dashoffset``
    from the full length (nothing drawn) to zero (the whole path
    drawn); the offset at a frame is the length still undrawn
    past that frame's point.
    """
    lengths = [0.0]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        lengths.append(lengths[-1] + ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    total = lengths[-1]
    return total, [total - length for length in lengths]


def _alarm_history(frame_drift: list[dict[str, tuple[float, str | None]]]
                   ) -> dict[str, tuple[int, int, float, str | None]]:
    """Each beacon's alarm history over the replay's frames.

    Returns beacon id -> (first alarming frame, last alarming
    frame, rolling mean residual in dB at that last frame, verdict
    at that last frame). A beacon that never alarms is absent, so
    an empty dict means no alarm was recorded on any frame.
    """
    history: dict[str, tuple[int, int, float, str | None]] = {}
    for i, state in enumerate(frame_drift):
        for beacon_id, (mean_db, verdict) in state.items():
            first = (history[beacon_id][0]
                     if beacon_id in history else i)
            history[beacon_id] = (first, i, mean_db, verdict)
    return history


def _alarm_ranges(frame_drift: list[dict[str, tuple[float, str | None]]]
                    ) -> dict[str, list[tuple[int, int, str | None]]]:
    """Each beacon's alarming frames as contiguous runs.

    Returns beacon id -> [(first frame, last frame, verdict),
    ...] in frame order, so a beacon that alarmed, cleared and
    alarmed again gets one run per alarm, and a beacon that never
    alarms is absent. A run also splits when the verdict changes
    mid-alarm, so every run carries a single verdict — the same
    per-frame verdict the replay's badges carry on those frames,
    so the timeline strip can never disagree with them. Built
    from the same per-frame alarm data the badges use; the
    verdict is ``None`` when no classification was recorded for
    the alarm.
    """
    runs: dict[str, list[tuple[int, int, str | None]]] = {}
    for i, state in enumerate(frame_drift):
        for beacon_id, (_mean_db, verdict) in state.items():
            beacon_runs = runs.setdefault(beacon_id, [])
            if (beacon_runs and beacon_runs[-1][1] == i - 1
                    and beacon_runs[-1][2] == verdict):
                beacon_runs[-1] = (beacon_runs[-1][0], i, verdict)
            else:
                beacon_runs.append((i, i, verdict))
    return runs


def _merge_frame_ranges(
        runs: list[tuple[int, int, str | None]]) -> list[tuple[int, int]]:
    """Verdict-split runs back into the alarm's frame spans.

    A run splits when the verdict changes, but the alarm itself
    never stopped — so the row label merges adjacent runs back
    into one span ("frames 4–39"), while the segments stay split
    by verdict. A real gap (the beacon cleared and alarmed
    again) stays two spans.
    """
    merged: list[tuple[int, int]] = []
    for first, last, _verdict in runs:
        if merged and merged[-1][1] == first - 1:
            merged[-1] = (merged[-1][0], last)
        else:
            merged.append((first, last))
    return merged


def _format_frame_ranges(ranges: list[tuple[int, int]]) -> str:
    """Contiguous frame runs as a short label, e.g. "frames 4–39"
    for one run, "frame 4" for a single frame, "frames 0, 2" for
    two disjoint single frames."""
    parts = [f"{first}–{last}" if first != last else f"{first}"
             for first, last in ranges]
    if len(parts) == 1:
        singular = ranges[0][0] == ranges[0][1]
        return f"frame {parts[0]}" if singular else f"frames {parts[0]}"
    return "frames " + ", ".join(parts)


def render_track_replay(rows: list[dict], beacons: list[Beacon],
                        width_m: float = 20.0, height_m: float = 12.0,
                        floor: int | None = None,
                        title: str | None = None,
                        seconds_per_step: float = 1.0,
                        loop: bool = True,
                        heatmap: bool = False,
                        heatmap_spacing: float = 2.0,
                        sigma_rssi_db: float = 2.0,
                        drift: bool = False) -> str:
    """Render an exported live track as an animated SVG string.

    ``rows`` are the row dicts ``load_track_json`` /
    ``load_track_csv`` return. ``beacons`` are the ward's beacons
    at their known positions. ``width_m`` / ``height_m`` are the
    ward's size in metres. ``seconds_per_step`` is the replay
    time per step (1 s by default); ``loop`` repeats the replay
    indefinitely, otherwise it plays once. With ``heatmap`` the
    replay also shades the ward grid by the positioning quality
    the beacon layout gives at each cell (see
    :func:`beacon.map_overlay.coverage_heatmap`). With ``drift``
    a beacon alarming on the track's last plotted row is marked
    with a red ring and a red "!" badge, plus a label with its
    rolling mean residual in dB and the drift classification
    verdict when the monitor gave one — the same marking the
    static overlay carries (see
    :func:`beacon.map_overlay.render_map_overlay`), but animated
    with the replay: the marking for a frame is visible only
    while that frame is current, so a beacon that alarmed
    mid-track and cleared is badged while it was alarming and
    unmarked once it cleared, and a beacon that only starts
    alarming late appears part-way through the replay. The
    legend carries the alarm history in one line — which
    beacons alarmed, on which frames, and the residual and
    verdict at their last alarming frame — so the replay says
    what the monitor recorded over the whole walk, not just at
    its end. Under the replay a timeline strip shows the same
    history as coloured segments — one row per beacon that
    alarmed, one segment per contiguous run of alarming frames,
    coloured by the drift classification verdict on those frames
    (moved, battery-sag, noisy, uncertain — red when no
    classification was recorded), labelled with the beacon and
    its frame ranges — so a viewer can see at a glance when each
    alarm started and ended, and what the monitor thought was
    wrong, without stepping through the frames; with no alarms
    the strip is an empty labelled track. A track recorded
    without a monitor marks nothing.

    Raises ValueError on an empty track, with no beacons, a
    non-positive ``seconds_per_step``, or when the chosen floor
    has no located points.
    """
    if not rows:
        raise ValueError("cannot render a track replay for an empty track")
    if not beacons:
        raise ValueError("cannot render a track replay with no beacons")
    if seconds_per_step <= 0:
        raise ValueError("seconds_per_step must be positive")
    if floor is None:
        floor = _most_common_floor(rows)
    floor_rows = [row for row in rows if row["floor"] == floor]
    if not floor_rows:
        raise ValueError(f"no located points on floor {floor}")
    frame_drift = ([_drift_state_at_row(row) for row in floor_rows]
                   if drift else [{} for _ in floor_rows])

    scale = min((SVG_WIDTH - 2 * MARGIN) / width_m,
                (SVG_HEIGHT - 2 * MARGIN) / height_m)
    ward_w = width_m * scale
    ward_h = height_m * scale
    coverage = floor_rows[0]["coverage"]
    duration, key_times = _frame_timing(floor_rows, seconds_per_step)
    boundaries = _frame_boundaries(key_times)
    repeat = "indefinite" if loop else "1"
    heading = title or (f"Live track replay \u2014 floor {floor} "
                        f"({len(floor_rows)} points, "
                        f"coverage {coverage:.0%})")

    parts = [
        f"<title>{escape(heading)}</title>",
        f'<text x="{MARGIN}" y="28" font-family="sans-serif" '
        f'font-size="16" font-weight="bold">{escape(heading)}</text>',
        # Ward outline.
        f'<rect x="{MARGIN}" y="{MARGIN}" width="{ward_w:.1f}" '
        f'height="{ward_h:.1f}" fill="#fafafa" stroke="#212121" '
        f'stroke-width="2"/>',
    ]
    # 5 m gridlines, so the eye can read distances off the plan.
    grid_x = 5.0
    while grid_x < width_m:
        gx, _ = _project(grid_x, 0.0, scale, width_m, height_m)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{MARGIN}" x2="{gx:.1f}" '
            f'y2="{MARGIN + ward_h:.1f}" stroke="#e0e0e0" '
            f'stroke-width="1"/>')
        parts.append(
            f'<text x="{gx:.1f}" y="{MARGIN + ward_h + 16:.1f}" '
            f'font-family="sans-serif" font-size="10" fill="#757575" '
            f'text-anchor="middle">{grid_x:g} m</text>')
        grid_x += 5.0
    grid_y = 5.0
    while grid_y < height_m:
        _, gy = _project(0.0, grid_y, scale, width_m, height_m)
        parts.append(
            f'<line x1="{MARGIN}" y1="{gy:.1f}" '
            f'x2="{MARGIN + ward_w:.1f}" y2="{gy:.1f}" '
            f'stroke="#e0e0e0" stroke-width="1"/>')
        parts.append(
            f'<text x="{MARGIN - 6:.1f}" y="{gy + 3:.1f}" '
            f'font-family="sans-serif" font-size="10" fill="#757575" '
            f'text-anchor="end">{grid_y:g} m</text>')
        grid_y += 5.0

    # Coverage heatmap, under everything else, as on the static
    # overlay. Degenerate cells are left unshaded rather than
    # coloured as if they were covered.
    summary = None
    if heatmap:
        cells = coverage_heatmap(beacons, width_m, height_m,
                                 heatmap_spacing, sigma_rssi_db)
        summary = heatmap_summary(cells)
        cell_px = heatmap_spacing * scale
        for cell in cells:
            if cell.semi_major is None:
                continue
            cx, cy = _project(cell.x, cell.y, scale, width_m, height_m)
            parts.append(
                f'<rect x="{cx - cell_px / 2:.1f}" '
                f'y="{cy - cell_px / 2:.1f}" width="{cell_px:.1f}" '
                f'height="{cell_px:.1f}" '
                f'fill="{confidence_colour(cell.semi_major)}" '
                f'fill-opacity="{HEATMAP_FILL_OPACITY}" '
                f'data-heatmap-cell="{cell.x:g},{cell.y:g}"/>')

    points = [_project(row["x"], row["y"], scale, width_m, height_m)
              for row in floor_rows]
    n = len(floor_rows)

    # The 95% ellipse for the current frame only, coloured by how
    # wide it is; a frame with no ellipse shows no ellipse rather
    # than a made-up one. A single-point track is static, so its
    # ellipse stays visible with no animation.
    for i, row in enumerate(floor_rows):
        if row["ellipse_semi_major_m"] is None:
            continue
        cx, cy = points[i]
        colour = confidence_colour(row["ellipse_semi_major_m"])
        angle = -(row["ellipse_angle_deg"] or 0.0)
        hidden = "" if n == 1 else ' visibility="hidden"'
        anim = ""
        if n > 1:
            anim = _visibility_animation(boundaries[i], boundaries[i + 1],
                                         duration, repeat)
        parts.append(
            f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" '
            f'rx="{row["ellipse_semi_major_m"] * scale:.1f}" '
            f'ry="{row["ellipse_semi_minor_m"] * scale:.1f}" '
            f'fill="{colour}" fill-opacity="0.18" '
            f'stroke="{colour}" stroke-width="1" '
            f'transform="rotate({angle:.1f} {cx:.1f} {cy:.1f})"'
            f'{hidden} data-frame="{i}">{anim}</ellipse>')

    # The trail draws itself behind the dot: a full polyline
    # whose dash offset animates from its length to zero on the
    # track's key times. A single point has no trail.
    if n > 1:
        total, offsets = _trail_offsets(points)
        trail = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        key_times_str = ";".join(f"{k:.4f}" for k in key_times)
        offsets_str = ";".join(f"{off:.1f}" for off in offsets)
        parts.append(
            f'<polyline points="{trail}" fill="none" '
            f'stroke="{TRACK_COLOUR}" stroke-width="2" '
            f'stroke-dasharray="{total:.1f}" '
            f'stroke-dashoffset="{total:.1f}" data-trail="true">'
            f'<animate attributeName="stroke-dashoffset" '
            f'values="{offsets_str}" keyTimes="{key_times_str}" '
            f'dur="{duration:g}s" repeatCount="{repeat}"/>'
            f"</polyline>")

    # The moving dot: its cx / cy animate through the projected
    # positions on the track's key times, gliding between frames.
    first = points[0]
    dot = [f'<circle cx="{first[0]:.1f}" cy="{first[1]:.1f}" r="5" '
           f'fill="{DOT_COLOUR}" stroke="#ffffff" stroke-width="1.5" '
           f'data-dot="true">']
    if n > 1:
        key_times_str = ";".join(f"{k:.4f}" for k in key_times)
        xs = ";".join(f"{px:.1f}" for px, _ in points)
        ys = ";".join(f"{py:.1f}" for _, py in points)
        dot.append(
            f'<animate attributeName="cx" values="{xs}" '
            f'keyTimes="{key_times_str}" dur="{duration:g}s" '
            f'repeatCount="{repeat}"/>')
        dot.append(
            f'<animate attributeName="cy" values="{ys}" '
            f'keyTimes="{key_times_str}" dur="{duration:g}s" '
            f'repeatCount="{repeat}"/>')
    dot.append("</circle>")
    parts.append("".join(dot))

    # One timestamp label per frame, visible only while its
    # frame is current, so the replay reads as a clock as well
    # as a moving dot.
    label_x = MARGIN + ward_w - 8
    for i, row in enumerate(floor_rows):
        hidden = "" if n == 1 else ' visibility="hidden"'
        anim = ""
        if n > 1:
            anim = _visibility_animation(boundaries[i], boundaries[i + 1],
                                         duration, repeat)
        parts.append(
            f'<text x="{label_x:.1f}" y="{MARGIN + ward_h - 10:.1f}" '
            f'font-family="sans-serif" font-size="12" '
            f'text-anchor="end" font-weight="bold"'
            f'{hidden} data-timestamp="{row["t"]}">'
            f't = {row["t"]:g} s{anim}</text>')

    # Start (S) and end (E) markers.
    last = points[-1]
    for (px, py), label in ((first, "S"), (last, "E")):
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="8" fill="#ffffff" '
            f'stroke="{TRACK_COLOUR}" stroke-width="2"/>')
        parts.append(
            f'<text x="{px:.1f}" y="{py + 4:.1f}" font-family="sans-serif" '
            f'font-size="11" font-weight="bold" text-anchor="middle">'
            f'{label}</text>')

    # Beacons: a diamond marker plus its id, so the replay can be
    # read against the hardware that located it. With drift, each
    # frame marks the beacons alarming on that frame's row with
    # the same red ring, red "!" badge and label as on the static
    # overlay — one group per alarming frame, visible only while
    # that frame is current, so the marking animates with the
    # replay: a beacon that alarmed mid-track and cleared is
    # badged while it was alarming and unmarked once it cleared.
    # A single-frame track has no animation, so its marking
    # stays visible with no animation, like its ellipse.
    for beacon in beacons:
        bx, by = _project(beacon.x, beacon.y, scale, width_m, height_m)
        frame_marks = []
        for i, row_state in enumerate(frame_drift):
            if beacon.id not in row_state:
                continue
            mean_db, verdict = row_state[beacon.id]
            hidden = "" if n == 1 else ' visibility="hidden"'
            anim = ""
            if n > 1:
                anim = _visibility_animation(boundaries[i],
                                             boundaries[i + 1],
                                             duration, repeat)
            ring = (
                f'<rect x="{bx - 9:.1f}" y="{by - 9:.1f}" width="18" '
                f'height="18" fill="none" stroke="{DRIFT_ALARM_COLOUR}" '
                f'stroke-width="2" '
                f'transform="rotate(45 {bx:.1f} {by:.1f})"/>')
            badge = (
                f'<circle cx="{bx + 13:.1f}" cy="{by - 13:.1f}" r="7" '
                f'fill="{DRIFT_ALARM_COLOUR}" '
                f'data-drift-alarm="{escape(str(beacon.id))}"/>'
                f'<text x="{bx + 13:.1f}" y="{by - 9:.1f}" '
                f'font-family="sans-serif" font-size="11" '
                f'font-weight="bold" fill="#ffffff" '
                f'text-anchor="middle">!</text>')
            verdict_attr = (
                f' data-drift-verdict="{escape(verdict)}"'
                if verdict is not None else "")
            drift_annotation = (
                f'<text x="{bx:.1f}" y="{by + 24:.1f}" '
                f'font-family="sans-serif" font-size="10" '
                f'font-weight="bold" fill="{DRIFT_ALARM_COLOUR}" '
                f'text-anchor="middle" '
                f'data-drift-alarm-label="{escape(str(beacon.id))}"'
                f"{verdict_attr}>"
                f'{escape(_drift_label(mean_db, verdict))}</text>')
            frame_marks.append(
                f'<g{hidden} data-frame="{i}" '
                f'data-drift-alarm-frame="{escape(str(beacon.id))}">'
                f"{ring}{badge}{drift_annotation}{anim}</g>")
        parts.append(
            f'<g data-beacon="{escape(str(beacon.id))}">'
            f'<rect x="{bx - 6:.1f}" y="{by - 6:.1f}" width="12" '
            f'height="12" fill="{BEACON_COLOUR}" '
            f'transform="rotate(45 {bx:.1f} {by:.1f})"/>'
            f'<text x="{bx:.1f}" y="{by - 12:.1f}" '
            f'font-family="sans-serif" font-size="12" '
            f'font-weight="bold" fill="{BEACON_COLOUR}" '
            f'text-anchor="middle">{escape(str(beacon.id))}</text>'
            f"{''.join(frame_marks)}"
            f"</g>")

    # Legend, bottom left under the ward.
    legend_y = MARGIN + ward_h + 34
    if any(frame_drift):
        # A drift label sits under a bottom-row beacon's diamond —
        # below the ward, where the legend starts — so the legend
        # drops a line to clear it instead of printing over it.
        legend_y += 14
    if drift:
        # Drift-alarm timeline strip, under the ward and above
        # the legend: one row per beacon that alarmed, with a
        # coloured segment per contiguous run of alarming frames,
        # labelled with the beacon and its frame ranges, so a
        # viewer can see at a glance when each alarm started and
        # ended without stepping through the frames. The strip
        # reuses the same per-frame alarm data the badges use, so
        # it can never disagree with them; a run splits when the
        # verdict changes mid-alarm, and each segment is coloured
        # by the verdict on its frames, so the strip's colours
        # are the badges' verdicts, not a second opinion; with
        # no alarms it is an empty labelled track, so the absence
        # of segments reads as no alarm recorded rather than a
        # missing strip.
        ranges = _alarm_ranges(frame_drift)
        strip_y = legend_y
        parts.append(
            f'<text x="{MARGIN}" y="{strip_y}" '
            f'font-family="sans-serif" font-size="12" '
            f'font-weight="bold">Drift alarm timeline</text>')
        track_x0 = MARGIN + 140
        track_x1 = MARGIN + ward_w - 8
        span_px = track_x1 - track_x0
        strip_rows = list(ranges.items()) or [(None, [])]
        for i, (beacon_id, beacon_ranges) in enumerate(strip_rows):
            row_y = strip_y + 18 + i * 16
            if beacon_id is None:
                label = "none"
            else:
                # The label is the alarm's span, merged across
                # verdict changes — the segments below stay split
                # by verdict, so the label says when it alarmed
                # and the colours say what the monitor thought.
                label = (f"{beacon_id} \u00b7 "
                         f"{_format_frame_ranges(
                             _merge_frame_ranges(beacon_ranges))}")
            parts.append(
                f'<line x1="{track_x0:.1f}" y1="{row_y:.1f}" '
                f'x2="{track_x1:.1f}" y2="{row_y:.1f}" '
                f'stroke="#e0e0e0" stroke-width="1"/>')
            parts.append(
                f'<text x="{MARGIN}" y="{row_y + 4:.1f}" '
                f'font-family="sans-serif" font-size="11">'
                f'{escape(label)}</text>')
            if beacon_id is None:
                continue
            row_parts = [
                f'<g data-drift-timeline="{escape(str(beacon_id))}">']
            for first, last, verdict in beacon_ranges:
                seg_x0 = track_x0 + (first / max(n - 1, 1)) * span_px
                seg_x1 = track_x0 + (last / max(n - 1, 1)) * span_px
                verdict_attr = (
                    f' data-drift-verdict="{escape(verdict)}"'
                    if verdict is not None else "")
                row_parts.append(
                    f'<rect x="{seg_x0:.1f}" y="{row_y - 4:.1f}" '
                    f'width="{max(seg_x1 - seg_x0, 3.0):.1f}" '
                    f'height="8" fill="{drift_verdict_colour(verdict)}" '
                    f'data-drift-timeline-range="{first}-{last}"'
                    f"{verdict_attr}/>")
            row_parts.append("</g>")
            parts.append("".join(row_parts))
        # The legend sits below the strip, so the canvas grows
        # with it the same way it already grows with the legend.
        legend_y = strip_y + 18 + (len(strip_rows) - 1) * 16 + 18
    parts.append(
        f'<text x="{MARGIN}" y="{legend_y}" font-family="sans-serif" '
        f'font-size="12" font-weight="bold">Legend</text>')
    legend_items = [
        (BEACON_COLOUR, "beacon"),
        (TRACK_COLOUR, "track path (S start, E end)"),
        (DOT_COLOUR, "live position"),
    ]
    if drift:
        legend_items.append(
            (DRIFT_ALARM_COLOUR,
             "drift alarm (red ! badge, per frame)"))
        # The strip's verdict colours, for the verdicts this
        # replay actually shows, so a viewer can read the
        # segments — one key per verdict present, plus the alarm
        # red for a segment with no classification recorded.
        for verdict in ("moved", "battery-sag", "noisy", "uncertain"):
            if any(verdict in [v for _first, _last, v in runs]
                   for runs in ranges.values()):
                legend_items.append(
                    (DRIFT_VERDICT_COLOURS[verdict],
                     f"timeline: {verdict}"))
        if any(v is None for runs in ranges.values()
               for _first, _last, v in runs):
            legend_items.append(
                (DRIFT_ALARM_COLOUR, "timeline: alarm, no verdict"))
    for _limit, colour, label in CONFIDENCE_BANDS:
        legend_items.append((colour, f"95% ellipse \u2014 {label}"))
    legend_items.append((NO_ELLIPSE_COLOUR, NO_ELLIPSE_LABEL))
    lx = MARGIN
    for colour, label in legend_items:
        parts.append(
            f'<rect x="{lx}" y="{legend_y + 8}" width="12" height="12" '
            f'fill="{colour}"/>')
        parts.append(
            f'<text x="{lx + 17}" y="{legend_y + 18}" '
            f'font-family="sans-serif" font-size="11">{escape(label)}</text>')
        lx += 17 + len(label) * 6 + 24
        if lx > SVG_WIDTH - MARGIN - 120:
            lx = MARGIN
            legend_y += 24
    legend_y += 24
    parts.append(
        f'<text x="{MARGIN}" y="{legend_y + 18}" '
        f'font-family="sans-serif" font-size="11" '
        f'fill="#757575">Replay: {duration:g} s'
        + (" loop" if loop else " once")
        + f", {n} frames, {seconds_per_step:g} s per step"
        + " — SMIL animation, no JavaScript"
        + "</text>")
    if summary is not None:
        legend_y += 24
        parts.append(
            f'<text x="{MARGIN}" y="{legend_y + 18}" '
            f'font-family="sans-serif" font-size="11" '
            f'fill="#757575">Coverage heatmap: mean 95% semi-major '
            f'{summary.mean_semi_major:.2f} m, worst '
            f'{summary.worst_semi_major:.2f} m at '
            f'({summary.worst_point[0]:g}, {summary.worst_point[1]:g}) m'
            f" — {summary.n_cells} cells"
            + (f", {summary.degenerate_cells} degenerate"
               if summary.degenerate_cells else "")
            + "</text>")
    if drift:
        # The alarm history over the whole replay, in one line —
        # which beacons alarmed, on which frames, with the
        # residual and verdict at their last alarming frame, so
        # the legend says what the replay marks even where the
        # badges are small and even for an alarm that cleared
        # before the walk ended.
        legend_y += 24
        history = _alarm_history(frame_drift)
        if history:
            alarm_text = ", ".join(
                f"{beacon_id} {mean_db:+.1f} dB"
                + (f" ({verdict})"
                   if verdict is not None and verdict != "healthy"
                   else "")
                + (f", frames {first}–{last}"
                   if first != last else f", frame {first}")
                for beacon_id, (first, last, mean_db, verdict)
                in history.items())
            drift_text = ("Drift alarms during replay: "
                          + alarm_text)
        else:
            drift_text = "Drift alarms during replay: none"
        parts.append(
            f'<text x="{MARGIN}" y="{legend_y + 18}" '
            f'font-family="sans-serif" font-size="11" '
            f'fill="#757575">{escape(drift_text)}</text>')
    # The legend grows with the options (heatmap, drift), so the
    # canvas grows with it: the ward is drawn to the fixed
    # SVG_HEIGHT scale, and the height is the taller of that and
    # the legend's last line plus a margin, so no legend line is
    # ever clipped below the canvas — the plain replay's own
    # "Replay:" line already ran past the 560 px canvas before
    # this change.
    height = int(max(SVG_HEIGHT, legend_y + 18 + 12))
    parts.insert(
        0,
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{SVG_WIDTH}" height="{height}" '
        f'viewBox="0 0 {SVG_WIDTH} {height}" role="img">')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_track_replay(track_path: str, beacons: list[Beacon],
                       output_path: str, width_m: float = 20.0,
                       height_m: float = 12.0,
                       floor: int | None = None,
                       seconds_per_step: float = 1.0,
                       loop: bool = True,
                       heatmap: bool = False,
                       heatmap_spacing: float = 2.0,
                       sigma_rssi_db: float = 2.0,
                       drift: bool = False) -> str:
    """Render the replay for ``track_path`` and write it to
    ``output_path``. Returns the SVG string. ``drift`` marks the
    beacons alarming at each frame's row (see
    :func:`render_track_replay`)."""
    svg = render_track_replay(load_track_for_overlay(track_path),
                              beacons, width_m=width_m,
                              height_m=height_m, floor=floor,
                              seconds_per_step=seconds_per_step,
                              loop=loop, heatmap=heatmap,
                              heatmap_spacing=heatmap_spacing,
                              sigma_rssi_db=sigma_rssi_db,
                              drift=drift)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(svg)
    return svg
