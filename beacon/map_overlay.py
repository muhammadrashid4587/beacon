"""Ward-map overlay: plot an exported live track on the ward plan.

A ward dashboard that received a live track as JSON or CSV (see
``beacon.export``) still has to picture where the device went. This
module renders that track as a self-contained SVG: the ward outline,
the beacons at their known positions, the track as a path through
the located points, and one confidence ellipse per point — the 95%
ellipse from the export, coloured by how wide it is, so a glance
shows where the track was tight and where it was guessing. A legend
explains the colours, and the header carries the floor, the step
counts and the coverage, so a track starved by dead beacons shows
up as thin rather than silently short.

The SVG is plain text with no dependencies, so a dashboard can
embed it, save it, or open it in a browser as-is. Rows on other
floors are not plotted on top of each other: the overlay renders
one floor, the most common floor in the track by default, or the
floor named explicitly.

Pure standard library, no dependencies.
"""

import os
from dataclasses import dataclass
from xml.sax.saxutils import escape

from .beacon import Beacon
from .export import load_track_csv, load_track_json
from .fingerprint import grid_points
from .placement import placement_uncertainty

# SVG canvas: the ward is drawn inside the margin, keeping its
# aspect ratio, so a 20 x 12 m ward does not stretch.
SVG_WIDTH = 860
SVG_HEIGHT = 560
MARGIN = 50

# Confidence colour by 95% ellipse semi-major axis: tight, loose,
# very loose. A point with no ellipse (degenerate geometry) is
# grey — its position is plotted, its confidence is not claimed.
CONFIDENCE_BANDS = (
    (2.0, "#2e7d32", "tight (semi-major \u2264 2 m)"),
    (5.0, "#e6a817", "loose (semi-major \u2264 5 m)"),
    (float("inf"), "#c0392b", "very loose (semi-major > 5 m)"),
)
NO_ELLIPSE_COLOUR = "#757575"
NO_ELLIPSE_LABEL = "no ellipse (degenerate geometry)"

BEACON_COLOUR = "#1565c0"
TRACK_COLOUR = "#212121"

HEATMAP_FILL_OPACITY = "0.22"


@dataclass(frozen=True)
class CoverageCell:
    """One heatmap cell: the positioning quality at ``(x, y)``.

    ``semi_major`` is the 95% confidence ellipse's semi-major axis
    in metres that the beacon layout would give at this cell (see
    :func:`beacon.placement.placement_uncertainty`) — ``None``
    where the ellipse is undefined, a degenerate geometry the
    heatmap makes no claim about.
    """

    x: float
    y: float
    semi_major: float | None


@dataclass(frozen=True)
class HeatmapSummary:
    """A heatmap's coverage in one line.

    ``mean_semi_major`` / ``worst_semi_major`` are the mean and
    worst 95% ellipse semi-major axis over the cells that gave a
    defined ellipse; ``worst_point`` is where the worst axis
    occurs; ``degenerate_cells`` counts cells with no defined
    ellipse, left out of both figures; ``n_cells`` is the grid
    size.
    """

    mean_semi_major: float
    worst_semi_major: float
    worst_point: tuple[float, float] | None
    degenerate_cells: int
    n_cells: int


def coverage_heatmap(beacons: list[Beacon], width_m: float = 20.0,
                     height_m: float = 12.0, spacing: float = 2.0,
                     sigma_rssi_db: float = 2.0
                     ) -> list[CoverageCell]:
    """Per-cell positioning quality for a beacon layout.

    The ward is gridded at ``spacing`` (1 m off the walls, so no
    cell sits on a corner beacon) and each cell gets the 95%
    ellipse semi-major axis the layout would give there, from
    :func:`beacon.placement.placement_uncertainty` — the same
    uncertainty model behind the placement advisor and the
    overlay's per-point ellipses, so the heatmap and the track
    agree about where the ward is well covered and where it is
    thin. Raises ValueError for fewer than three beacons or
    non-positive dimensions or spacing.
    """
    if len(beacons) < 3:
        raise ValueError("a coverage heatmap needs at least 3 beacons")
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    cells = []
    for x, y in grid_points(width_m, height_m, spacing):
        try:
            unc = placement_uncertainty(beacons, x, y, sigma_rssi_db)
        except ValueError:
            cells.append(CoverageCell(x, y, None))
            continue
        cells.append(CoverageCell(x, y, unc.semi_major))
    return cells


def heatmap_summary(cells: list[CoverageCell]) -> HeatmapSummary:
    """Summarise a heatmap: mean and worst semi-major axis.

    Cells with no defined ellipse are counted in
    ``degenerate_cells`` and left out of the mean and worst.
    Raises ValueError when no cell gave a defined ellipse.
    """
    defined = [cell for cell in cells if cell.semi_major is not None]
    if not defined:
        raise ValueError("no heatmap cell gave a defined uncertainty")
    worst = max(defined, key=lambda cell: cell.semi_major)
    return HeatmapSummary(
        mean_semi_major=sum(cell.semi_major for cell in defined)
        / len(defined),
        worst_semi_major=worst.semi_major,
        worst_point=(worst.x, worst.y),
        degenerate_cells=len(cells) - len(defined),
        n_cells=len(cells))


def confidence_colour(semi_major: float | None) -> str:
    """The confidence colour for one point's 95% ellipse.

    ``None`` (no ellipse) is grey; otherwise the band the
    semi-major axis falls in.
    """
    if semi_major is None:
        return NO_ELLIPSE_COLOUR
    for limit, colour, _label in CONFIDENCE_BANDS:
        if semi_major <= limit:
            return colour
    return CONFIDENCE_BANDS[-1][1]  # pragma: no cover - inf band


def _most_common_floor(rows: list[dict]) -> int:
    """The floor most rows are on — the overlay's floor by default."""
    counts: dict[int, int] = {}
    for row in rows:
        counts[row["floor"]] = counts.get(row["floor"], 0) + 1
    return max(counts, key=lambda floor: counts[floor])


def _project(x: float, y: float, scale: float,
             width_m: float, height_m: float) -> tuple[float, float]:
    """Ward metres to SVG pixels; y is flipped, so up stays up."""
    return (MARGIN + x * scale, MARGIN + (height_m - y) * scale)


def render_map_overlay(rows: list[dict], beacons: list[Beacon],
                       width_m: float = 20.0, height_m: float = 12.0,
                       floor: int | None = None,
                       title: str | None = None,
                       heatmap: bool = False,
                       heatmap_spacing: float = 2.0,
                       sigma_rssi_db: float = 2.0) -> str:
    """Render an exported live track as a ward-map SVG string.

    ``rows`` are the row dicts ``load_track_json`` /
    ``load_track_csv`` return. ``beacons`` are the ward's beacons
    at their known positions; beacons on other floors are drawn
    too, since the plan is the same ward — only the track is
    filtered to one floor. ``width_m`` / ``height_m`` are the
    ward's size in metres. With ``heatmap`` the overlay also
    shades the ward grid by the positioning quality the beacon
    layout gives at each cell (see :func:`coverage_heatmap`),
    coloured by the same confidence bands as the per-point
    ellipses, so a dashboard can see at a glance where the ward
    is well covered and where a device would be guessing — before
    any device walks there.

    Raises ValueError on an empty track, with no beacons, or when
    the chosen floor has no located points.
    """
    if not rows:
        raise ValueError("cannot render a map overlay for an empty track")
    if not beacons:
        raise ValueError("cannot render a map overlay with no beacons")
    if floor is None:
        floor = _most_common_floor(rows)
    floor_rows = [row for row in rows if row["floor"] == floor]
    if not floor_rows:
        raise ValueError(f"no located points on floor {floor}")

    scale = min((SVG_WIDTH - 2 * MARGIN) / width_m,
                (SVG_HEIGHT - 2 * MARGIN) / height_m)
    ward_w = width_m * scale
    ward_h = height_m * scale
    coverage = floor_rows[0]["coverage"]
    heading = title or (f"Live track \u2014 floor {floor} "
                        f"({len(floor_rows)} points, "
                        f"coverage {coverage:.0%})")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{SVG_WIDTH}" height="{SVG_HEIGHT}" '
        f'viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" role="img">',
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

    # Coverage heatmap, under everything else: one shaded cell
    # per grid point, coloured by the same confidence bands as
    # the per-point ellipses. Degenerate cells are left unshaded
    # rather than coloured as if they were covered.
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

    # Confidence ellipses, then the track path, then the points —
    # the path sits on top of the ellipses, the points on top of
    # the path.
    for row in floor_rows:
        if row["ellipse_semi_major_m"] is None:
            continue
        cx, cy = _project(row["x"], row["y"], scale, width_m, height_m)
        colour = confidence_colour(row["ellipse_semi_major_m"])
        angle = -(row["ellipse_angle_deg"] or 0.0)
        parts.append(
            f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" '
            f'rx="{row["ellipse_semi_major_m"] * scale:.1f}" '
            f'ry="{row["ellipse_semi_minor_m"] * scale:.1f}" '
            f'fill="{colour}" fill-opacity="0.18" '
            f'stroke="{colour}" stroke-width="1" '
            f'transform="rotate({angle:.1f} {cx:.1f} {cy:.1f})"/>')

    points = " ".join(
        f"{_project(row['x'], row['y'], scale, width_m, height_m)[0]:.1f},"
        f"{_project(row['x'], row['y'], scale, width_m, height_m)[1]:.1f}"
        for row in floor_rows)
    parts.append(
        f'<polyline points="{points}" fill="none" '
        f'stroke="{TRACK_COLOUR}" stroke-width="2"/>')
    for i, row in enumerate(floor_rows):
        cx, cy = _project(row["x"], row["y"], scale, width_m, height_m)
        colour = confidence_colour(row["ellipse_semi_major_m"])
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" '
            f'fill="{colour}" data-t="{row["t"]}" '
            f'data-index="{i}"/>')
    # Start (S) and end (E) markers.
    first = _project(floor_rows[0]["x"], floor_rows[0]["y"],
                     scale, width_m, height_m)
    last = _project(floor_rows[-1]["x"], floor_rows[-1]["y"],
                    scale, width_m, height_m)
    for (px, py), label in ((first, "S"), (last, "E")):
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="8" fill="#ffffff" '
            f'stroke="{TRACK_COLOUR}" stroke-width="2"/>')
        parts.append(
            f'<text x="{px:.1f}" y="{py + 4:.1f}" font-family="sans-serif" '
            f'font-size="11" font-weight="bold" text-anchor="middle">'
            f'{label}</text>')

    # Beacons: a diamond marker plus its id, so the track can be
    # read against the hardware that located it.
    for beacon in beacons:
        bx, by = _project(beacon.x, beacon.y, scale, width_m, height_m)
        parts.append(
            f'<g data-beacon="{escape(str(beacon.id))}">'
            f'<rect x="{bx - 6:.1f}" y="{by - 6:.1f}" width="12" '
            f'height="12" fill="{BEACON_COLOUR}" '
            f'transform="rotate(45 {bx:.1f} {by:.1f})"/>'
            f'<text x="{bx:.1f}" y="{by - 12:.1f}" '
            f'font-family="sans-serif" font-size="12" '
            f'font-weight="bold" fill="{BEACON_COLOUR}" '
            f'text-anchor="middle">{escape(str(beacon.id))}</text>'
            f"</g>")

    # Legend, bottom left under the ward.
    legend_y = MARGIN + ward_h + 34
    parts.append(
        f'<text x="{MARGIN}" y="{legend_y}" font-family="sans-serif" '
        f'font-size="12" font-weight="bold">Legend</text>')
    legend_items = [
        (BEACON_COLOUR, "beacon"),
        (TRACK_COLOUR, "track path (S start, E end)"),
    ]
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
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def load_track_for_overlay(path: str) -> list[dict]:
    """Read back an exported track for the overlay, by extension.

    ``.json`` reads a ``export_track_json`` file, ``.csv`` a
    ``export_track_csv`` file; anything else raises ValueError
    instead of guessing.
    """
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".json":
        return load_track_json(path)
    if suffix == ".csv":
        return load_track_csv(path)
    raise ValueError(
        f"cannot render a map overlay from {path!r}: "
        f"expected a .json or .csv live-track export")


def write_map_overlay(track_path: str, beacons: list[Beacon],
                      output_path: str, width_m: float = 20.0,
                      height_m: float = 12.0,
                      floor: int | None = None,
                      heatmap: bool = False,
                      heatmap_spacing: float = 2.0,
                      sigma_rssi_db: float = 2.0) -> str:
    """Render the overlay for ``track_path`` and write it to
    ``output_path``. Returns the SVG string."""
    svg = render_map_overlay(load_track_for_overlay(track_path),
                             beacons, width_m=width_m,
                             height_m=height_m, floor=floor,
                             heatmap=heatmap,
                             heatmap_spacing=heatmap_spacing,
                             sigma_rssi_db=sigma_rssi_db)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(svg)
    return svg
