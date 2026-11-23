"""Per-floor beacon layout validation: can every floor triangulate?

The engine localises within one floor's 2D plane, so each floor
needs its own beacons — three at least, spread out enough that the
geometry pins a device down. A floor that fails that test does not
fail loudly today: with fewer than three beacons heard on the
winning floor the engine falls back to beacons from other floors,
whose cross-floor distances are inflated by the slab attenuation,
so positions on that floor degrade silently. A floor whose beacons
are clustered on one wall is just as quiet about it — the
estimates still come out, only with a very wide confidence
ellipse in the thin corners.

This module scores every floor before deployment, with the same
95% confidence-ellipse model as the placement advisor
(:mod:`beacon.placement`), and reports a verdict per floor:
``ok``, ``thin`` (it triangulates, but the worst point in the room
is looser than the limit), or ``cannot_triangulate`` (fewer than
three beacons, or a geometry — collinear or coincident beacons —
under which no test point gives a defined uncertainty). Each
verdict carries its warnings in plain language, so a ward team
knows which floor to fix and why.

The verdict is only as good as the model: it assumes the
log-distance path-loss model with the beacons' own exponents and
the stated RSSI noise, and it knows nothing about walls, metal
beds, or dead zones in the real ward — the same limits as the
placement advisor. A floor that validates ``ok`` still needs a
walked survey and calibration before the numbers are trusted.
"""

from dataclasses import dataclass

from .beacon import Beacon
from .placement import LayoutScore, score_layout

#: A floor whose worst 95% ellipse semi-major axis is above this
#: is ``thin``: it triangulates, but some corner of the room is
#: located loosely. 10 m is the default limit — the default
#: four-corner layout on a 20 x 12 m ward worsts at 8.33 m, while
#: the placement advisor's mean-optimal layout worsts at 11.16 m
#: at its far corner (it trades the worst point for the mean, as
#: the advisor's own notes say), so the default limit calls the
#: corners fine and the advised layout thin at that corner.
DEFAULT_MAX_WORST_SEMI_MAJOR_M = 10.0


@dataclass(frozen=True)
class FloorLayoutVerdict:
    """One floor's layout verdict.

    ``floor`` is the floor number and ``n_beacons`` how many
    beacons it carries. ``status`` is ``ok``, ``thin``, or
    ``cannot_triangulate``. ``score`` is the floor's
    :class:`beacon.placement.LayoutScore` over the test grid, or
    ``None`` when the floor cannot triangulate and so has no
    score. ``warnings`` are the plain-language reasons behind
    the status, empty when the floor is ``ok``.
    """

    floor: int
    n_beacons: int
    status: str
    score: LayoutScore | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LayoutValidation:
    """The verdict for every floor in a building.

    ``verdicts`` are in floor order. ``ok`` is true only when
    every floor is ``ok``; ``warnings()`` collects every floor's
    warnings, and ``by_floor(floor)`` returns one floor's
    verdict (or ``None`` when the building has no such floor).
    """

    verdicts: tuple[FloorLayoutVerdict, ...]

    @property
    def ok(self) -> bool:
        return all(v.status == "ok" for v in self.verdicts)

    def warnings(self) -> list[str]:
        """Every floor's warnings, in floor order."""
        return [w for v in self.verdicts for w in v.warnings]

    def by_floor(self, floor: int) -> FloorLayoutVerdict | None:
        for verdict in self.verdicts:
            if verdict.floor == floor:
                return verdict
        return None


def validate_floor_layouts(
        beacons: list[Beacon], width: float, height: float,
        max_worst_semi_major: float = DEFAULT_MAX_WORST_SEMI_MAJOR_M,
        test_spacing: float = 1.0,
        sigma_rssi_db: float = 2.0,
        margin: float = 0.5) -> LayoutValidation:
    """Validate every floor's beacon layout for a room.

    ``beacons`` carry their floor (``Beacon.floor``, default 0);
    ``width`` and ``height`` are the room dimensions in metres,
    the same on every floor. Each floor is scored with
    :func:`beacon.placement.score_layout` over a test grid at
    ``test_spacing`` with the given ``margin`` and RSSI noise
    ``sigma_rssi_db``. The margin defaults to 0.5 m, the same as
    the placement advisor's test grid, so a beacon never lands
    exactly on a test point and is never reported as a coverage
    hole for merely coinciding with the grid. A floor with fewer
    than three beacons, or whose beacons give no test point a
    defined uncertainty, is ``cannot_triangulate``; a floor
    whose worst 95% ellipse semi-major axis exceeds
    ``max_worst_semi_major`` is ``thin``; the rest are ``ok``.
    Raises ValueError for no beacons, non-positive dimensions,
    or a non-positive limit.
    """
    if not beacons:
        raise ValueError("validating a layout needs at least one beacon")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if max_worst_semi_major <= 0:
        raise ValueError("max_worst_semi_major must be positive")

    by_floor: dict[int, list[Beacon]] = {}
    for beacon in beacons:
        by_floor.setdefault(beacon.floor, []).append(beacon)

    verdicts: list[FloorLayoutVerdict] = []
    for floor in sorted(by_floor):
        floor_beacons = by_floor[floor]
        n = len(floor_beacons)
        if n < 3:
            verdicts.append(FloorLayoutVerdict(
                floor=floor, n_beacons=n,
                status="cannot_triangulate", score=None,
                warnings=(f"floor {floor} has only {n} beacon"
                          f"{'s' if n != 1 else ''} — the engine needs "
                          f"3 to triangulate, so it falls back to "
                          f"beacons on other floors, whose cross-floor "
                          f"distances are inflated by the slab "
                          f"attenuation",)))
            continue
        try:
            score = score_layout(floor_beacons, width, height,
                                 test_spacing, sigma_rssi_db, margin)
        except ValueError:
            verdicts.append(FloorLayoutVerdict(
                floor=floor, n_beacons=n,
                status="cannot_triangulate", score=None,
                warnings=(f"floor {floor} has {n} beacons but no test "
                          f"point gives a defined uncertainty — the "
                          f"beacons are collinear or coincide, so the "
                          f"floor cannot triangulate",)))
            continue
        warnings: list[str] = []
        if score.degenerate_points:
            warnings.append(
                f"floor {floor} has {score.degenerate_points} of "
                f"{score.n_points} test points with undefined "
                f"uncertainty (on a beacon or on the beacon line)")
        if score.worst_semi_major > max_worst_semi_major:
            worst = score.worst_point
            where = (f" at ({worst[0]:.1f}, {worst[1]:.1f}) m"
                     if worst is not None else "")
            warnings.append(
                f"floor {floor}'s worst 95% error is "
                f"{score.worst_semi_major:.2f} m{where}, above the "
                f"{max_worst_semi_major:.1f} m limit — add a beacon "
                f"or move one toward that corner")
        verdicts.append(FloorLayoutVerdict(
            floor=floor, n_beacons=n,
            status="thin" if warnings else "ok", score=score,
            warnings=tuple(warnings)))
    return LayoutValidation(verdicts=tuple(verdicts))
