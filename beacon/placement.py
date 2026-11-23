"""Beacon placement advisor: where to mount the beacons.

A ward's beacons are usually screwed to the walls where it is
convenient, not where the geometry is good — and geometry is what
decides how well a device can be located. The confidence ellipse
from :mod:`beacon.uncertainty` measures that directly: at any point
in the room, the ellipse's semi-major axis is the worst-case
95% positioning error there, given the beacon layout and the RSSI
noise. A good layout keeps that axis small everywhere in the room,
not just in the middle.

This module scores a layout by that measure — the mean and worst
semi-major axis over a grid of test points — and picks a layout
greedily: the first beacon goes at the room centre, the second as
far from it as possible (two beacons cannot localise, so the
ellipse cannot rank them yet), and each further beacon goes where
it most reduces the mean semi-major axis over the test grid.
Adding a beacon can only add information to the weighted
least-squares system, so the mean uncertainty never gets worse as
beacons are added; the greedy choice just takes the biggest
improvement available at each step.

The advice is only as good as the model: it assumes the
log-distance path-loss model with the beacons' own exponents and
the stated RSSI noise, and it knows nothing about walls, metal
beds, or dead zones in the real ward. Treat the suggested spots as
a starting point, then survey and calibrate.
"""

import math
from dataclasses import dataclass

from .beacon import Beacon
from .fingerprint import grid_points
from .uncertainty import PositionUncertainty, estimate_uncertainty


@dataclass(frozen=True)
class LayoutScore:
    """How well a beacon layout covers a room.

    ``mean_semi_major`` / ``worst_semi_major`` are the mean and
    worst 95% ellipse semi-major axis in metres over the test
    grid; ``worst_point`` is where the worst axis occurs.
    ``degenerate_points`` counts test points where the ellipse is
    undefined — a point sitting exactly on a beacon, or a
    collinear geometry — which are left out of the mean and worst
    figures; ``n_points`` is the size of the test grid.
    """

    mean_semi_major: float
    worst_semi_major: float
    worst_point: tuple[float, float] | None
    degenerate_points: int
    n_points: int


@dataclass(frozen=True)
class PlacementAdvice:
    """A suggested beacon layout for a room.

    ``beacons`` are in placement order with ids ``P1`` .. ``Pn``;
    ``score`` is the layout's coverage over the test grid, and
    ``width`` / ``height`` are the room dimensions in metres.
    """

    beacons: list[Beacon]
    score: LayoutScore
    width: float
    height: float


def placement_uncertainty(beacons: list[Beacon], x: float, y: float,
                          sigma_rssi_db: float = 2.0
                          ) -> PositionUncertainty:
    """95% confidence ellipse for a device at ``(x, y)``.

    The distances are the true geometric distances from the point
    to each beacon — no observation is involved, so this is the
    uncertainty a layout *would* give at that point, given the
    RSSI noise ``sigma_rssi_db`` in dB (default 2.0, matching the
    simulator's default noise). Each beacon contributes with its
    own path-loss exponent. Raises ValueError for fewer than
    three beacons or a degenerate geometry, as
    :func:`beacon.uncertainty.estimate_uncertainty` does.
    """
    positions = [b.position for b in beacons]
    distances = [math.hypot(x - b.x, y - b.y) for b in beacons]
    exponents = [b.path_loss_exponent for b in beacons]
    return estimate_uncertainty(positions, (x, y), distances,
                                exponents, sigma_rssi_db)


def score_layout(beacons: list[Beacon], width: float, height: float,
                 test_spacing: float = 1.0,
                 sigma_rssi_db: float = 2.0,
                 margin: float = 1.0) -> LayoutScore:
    """Score a beacon layout over a room.

    The test grid is :func:`beacon.fingerprint.grid_points` at
    ``test_spacing`` with the given ``margin``; each point gets
    its 95% ellipse semi-major axis from
    :func:`placement_uncertainty`. Points where the ellipse is
    undefined are counted in ``degenerate_points`` and left out
    of the mean and worst figures. Raises ValueError for fewer
    than three beacons or non-positive dimensions.
    """
    if len(beacons) < 3:
        raise ValueError("scoring a layout needs at least 3 beacons")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    points = grid_points(width, height, test_spacing, margin)
    semi_majors: list[float] = []
    worst_point: tuple[float, float] | None = None
    worst = 0.0
    degenerate = 0
    for point in points:
        try:
            unc = placement_uncertainty(beacons, point[0], point[1],
                                        sigma_rssi_db)
        except ValueError:
            degenerate += 1
            continue
        semi_majors.append(unc.semi_major)
        if unc.semi_major > worst:
            worst = unc.semi_major
            worst_point = point
    if not semi_majors:
        raise ValueError("no test point gave a defined uncertainty")
    return LayoutScore(
        mean_semi_major=sum(semi_majors) / len(semi_majors),
        worst_semi_major=worst,
        worst_point=worst_point,
        degenerate_points=degenerate,
        n_points=len(points))


def _score_key(score: LayoutScore) -> tuple[int, float]:
    """Greedy comparison key: fewer degenerate points first, then
    a smaller mean semi-major axis, so trials that leave a test
    point undefined never win over ones that cover the room."""
    return (score.degenerate_points, score.mean_semi_major)


def advise_placement(width: float, height: float, n_beacons: int,
                     candidate_spacing: float = 2.0,
                     test_spacing: float = 1.0,
                     sigma_rssi_db: float = 2.0) -> PlacementAdvice:
    """Suggest where to mount ``n_beacons`` beacons in a room.

    Candidate spots are a grid over the ``width`` x ``height``
    room at ``candidate_spacing`` (1 m off the walls); the layout
    is scored over a test grid at ``test_spacing`` (0.5 m off the
    walls, so test points never sit on a candidate spot). The
    first beacon goes at the candidate nearest the room centre,
    the second at the candidate farthest from it, and each further
    beacon at the spot that most reduces the mean 95% ellipse
    semi-major axis over the test grid. Raises ValueError for
    fewer than three beacons, non-positive dimensions or spacings,
    or more beacons than candidate spots.
    """
    if n_beacons < 3:
        raise ValueError("placement needs at least 3 beacons")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if candidate_spacing <= 0 or test_spacing <= 0:
        raise ValueError("spacings must be positive")
    candidates = grid_points(width, height, candidate_spacing,
                             margin=1.0)
    if n_beacons > len(candidates):
        raise ValueError(
            f"{n_beacons} beacons need at least {n_beacons} candidate "
            f"spots, only {len(candidates)} fit the room")
    centre = (width / 2.0, height / 2.0)

    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # First beacon: nearest the room centre. Second: farthest from
    # the first, so the pair spans the room. Neither can be ranked
    # by the ellipse — two beacons cannot localise — so spread is
    # the honest criterion for both.
    first = min(candidates, key=lambda c: _dist(c, centre))
    placed = [first]
    remaining = [c for c in candidates if c != first]
    second = max(remaining, key=lambda c: _dist(c, first))
    placed.append(second)
    remaining.remove(second)

    # Greedy from the third beacon on: try every remaining spot,
    # keep the one that most reduces the mean semi-major axis.
    while len(placed) < n_beacons:
        best_spot = None
        best_score = None
        for spot in remaining:
            trial = [Beacon(f"P{i + 1}", x, y)
                     for i, (x, y) in enumerate(placed + [spot])]
            score = score_layout(trial, width, height, test_spacing,
                                 sigma_rssi_db, margin=0.5)
            if best_score is None or _score_key(score) < _score_key(
                    best_score):
                best_spot, best_score = spot, score
        placed.append(best_spot)
        remaining.remove(best_spot)

    beacons = [Beacon(f"P{i + 1}", x, y)
               for i, (x, y) in enumerate(placed)]
    score = score_layout(beacons, width, height, test_spacing,
                         sigma_rssi_db, margin=0.5)
    return PlacementAdvice(beacons=beacons, score=score,
                           width=width, height=height)
