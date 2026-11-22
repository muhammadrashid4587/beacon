"""Trilateration: estimate a position from distances to known points.

Given three or more beacons at known positions and estimated distances
to each, solve for the (x, y) position. The circle equations are
linearised by subtracting the first beacon's equation from the rest,
giving a 2-column linear system that is solved in the least-squares
sense via the normal equations. With exactly three beacons the result
is exact (up to floating-point error); with more beacons it is the
least-squares best fit.

Raises ValueError when fewer than three beacons are given or when the
beacons are collinear (the linearised system is singular, so the
position is not determined).

The linear solution seeds a Gauss-Newton refinement on the true
distance residuals. The linearisation distorts the noise (a small RSSI
error becomes a large distance error far from a beacon), so the raw
linear estimate is noisy; refining against the actual residuals
roughly halves the error on noisy data.
"""

import math


def trilaterate(beacons: list[tuple[float, float]],
                distances: list[float]) -> tuple[float, float]:
    """Estimate (x, y) from beacon positions and distances to them.

    ``beacons`` is a list of (x, y) positions in metres, ``distances``
    the matching distances in metres.
    """
    if len(beacons) != len(distances):
        raise ValueError("beacons and distances must have the same length")
    if len(beacons) < 3:
        raise ValueError("trilateration needs at least 3 beacons")
    if any(d < 0 for d in distances):
        raise ValueError("distances must be non-negative")

    x1, y1 = beacons[0]
    d1 = distances[0]

    # Build A (2 columns) and b by subtracting the first circle equation
    # from each of the others:
    #   2*(xi - x1)*x + 2*(yi - y1)*y
    #       = d1^2 - di^2 + xi^2 - x1^2 + yi^2 - y1^2
    ata_00 = ata_01 = ata_11 = 0.0
    atb_0 = atb_1 = 0.0
    for (xi, yi), di in zip(beacons[1:], distances[1:]):
        a0 = 2.0 * (xi - x1)
        a1 = 2.0 * (yi - y1)
        b = d1 * d1 - di * di + xi * xi - x1 * x1 + yi * yi - y1 * y1
        ata_00 += a0 * a0
        ata_01 += a0 * a1
        ata_11 += a1 * a1
        atb_0 += a0 * b
        atb_1 += a1 * b

    det = ata_00 * ata_11 - ata_01 * ata_01
    if abs(det) < 1e-12:
        raise ValueError("beacons are collinear; position is not determined")

    x = (atb_0 * ata_11 - atb_1 * ata_01) / det
    y = (ata_00 * atb_1 - ata_01 * atb_0) / det
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("trilateration produced a non-finite position")
    return _gauss_newton(beacons, distances, (x, y))


def _gauss_newton(beacons: list[tuple[float, float]],
                  distances: list[float],
                  start: tuple[float, float],
                  iterations: int = 10) -> tuple[float, float]:
    """Refine ``start`` by minimising sum((|p - bi| - di)^2)."""
    x, y = start
    for _ in range(iterations):
        ata_00 = ata_01 = ata_11 = 0.0
        atb_0 = atb_1 = 0.0
        for (xi, yi), di in zip(beacons, distances):
            ri = math.hypot(x - xi, y - yi)
            if ri < 1e-9:
                continue
            residual = ri - di
            jx, jy = (x - xi) / ri, (y - yi) / ri
            ata_00 += jx * jx
            ata_01 += jx * jy
            ata_11 += jy * jy
            atb_0 += jx * residual
            atb_1 += jy * residual
        det = ata_00 * ata_11 - ata_01 * ata_01
        if abs(det) < 1e-12:
            break
        dx = -(atb_0 * ata_11 - atb_1 * ata_01) / det
        dy = -(ata_00 * atb_1 - ata_01 * atb_0) / det
        x += dx
        y += dy
        if abs(dx) + abs(dy) < 1e-9:
            break
    if not (math.isfinite(x) and math.isfinite(y)):
        return start
    return (x, y)
