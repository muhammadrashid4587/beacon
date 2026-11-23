"""Position uncertainty: a confidence ellipse for a trilateration estimate.

A trilateration result is only half the answer: a ward dashboard also
needs to know how much to trust it. A device heard by four beacons
spread around it is pinned down tightly; the same device heard by
three beacons clustered on one wall is well located along the wall
and barely located across it. This module turns that geometry into a
95% confidence ellipse around the estimate.

Derivation. The estimate minimises the squared distance residuals,
so near the solution the position error is approximately linear in
the distance errors, with Jacobian rows equal to the unit vectors
from each beacon to the estimate. Each distance comes from an RSSI
reading through the log-distance model, so an RSSI noise of
``sigma_rssi_db`` dB becomes a distance noise of

    sigma_d = sigma_rssi_db * d * ln(10) / (10 * n)

per beacon (``d`` is that beacon's estimated distance, ``n`` its
path-loss exponent — a steeper exponent means the same RSSI noise
moves the distance less). Weighting each Jacobian row by
``1 / sigma_d`` gives the weighted least-squares covariance
``C = (J^T W J)^-1``; its eigenvalues are the variances along the
ellipse axes, scaled by the 95% chi-square quantile for 2 degrees
of freedom (5.991), and the major axis points along the eigenvector
of the larger eigenvalue.

The ellipse describes one raw trilateration, not a filtered track:
it is the uncertainty of the geometry and the RSSI noise at this
step. A Kalman-filtered position has its own (smaller) covariance
that shrinks as the filter accumulates steps; that is a separate
feature.
"""

import math
from dataclasses import dataclass

# 95% chi-square quantile, 2 degrees of freedom: a point drawn from
# the estimated 2-D Gaussian lands inside the scaled ellipse with
# probability 0.95.
_CHI2_95_2DOF = 5.991


@dataclass(frozen=True)
class PositionUncertainty:
    """A 95% confidence ellipse around a position estimate.

    ``semi_major`` / ``semi_minor`` are the ellipse half-axes in
    metres; ``angle_deg`` is the orientation of the semi-major axis
    in degrees from the +x axis, in (-90, 90].
    """

    semi_major: float
    semi_minor: float
    angle_deg: float

    def contains(self, x: float, y: float,
                 centre: tuple[float, float]) -> bool:
        """True when ``(x, y)`` lies inside this ellipse centred at
        ``centre``."""
        dx, dy = x - centre[0], y - centre[1]
        theta = math.radians(self.angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # Rotate into the ellipse's own axes, then check the
        # normalised squared distance against 1.
        u = cos_t * dx + sin_t * dy
        v = -sin_t * dx + cos_t * dy
        return ((u / self.semi_major) ** 2
                + (v / self.semi_minor) ** 2) <= 1.0


def estimate_uncertainty(beacons: list[tuple[float, float]],
                         position: tuple[float, float],
                         distances: list[float],
                         path_loss_exponents: list[float] | None = None,
                         sigma_rssi_db: float = 2.0
                         ) -> PositionUncertainty:
    """95% confidence ellipse for a trilateration at ``position``.

    ``beacons`` are the (x, y) beacon positions and ``distances``
    their estimated distances to the device, in the same order.
    ``path_loss_exponents`` gives each beacon's exponent (default
    2.0 for all); ``sigma_rssi_db`` is the assumed RSSI noise
    standard deviation in dB (default 2.0, matching the simulator's
    default noise). Raises ValueError for fewer than three beacons
    or for a geometry whose covariance is singular (collinear
    beacons, or the estimate sitting exactly on a beacon).
    """
    if len(beacons) != len(distances):
        raise ValueError("beacons and distances must have the same length")
    if len(beacons) < 3:
        raise ValueError("uncertainty needs at least 3 beacons")
    if sigma_rssi_db <= 0:
        raise ValueError("sigma_rssi_db must be positive")
    exponents = (path_loss_exponents if path_loss_exponents is not None
                 else [2.0] * len(beacons))
    if len(exponents) != len(beacons):
        raise ValueError(
            "path_loss_exponents must match beacons in length")
    if any(n <= 0 for n in exponents):
        raise ValueError("path_loss_exponents must be positive")

    # Weighted normal matrix J^T W J, built from the unit vectors
    # from each beacon to the estimate, weighted by 1 / sigma_d.
    jtwj_00 = jtwj_01 = jtwj_11 = 0.0
    rows = 0
    for (bx, by), d, n in zip(beacons, distances, exponents):
        r = math.hypot(position[0] - bx, position[1] - by)
        if r < 1e-9:
            # The estimate sits on this beacon: the Jacobian row
            # is undefined, so this beacon cannot contribute.
            continue
        sigma_d = sigma_rssi_db * d * math.log(10) / (10.0 * n)
        if sigma_d <= 0:
            continue
        w = 1.0 / sigma_d
        jx, jy = (position[0] - bx) / r, (position[1] - by) / r
        jtwj_00 += w * w * jx * jx
        jtwj_01 += w * w * jx * jy
        jtwj_11 += w * w * jy * jy
        rows += 1
    if rows < 3:
        raise ValueError(
            "uncertainty needs at least 3 beacons away from the estimate")
    det = jtwj_00 * jtwj_11 - jtwj_01 * jtwj_01
    if abs(det) < 1e-12:
        raise ValueError(
            "beacon geometry is degenerate; uncertainty is not determined")

    # Covariance C = (J^T W J)^-1, then its eigenvalues analytically
    # (2x2 symmetric): the variances along the ellipse axes.
    c_00 = jtwj_11 / det
    c_01 = -jtwj_01 / det
    c_11 = jtwj_00 / det
    trace = c_00 + c_11
    disc = math.sqrt(max(0.0, (c_00 - c_11) ** 2 / 4.0 + c_01 ** 2))
    var_major = trace / 2.0 + disc
    var_minor = trace / 2.0 - disc
    if var_minor <= 0:
        raise ValueError(
            "beacon geometry is degenerate; uncertainty is not determined")
    scale = math.sqrt(_CHI2_95_2DOF)
    semi_major = scale * math.sqrt(var_major)
    semi_minor = scale * math.sqrt(var_minor)
    # Orientation of the major axis: the eigenvector of the larger
    # eigenvalue, from atan2(2*c_01, c_00 - c_11) / 2.
    angle_deg = 0.5 * math.degrees(
        math.atan2(2.0 * c_01, c_00 - c_11))
    return PositionUncertainty(semi_major=semi_major,
                               semi_minor=semi_minor,
                               angle_deg=angle_deg)
