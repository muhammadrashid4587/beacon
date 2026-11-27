"""Smoothing filters for position estimates.

RSSI is noisy, so raw trilateration output jitters from step to step
even when the device is standing still. Two filters are provided:

- ``MovingAverageFilter``: a moving average over the last ``window``
  estimates. It trades a little lag for a much steadier position.
- ``KalmanFilter``: a constant-velocity Kalman filter. It models the
  device as moving at a steady velocity and weighs each new estimate
  against that prediction, so it tracks a walking device with less
  lag than the moving average and still smooths a stationary one.
  Its ``smooth`` method runs the Rauch-Tung-Striebel backward
  recursion over the recorded forward pass, revising each past
  position with the observations that came after it — for a
  recorded walk, where the future is already known.
"""

import math
from collections import deque

# Measured measurement noise for the smoother, in metres (standard
# deviation): on the default demo walk at 2 dB RSSI noise the live
# estimates err by about 2 m, so a filter that assumes the 1 m
# default understates the noise and its 95% ellipse is overconfident
# — only 20 of 40 true positions fell inside the smoothed ellipse
# on seed 7, against 33 of 40 inside the raw geometry ellipse.
# The smoothing path (see ``beacon.export.smooth_track_rows``)
# defaults to this measured value instead of the filter's 1 m
# default; the live filter keeps the 1 m default, since its
# positions carry no covariance-based ellipse of their own.
MEASURED_MEASUREMENT_NOISE_M = 2.0


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Matrix product of two small dense matrices."""
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def _transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*a)]


def _matvec(a: list[list[float]], v: list[float]) -> list[float]:
    return [sum(row[k] * v[k] for k in range(len(v))) for row in a]


def _matinv(a: list[list[float]]) -> list[list[float]]:
    """Inverse of a small dense square matrix (Gauss-Jordan with
    partial pivoting). Raises ValueError on a singular matrix."""
    n = len(a)
    aug = [list(map(float, a[i])) + [1.0 if i == j else 0.0
                                     for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) == 0.0:
            raise ValueError("singular matrix")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for row in range(n):
            if row != col and aug[row][col] != 0.0:
                factor = aug[row][col]
                aug[row] = [v - factor * w for v, w in
                            zip(aug[row], aug[col])]
    return [row[n:] for row in aug]


def _mat_sub(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))]
            for i in range(len(a))]


def _symmetrise(cov: list[list[float]]) -> list[list[float]]:
    """A covariance averaged with its own transpose.

    A covariance is symmetric by definition, but the update
    arithmetic does not preserve that bit-for-bit: the short-form
    update ``(I - K H) P`` and the smoother's backward subtraction
    both drift off symmetry by rounding, and an asymmetric
    covariance feeds the next step a gain that is wrong in the
    last bits — and, with a strongly anisotropic measurement
    noise, wrong enough to drive the smoothed covariance
    degenerate (see :meth:`KalmanFilter.update`). Averaging
    with the transpose after each step keeps the stored
    covariances exactly symmetric at no cost to the maths.
    """
    n = len(cov)
    return [[0.5 * (cov[i][j] + cov[j][i]) for j in range(n)]
            for i in range(n)]


def _floor_position_block(cov: list[list[float]],
                          floor: float = 1e-6
                          ) -> tuple[list[list[float]], bool]:
    """Floor the 2x2 position block's eigenvalues at ``floor`` m^2.

    Returns the (possibly repaired) covariance and whether a
    repair happened. The Rauch-Tung-Striebel backward recursion
    subtracts one covariance from another, so floating-point
    rounding can leave the smoothed position block with a
    non-positive minor eigenvalue even when every forward
    covariance is positive definite — a degenerate covariance
    no ellipse can be drawn from. When that happens the block
    is rebuilt from its own eigen-decomposition with the minor
    variance raised to the floor (1 mm standard deviation), the
    smallest change that restores positive-definiteness, and
    the major variance raised to the floor too if it is smaller.
    A healthy block passes through untouched. On the demo walks
    the Joseph-form update and the symmetrisation below already
    keep every smoothed covariance healthy, so the floor never
    fires there — it is a backstop for other walks, not a
    correction to the numbers.
    """
    c_00, c_01, c_11 = cov[0][0], cov[0][1], cov[1][1]
    trace = c_00 + c_11
    disc = math.sqrt(max(0.0, (c_00 - c_11) ** 2 / 4.0 + c_01 ** 2))
    var_major = trace / 2.0 + disc
    var_minor = trace / 2.0 - disc
    if var_minor > 0.0:
        return cov, False
    var_minor = floor
    var_major = max(var_major, floor)
    angle = 0.5 * math.atan2(2.0 * c_01, c_00 - c_11)
    cos_t, sin_t = math.cos(angle), math.sin(angle)
    fixed = [list(row) for row in cov]
    fixed[0][0] = (cos_t * cos_t * var_major
                   + sin_t * sin_t * var_minor)
    fixed[0][1] = fixed[1][0] = cos_t * sin_t * (var_major - var_minor)
    fixed[1][1] = (sin_t * sin_t * var_major
                   + cos_t * cos_t * var_minor)
    return fixed, True


def _copy_state(state: list[float]) -> list[float]:
    return list(state)


def _copy_cov(cov: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in cov]


class MovingAverageFilter:
    """Moving average over the last ``window`` (x, y) estimates."""

    def __init__(self, window: int = 5):
        if window < 1:
            raise ValueError("window must be at least 1")
        self.window = window
        self._history: deque[tuple[float, float]] = deque(maxlen=window)

    def update(self, position: tuple[float, float]) -> tuple[float, float]:
        """Feed a new estimate, return the smoothed position."""
        self._history.append(position)
        n = len(self._history)
        return (
            sum(p[0] for p in self._history) / n,
            sum(p[1] for p in self._history) / n,
        )

    def reset(self) -> None:
        """Forget all history (e.g. when tracking a new device)."""
        self._history.clear()


class KalmanFilter:
    """Constant-velocity Kalman filter over (x, y) estimates.

    State is ``(x, y, vx, vy)`` with one step per observation
    (``dt = 1``). The filter predicts the next position from the
    estimated velocity, then corrects with the new measurement. The
    first update initialises the state at the measurement with zero
    velocity, so it returns the measurement unchanged.

    ``process_noise`` is the variance of the (unknown) acceleration
    between steps: larger values trust the measurements more and
    follow turns faster, smaller values smooth more. ``measurement_noise``
    is the variance of the position estimates the filter is fed:
    the default 1.0 (a 1 m standard deviation) is tuned for a walking
    device with ~1 m measurement noise, but the live estimates on
    the demo walk err by about 2 m, so the smoothing path uses
    ``MEASURED_MEASUREMENT_NOISE_M`` (2 m, variance 4.0) instead —
    a filter that assumes less noise than the estimates really
    carry reports a tighter posterior than it has earned.
    ``update`` also takes a per-step ``measurement_noise`` that
    overrides this value for one update: the smoothing path
    feeds each step its own geometry-derived variance, so the
    filter trusts geometrically strong steps more than weak
    ones instead of one isotropic number for the whole walk.
    It also takes a per-step ``measurement_covariance`` — the
    full 2x2 measurement covariance, so a step can be trusted
    more along one axis than the other (the live confidence
    ellipse's own covariance, via
    :func:`beacon.uncertainty.ellipse_covariance`). The
    covariance update is the Joseph form, ``(I - K H) P (I - K
    H)^T + K R K^T``, and every stored covariance is
    symmetrised: the short form ``(I - K H) P`` lost symmetry
    and positive-definiteness in the last bits, and with a
    strongly anisotropic ``R`` that was enough to drive 33–37
    of 40 smoothed covariances degenerate on the demo walks
    (the v0.56.0 attempt, abandoned) — the Joseph form plus
    symmetrisation keeps all 40 healthy. The full anisotropic
    weighting still measures worse than the per-step isotropic
    noise on the demo walks (see the README), so the smoothing
    path does not default to it; the parameter is the stable
    mechanism, not the default.
    """

    # State-transition (constant velocity, dt = 1) and observation
    # (position only) matrices.
    _F = [[1.0, 0.0, 1.0, 0.0],
          [0.0, 1.0, 0.0, 1.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0]]
    _H = [[1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0]]

    def __init__(self, process_noise: float = 0.1,
                 measurement_noise: float = 1.0):
        if process_noise <= 0.0:
            raise ValueError("process_noise must be positive")
        if measurement_noise <= 0.0:
            raise ValueError("measurement_noise must be positive")
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        # Discrete white-noise acceleration model, dt = 1.
        q = process_noise
        self._Q = [[q / 4, 0.0, q / 2, 0.0],
                   [0.0, q / 4, 0.0, q / 2],
                   [q / 2, 0.0, q, 0.0],
                   [0.0, q / 2, 0.0, q]]
        self._R = [[measurement_noise, 0.0],
                   [0.0, measurement_noise]]
        self._state: list[float] | None = None
        self._covariance: list[list[float]] | None = None
        # Forward-pass history for trajectory smoothing: one entry
        # per update, (predicted state, predicted covariance,
        # filtered state, filtered covariance). ``smooth`` runs the
        # Rauch-Tung-Striebel backward recursion over it.
        self._history: list[tuple[
            list[float], list[list[float]],
            list[float], list[list[float]]]] = []

    def update(self, position: tuple[float, float],
               measurement_noise: float | None = None,
               measurement_covariance: list[list[float]] | None = None
               ) -> tuple[float, float]:
        """Feed a new estimate, return the filtered position.

        ``measurement_noise`` is this step's measurement-noise
        variance, overriding the constructor's value for this
        update only — the smoother's per-step geometry noise
        (see :func:`beacon.uncertainty.ellipse_rms_sigma`):
        a step whose live estimate is geometrically weak is fed
        with a larger variance, so the filter leans harder on
        its velocity model there, and a geometrically strong
        step with a smaller one. It must be positive when
        given; the constructor's value is used when it is not.

        ``measurement_covariance`` is instead the full 2x2
        measurement covariance for this step — the live
        confidence ellipse's own covariance (see
        :func:`beacon.uncertainty.ellipse_covariance`) — so
        the filter can trust a step more along its strong axis
        than its weak one. It must be a symmetric
        positive-definite 2x2 matrix; it cannot be given
        together with ``measurement_noise``. A diagonal
        covariance with equal entries is exactly the scalar
        override with that variance.
        """
        if (measurement_noise is not None
                and measurement_covariance is not None):
            raise ValueError(
                "measurement_noise and measurement_covariance "
                "cannot both be given")
        if (measurement_noise is not None
                and measurement_noise <= 0.0):
            raise ValueError("measurement_noise must be positive")
        if measurement_covariance is not None:
            if (len(measurement_covariance) != 2
                    or len(measurement_covariance[0]) != 2
                    or len(measurement_covariance[1]) != 2
                    or measurement_covariance[0][1]
                    != measurement_covariance[1][0]):
                raise ValueError(
                    "measurement_covariance must be a symmetric "
                    "2x2 matrix")
            if (measurement_covariance[0][0] <= 0.0
                    or measurement_covariance[1][1] <= 0.0
                    or (measurement_covariance[0][0]
                        * measurement_covariance[1][1]
                        - measurement_covariance[0][1] ** 2) <= 0.0):
                raise ValueError(
                    "measurement_covariance must be positive "
                    "definite")
            step_r = [list(row) for row in measurement_covariance]
        else:
            step_r = ([[measurement_noise, 0.0], [0.0, measurement_noise]]
                      if measurement_noise is not None else self._R)
        if self._state is None:
            # First observation: start here, at rest, with a wide
            # velocity covariance so the filter can learn the speed.
            self._state = [position[0], position[1], 0.0, 0.0]
            self._covariance = [[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 100.0, 0.0],
                                [0.0, 0.0, 0.0, 100.0]]
            self._history.append((_copy_state(self._state),
                                  _copy_cov(self._covariance),
                                  _copy_state(self._state),
                                  _copy_cov(self._covariance)))
            return position
        state = self._state
        cov = self._covariance
        # Predict.
        state = _matvec(self._F, state)
        cov = _matmul(_matmul(self._F, cov), _transpose(self._F))
        cov = [[cov[i][j] + self._Q[i][j] for j in range(4)]
               for i in range(4)]
        cov = _symmetrise(cov)
        predicted_state = _copy_state(state)
        predicted_cov = _copy_cov(cov)
        # Update: innovation, its covariance, Kalman gain.
        hx = _matvec(self._H, state)
        innovation = [position[0] - hx[0], position[1] - hx[1]]
        s = _matmul(_matmul(self._H, cov), _transpose(self._H))
        s = [[s[i][j] + step_r[i][j] for j in range(2)] for i in range(2)]
        det = s[0][0] * s[1][1] - s[0][1] * s[1][0]
        if det == 0.0:
            raise ValueError("singular innovation covariance")
        s_inv = [[s[1][1] / det, -s[0][1] / det],
                 [-s[1][0] / det, s[0][0] / det]]
        gain = _matmul(_matmul(cov, _transpose(self._H)), s_inv)
        state = [state[i] + gain[i][0] * innovation[0]
                 + gain[i][1] * innovation[1] for i in range(4)]
        # Joseph form: P = (I - K H) P_pred (I - K H)^T + K R K^T.
        # The short form (I - K H) P_pred is algebraically the
        # same update, but it is not symmetric by construction:
        # rounding drifts it off symmetry and, with a strongly
        # anisotropic R, off positive-definiteness too — the
        # v0.56.0 anisotropic attempt degenerated 33–37 of 40
        # smoothed covariances on the demo walks through exactly
        # this. The Joseph form keeps the covariance symmetric
        # and positive semi-definite by construction, and the
        # explicit symmetrisation keeps it exactly symmetric.
        kh = _matmul(gain, self._H)
        i_kh = [[(1.0 if i == j else 0.0) - kh[i][j] for j in range(4)]
                for i in range(4)]
        cov = _matmul(_matmul(i_kh, cov), _transpose(i_kh))
        krk = _matmul(_matmul(gain, step_r), _transpose(gain))
        cov = [[cov[i][j] + krk[i][j] for j in range(4)]
               for i in range(4)]
        cov = _symmetrise(cov)
        self._state = state
        self._covariance = cov
        self._history.append((predicted_state, predicted_cov,
                              _copy_state(state), _copy_cov(cov)))
        return (state[0], state[1])

    def smooth(self) -> list[tuple[float, float]]:
        """Rauch-Tung-Striebel smoothed positions for the recorded
        history: one (x, y) per update so far.

        The forward filter is causal — each estimate only uses
        observations up to that step, so it lags behind turns and
        keeps some jitter. The smoother revises every past estimate
        with the observations that came after it, so a recorded walk
        comes out steadier and closer to the true path. The last
        smoothed position is the last filtered one (nothing comes
        after it to revise it with). Returns [] when no updates
        have been fed yet.
        """
        return [(x, y) for x, y, _ in self.smooth_with_covariance()]

    def smooth_with_covariance(
            self) -> list[tuple[float, float, list[list[float]]]]:
        """Rauch-Tung-Striebel smoothed positions with their
        posterior covariances: one (x, y, 2x2 covariance) per
        update so far.

        The same backward recursion as :meth:`smooth`, keeping
        the smoothed covariance at each step alongside the
        smoothed position — the covariance the smoother's own
        posterior gives the revised position, so a smoothed track
        can carry its own 95% ellipse (see
        :func:`beacon.uncertainty.ellipse_from_covariance`)
        instead of reusing the live estimate's. The covariance
        is the 2x2 position block of the smoothed 4x4 state
        covariance. Returns [] when no updates have been fed
        yet.

        Each smoothed covariance is symmetrised, and its
        position block's eigenvalues are floored at a small
        positive value (see :func:`_floor_position_block`):
        the backward step subtracts one covariance from
        another, so rounding can leave the result with a
        non-positive minor eigenvalue even when every forward
        covariance is healthy — with an anisotropic measurement
        noise that happened on 33–37 of 40 steps per demo walk
        before the Joseph-form update and this symmetrisation.
        """
        n = len(self._history)
        if n == 0:
            return []
        smoothed_states: list[list[float]] = [None] * n  # type: ignore[list-item]
        smoothed_covs: list[list[list[float]]] = [None] * n  # type: ignore[list-item]
        smoothed_states[-1] = self._history[-1][2]
        smoothed_covs[-1] = _symmetrise(self._history[-1][3])
        smoothed_covs[-1], _ = _floor_position_block(
            smoothed_covs[-1])
        for k in range(n - 2, -1, -1):
            _, _, filt_state, filt_cov = self._history[k]
            pred_state, pred_cov, _, _ = self._history[k + 1]
            # Smoother gain G = P_k|k F^T P_{k+1}|k^-1.
            gain = _matmul(_matmul(filt_cov, _transpose(self._F)),
                           _matinv(pred_cov))
            diff = [smoothed_states[k + 1][i] - pred_state[i]
                    for i in range(4)]
            smoothed_states[k] = [filt_state[i]
                                  + sum(gain[i][j] * diff[j]
                                        for j in range(4))
                                  for i in range(4)]
            cov_diff = _mat_sub(smoothed_covs[k + 1], pred_cov)
            smoothed_covs[k] = _symmetrise([
                [filt_cov[i][j]
                 + sum(gain[i][m] * cov_diff[m][l] * gain[j][l]
                       for m in range(4) for l in range(4))
                 for j in range(4)] for i in range(4)])
            smoothed_covs[k], _ = _floor_position_block(
                smoothed_covs[k])
        return [(s[0], s[1],
                 [[cov[0][0], cov[0][1]], [cov[1][0], cov[1][1]]])
                for s, cov in zip(smoothed_states, smoothed_covs)]

    @property
    def velocity(self) -> tuple[float, float]:
        """Current estimated velocity (m per step), (0, 0) before any update."""
        if self._state is None:
            return (0.0, 0.0)
        return (self._state[2], self._state[3])

    def reset(self) -> None:
        """Forget all state (e.g. when tracking a new device)."""
        self._state = None
        self._covariance = None
        self._history.clear()
