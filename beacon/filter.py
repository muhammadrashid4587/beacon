"""Smoothing filters for position estimates.

RSSI is noisy, so raw trilateration output jitters from step to step
even when the device is standing still. Two filters are provided:

- ``MovingAverageFilter``: a moving average over the last ``window``
  estimates. It trades a little lag for a much steadier position.
- ``KalmanFilter``: a constant-velocity Kalman filter. It models the
  device as moving at a steady velocity and weighs each new estimate
  against that prediction, so it tracks a walking device with less
  lag than the moving average and still smooths a stationary one.
"""

from collections import deque


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Matrix product of two small dense matrices."""
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def _transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*a)]


def _matvec(a: list[list[float]], v: list[float]) -> list[float]:
    return [sum(row[k] * v[k] for k in range(len(v))) for row in a]


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
    follow turns faster, smaller values smooth more. Defaults are
    tuned for a walking device with ~1 m measurement noise.
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

    def update(self, position: tuple[float, float]) -> tuple[float, float]:
        """Feed a new estimate, return the filtered position."""
        if self._state is None:
            # First observation: start here, at rest, with a wide
            # velocity covariance so the filter can learn the speed.
            self._state = [position[0], position[1], 0.0, 0.0]
            self._covariance = [[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 100.0, 0.0],
                                [0.0, 0.0, 0.0, 100.0]]
            return position
        state = self._state
        cov = self._covariance
        # Predict.
        state = _matvec(self._F, state)
        cov = _matmul(_matmul(self._F, cov), _transpose(self._F))
        cov = [[cov[i][j] + self._Q[i][j] for j in range(4)]
               for i in range(4)]
        # Update: innovation, its covariance, Kalman gain.
        hx = _matvec(self._H, state)
        innovation = [position[0] - hx[0], position[1] - hx[1]]
        s = _matmul(_matmul(self._H, cov), _transpose(self._H))
        s = [[s[i][j] + self._R[i][j] for j in range(2)] for i in range(2)]
        det = s[0][0] * s[1][1] - s[0][1] * s[1][0]
        if det == 0.0:
            raise ValueError("singular innovation covariance")
        s_inv = [[s[1][1] / det, -s[0][1] / det],
                 [-s[1][0] / det, s[0][0] / det]]
        gain = _matmul(_matmul(cov, _transpose(self._H)), s_inv)
        state = [state[i] + gain[i][0] * innovation[0]
                 + gain[i][1] * innovation[1] for i in range(4)]
        # P = (I - K H) P.
        kh = _matmul(gain, self._H)
        cov = _matmul(
            [[(1.0 if i == j else 0.0) - kh[i][j] for j in range(4)]
             for i in range(4)], cov)
        self._state = state
        self._covariance = cov
        return (state[0], state[1])

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
