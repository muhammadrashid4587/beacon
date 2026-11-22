"""Smoothing filter for position estimates.

RSSI is noisy, so raw trilateration output jitters from step to step
even when the device is standing still. A moving average over the
last ``window`` estimates trades a little lag for a much steadier
position. The window is small (default 5) so a walking device is
still tracked promptly.
"""

from collections import deque


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
