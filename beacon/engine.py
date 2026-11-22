"""Positioning engine: RSSI observations -> smoothed (x, y).

One engine instance tracks one device. Feed it a dict of
``{beacon_id: rssi_dbm}`` per step and it returns the smoothed
position estimate: RSSI to distance (path-loss model), trilateration,
then a constant-velocity Kalman filter (a moving average is still
available as an opt-out via ``filter="moving_average"``).
"""

from .beacon import Beacon
from .filter import KalmanFilter, MovingAverageFilter
from .rssi import rssi_to_distance
from .trilateration import trilaterate


class PositioningEngine:
    """Estimate a device's position from RSSI observations."""

    def __init__(self, beacons: list[Beacon],
                 path_loss_exponent: float | None = None,
                 filter_window: int = 5,
                 filter: str = "kalman"):
        if len(beacons) < 3:
            raise ValueError("positioning needs at least 3 beacons")
        self.beacons = {b.id: b for b in beacons}
        # None means "use each beacon's own calibrated exponent".
        # Passing a value overrides every beacon (the pre-calibration
        # behaviour, when all beacons shared one exponent).
        self.path_loss_exponent = path_loss_exponent
        # The filter in use ("kalman" by default, "moving_average"
        # as an opt-out); kept so callers can inspect the default.
        self.filter = filter
        if filter == "moving_average":
            self._filter = MovingAverageFilter(window=filter_window)
        elif filter == "kalman":
            self._filter = KalmanFilter()
        else:
            raise ValueError(
                f"unknown filter: {filter!r} "
                "(expected 'moving_average' or 'kalman')")

    def estimate_raw(self, observations: dict[str, float]) -> tuple[float, float]:
        """Unfiltered position estimate from one set of observations."""
        positions: list[tuple[float, float]] = []
        distances: list[float] = []
        for beacon_id, rssi in observations.items():
            beacon = self.beacons.get(beacon_id)
            if beacon is None:
                raise ValueError(f"unknown beacon id: {beacon_id!r}")
            positions.append(beacon.position)
            exponent = (self.path_loss_exponent
                        if self.path_loss_exponent is not None
                        else beacon.path_loss_exponent)
            distances.append(rssi_to_distance(
                rssi, beacon.rssi_at_1m, exponent))
        return trilaterate(positions, distances)

    def estimate(self, observations: dict[str, float]) -> tuple[float, float]:
        """Smoothed position estimate from one set of observations."""
        return self._filter.update(self.estimate_raw(observations))

    def reset(self) -> None:
        """Forget filter history (e.g. when tracking a new device)."""
        self._filter.reset()
