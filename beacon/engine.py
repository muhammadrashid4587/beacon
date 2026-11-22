"""Positioning engine: RSSI observations -> smoothed (x, y).

One engine instance tracks one device. Feed it a dict of
``{beacon_id: rssi_dbm}`` per step and it returns the smoothed
position estimate: RSSI to distance (path-loss model), trilateration,
then a moving-average filter.
"""

from .beacon import Beacon
from .filter import MovingAverageFilter
from .rssi import rssi_to_distance
from .trilateration import trilaterate


class PositioningEngine:
    """Estimate a device's position from RSSI observations."""

    def __init__(self, beacons: list[Beacon],
                 path_loss_exponent: float = 2.0,
                 filter_window: int = 5):
        if len(beacons) < 3:
            raise ValueError("positioning needs at least 3 beacons")
        self.beacons = {b.id: b for b in beacons}
        self.path_loss_exponent = path_loss_exponent
        self._filter = MovingAverageFilter(window=filter_window)

    def estimate_raw(self, observations: dict[str, float]) -> tuple[float, float]:
        """Unfiltered position estimate from one set of observations."""
        positions: list[tuple[float, float]] = []
        distances: list[float] = []
        for beacon_id, rssi in observations.items():
            beacon = self.beacons.get(beacon_id)
            if beacon is None:
                raise ValueError(f"unknown beacon id: {beacon_id!r}")
            positions.append(beacon.position)
            distances.append(rssi_to_distance(
                rssi, beacon.rssi_at_1m, self.path_loss_exponent))
        return trilaterate(positions, distances)

    def estimate(self, observations: dict[str, float]) -> tuple[float, float]:
        """Smoothed position estimate from one set of observations."""
        return self._filter.update(self.estimate_raw(observations))

    def reset(self) -> None:
        """Forget filter history (e.g. when tracking a new device)."""
        self._filter.reset()
