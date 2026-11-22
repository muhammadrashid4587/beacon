"""Fixed BLE beacons with known positions.

A beacon is a fixed transmitter mounted at a known (x, y) position in
the building, in metres. ``rssi_at_1m`` is the received signal strength
a device would observe 1 metre away from this beacon, in dBm. It is a
calibration constant: measure it once per beacon model in the real
space, or use the default (-59 dBm, a typical BLE value) for the
simulator.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Beacon:
    """A fixed BLE beacon at a known position."""

    id: str
    x: float
    y: float
    rssi_at_1m: float = -59.0
    path_loss_exponent: float = 2.0

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)
