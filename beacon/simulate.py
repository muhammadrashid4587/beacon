"""Simulator: generate noisy RSSI readings from a true path.

No BLE hardware is needed. Given beacons and a true path (a list of
(x, y) positions in metres), the simulator computes the true distance
to each beacon at every step, converts it to RSSI with the path-loss
model, and adds Gaussian noise (``noise_db`` standard deviation) to
model multipath and measurement error. Pass a ``seed`` for a
reproducible run.
"""

import math
import random

from .beacon import Beacon
from .rssi import distance_to_rssi


def simulate_walk(path: list[tuple[float, float]], beacons: list[Beacon],
                  noise_db: float = 2.0,
                  path_loss_exponent: float = 2.0,
                  seed: int | None = None) -> list[dict]:
    """Simulate RSSI observations along ``path``.

    Returns one dict per step: ``{"true": (x, y), "rssi": {id: rssi}}``.
    """
    rng = random.Random(seed)
    steps = []
    for x, y in path:
        observations: dict[str, float] = {}
        for beacon in beacons:
            d = math.hypot(x - beacon.x, y - beacon.y)
            rssi = distance_to_rssi(d, beacon.rssi_at_1m, path_loss_exponent)
            if noise_db:
                rssi += rng.gauss(0.0, noise_db)
            observations[beacon.id] = rssi
        steps.append({"true": (x, y), "rssi": observations})
    return steps


def simulate_multifloor_walk(path: list[tuple[float, float, int]],
                             beacons: list[Beacon],
                             noise_db: float = 2.0,
                             path_loss_exponent: float = 2.0,
                             floor_height_m: float = 3.0,
                             floor_attenuation_db: float = 15.0,
                             seed: int | None = None) -> list[dict]:
    """Simulate RSSI observations along a multi-floor ``path``.

    ``path`` is a list of ``(x, y, floor)`` points. The distance to a
    beacon is the 3D distance — the vertical gap is ``floor_height_m``
    per floor of separation — and a beacon on a different floor from
    the device loses a further ``floor_attenuation_db`` (the slab
    between floors absorbs signal, so a beacon heard through a floor
    looks much farther away than its 3D distance alone would say).
    That attenuation is what makes floor detection possible: without
    it, a beacon directly below the device would be heard as loudly
    as one a few metres away on the same floor.

    Returns one dict per step: ``{"true": (x, y, floor),
    "rssi": {id: rssi}}``.
    """
    rng = random.Random(seed)
    steps = []
    for x, y, floor in path:
        observations: dict[str, float] = {}
        for beacon in beacons:
            dz = floor_height_m * abs(floor - beacon.floor)
            d = math.hypot(x - beacon.x, y - beacon.y, dz)
            rssi = distance_to_rssi(d, beacon.rssi_at_1m,
                                    path_loss_exponent)
            if beacon.floor != floor:
                rssi -= floor_attenuation_db
            if noise_db:
                rssi += rng.gauss(0.0, noise_db)
            observations[beacon.id] = rssi
        steps.append({"true": (x, y, floor), "rssi": observations})
    return steps


def two_floor_walk(width: float, height: float, steps_per_side: int,
                   margin: float = 2.0) -> list[tuple[float, float, int]]:
    """A rectangular loop on floor 0, then the same loop on floor 1.

    The device walks the ward, changes floor (the stairwell step is
    the jump between the two loops — the path does not model the
    stairs themselves), and walks the same loop upstairs. Useful for
    testing floor detection and within-floor accuracy in one run.
    """
    loop = rectangular_walk(width, height, steps_per_side, margin)
    return [(x, y, 0) for x, y in loop] + [(x, y, 1) for x, y in loop]


def rectangular_walk(width: float, height: float, steps_per_side: int,
                     margin: float = 2.0) -> list[tuple[float, float]]:
    """A rectangular loop inside a room, inset ``margin`` from the walls.

    Returns the path as a list of (x, y) points, starting at the
    bottom-left corner and walking clockwise.
    """
    corners = [
        (margin, margin),
        (width - margin, margin),
        (width - margin, height - margin),
        (margin, height - margin),
        (margin, margin),
    ]
    path: list[tuple[float, float]] = []
    for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
        for i in range(steps_per_side):
            t = i / steps_per_side
            path.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return path
