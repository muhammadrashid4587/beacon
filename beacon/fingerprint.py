"""Radio-map fingerprinting: RSSI vectors per survey cell, k-NN locate.

Trilateration inverts RSSI into distances and then geometry; it falls
apart when the path-loss model is wrong (a mismatched exponent sends
the distance estimate, and the position with it, orders of magnitude
off). Fingerprinting sidesteps the model entirely: survey the space
once, recording the RSSI vector observed from each beacon at each
known grid cell, and locate a device by matching its live RSSI vector
against the map — the cell whose fingerprint is closest in RSSI
space is where the device most likely is.

The map stores one averaged RSSI vector per cell: repeated survey
passes over the same cell are averaged, which smooths noise the same
way a real survey would. Matching is k-nearest-neighbour in RSSI
space (root-mean-square distance over the beacon vector);
``locate`` returns the inverse-distance-weighted centroid of the
``k`` nearest cells, so ``k=1`` returns the best-matching cell
exactly and ``k=3`` (the default) interpolates between neighbouring
cells instead of snapping to the grid. A beacon missing from an
observation is simply left out of the comparison — the match uses
only the beacons actually heard, so a partially-heard vector still
locates instead of being dragged toward the cells farthest from the
unheard beacon.

Pure standard library, no dependencies. Typical use: build the map
from a surveyed grid, then locate live observations against it:

    from beacon import build_fingerprint_map, grid_points
    from beacon.simulate import simulate_walk

    cells = grid_points(20.0, 12.0, spacing=2.0)
    survey = simulate_walk(cells, beacons, noise_db=0.0, seed=7)
    radio_map = build_fingerprint_map(beacons, survey)
    x, y = radio_map.locate({"A": -62.1, "B": -71.4, "C": -68.0, "D": -59.8})
"""

import math


def grid_points(width: float, height: float, spacing: float,
                margin: float = 1.0) -> list[tuple[float, float]]:
    """Grid cells covering a ``width`` x ``height`` rectangle.

    Points run from ``margin`` to ``width - margin`` (and likewise
    for height) in steps of ``spacing``, inclusive of both ends, so a
    20 x 12 m ward at 2 m spacing with the default 1 m margin is a
    10 x 6 grid of 60 cells. The margin keeps cells off the walls —
    and off corner beacons, where the distance (and so the RSSI
    model) is undefined.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if margin < 0 or 2 * margin >= min(width, height):
        raise ValueError("margin must leave room for at least one cell")
    xs = _axis(margin, width - margin, spacing)
    ys = _axis(margin, height - margin, spacing)
    return [(x, y) for y in ys for x in xs]


def _axis(start: float, end: float, spacing: float) -> list[float]:
    points = []
    value = start
    while value < end:
        points.append(value)
        value += spacing
    points.append(end)
    return points


class FingerprintMap:
    """A radio map: one averaged RSSI vector per surveyed cell."""

    def __init__(self, beacon_ids: list[str]):
        if not beacon_ids:
            raise ValueError("fingerprinting needs at least one beacon")
        if len(set(beacon_ids)) != len(beacon_ids):
            raise ValueError("beacon ids must be unique")
        self.beacon_ids = list(beacon_ids)
        # cell -> (sums, counts): per-beacon RSSI sum and the number
        # of samples that actually heard that beacon, so a beacon
        # unheard at a cell is simply absent from its fingerprint
        # instead of being filled with a made-up value.
        self._cells: dict[tuple[float, float],
                          tuple[list[float], list[int]]] = {}

    def add(self, point: tuple[float, float],
            observations: dict[str, float]) -> None:
        """Add one survey sample at ``point`` (averaged with repeats).

        ``observations`` maps beacon id to RSSI in dBm. A beacon
        missing from the sample is not recorded for that sample, so
        every cell's fingerprint covers only the beacons heard there.
        """
        for beacon_id in observations:
            if beacon_id not in self.beacon_ids:
                raise ValueError(f"unknown beacon id: {beacon_id!r}")
        sums, counts = self._cells.get(
            point, ([0.0] * len(self.beacon_ids), [0] * len(self.beacon_ids)))
        for i, beacon_id in enumerate(self.beacon_ids):
            if beacon_id in observations:
                sums[i] += observations[beacon_id]
                counts[i] += 1
        self._cells[point] = (sums, counts)

    @classmethod
    def from_survey(cls, beacon_ids: list[str],
                    steps: list[dict]) -> "FingerprintMap":
        """Build a map from walked survey steps.

        ``steps`` is a list of ``{"true": (x, y), "rssi": {id: rssi}}``
        dicts — exactly the shape ``simulate_walk`` returns, so a
        simulated survey doubles as a test of the matcher.
        """
        radio_map = cls(beacon_ids)
        for step in steps:
            radio_map.add(step["true"], step["rssi"])
        return radio_map

    @property
    def cells(self) -> list[tuple[float, float]]:
        """The surveyed grid cells, in the order they were added."""
        return list(self._cells)

    def __len__(self) -> int:
        return len(self._cells)

    def fingerprint(self, point: tuple[float, float]) -> dict[str, float]:
        """The averaged RSSI vector recorded at ``point``.

        Only the beacons heard at that cell are included.
        """
        sums, counts = self._cells[point]
        return {beacon_id: sums[i] / counts[i]
                for i, beacon_id in enumerate(self.beacon_ids)
                if counts[i] > 0}

    def nearest(self, observations: dict[str, float], k: int = 3
                ) -> list[tuple[tuple[float, float], float]]:
        """The ``k`` nearest cells to ``observations`` in RSSI space.

        Returns ``[((x, y), distance_db), ...]`` sorted nearest first,
        where the distance is the root-mean-square difference between
        the observed RSSI vector and the cell's fingerprint, over the
        beacons heard in both — the RMS (rather than the raw
        Euclidean distance) keeps cells comparable when different
        cells were surveyed with different beacons heard.
        """
        if not self._cells:
            raise ValueError("fingerprint map is empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        for beacon_id in observations:
            if beacon_id not in self.beacon_ids:
                raise ValueError(f"unknown beacon id: {beacon_id!r}")
        scored = []
        for point, (sums, counts) in self._cells.items():
            squared = []
            for i, beacon_id in enumerate(self.beacon_ids):
                if counts[i] > 0 and beacon_id in observations:
                    squared.append(
                        (observations[beacon_id] - sums[i] / counts[i]) ** 2)
            if not squared:
                continue
            scored.append((point, math.sqrt(sum(squared) / len(squared))))
        if not scored:
            raise ValueError(
                "observations share no beacons with the fingerprint map")
        scored.sort(key=lambda item: item[1])
        return scored[:min(k, len(scored))]

    def locate(self, observations: dict[str, float], k: int = 3
               ) -> tuple[float, float]:
        """Estimate a position from one set of RSSI observations.

        Inverse-distance-weighted centroid of the ``k`` nearest
        cells in RSSI space. An exact fingerprint match (distance
        zero) returns that cell directly; ``k=1`` always returns the
        single best-matching cell.
        """
        neighbours = self.nearest(observations, k)
        if neighbours[0][1] == 0.0:
            return neighbours[0][0]
        total_weight = 0.0
        weighted_x = weighted_y = 0.0
        for (x, y), distance in neighbours:
            weight = 1.0 / distance
            total_weight += weight
            weighted_x += weight * x
            weighted_y += weight * y
        return (weighted_x / total_weight, weighted_y / total_weight)


def build_fingerprint_map(beacons, steps: list[dict]) -> FingerprintMap:
    """Build a radio map from a walked survey.

    ``beacons`` is the list of surveyed beacons and ``steps`` a list
    of ``{"true": (x, y), "rssi": {id: rssi}}`` dicts — exactly the
    shape ``simulate_walk`` returns. A convenience wrapper around
    ``FingerprintMap.from_survey`` matching ``calibrate_survey``.
    """
    return FingerprintMap.from_survey([b.id for b in beacons], steps)
