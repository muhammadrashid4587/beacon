"""Per-beacon calibration from a walked survey.

The positioning engine is only as accurate as its path-loss model:
each beacon needs its own ``rssi_at_1m`` and path-loss exponent ``n``
fitted to the real space (walls, people, metal beds all shift RSSI).
This module fits those two constants from survey data — a walk
through the space with known positions and the RSSI observed from
each beacon at each stop.

The model is linear in ``log10(d)``:

    RSSI(d) = rssi_at_1m - 10 * n * log10(d)

so fitting is ordinary least squares on ``x = log10(d)``:
``RSSI = a + b*x`` with ``a = rssi_at_1m`` and ``n = -b / 10``.
Pure standard library, no dependencies.

Typical use: walk the survey path once, record the RSSI seen from
each beacon at each known stop (the simulator produces steps in
exactly this shape), then calibrate:

    from beacon import calibrate_survey

    calibrated = calibrate_survey(beacons, survey_steps)
    engine = PositioningEngine(calibrated)
"""

import dataclasses
import math

from .beacon import Beacon

MIN_DISTINCT_DISTANCES = 2


def fit_path_loss(samples: list[tuple[float, float]]) -> tuple[float, float]:
    """Fit (rssi_at_1m, path_loss_exponent) to (distance_m, rssi_dbm) pairs.

    Ordinary least squares on RSSI against log10(distance). Returns
    ``(rssi_at_1m, n)``. Raises ValueError when the samples cannot
    determine a fit: fewer than two samples, fewer than two distinct
    distances, or RSSI that does not decrease with distance (which
    would give a non-positive path-loss exponent).
    """
    if len(samples) < 2:
        raise ValueError("calibration needs at least 2 samples")
    xs = []
    ys = []
    for distance_m, rssi_dbm in samples:
        if distance_m <= 0:
            raise ValueError("distance must be positive")
        xs.append(math.log10(max(distance_m, 1e-9)))
        ys.append(rssi_dbm)
    if len(set(xs)) < MIN_DISTINCT_DISTANCES:
        raise ValueError("calibration needs at least 2 distinct distances")

    n_samples = len(xs)
    mean_x = sum(xs) / n_samples
    mean_y = sum(ys) / n_samples
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx == 0:
        raise ValueError("calibration needs at least 2 distinct distances")
    slope = sxy / sxx
    if slope >= 0:
        raise ValueError(
            "RSSI does not decrease with distance in these samples; "
            "cannot fit a path-loss exponent"
        )
    intercept = mean_y - slope * mean_x
    return intercept, -slope / 10.0


def calibrate_beacon(beacon: Beacon,
                     samples: list[tuple[tuple[float, float], float]]
                     ) -> Beacon:
    """Return a new beacon with path-loss constants fitted to a survey.

    ``samples`` is a list of ``((x, y), rssi_dbm)`` pairs: the known
    position the RSSI was observed from, and the observed RSSI. The
    original beacon is unchanged (it is a frozen dataclass).
    """
    distances = [
        (math.hypot(x - beacon.x, y - beacon.y), rssi)
        for (x, y), rssi in samples
    ]
    rssi_at_1m, exponent = fit_path_loss(distances)
    return dataclasses.replace(beacon, rssi_at_1m=rssi_at_1m,
                               path_loss_exponent=exponent)


def calibrate_survey(beacons: list[Beacon],
                     steps: list[dict]) -> list[Beacon]:
    """Calibrate every beacon from a walked survey.

    ``steps`` is a list of ``{"true": (x, y), "rssi": {id: rssi}}``
    dicts — exactly the shape ``simulate_walk`` returns, so a
    simulated survey doubles as a test of the fit. A beacon with too
    little survey data (fewer than two distinct distances) is left
    unchanged rather than failing the whole survey.
    """
    by_id: dict[str, list[tuple[tuple[float, float], float]]] = {
        b.id: [] for b in beacons
    }
    for step in steps:
        position = step["true"]
        for beacon_id, rssi in step["rssi"].items():
            if beacon_id in by_id:
                by_id[beacon_id].append((position, rssi))
    calibrated = []
    for beacon in beacons:
        try:
            calibrated.append(calibrate_beacon(beacon, by_id[beacon.id]))
        except ValueError:
            calibrated.append(beacon)
    return calibrated
