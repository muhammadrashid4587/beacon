"""Beacon health: spot dead or drifting beacons from a walked survey.

A beacon that has died, been moved, or had its transmit power changed
poisons every position estimate it feeds: its RSSI no longer matches
the path-loss model the engine trusts, so trilateration bends toward
(or away from) a beacon that is lying. This module checks each beacon
against a survey — a walk through the space with known positions and
the RSSI observed from each beacon at each stop, exactly the shape
``simulate_walk`` returns — and reports, per beacon:

- ``ok``: heard throughout the survey and its RSSI matches the model
  (mean residual within the drift threshold).
- ``drifting``: heard, but its mean RSSI residual is larger than the
  threshold — the beacon moved, its transmit power changed, or
  something is shielding it. Recalibrate it, or exclude it.
- ``dead``: never heard during the survey. It is off, removed, or
  out of range everywhere the survey went.

The residual for one observation is ``observed RSSI - predicted
RSSI`` at the known position, using the beacon's own calibrated
``rssi_at_1m`` and path-loss exponent. A healthy beacon's residuals
average near zero (the simulator's 2 dB noise averages out over a
walk); a beacon whose transmit power rose 10 dB averages about
+10 dB. The default drift threshold is 5 dB.

Typical use: survey the ward, check health, then build the engine
from the healthy beacons only, so a dead or drifting beacon cannot
drag estimates around:

    from beacon import assess_beacon_health, healthy_beacons
    from beacon.simulate import rectangular_walk, simulate_walk

    survey = simulate_walk(rectangular_walk(20, 12, 10), beacons,
                           noise_db=2.0, seed=7)
    report = assess_beacon_health(beacons, survey)
    engine = PositioningEngine(healthy_beacons(beacons, report))

``healthy_beacons`` keeps only the ``ok`` beacons. If fewer than
three beacons are healthy, ``PositioningEngine`` still raises — a
ward with two working beacons cannot be localised, and silently
continuing would be worse than failing loudly.
"""

import math
from dataclasses import dataclass

from .beacon import Beacon
from .rssi import distance_to_rssi

DEFAULT_DRIFT_THRESHOLD_DB = 5.0


@dataclass(frozen=True)
class BeaconHealth:
    """One beacon's health verdict from a survey.

    ``status`` is ``"ok"``, ``"drifting"``, or ``"dead"``.
    ``mean_residual_db`` is the mean of ``observed - predicted``
    RSSI over the steps where the beacon was heard, or ``None`` for
    a dead beacon (nothing was heard, so there is no residual).
    ``samples`` is how many survey steps heard the beacon;
    ``expected_samples`` is how many steps the survey had.
    """

    id: str
    status: str
    mean_residual_db: float | None
    samples: int
    expected_samples: int


def assess_beacon_health(
        beacons: list[Beacon],
        steps: list[dict],
        drift_threshold_db: float = DEFAULT_DRIFT_THRESHOLD_DB
        ) -> list[BeaconHealth]:
    """Assess each beacon's health from a walked survey.

    ``steps`` is a list of ``{"true": (x, y), "rssi": {id: rssi}}``
    dicts — exactly the shape ``simulate_walk`` returns. Only
    single-floor ``(x, y)`` survey points are supported: a
    multi-floor survey's slab attenuation would look like drift,
    so those steps are rejected rather than misread.
    """
    if not steps:
        raise ValueError("health check needs at least one survey step")
    if drift_threshold_db <= 0:
        raise ValueError("drift_threshold_db must be positive")

    residuals: dict[str, list[float]] = {b.id: [] for b in beacons}
    by_id = {b.id: b for b in beacons}
    for step in steps:
        true = step["true"]
        if len(true) != 2:
            raise ValueError(
                "health check needs single-floor (x, y) survey points; "
                "multi-floor surveys are not supported")
        x, y = true
        for beacon_id, rssi in step["rssi"].items():
            if beacon_id in residuals:
                beacon = by_id[beacon_id]
                distance = math.hypot(x - beacon.x, y - beacon.y)
                predicted = distance_to_rssi(
                    distance, beacon.rssi_at_1m, beacon.path_loss_exponent)
                residuals[beacon_id].append(rssi - predicted)

    report = []
    for beacon in beacons:
        heard = residuals[beacon.id]
        if not heard:
            report.append(BeaconHealth(
                id=beacon.id, status="dead", mean_residual_db=None,
                samples=0, expected_samples=len(steps)))
            continue
        mean_residual = sum(heard) / len(heard)
        status = ("drifting" if abs(mean_residual) > drift_threshold_db
                  else "ok")
        report.append(BeaconHealth(
            id=beacon.id, status=status,
            mean_residual_db=mean_residual,
            samples=len(heard), expected_samples=len(steps)))
    return report


def healthy_beacons(beacons: list[Beacon],
                    report: list[BeaconHealth]) -> list[Beacon]:
    """Return the beacons whose health status is ``"ok"``.

    Use the result to build a ``PositioningEngine`` that a dead or
    drifting beacon cannot poison. Raises ValueError when the report
    does not cover exactly the beacons given (a stale report from a
    different beacon set would silently drop the wrong beacons).
    """
    by_id = {entry.id: entry for entry in report}
    if set(by_id) != {b.id for b in beacons}:
        raise ValueError("health report does not match the beacons given")
    return [b for b in beacons if by_id[b.id].status == "ok"]
