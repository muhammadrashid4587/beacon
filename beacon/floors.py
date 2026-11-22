"""Multi-floor positioning: which floor is the device on?

A building has one set of beacons per floor, all sharing the same
(x, y) plane. Before localising within a floor, the engine has to
decide which floor the device is on — and it has to do it from RSSI
alone, since a phone's barometer is not part of this engine.

Floor detection is a vote. Each heard beacon votes for its own
floor, weighted by closeness: the weight is ``1 / distance``, where
the distance is the path-loss estimate from the observed RSSI.
Beacons on the device's own floor are heard loud (short apparent
distance, heavy vote); beacons on an adjacent floor are heard
through the slab, so their apparent distance is inflated by the
floor attenuation and they vote weakly. The floor with the highest
total weight wins, and the confidence is the winner's share of the
total weight — 1.0 when every heard beacon is on one floor, near
0.5 when two floors are heard equally loudly and the vote is a
coin toss.

The vote is per-observation, so a single noisy step can disagree
with its neighbours. The engine therefore only *switches* floors
when the winning floor's confidence clears a threshold (0.6 by
default); a low-confidence vote leaves the current floor in place.
That hysteresis is what stops the reported floor flickering when
the device stands near a stairwell and hears both floors at similar
strength.

Honest edge cases, handled rather than papered over:

- Beacons heard from an adjacent floor: they vote, but weakly,
  because the slab attenuation inflates their apparent distance.
- A floor with no beacons heard: it scores zero and cannot win,
  so the vote only ever picks a floor that was actually heard.
- Fewer than three beacons heard on the winning floor:
  trilateration needs three, so the engine falls back to all heard
  beacons (their (x, y) geometry is still valid; the cross-floor
  distances are inflated, so the estimate degrades gracefully
  instead of raising).
- A floor change resets the smoothing filter: the device has moved
  to a different plane, so the filter's velocity state from the old
  floor would drag the new floor's estimates backwards.
"""

from dataclasses import dataclass

from .beacon import Beacon
from .rssi import rssi_to_distance

# The winning floor must clear this confidence before the engine
# switches to it; below it, a disagreeing vote is treated as noise
# and the current floor is kept.
DEFAULT_FLOOR_SWITCH_CONFIDENCE = 0.6


@dataclass(frozen=True)
class FloorEstimate:
    """The result of one floor vote: the floor, its confidence, and
    every floor's score (the summed beacon weights)."""

    floor: int
    confidence: float
    scores: dict[int, float]


@dataclass(frozen=True)
class Position:
    """A localised device: (x, y) in metres on ``floor``, with the
    confidence of the floor vote that put it there."""

    x: float
    y: float
    floor: int
    floor_confidence: float


def _beacon_map(beacons) -> dict[str, Beacon]:
    if isinstance(beacons, dict):
        return beacons
    return {b.id: b for b in beacons}


def estimate_floor(observations: dict[str, float],
                   beacons) -> FloorEstimate:
    """Vote on which floor the device is on, from one observation set.

    ``observations`` maps beacon id to RSSI in dBm; ``beacons`` is the
    engine's beacon dict (or a list of Beacons). Each heard beacon
    votes for its floor with weight ``1 / distance`` from the
    path-loss model, so near beacons count more than far ones and
    cross-floor beacons — heard through the slab, so apparently far
    — count less. Returns the winning floor, its share of the total
    weight as confidence, and every floor's score.

    Raises ValueError on an unknown beacon id or an empty
    observation set.
    """
    beacon_map = _beacon_map(beacons)
    if not observations:
        raise ValueError("floor estimation needs at least one observation")
    scores: dict[int, float] = {}
    for beacon_id, rssi in observations.items():
        beacon = beacon_map.get(beacon_id)
        if beacon is None:
            raise ValueError(f"unknown beacon id: {beacon_id!r}")
        distance = rssi_to_distance(rssi, beacon.rssi_at_1m,
                                    beacon.path_loss_exponent)
        weight = 1.0 / max(distance, 1.0)
        scores[beacon.floor] = scores.get(beacon.floor, 0.0) + weight
    total = sum(scores.values())
    floor = max(scores, key=lambda f: scores[f])
    return FloorEstimate(floor=floor,
                         confidence=scores[floor] / total,
                         scores=dict(scores))
