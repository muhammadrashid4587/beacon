"""Positioning engine: RSSI observations -> smoothed (x, y).

One engine instance tracks one device. Feed it a dict of
``{beacon_id: rssi_dbm}`` per step and it returns the smoothed
position estimate: RSSI to distance (path-loss model), trilateration,
then a constant-velocity Kalman filter (a moving average is still
available as an opt-out via ``filter="moving_average"``).

Multi-floor buildings: when the beacons span more than one floor,
the engine first votes on which floor the device is on (see
``beacon.floors``) and then localises within that floor, using only
that floor's beacons when at least three were heard. The floor only
changes when the winning floor's confidence clears
``floor_switch_confidence`` (0.6 by default); a low-confidence vote
keeps the current floor, so the reported floor does not flicker near
a stairwell. A floor change resets the smoothing filter — the device
has moved to a different plane, and the old floor's velocity state
would drag the new floor's estimates backwards. ``estimate`` still
returns ``(x, y)`` as before; ``estimate_with_floor`` returns a
``Position`` carrying the floor and its confidence.
"""

from .beacon import Beacon
from .filter import KalmanFilter, MovingAverageFilter
from .floors import (DEFAULT_FLOOR_SWITCH_CONFIDENCE, FloorEstimate,
                     Position, estimate_floor as _vote_floor)
from .rssi import rssi_to_distance
from .trilateration import trilaterate
from .uncertainty import PositionUncertainty, estimate_uncertainty


class PositioningEngine:
    """Estimate a device's position from RSSI observations."""

    def __init__(self, beacons: list[Beacon],
                 path_loss_exponent: float | None = None,
                 filter_window: int = 5,
                 filter: str = "kalman",
                 floor_switch_confidence: float =
                 DEFAULT_FLOOR_SWITCH_CONFIDENCE):
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
        if not 0.0 < floor_switch_confidence <= 1.0:
            raise ValueError(
                "floor_switch_confidence must be in (0, 1]")
        self.floor_switch_confidence = floor_switch_confidence
        # The floor the device is currently believed to be on, or
        # None before the first observation. A single-floor building
        # never leaves its only floor.
        self._current_floor: int | None = None

    @property
    def current_floor(self) -> int | None:
        """The floor the device is currently believed to be on
        (None before the first observation)."""
        return self._current_floor

    def _distances(self, observations: dict[str, float],
                   beacons: list[Beacon]) -> tuple[list, list]:
        """Beacon positions and path-loss distances for ``observations``,
        restricted to ``beacons``. Raises on an unknown beacon id."""
        positions: list[tuple[float, float]] = []
        distances: list[float] = []
        for beacon in beacons:
            if beacon.id not in observations:
                continue
            positions.append(beacon.position)
            exponent = (self.path_loss_exponent
                        if self.path_loss_exponent is not None
                        else beacon.path_loss_exponent)
            distances.append(rssi_to_distance(
                observations[beacon.id], beacon.rssi_at_1m, exponent))
        return positions, distances

    def estimate_raw(self, observations: dict[str, float]) -> tuple[float, float]:
        """Unfiltered position estimate from one set of observations."""
        beacons = [self.beacons[beacon_id] for beacon_id in observations
                   if beacon_id in self.beacons]
        for beacon_id in observations:
            if beacon_id not in self.beacons:
                raise ValueError(f"unknown beacon id: {beacon_id!r}")
        positions, distances = self._distances(observations, beacons)
        return trilaterate(positions, distances)

    def estimate_with_uncertainty(
            self, observations: dict[str, float],
            sigma_rssi_db: float = 2.0
            ) -> tuple[float, float, PositionUncertainty]:
        """Raw position estimate plus its 95% confidence ellipse.

        Like ``estimate_raw`` this is unfiltered — one observation
        set, all heard beacons — so the ellipse describes the
        geometry and RSSI noise of this step, not the filtered
        track. ``sigma_rssi_db`` is the assumed RSSI noise in dB
        (default 2.0, matching the simulator's default noise).
        """
        beacons = [self.beacons[beacon_id] for beacon_id in observations
                   if beacon_id in self.beacons]
        for beacon_id in observations:
            if beacon_id not in self.beacons:
                raise ValueError(f"unknown beacon id: {beacon_id!r}")
        positions, distances = self._distances(observations, beacons)
        exponents = [
            (self.path_loss_exponent
             if self.path_loss_exponent is not None
             else beacon.path_loss_exponent)
            for beacon in beacons]
        x, y = trilaterate(positions, distances)
        return (x, y, estimate_uncertainty(
            positions, (x, y), distances, exponents, sigma_rssi_db))

    def estimate_floor(self, observations: dict[str, float]) -> FloorEstimate:
        """Vote on which floor the device is on, from one observation
        set. Stateless: this is the raw vote, before the hysteresis
        in ``estimate_with_floor`` decides whether to switch."""
        return _vote_floor(observations, self.beacons)

    def _decide_floor(self, vote: FloorEstimate) -> int:
        """Apply hysteresis to a floor vote: adopt the winner on the
        first observation, switch to a different winner only when
        its confidence clears the threshold, and otherwise keep the
        current floor. Returns the floor to localise on."""
        if self._current_floor is None:
            self._current_floor = vote.floor
        elif (vote.floor != self._current_floor
              and vote.confidence >= self.floor_switch_confidence):
            self._current_floor = vote.floor
            # New plane, new filter state: the old floor's velocity
            # would drag the new floor's estimates backwards.
            self._filter.reset()
        return self._current_floor

    def estimate_with_floor(self, observations: dict[str, float]) -> Position:
        """Smoothed position estimate with the floor attached.

        Votes on the floor, localises within it using only that
        floor's beacons when at least three were heard, and falls
        back to all heard beacons when fewer than three were heard
        on the floor (their (x, y) geometry is still valid; the
        cross-floor distances are inflated by the slab, so the
        estimate degrades gracefully instead of raising).
        """
        vote = self.estimate_floor(observations)
        floor = self._decide_floor(vote)
        on_floor = [b for b in self.beacons.values()
                    if b.floor == floor and b.id in observations]
        if len(on_floor) >= 3:
            positions, distances = self._distances(observations, on_floor)
        else:
            # Too few beacons heard on this floor to trilaterate:
            # use every heard beacon rather than failing.
            positions, distances = self._distances(
                observations, [self.beacons[i] for i in observations])
        x, y = self._filter.update(trilaterate(positions, distances))
        confidence = (vote.scores.get(floor, 0.0)
                      / sum(vote.scores.values()))
        return Position(x=x, y=y, floor=floor,
                        floor_confidence=confidence)

    def estimate(self, observations: dict[str, float]) -> tuple[float, float]:
        """Smoothed position estimate from one set of observations.

        In a multi-floor building this first votes on the floor and
        localises within it; in a single-floor building the vote is
        trivial and the result is identical to before.
        """
        position = self.estimate_with_floor(observations)
        return (position.x, position.y)

    def smooth_walk(self, observations: list[dict[str, float]]
                    ) -> list[tuple[float, float]]:
        """Rauch-Tung-Striebel smoothed positions for a recorded walk.

        Runs the whole walk through the live (causal) filter, then
        revises every past position with the observations that came
        after it, so the returned path is steadier and closer to the
        true path than the step-by-step estimates. Returns one
        (x, y) per observation, in order; [] for an empty walk. The
        engine is reset first, so a previous walk does not leak into
        this one. Requires the Kalman filter: the moving-average
        opt-out keeps no forward-pass history to smooth over. A walk
        that changes floor is rejected — split it per floor instead,
        since a floor change resets the filter and would leave only
        the final segment's history to smooth.
        """
        if self.filter != "kalman":
            raise ValueError(
                "smooth_walk needs the Kalman filter "
                f"(engine uses {self.filter!r})")
        self.reset()
        floors: set[int] = set()
        for obs in observations:
            floors.add(self.estimate_with_floor(obs).floor)
        if len(floors) > 1:
            raise ValueError(
                "smooth_walk does not support a walk that changes "
                "floor; split the walk per floor")
        return self._filter.smooth()

    def reset(self) -> None:
        """Forget filter history (e.g. when tracking a new device)."""
        self._filter.reset()
        self._current_floor = None
