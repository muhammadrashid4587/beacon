"""Live calibration-drift alarm: catch a beacon going bad while running.

``assess_beacon_health`` is a one-shot check over a finished survey:
walk the ward, then find out a beacon drifted. A ward that is already
running cannot afford to wait for the next re-walk — a beacon whose
transmit power changed, that was moved, or that something is
shielding poisons every estimate it feeds from the moment it goes
bad. This module watches the live stream instead: feed it each
observation as it arrives, together with the device's position (from
a reference tag, a known waypoint, or any ground truth the deployment
has), and it keeps a rolling window of RSSI residuals per beacon —
``observed - predicted`` under that beacon's calibrated constants —
and raises an alarm when a beacon's rolling mean residual crosses
the drift threshold, the same 5 dB default the survey check uses.

An alarm clears on its own when the rolling mean comes back inside
the threshold, so a beacon that recovers (a trolley moved away, a
door closed) stops alarming without anyone resetting the monitor.
A single bad reading never alarms: a beacon needs ``min_samples``
readings in its window before it can alarm, so noise spikes pass
unnoticed and only a sustained shift trips it.

Typical use: run the monitor alongside the engine, feeding it the
same observations plus a position. In a deployment the position
would come from a reference tag at a known spot; in tests the
simulator's true path plays that role:

    from beacon import DriftMonitor
    from beacon.simulate import rectangular_walk, simulate_walk

    monitor = DriftMonitor(beacons)
    for step in simulate_walk(rectangular_walk(20, 12, 10), beacons,
                              noise_db=2.0, seed=7):
        monitor.feed(step)
    monitor.alarms()   # [] while every beacon matches its calibration
"""

import math
from collections import deque
from dataclasses import dataclass

from .beacon import Beacon
from .health import DEFAULT_DRIFT_THRESHOLD_DB
from .rssi import distance_to_rssi

DEFAULT_WINDOW = 20
DEFAULT_MIN_SAMPLES = 5


@dataclass(frozen=True)
class DriftAlarm:
    """A beacon whose RSSI baseline has shifted.

    ``id`` is the beacon, ``mean_residual_db`` is the mean of
    ``observed - predicted`` RSSI over the rolling window (positive
    means the beacon reads louder than its calibration says it
    should — power up, or moved closer), and ``samples`` is how many
    readings are in the window.
    """

    id: str
    mean_residual_db: float
    samples: int


class DriftMonitor:
    """Watch a stream of observations for beacons drifting off calibration.

    ``window`` is how many recent readings per beacon the rolling
    mean covers; ``drift_threshold_db`` is the mean residual that
    trips an alarm (5 dB by default, matching the survey health
    check); ``min_samples`` is how many readings a beacon needs in
    its window before it can alarm at all.
    """

    def __init__(self, beacons: list[Beacon],
                 window: int = DEFAULT_WINDOW,
                 drift_threshold_db: float = DEFAULT_DRIFT_THRESHOLD_DB,
                 min_samples: int = DEFAULT_MIN_SAMPLES):
        if not beacons:
            raise ValueError("drift monitor needs at least one beacon")
        if window < 1:
            raise ValueError("window must be at least 1")
        if drift_threshold_db <= 0:
            raise ValueError("drift_threshold_db must be positive")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        if min_samples > window:
            raise ValueError("min_samples cannot exceed window")
        self._beacons = {b.id: b for b in beacons}
        self._window = window
        self._threshold = drift_threshold_db
        self._min_samples = min_samples
        self._residuals: dict[str, deque[float]] = {
            b.id: deque(maxlen=window) for b in beacons}

    def observe(self, position: tuple[float, float],
                rssi: dict[str, float]) -> None:
        """Feed one observation: the device's position and the RSSI heard.

        ``position`` is a single-floor ``(x, y)``; multi-floor
        positions are rejected, since the slab attenuation would look
        like drift. Beacons not in the monitor's beacon set are
        ignored rather than failing the stream.
        """
        if len(position) != 2:
            raise ValueError(
                "drift monitor needs single-floor (x, y) positions; "
                "multi-floor positions are not supported")
        x, y = position
        for beacon_id, observed in rssi.items():
            beacon = self._beacons.get(beacon_id)
            if beacon is None:
                continue
            distance = math.hypot(x - beacon.x, y - beacon.y)
            predicted = distance_to_rssi(
                distance, beacon.rssi_at_1m, beacon.path_loss_exponent)
            self._residuals[beacon_id].append(observed - predicted)

    def feed(self, step: dict) -> None:
        """Feed one simulator step: ``{"true": (x, y), "rssi": {...}}``."""
        self.observe(step["true"], step["rssi"])

    def status(self, beacon_id: str) -> DriftAlarm | None:
        """The rolling residual for one beacon, or ``None`` if unheard."""
        residuals = self._residuals.get(beacon_id)
        if residuals is None:
            raise ValueError(f"unknown beacon: {beacon_id!r}")
        if not residuals:
            return None
        return DriftAlarm(
            id=beacon_id,
            mean_residual_db=sum(residuals) / len(residuals),
            samples=len(residuals))

    def alarm(self, beacon_id: str) -> DriftAlarm | None:
        """The active alarm for one beacon, or ``None`` if it is not alarming.

        A beacon alarms once it has at least ``min_samples`` readings
        in its window and the rolling mean residual is beyond the
        drift threshold in either direction.
        """
        state = self.status(beacon_id)
        if state is None or state.samples < self._min_samples:
            return None
        if abs(state.mean_residual_db) > self._threshold:
            return state
        return None

    def alarms(self) -> list[DriftAlarm]:
        """Every beacon currently alarming, in beacon order."""
        return [a for a in (self.alarm(b.id) for b in self._beacons.values())
                if a is not None]

    def is_alarming(self, beacon_id: str) -> bool:
        """Whether the beacon is currently alarming."""
        return self.alarm(beacon_id) is not None
