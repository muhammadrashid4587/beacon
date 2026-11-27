"""Live calibration-drift alarm and drift classification.

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

An alarm says *that* a beacon drifted, not *why*. The cause matters
for what happens next — a dying battery means send someone with a
replacement cell, a moved beacon means re-survey its position, a
noisy beacon means check the radio — and the three leave different
signatures in the residuals, so ``classify`` names the cause:

- **battery-sag**: the beacon reads weaker over time at the *same*
  spot. ``BatteryModel`` loses RSSI with the square of the charge
  used and adds noise as the cell drains, so a sagging beacon's
  residual falls steadily while the device revisits the same
  positions. The test is the revisit delta: pair each recent
  reading with the closest earlier reading at (nearly) the same
  position, at least ``revisit_gap`` readings earlier, and ask
  whether the residual at that spot has fallen — a move cannot
  change what a beacon reads like at a fixed spot, a draining
  battery can. The beacon must also read weaker overall (mean
  residual below -1.5 dB); a sag only ever weakens.
- **moved**: the residual is stable at the same spot (the revisit
  delta is near zero) but wrong by a position-dependent amount —
  the beacon reads louder near its new home and quieter far from
  it, so the window's residuals spread out (std above 2.5 dB)
  around a shifted mean (beyond 1.5 dB). A beacon moved *towards*
  the walked area reads louder on average, which is what the
  no-revisit fallback looks for; a beacon moved away reads like a
  steady offset and is honestly called ``uncertain`` there.
- **noisy**: the residual spread is more than twice the expected
  RSSI noise (``noise_db``, 2 dB by default, so above 4 dB) with
  no coherent time trend — the beacon is still roughly calibrated
  on average, but its readings have gone unreliable.
- **uncertain**: the beacon is off calibration — its mean residual
  is beyond the alarm threshold, or more than three standard
  errors from zero (and at least 1.5 dB) — but no signature fits.
  A steady offset is the classic case: a beacon whose battery
  sagged and then stabilised looks exactly like a beacon that was
  miscalibrated, and no statistic in the window can tell them
  apart. The classifier says so instead of guessing.
- **healthy**: nothing is off — the mean residual is inside noise
  and no signature fires.

The revisit test is what separates a sag from a move: within a
single pass, time and position are confounded (the device walks
from one side of the ward to the other as the battery drains), so
a declining residual could be either. Walk the loop twice and the
two come apart — the same spot, an hour later, reads weaker only
if the battery is draining. Without revisits the classifier falls
back to the within-window trend (a strong monotonic decline, or a
second half of the window much weaker than the first), which
catches a fast sag most of the time but cannot rule a move out;
the verdict's reason says which evidence was used.

On a multi-floor walk the rolling window is per floor, not one
window per beacon: the classifier's evidence is the window, and a
single window spanning both floors mixes the beacon's own-floor
readings — where a move's position-dependent signature lives —
with cross-floor readings, where the slab compresses it, so the
verdict follows whichever floor the window happens to cover
instead of the beacon. Each beacon keeps one rolling window per
floor it was heard on, and ``classify`` — and the alarm — read the
window for the beacon's own floor by default, the floor the beacon
is mounted on, so the verdict always rests on the beacon's own
geometry. ``classify(id, floor=...)`` reads another floor's window
instead, for inspection, and the classification carries the floor
it was taken on. A beacon never heard on its own floor is
``uncertain`` — no readings to classify — not healthy.

Measured behaviour (simulator, 2 dB RSSI noise, two-loop ward
walk, seeds 1, 2, 3, 7, 11 — see the README for the full table):
a slowly draining beacon (0.3 per hour, 120 s steps) is called
battery-sag on 5/5 seeds, a beacon moved 7 m across the ward is
called moved on 4/5 (seed 11 comes back uncertain — the same
noise fluke that fakes a decline also blocks the moved call), a
beacon with 5 dB of extra noise is called noisy on 5/5, and a
steady -6 dB offset is called uncertain on 4/5 and misread as
battery-sag on seed 11, where the noise itself drifts down over
the walk. A move under about 2 m is invisible at 2 dB noise and
is called healthy or uncertain, which is a limit of the data, not
a bug in the classifier.
"""

import math
from collections import deque
from dataclasses import dataclass

from .beacon import Beacon
from .health import DEFAULT_DRIFT_THRESHOLD_DB
from .rssi import distance_to_rssi

DEFAULT_WINDOW = 20
DEFAULT_MIN_SAMPLES = 5

# Expected RSSI noise (stddev, dB) — the deployment's nominal noise,
# matching the simulator's default. The noisy verdict fires when the
# window's residual spread is more than twice this.
DEFAULT_NOISE_DB = 2.0

# How many readings per beacon the classifier keeps for revisit
# matching — long enough to cover two loops of the default walk.
DEFAULT_HISTORY = 500

# The floor model the monitor uses for multi-floor positions —
# the same numbers the simulator uses: 3 m of vertical separation
# per floor and 15 dB of slab attenuation for a beacon heard
# through a floor. A beacon on another floor reads weaker by the
# slab and farther by the vertical gap, and the monitor's predicted
# RSSI accounts for both, so a healthy cross-floor reading does
# not look like drift.
DEFAULT_FLOOR_HEIGHT_M = 3.0
DEFAULT_FLOOR_ATTENUATION_DB = 15.0

# A revisit pair needs this many readings between the two visits,
# so two readings from the same pass never count as a revisit.
DEFAULT_REVISIT_GAP = 10

# Two readings count as the same spot when they are this close.
DEFAULT_REVISIT_RADIUS_M = 1.0

# A beacon reads weaker overall than this before battery-sag is
# even considered — a sag only ever weakens, so a beacon reading
# louder on average cannot be sagging.
SAG_MEAN_MAX_DB = -1.5

# The revisit delta must be at least this negative, and at least
# this many standard errors from zero, to call a decline real.
SAG_DELTA_MAX_DB = -1.0
SAG_DELTA_MIN_T = -1.5

# Without revisits, a decline this steep (dB per reading) or this
# large between the window's halves counts as a sag.
SAG_SLOPE_MAX_DB_PER_STEP = -0.15
SAG_HALF_DELTA_MAX_DB = -1.5

# A moved beacon's residuals spread out (geometry wrong in
# different directions) around a shifted mean.
MOVED_STD_MIN_DB = 2.5
MOVED_MEAN_MIN_DB = 1.5

# A beacon is "off calibration but unclassifiable" when its mean
# residual is this far from zero in standard errors (and at least
# this many dB) — the noise floor for calling something wrong
# without being able to name it.
UNCERTAIN_MIN_DB = 1.5
UNCERTAIN_MIN_T = 3.0

VERDICTS = ("healthy", "battery-sag", "moved", "noisy", "uncertain")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _stdev(values: list[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of ``ys`` on ``xs`` (0.0 if degenerate)."""
    mx, my = _mean(xs), _mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def _correlation(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation of ``xs`` and ``ys`` (0.0 if degenerate)."""
    mx, my = _mean(xs), _mean(ys)
    den = math.sqrt(sum((x - mx) ** 2 for x in xs)
                    * sum((y - my) ** 2 for y in ys))
    if den == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


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

    def to_dict(self) -> dict:
        """JSON-friendly form, for the stream output and dashboards."""
        return {
            "id": self.id,
            "mean_residual_db": self.mean_residual_db,
            "samples": self.samples,
        }


@dataclass(frozen=True)
class DriftClassification:
    """What kind of drift a beacon has, and the evidence for it.

    ``verdict`` is one of ``"healthy"``, ``"battery-sag"``,
    ``"moved"``, ``"noisy"`` or ``"uncertain"`` — see the module
    docstring for what each signature looks like. ``floor`` is the
    floor whose window the verdict rests on — the beacon's own
    floor by default, or the floor asked for. ``mean_residual_db``
    and ``residual_std_db`` are the rolling window's mean and
    spread; ``time_slope_db_per_step`` and ``time_correlation``
    are the window's residual trend over time;
    ``revisit_delta_db`` is the mean change in residual at the same
    spot between visits (None when the device never revisited a
    spot — under five pairs — so the time trend is the only
    evidence); ``revisit_pairs`` is how many matched pairs that
    delta rests on; ``samples`` is the window size; ``reason`` says
    in words why this verdict was reached.
    """

    id: str
    floor: int
    verdict: str
    mean_residual_db: float
    residual_std_db: float
    time_slope_db_per_step: float
    time_correlation: float
    revisit_delta_db: float | None
    revisit_pairs: int
    samples: int
    reason: str

    def to_dict(self) -> dict:
        """JSON-friendly form, for the stream output, the exported
        track rows, and dashboards. ``revisit_delta_db`` is null
        when the device never revisited a spot, so no delta was
        measured."""
        return {
            "id": self.id,
            "floor": self.floor,
            "verdict": self.verdict,
            "mean_residual_db": self.mean_residual_db,
            "residual_std_db": self.residual_std_db,
            "time_slope_db_per_step": self.time_slope_db_per_step,
            "time_correlation": self.time_correlation,
            "revisit_delta_db": self.revisit_delta_db,
            "revisit_pairs": self.revisit_pairs,
            "samples": self.samples,
            "reason": self.reason,
        }


class DriftMonitor:
    """Watch a stream of observations for beacons drifting off calibration.

    ``window`` is how many recent readings per beacon per floor the
    rolling mean covers — each beacon keeps one window per floor it
    was heard on, and the alarm and the classification read the
    window for the beacon's own floor by default, so a verdict on
    a multi-floor walk always rests on the beacon's own geometry
    instead of whichever floor the window happens to cover;
    ``drift_threshold_db`` is the mean residual that trips an alarm
    (5 dB by default, matching the survey health check);
    ``min_samples`` is how many readings a beacon needs in its
    window before it can alarm at all. ``noise_db`` is the
    deployment's expected RSSI noise (2 dB by default, matching the
    simulator) — the noisy verdict fires when a beacon's residual
    spread is more than twice it. ``history`` is how many readings
    per beacon are kept for revisit matching (500 by default, long
    enough for two loops of the default walk); ``revisit_gap`` is
    how many readings must separate two visits to the same spot
    before they count as a revisit, and ``revisit_radius_m`` is how
    close two readings must be to count as the same spot.
    ``floor_height_m`` and ``floor_attenuation_db`` are the floor
    model for multi-floor positions (3 m and 15 dB by default,
    matching the simulator): the predicted RSSI for a beacon on
    another floor accounts for the vertical gap and the slab, and
    revisit pairs only match readings on the same floor.
    """

    def __init__(self, beacons: list[Beacon],
                 window: int = DEFAULT_WINDOW,
                 drift_threshold_db: float = DEFAULT_DRIFT_THRESHOLD_DB,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 noise_db: float = DEFAULT_NOISE_DB,
                 history: int = DEFAULT_HISTORY,
                 revisit_gap: int = DEFAULT_REVISIT_GAP,
                 revisit_radius_m: float = DEFAULT_REVISIT_RADIUS_M,
                 floor_height_m: float = DEFAULT_FLOOR_HEIGHT_M,
                 floor_attenuation_db: float = DEFAULT_FLOOR_ATTENUATION_DB):
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
        if noise_db <= 0:
            raise ValueError("noise_db must be positive")
        if history < window:
            raise ValueError(
                "history must be at least window: "
                f"got history={history}, window={window}")
        if revisit_gap < 1:
            raise ValueError("revisit_gap must be at least 1")
        if revisit_radius_m <= 0:
            raise ValueError("revisit_radius_m must be positive")
        if floor_height_m <= 0:
            raise ValueError("floor_height_m must be positive")
        if floor_attenuation_db < 0:
            raise ValueError("floor_attenuation_db cannot be negative")
        self._beacons = {b.id: b for b in beacons}
        self._window = window
        self._threshold = drift_threshold_db
        self._min_samples = min_samples
        self._noise_db = noise_db
        self._revisit_gap = revisit_gap
        self._revisit_radius_m = revisit_radius_m
        self._floor_height_m = floor_height_m
        self._floor_attenuation_db = floor_attenuation_db
        # (sequence, x, y, floor, residual) per beacon per floor: the
        # rolling window for the alarm and the classification — one
        # window per floor, so the verdict on a multi-floor walk
        # rests on the beacon's own floor instead of whichever floor
        # a single mixed window happens to cover — and the longer
        # history, per beacon, for classification.
        self._observations: dict[str, dict[int, deque[tuple[int, float, float, int, float]]]] = {
            b.id: {} for b in beacons}
        self._history: dict[str, deque[tuple[int, float, float, int, float]]] = {
            b.id: deque(maxlen=history) for b in beacons}
        self._seq = 0

    def _window_floor(self, beacon_id: str, floor: int | None) -> int:
        """Which floor's window to read: the floor asked for, or the
        beacon's own floor — the floor it is mounted on."""
        beacon = self._beacons[beacon_id]
        return beacon.floor if floor is None else floor

    def _window_for(self, beacon_id: str, floor: int
                ) -> deque[tuple[int, float, float, int, float]]:
        """The rolling window for one beacon on one floor — an empty
        deque when it was never heard there."""
        return self._observations[beacon_id].get(
            floor, deque(maxlen=self._window))

    def observe(self, position: tuple[float, float]
                | tuple[float, float, int],
                rssi: dict[str, float]) -> None:
        """Feed one observation: the device's position and the RSSI heard.

        ``position`` is ``(x, y)`` on a single floor, or ``(x, y,
        floor)`` in a multi-floor building (a two-tuple means floor
        0). The predicted RSSI is the 3D distance — the vertical gap
        is ``floor_height_m`` per floor of separation — minus the
        slab attenuation for a beacon on a different floor from the
        device, the same model the simulator uses, so a healthy
        beacon heard through a floor does not read as drift. The
        reading lands in that floor's window for the beacon, so
        each floor's window holds only readings taken on that
        floor. Beacons not in the monitor's beacon set are ignored
        rather than failing the stream.
        """
        if len(position) == 2:
            x, y, floor = position[0], position[1], 0
        elif len(position) == 3:
            x, y, floor = position
        else:
            raise ValueError(
                "drift monitor needs an (x, y) or (x, y, floor) "
                f"position; got {len(position)} values")
        self._seq += 1
        for beacon_id, observed in rssi.items():
            beacon = self._beacons.get(beacon_id)
            if beacon is None:
                continue
            dz = self._floor_height_m * abs(floor - beacon.floor)
            distance = math.hypot(x - beacon.x, y - beacon.y, dz)
            predicted = distance_to_rssi(
                distance, beacon.rssi_at_1m, beacon.path_loss_exponent)
            if beacon.floor != floor:
                predicted -= self._floor_attenuation_db
            entry = (self._seq, x, y, floor, observed - predicted)
            windows = self._observations[beacon_id]
            if floor not in windows:
                windows[floor] = deque(maxlen=self._window)
            windows[floor].append(entry)
            self._history[beacon_id].append(entry)

    def feed(self, step: dict) -> None:
        """Feed one simulator step: ``{"true": (x, y[, floor]),
        "rssi": {...}}``."""
        self.observe(step["true"], step["rssi"])

    def reset(self) -> None:
        """Forget every reading — the rolling windows, the revisit
        history, and the sequence — so a new device starts with a
        clean monitor, as a new ``LiveTracker`` would."""
        for windows in self._observations.values():
            for observations in windows.values():
                observations.clear()
        for history in self._history.values():
            history.clear()
        self._seq = 0

    def fresh(self) -> "DriftMonitor":
        """An empty monitor with the same tuning as this one — the
        same window, threshold, minimum samples, noise level and
        revisit settings, but no readings yet.

        A ward tracks several devices at once, and each device's
        track is its own observation stream: one monitor per
        device, so one device's healthy readings never dilute
        another's alarm and resetting one device never clears
        another's window. ``MultiDeviceTracker`` builds each
        device's monitor this way from the monitor it was given.
        """
        return DriftMonitor(
            list(self._beacons.values()),
            window=self._window,
            drift_threshold_db=self._threshold,
            min_samples=self._min_samples,
            noise_db=self._noise_db,
            history=max(self._history[b].maxlen
                        for b in self._history),
            revisit_gap=self._revisit_gap,
            revisit_radius_m=self._revisit_radius_m,
            floor_height_m=self._floor_height_m,
            floor_attenuation_db=self._floor_attenuation_db)

    def status(self, beacon_id: str, floor: int | None = None
               ) -> DriftAlarm | None:
        """The rolling residual for one beacon, or ``None`` if unheard.

        ``floor`` names which floor's window to read; by default it
        is the beacon's own floor, so the reading rests on the
        beacon's own geometry rather than cross-floor readings.
        """
        if beacon_id not in self._beacons:
            raise ValueError(f"unknown beacon: {beacon_id!r}")
        observations = self._window_for(
            beacon_id, self._window_floor(beacon_id, floor))
        if not observations:
            return None
        residuals = [entry[4] for entry in observations]
        return DriftAlarm(
            id=beacon_id,
            mean_residual_db=_mean(residuals),
            samples=len(residuals))

    def alarm(self, beacon_id: str, floor: int | None = None
              ) -> DriftAlarm | None:
        """The active alarm for one beacon, or ``None`` if it is not alarming.

        A beacon alarms once it has at least ``min_samples`` readings
        in its window and the rolling mean residual is beyond the
        drift threshold in either direction. ``floor`` names which
        floor's window to read; by default it is the beacon's own
        floor.
        """
        state = self.status(beacon_id, floor)
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

    def _revisit_delta(self, beacon_id: str, floor: int,
                       window: list[tuple[int, float, float, int, float]]
                       ) -> tuple[float | None, float | None, int]:
        """Mean change in residual at the same spot between visits.

        Each reading in the floor's rolling window is matched to the
        closest earlier reading at (nearly) the same position, at
        least ``revisit_gap`` readings earlier — on the same floor,
        so a reading upstairs never counts as revisiting the same
        spot downstairs. Returns ``(mean_delta,
        standard_error, pairs)`` — or ``(None, None, 0)`` when fewer
        than five pairs match, which is too few to trust.
        """
        history = self._history[beacon_id]
        deltas = []
        for seq, x, y, _, residual in window:
            best = None
            for old_seq, old_x, old_y, old_floor, old_residual in history:
                if old_seq > seq - self._revisit_gap:
                    break
                if old_floor != floor:
                    continue
                distance = math.hypot(x - old_x, y - old_y)
                if best is None or distance < best[0]:
                    best = (distance, old_residual)
            if best is not None and best[0] <= self._revisit_radius_m:
                deltas.append(residual - best[1])
        if len(deltas) < 5:
            return None, None, len(deltas)
        mean_delta = _mean(deltas)
        se = _stdev(deltas) / math.sqrt(len(deltas))
        return mean_delta, se, len(deltas)

    def classify(self, beacon_id: str, floor: int | None = None
                 ) -> DriftClassification:
        """Classify one beacon's drift: sag, move, noise, or unclear.

        See the module docstring for the signatures. ``floor`` names
        which floor's window the verdict rests on; by default it is
        the beacon's own floor, so a verdict on a multi-floor walk
        always reads the beacon's own geometry — a verdict taken
        from another floor's window is the same beacon seen through
        the slab, and the classification says which floor it was
        taken on. The verdict is ``"healthy"`` when nothing is off,
        ``"battery-sag"`` when the beacon reads weaker over time at
        the same spot, ``"moved"`` when the residual is stable at
        the same spot but wrong by a position-dependent amount,
        ``"noisy"`` when the residual spread is more than twice the
        expected noise, and ``"uncertain"`` when the beacon is off
        calibration but no signature fits — including when it has
        not been heard enough yet to tell, or not heard on its own
        floor at all.
        """
        if beacon_id not in self._beacons:
            raise ValueError(f"unknown beacon: {beacon_id!r}")
        floor = self._window_floor(beacon_id, floor)
        window = list(self._window_for(beacon_id, floor))
        samples = len(window)
        if samples == 0:
            return DriftClassification(
                id=beacon_id, floor=floor, verdict="uncertain",
                mean_residual_db=0.0, residual_std_db=0.0,
                time_slope_db_per_step=0.0, time_correlation=0.0,
                revisit_delta_db=None, revisit_pairs=0, samples=0,
                reason=f"beacon unheard on floor {floor} — no "
                       "readings to classify")
        residuals = [entry[4] for entry in window]
        mean = _mean(residuals)
        std = _stdev(residuals)
        steps = list(range(samples))
        slope = _slope(steps, residuals)
        correlation = _correlation(steps, residuals)
        half = samples // 2
        half_delta = (_mean(residuals[half:]) - _mean(residuals[:half])
                      if half else 0.0)
        delta, delta_se, pairs = self._revisit_delta(beacon_id, floor,
                                                     window)

        def result(verdict: str, reason: str) -> DriftClassification:
            return DriftClassification(
                id=beacon_id, floor=floor, verdict=verdict,
                mean_residual_db=mean, residual_std_db=std,
                time_slope_db_per_step=slope,
                time_correlation=correlation,
                revisit_delta_db=delta, revisit_pairs=pairs,
                samples=samples, reason=reason)

        if samples < self._min_samples:
            return result(
                "uncertain",
                f"only {samples} readings — need at least "
                f"{self._min_samples} to classify")

        # Battery sag: weaker overall, and falling at the same spot.
        # With revisits the delta is the evidence; without them the
        # within-window trend has to carry it, and a move cannot be
        # ruled out, so the reason says so.
        if mean < SAG_MEAN_MAX_DB:
            if pairs >= 5:
                if (delta is not None and delta_se is not None
                        and delta < SAG_DELTA_MAX_DB
                        and delta < SAG_DELTA_MIN_T * delta_se):
                    return result(
                        "battery-sag",
                        f"mean residual {mean:.2f} dB and the residual "
                        f"at revisited spots fell {delta:.2f} dB "
                        f"({pairs} pairs) — the beacon reads weaker "
                        "at the same place over time")
            elif (slope <= SAG_SLOPE_MAX_DB_PER_STEP
                    or half_delta <= SAG_HALF_DELTA_MAX_DB):
                return result(
                    "battery-sag",
                    f"mean residual {mean:.2f} dB and the residual "
                    f"falls {slope:.2f} dB per reading within the "
                    "window — no revisits to confirm against, so "
                    "a move cannot be ruled out")

        # Noise: spread more than twice the expected RSSI noise,
        # with no coherent decline.
        if std > 2.0 * self._noise_db:
            return result(
                "noisy",
                f"residual spread {std:.2f} dB is more than twice "
                f"the expected {self._noise_db:.2f} dB noise, with "
                f"no steady decline — the beacon is unreliable, "
                "not miscalibrated")

        # Moved: stable at the same spot, wrong by a
        # position-dependent amount — the residuals spread out
        # around a shifted mean. Without revisits only a move
        # towards the walked area shows up (it reads louder), so
        # the fallback requires a positive mean.
        if std > MOVED_STD_MIN_DB and abs(mean) > MOVED_MEAN_MIN_DB:
            if pairs >= 5:
                if delta is not None and abs(delta) < 1.0:
                    return result(
                        "moved",
                        f"residual at revisited spots is stable "
                        f"({delta:+.2f} dB over {pairs} pairs) but "
                        f"the window spreads {std:.2f} dB around a "
                        f"{mean:+.2f} dB mean — the geometry is "
                        "wrong, not the power")
            elif mean > MOVED_MEAN_MIN_DB:
                return result(
                    "moved",
                    f"residuals spread {std:.2f} dB around a "
                    f"{mean:+.2f} dB mean with no steady decline "
                    "— no revisits to confirm against, so a "
                    "power change cannot be ruled out")

        # Off calibration, but no signature fits — say so.
        se_mean = std / math.sqrt(samples) if samples else 0.0
        if (abs(mean) > self._threshold
                or abs(mean) > max(UNCERTAIN_MIN_DB,
                                   UNCERTAIN_MIN_T * se_mean)):
            return result(
                "uncertain",
                f"mean residual {mean:+.2f} dB is off calibration "
                f"but the residual is steady at revisited spots "
                f"and shows no position pattern — a stabilised "
                "sag and a miscalibration look identical here")

        return result(
            "healthy",
            f"mean residual {mean:+.2f} dB and spread {std:.2f} dB "
            "are inside noise — the beacon matches its calibration")

    def classifications(self) -> list[DriftClassification]:
        """Every beacon's drift classification, in beacon order."""
        return [self.classify(b.id) for b in self._beacons.values()]
