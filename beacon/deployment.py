"""Deployment planner: when does each beacon go dark?

A ward full of battery beacons needs a replacement schedule, not a
surprise. ``BatteryModel`` already says how one beacon declines —
linear drain, RSSI loss growing with the square of the charge used,
extra noise growing linearly with it, dark below the cutoff — so
the time to each of those milestones is just algebra on the model,
not a simulation: the level falls linearly, so the hours until any
target level are ``(level_now - level_target) / drain_rate``.

``forecast_beacon`` predicts one beacon's milestones from its model
and its current charge (a measured level, or the model's ``initial``
when the beacon is fresh): hours until it goes dark, hours until
its RSSI loss crosses a threshold (3 dB by default — the point
where the beacon starts pulling estimates noticeably), and hours
until its extra noise crosses a threshold (2 dB by default). A
milestone the beacon never reaches — no drain at all, or dark
before the loss gets that bad — is ``math.inf``, not a made-up
number. ``plan_deployment`` forecasts a whole ward at once,
sorted soonest-dark first, and says when the ward drops below
enough beacons to triangulate.

Pure standard library, no dependencies.
"""

import math
from dataclasses import dataclass

from .battery import BatteryModel

# RSSI loss at which a beacon starts pulling estimates noticeably.
# The drift alarm trips at 5 dB; 3 dB is the point worth planning
# around, while the beacon can still be trusted a little longer.
DEFAULT_LOSS_THRESHOLD_DB = 3.0

# Extra RSSI noise at which a beacon's readings are about as noisy
# as the simulator's own base noise (2 dB) — past this, the beacon
# is as much noise as signal.
DEFAULT_NOISE_THRESHOLD_DB = 2.0

# Beacons needed to triangulate. A ward with fewer lit beacons than
# this cannot locate at all, so the plan reports when that happens.
MIN_BEACONS_TO_TRIANGULATE = 3


@dataclass(frozen=True)
class BeaconForecast:
    """When one beacon goes dark, and when it starts lying first.

    ``hours_to_dark`` is the time until the beacon's level hits its
    model's cutoff; ``hours_to_rssi_loss_db`` the time until its
    RSSI loss reaches ``loss_threshold_db``; ``hours_to_extra_noise_db``
    the time until its extra noise reaches ``noise_threshold_db``.
    Any of them is ``math.inf`` when the beacon never gets there —
    a beacon that does not drain never goes dark, and a beacon that
    goes dark first never reaches a loss its cutoff level cannot
    produce. ``status`` is ``dark`` when the beacon is already below
    its cutoff, ``degrading`` when its RSSI loss is already past the
    threshold, and ``ok`` otherwise.
    """

    beacon_id: str
    level: float
    drain_rate_per_hour: float
    hours_to_dark: float
    hours_to_rssi_loss_db: float
    hours_to_extra_noise_db: float
    loss_threshold_db: float = DEFAULT_LOSS_THRESHOLD_DB
    noise_threshold_db: float = DEFAULT_NOISE_THRESHOLD_DB
    status: str = "ok"

    def to_dict(self) -> dict:
        """JSON-friendly form; never-reached horizons are null."""
        def _hours(value: float) -> float | None:
            return None if math.isinf(value) else value

        return {
            "beacon": self.beacon_id,
            "level": self.level,
            "drain_rate_per_hour": self.drain_rate_per_hour,
            "hours_to_dark": _hours(self.hours_to_dark),
            "hours_to_rssi_loss_db": _hours(self.hours_to_rssi_loss_db),
            "hours_to_extra_noise_db": _hours(self.hours_to_extra_noise_db),
            "loss_threshold_db": self.loss_threshold_db,
            "noise_threshold_db": self.noise_threshold_db,
            "status": self.status,
        }


@dataclass(frozen=True)
class DeploymentPlan:
    """A whole ward's battery forecast, soonest-dark first.

    ``forecasts`` are sorted by ``hours_to_dark`` ascending (a
    beacon that never goes dark sorts last), and
    ``earliest_dark_hours`` is the soonest any beacon goes dark —
    ``math.inf`` when none of them ever does.
    """

    forecasts: tuple[BeaconForecast, ...]
    earliest_dark_hours: float

    def hours_until_fewer_than(self, n_beacons: int) -> float:
        """Hours until fewer than ``n_beacons`` are still lit.

        The ward loses its nth beacon when the (len - n + 1)-th
        soonest beacon goes dark: with four beacons, fewer than
        three remain once the second beacon goes dark. Returns
        ``math.inf`` when that never happens — not enough beacons
        drain — and 0.0 when the ward is already below the count.
        """
        if n_beacons < 1:
            raise ValueError(
                f"n_beacons must be at least 1: got {n_beacons}")
        dark_hours = sorted(f.hours_to_dark for f in self.forecasts)
        already_dark = sum(1 for h in dark_hours if h == 0.0)
        lit = len(dark_hours) - already_dark
        if lit < n_beacons:
            return 0.0
        # The (len - n + 1)-th darkening drops the lit count to n - 1.
        index = len(dark_hours) - n_beacons
        return dark_hours[index]

    def to_dict(self) -> dict:
        """JSON-friendly form; never-reached horizons are null."""
        return {
            "forecasts": [f.to_dict() for f in self.forecasts],
            "earliest_dark_hours": (
                None if math.isinf(self.earliest_dark_hours)
                else self.earliest_dark_hours),
            "hours_until_fewer_than_three": (
                None
                if math.isinf(
                    self.hours_until_fewer_than(
                        MIN_BEACONS_TO_TRIANGULATE))
                else self.hours_until_fewer_than(
                    MIN_BEACONS_TO_TRIANGULATE)),
        }


def _hours_until_level(model: BatteryModel, level_now: float,
                       target_level: float) -> float:
    """Hours until the level falls to ``target_level``.

    Zero when already there or past it; infinite when the beacon
    does not drain at all.
    """
    if target_level >= level_now:
        return 0.0
    if model.drain_rate_per_hour == 0.0:
        return math.inf
    return (level_now - target_level) / model.drain_rate_per_hour


def forecast_beacon(
        beacon_id: str, model: BatteryModel, level: float | None = None,
        loss_threshold_db: float = DEFAULT_LOSS_THRESHOLD_DB,
        noise_threshold_db: float = DEFAULT_NOISE_THRESHOLD_DB,
) -> BeaconForecast:
    """Predict one beacon's dark and degradation milestones.

    ``level`` is the beacon's current charge (0 to 1); it defaults
    to the model's ``initial``, for a fresh beacon. A level above
    the model's initial is rejected — a beacon cannot hold more
    charge than it started with — and so are negative thresholds.
    """
    if level is None:
        level = model.initial
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"level must be in [0, 1]: got {level}")
    if level > model.initial:
        raise ValueError(
            f"level {level} is above the model's initial "
            f"{model.initial}")
    if loss_threshold_db < 0:
        raise ValueError(
            "loss_threshold_db must not be negative: got "
            f"{loss_threshold_db}")
    if noise_threshold_db < 0:
        raise ValueError(
            "noise_threshold_db must not be negative: got "
            f"{noise_threshold_db}")

    hours_to_dark = _hours_until_level(model, level, model.cutoff)

    # RSSI loss grows with the square of the charge used, so the
    # level at which the loss reaches the threshold is
    # 1 - sqrt(threshold / max_loss). But the beacon goes dark at
    # its cutoff first: if even the loss at the cutoff is below
    # the threshold, the beacon never reads that weak — it just
    # goes quiet.
    if model.max_rssi_loss_db == 0.0 or loss_threshold_db == 0.0:
        hours_to_loss = 0.0 if loss_threshold_db == 0.0 else math.inf
    elif model.rssi_loss_db(model.cutoff) < loss_threshold_db:
        hours_to_loss = math.inf
    else:
        level_at_loss = 1.0 - math.sqrt(
            loss_threshold_db / model.max_rssi_loss_db)
        hours_to_loss = _hours_until_level(model, level, level_at_loss)

    # Extra noise grows linearly with the charge used, so the level
    # at which it reaches the threshold is 1 - threshold / max.
    # The same cutoff rule applies: dark first means never.
    if model.max_extra_noise_db == 0.0 or noise_threshold_db == 0.0:
        hours_to_noise = 0.0 if noise_threshold_db == 0.0 else math.inf
    elif model.extra_noise_db(model.cutoff) < noise_threshold_db:
        hours_to_noise = math.inf
    else:
        level_at_noise = 1.0 - (
            noise_threshold_db / model.max_extra_noise_db)
        hours_to_noise = _hours_until_level(model, level, level_at_noise)

    if model.is_dark(level):
        status = "dark"
    elif model.rssi_loss_db(level) >= loss_threshold_db:
        status = "degrading"
    else:
        status = "ok"

    return BeaconForecast(
        beacon_id=beacon_id,
        level=level,
        drain_rate_per_hour=model.drain_rate_per_hour,
        hours_to_dark=hours_to_dark,
        hours_to_rssi_loss_db=hours_to_loss,
        hours_to_extra_noise_db=hours_to_noise,
        loss_threshold_db=loss_threshold_db,
        noise_threshold_db=noise_threshold_db,
        status=status,
    )


def plan_deployment(
        models: dict[str, BatteryModel],
        levels: dict[str, float] | None = None,
        loss_threshold_db: float = DEFAULT_LOSS_THRESHOLD_DB,
        noise_threshold_db: float = DEFAULT_NOISE_THRESHOLD_DB,
) -> DeploymentPlan:
    """Forecast a whole ward's beacons, soonest-dark first.

    ``models`` maps beacon id to its ``BatteryModel``; ``levels``
    optionally overrides each beacon's current charge (a measured
    level, rather than the model's fresh-cell initial). Unknown
    ids in ``levels`` are rejected rather than silently ignored.
    """
    levels = levels or {}
    unknown = sorted(set(levels) - set(models))
    if unknown:
        raise ValueError(
            f"levels given for unknown beacons: {', '.join(unknown)}")
    forecasts = tuple(sorted(
        (forecast_beacon(beacon_id, model,
                         level=levels.get(beacon_id),
                         loss_threshold_db=loss_threshold_db,
                         noise_threshold_db=noise_threshold_db)
         for beacon_id, model in models.items()),
        key=lambda f: f.hours_to_dark))
    earliest = (forecasts[0].hours_to_dark if forecasts
                else math.inf)
    return DeploymentPlan(forecasts=forecasts, earliest_dark_hours=earliest)
