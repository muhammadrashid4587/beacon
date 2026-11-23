"""Battery-swap replacement schedule: when to change each cell.

The deployment planner forecasts when each beacon goes dark and
when it starts lying first; a ward still needs that as a work
order — which beacon to swap, by when, and why. ``replacement_
schedule`` turns a ``DeploymentPlan`` into that list: one entry
per beacon, soonest swap first, each with the forecast horizon
that drives it (the beacon goes dark, its RSSI loss crosses
3 dB, or its extra noise crosses 2 dB — whichever comes first,
since a beacon that is already lying needs a new cell just as
much as one that is about to go quiet).

The recommended swap is by the first milestone, not by the dark:
by the time a beacon goes dark it has already been pulling
estimates off for hours. ``lead_time_hours`` pulls every swap
earlier by the same margin — order the cells a day ahead and
the schedule says so — floored at zero, so a beacon already
past its milestone is due now, not in the past. A beacon that
never drains never needs a swap, and says so with ``math.inf``
and the reason ``never`` rather than a made-up date.

Pure standard library, no dependencies.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from .deployment import DeploymentPlan, MIN_BEACONS_TO_TRIANGULATE

# Reasons a beacon is due a swap, in the order they win a tie:
# going dark beats reading weak, and reading weak beats reading
# noisy — the more severe failure drives the work order.
REASON_DARK = "dark"
REASON_LOSS = "3 dB loss"
REASON_NOISE = "2 dB noise"
REASON_NEVER = "never"


@dataclass(frozen=True)
class SwapEntry:
    """One beacon's recommended battery swap.

    ``swap_by_hours`` is the hours from now until the swap is due
    — the soonest of the beacon's forecast milestones, less any
    lead time — or ``math.inf`` when the beacon never needs one.
    ``reason`` names the milestone that drives the date: ``dark``,
    ``3 dB loss``, ``2 dB noise``, or ``never``. The three
    horizons ride along so a dashboard can show why the date is
    what it is, and ``status`` is the forecast's own status
    (``ok``, ``degrading`` or ``dark``).
    """

    beacon_id: str
    swap_by_hours: float
    reason: str
    hours_to_dark: float
    hours_to_rssi_loss_db: float
    hours_to_extra_noise_db: float
    status: str

    def due_at(self, now: datetime | None = None) -> datetime | None:
        """The clock time the swap is due, or None when never."""
        if math.isinf(self.swap_by_hours):
            return None
        now = now or datetime.now()
        return now + timedelta(hours=self.swap_by_hours)

    def to_dict(self) -> dict:
        """JSON-friendly form; a never-due swap is null."""
        def _hours(value: float) -> float | None:
            return None if math.isinf(value) else value

        return {
            "beacon": self.beacon_id,
            "swap_by_hours": _hours(self.swap_by_hours),
            "reason": self.reason,
            "hours_to_dark": _hours(self.hours_to_dark),
            "hours_to_rssi_loss_db": _hours(self.hours_to_rssi_loss_db),
            "hours_to_extra_noise_db": (
                _hours(self.hours_to_extra_noise_db)),
            "status": self.status,
        }


@dataclass(frozen=True)
class ReplacementSchedule:
    """A whole ward's battery swaps, soonest first.

    ``entries`` are sorted by ``swap_by_hours`` ascending (a beacon
    that never needs a swap sorts last), and
    ``ward_deadline_hours`` is when the ward drops below three lit
    beacons and can no longer triangulate at all — the hard
    deadline the swaps must beat, ``math.inf`` when that never
    happens.
    """

    entries: tuple[SwapEntry, ...]
    ward_deadline_hours: float

    def to_dict(self) -> dict:
        """JSON-friendly form; a never-reached deadline is null."""
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "ward_deadline_hours": (
                None if math.isinf(self.ward_deadline_hours)
                else self.ward_deadline_hours),
        }


def replacement_schedule(
        plan: DeploymentPlan, lead_time_hours: float = 0.0,
) -> ReplacementSchedule:
    """Turn a deployment plan into a battery-swap work order.

    Each beacon's swap is due at its soonest forecast milestone —
    dark, 3 dB of RSSI loss, or 2 dB of extra noise — pulled
    earlier by ``lead_time_hours`` and floored at zero, so a
    beacon already past a milestone is due now. A negative lead
    time is rejected: a schedule cannot push a swap later than
    the beacon's own forecast.
    """
    if lead_time_hours < 0:
        raise ValueError(
            "lead_time_hours must not be negative: got "
            f"{lead_time_hours}")

    entries = []
    for forecast in plan.forecasts:
        milestones = (
            (forecast.hours_to_dark, REASON_DARK),
            (forecast.hours_to_rssi_loss_db, REASON_LOSS),
            (forecast.hours_to_extra_noise_db, REASON_NOISE),
        )
        soonest_hours, reason = min(milestones, key=lambda m: m[0])
        if math.isinf(soonest_hours):
            swap_by = math.inf
            reason = REASON_NEVER
        else:
            swap_by = max(0.0, soonest_hours - lead_time_hours)
        entries.append(SwapEntry(
            beacon_id=forecast.beacon_id,
            swap_by_hours=swap_by,
            reason=reason,
            hours_to_dark=forecast.hours_to_dark,
            hours_to_rssi_loss_db=forecast.hours_to_rssi_loss_db,
            hours_to_extra_noise_db=forecast.hours_to_extra_noise_db,
            status=forecast.status,
        ))
    entries.sort(key=lambda entry: entry.swap_by_hours)
    return ReplacementSchedule(
        entries=tuple(entries),
        ward_deadline_hours=plan.hours_until_fewer_than(
            MIN_BEACONS_TO_TRIANGULATE),
    )
