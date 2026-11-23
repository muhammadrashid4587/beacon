"""Battery degradation: a beacon's RSSI as its battery drains.

A BLE beacon does not die all at once. As its cell drains, its
transmit power sags — the signal gets weaker — and the readings
get noisier, until the cell can no longer power the radio at all
and the beacon goes dark. That slow decline poisons positioning
long before the beacon disappears: the engine trusts the
path-loss model, so a beacon reading weaker than its calibration
says looks farther away than it is, and the estimate is pulled
towards it.

``BatteryModel`` describes one beacon's decline:

- ``initial`` is the charge when the simulation starts, as a
  fraction of full (1.0 is a fresh cell).
- ``drain_rate_per_hour`` is the fraction of a full charge the
  beacon burns per hour — a beacon that lasts a year on a coin
  cell drains at about 0.0001 per hour; a beacon with a day left
  drains at about 0.04. The drain is linear in time, which is
  close enough for a coin cell over the short window a walk
  covers.
- ``cutoff`` is the level at or below which the beacon goes
  dark (0.05 by default — the radio browns out before the cell
  is truly empty).
- ``max_rssi_loss_db`` is how much weaker the beacon reads at
  an empty cell than at a full one (6 dB by default — the
  beacon looks about twice as far away). The loss grows with
  the square of the charge used, so a half-drained beacon loses
  only a quarter of it and the decline accelerates near the end,
  the way a lithium cell's voltage holds up and then falls off
  a cliff.
- ``max_extra_noise_db`` is how much extra RSSI noise (standard
  deviation) an empty cell adds on top of the simulator's base
  noise (3 dB by default — a dying radio is a noisy radio). The
  extra noise grows linearly with the charge used.

Pure standard library, no dependencies.
"""

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class BatteryModel:
    """How one beacon's battery drains, and what that does to RSSI.

    ``level_at(elapsed_s)`` is the remaining charge after that
    many seconds, ``rssi_loss_db(level)`` is how many dB weaker
    the beacon reads at that level, ``extra_noise_db(level)`` is
    the extra RSSI noise it adds, and ``is_dark(level)`` says
    whether the beacon has gone quiet. ``apply`` does all of it
    to one reading: it returns None when the beacon is dark, and
    the degraded RSSI otherwise.
    """

    initial: float = 1.0
    drain_rate_per_hour: float = 0.05
    cutoff: float = 0.05
    max_rssi_loss_db: float = 6.0
    max_extra_noise_db: float = 3.0

    def __post_init__(self) -> None:
        if not 0.0 < self.initial <= 1.0:
            raise ValueError(
                f"initial must be in (0, 1]: got {self.initial}")
        if self.drain_rate_per_hour < 0:
            raise ValueError(
                "drain_rate_per_hour must not be negative: got "
                f"{self.drain_rate_per_hour}")
        if not 0.0 <= self.cutoff < self.initial:
            raise ValueError(
                f"cutoff must be in [0, initial): got {self.cutoff}")
        if self.max_rssi_loss_db < 0:
            raise ValueError(
                "max_rssi_loss_db must not be negative: got "
                f"{self.max_rssi_loss_db}")
        if self.max_extra_noise_db < 0:
            raise ValueError(
                "max_extra_noise_db must not be negative: got "
                f"{self.max_extra_noise_db}")

    def level_at(self, elapsed_s: float) -> float:
        """Remaining charge after ``elapsed_s`` seconds, 0 to 1.

        Linear drain, floored at zero — a dead cell stays dead,
        it does not go negative.
        """
        if elapsed_s < 0:
            raise ValueError(
                f"elapsed_s must not be negative: got {elapsed_s}")
        return max(0.0, self.initial
                   - self.drain_rate_per_hour * elapsed_s / 3600.0)

    def rssi_loss_db(self, level: float) -> float:
        """How many dB weaker the beacon reads at ``level``.

        Zero on a full cell, ``max_rssi_loss_db`` on an empty
        one, growing with the square of the charge used — the
        decline stays mild while the cell has charge left and
        accelerates near the end.
        """
        used = 1.0 - level
        return self.max_rssi_loss_db * used * used

    def extra_noise_db(self, level: float) -> float:
        """Extra RSSI noise (stddev, dB) at ``level``.

        Zero on a full cell, ``max_extra_noise_db`` on an empty
        one, growing linearly with the charge used.
        """
        return self.max_extra_noise_db * (1.0 - level)

    def is_dark(self, level: float) -> bool:
        """Has the beacon gone quiet at ``level``?"""
        return level <= self.cutoff

    def apply(self, rssi: float, level: float,
              rng: random.Random, base_noise_db: float = 0.0) -> float | None:
        """Degrade one RSSI reading for a beacon at ``level``.

        Returns None when the beacon is dark at this level — a
        dead beacon is not heard at all. Otherwise returns the
        reading with the battery loss applied and the extra
        noise added on top of ``base_noise_db`` (the simulator's
        own noise): one Gaussian draw with the combined
        standard deviation, so the noise stays a single draw
        rather than two stacked ones.
        """
        if self.is_dark(level):
            return None
        degraded = rssi - self.rssi_loss_db(level)
        noise = base_noise_db + self.extra_noise_db(level)
        if noise:
            degraded += rng.gauss(0.0, noise)
        return degraded
