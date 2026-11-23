"""Live tracking mode: consume a stream of RSSI readings in real time.

The engine so far answers one-shot questions — one observation set,
one position. A ward dashboard needs the opposite: a device that
keeps reporting, and a position that keeps updating as readings
arrive. ``LiveTracker`` wraps a ``PositioningEngine`` and turns the
observation stream into a live track:

- Feed it each reading as it arrives — ``feed(rssi, t)`` — and it
  emits a ``LivePosition`` per reading: the smoothed (x, y) from
  the engine's filter, the floor, and the 95% confidence ellipse
  for that step's geometry.
- Readings are merged across beacons, so a reading that only hears
  two beacons still updates the track from the beacons heard a
  moment ago — BLE scans rarely hear every beacon on every scan.
- A reading older than ``stale_after`` seconds (5 s by default)
  ages out of the merged set, so a beacon that has gone quiet
  stops dragging the estimate towards where the device was when
  it was last heard.
- When fewer than three fresh beacons remain, trilateration is
  impossible and ``feed`` returns ``None`` instead of a made-up
  position; the last good position stays available as
  ``last_position`` until a fresh reading revives the track.

The stream format is JSON lines, one reading per line — the same
shape a BLE scanner would emit, and easy to pipe from anything:

    {"t": 0.0, "rssi": {"A": -62.1, "B": -71.4, "C": -68.0, "D": -59.8}}

Run it as a module: ``python -m beacon.tracking`` reads those lines
from stdin and prints one JSON line per reading with the live
position attached (``position`` is null on readings too thin to
locate). ``--demo`` runs a simulated walk through the tracker
instead, so the mode can be tried with no scanner at all.
"""

import argparse
import json
import sys
from dataclasses import dataclass

from .beacon import Beacon
from .engine import PositioningEngine
from .simulate import rectangular_walk, simulate_walk
from .uncertainty import PositionUncertainty

# A reading older than this many seconds no longer describes where
# the device is, so it leaves the merged observation set.
DEFAULT_STALE_AFTER_S = 5.0

# Assumed RSSI noise for the confidence ellipse, matching the
# simulator's default noise and the engine's default.
DEFAULT_SIGMA_RSSI_DB = 2.0


@dataclass(frozen=True)
class LivePosition:
    """One live position estimate, emitted per reading.

    ``t`` is the timestamp of the reading that produced it (seconds,
    same clock as the feed). ``x`` / ``y`` are the filter-smoothed
    metres on ``floor`` (``floor_confidence`` is the floor vote's
    confidence). ``uncertainty`` is the 95% confidence ellipse for
    this step's raw geometry — None when the geometry is degenerate
    (the filtered position is still valid). ``heard`` is the fresh
    beacons the estimate was built from, in id order.
    """

    t: float
    x: float
    y: float
    floor: int
    floor_confidence: float
    uncertainty: PositionUncertainty | None
    heard: tuple[str, ...]

    def to_dict(self) -> dict:
        """JSON-friendly form, for the stream output and dashboards."""
        return {
            "t": self.t,
            "x": self.x,
            "y": self.y,
            "floor": self.floor,
            "floor_confidence": self.floor_confidence,
            "uncertainty": (
                None if self.uncertainty is None else {
                    "semi_major": self.uncertainty.semi_major,
                    "semi_minor": self.uncertainty.semi_minor,
                    "angle_deg": self.uncertainty.angle_deg,
                }),
            "heard": list(self.heard),
        }


class LiveTracker:
    """Turn a stream of RSSI readings into a live position track.

    ``engine`` is the ``PositioningEngine`` doing the localisation;
    one tracker tracks one device. ``stale_after`` is how long a
    reading stays in the merged observation set, in seconds.
    ``sigma_rssi_db`` is the assumed RSSI noise for the confidence
    ellipse (default 2.0 dB, matching the simulator).
    """

    def __init__(self, engine: PositioningEngine,
                 stale_after: float = DEFAULT_STALE_AFTER_S,
                 sigma_rssi_db: float = DEFAULT_SIGMA_RSSI_DB):
        if stale_after <= 0:
            raise ValueError("stale_after must be positive")
        if sigma_rssi_db <= 0:
            raise ValueError("sigma_rssi_db must be positive")
        self.engine = engine
        self._stale_after = stale_after
        self._sigma_rssi_db = sigma_rssi_db
        # Beacon id -> (rssi, t) of its most recent reading.
        self._latest: dict[str, tuple[float, float]] = {}
        self._last_t: float | None = None
        self._last: LivePosition | None = None

    @property
    def last_position(self) -> LivePosition | None:
        """The most recent live position, or None before the first
        locatable reading."""
        return self._last

    @property
    def heard(self) -> tuple[str, ...]:
        """The beacons currently fresh enough to localise from,
        in id order."""
        return tuple(sorted(self._latest))

    def feed(self, rssi: dict[str, float], t: float) -> LivePosition | None:
        """Feed one reading and return the live position it gives.

        ``rssi`` maps beacon id to RSSI in dBm (a partial scan is
        fine — only the beacons heard need be present), ``t`` is the
        reading's timestamp in seconds. Unknown beacon ids are
        ignored rather than failing the stream. Timestamps must not
        go backwards: a reading older than the previous one is a
        replayed or misordered scan, not a live one.

        Returns the live position, or None when fewer than three
        fresh beacons remain — trilateration needs three, and a
        made-up position is worse than no position.
        """
        if self._last_t is not None and t < self._last_t:
            raise ValueError(
                f"timestamps must not go backwards: {t} after {self._last_t}")
        self._last_t = t
        for beacon_id, value in rssi.items():
            if beacon_id not in self.engine.beacons:
                continue
            self._latest[beacon_id] = (value, t)
        # Age out readings that no longer describe where the device
        # is: a beacon heard a while ago must not anchor the track.
        self._latest = {
            beacon_id: reading
            for beacon_id, reading in self._latest.items()
            if t - reading[1] <= self._stale_after}
        fresh = {beacon_id: reading[0]
                 for beacon_id, reading in self._latest.items()}
        if len(fresh) < 3:
            return None
        position = self.engine.estimate_with_floor(fresh)
        # The ellipse describes this step's raw geometry, from the
        # same beacon subset the engine localised on: the winning
        # floor's beacons when at least three were heard there, else
        # every heard beacon (the engine's own fallback).
        on_floor = [b for b in self.engine.beacons.values()
                    if b.floor == position.floor and b.id in fresh]
        subset = (on_floor if len(on_floor) >= 3
                  else [self.engine.beacons[i] for i in fresh])
        try:
            _, _, uncertainty = self.engine.estimate_with_uncertainty(
                {b.id: fresh[b.id] for b in subset},
                sigma_rssi_db=self._sigma_rssi_db)
        except ValueError:
            # Degenerate geometry (collinear beacons, or the estimate
            # sitting on a beacon): the filtered position is still
            # valid, it just has no ellipse this step.
            uncertainty = None
        live = LivePosition(
            t=t, x=position.x, y=position.y, floor=position.floor,
            floor_confidence=position.floor_confidence,
            uncertainty=uncertainty, heard=tuple(sorted(fresh)))
        self._last = live
        return live

    def reset(self) -> None:
        """Forget the track (e.g. when tracking a new device)."""
        self._latest.clear()
        self._last_t = None
        self._last = None
        self.engine.reset()


def parse_line(line: str) -> tuple[float, dict[str, float]]:
    """Parse one stream line: ``{"t": ..., "rssi": {...}}``.

    Returns ``(t, rssi)``. Raises ValueError on a line that is not
    JSON, or that lacks a numeric ``t`` or a ``rssi`` object.
    """
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not a JSON line: {line.strip()!r}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"stream line must be a JSON object: {line.strip()!r}")
    t = record.get("t")
    rssi = record.get("rssi")
    if not isinstance(t, (int, float)) or isinstance(t, bool):
        raise ValueError(f"stream line needs a numeric 't': {line.strip()!r}")
    if not isinstance(rssi, dict) or not rssi:
        raise ValueError(f"stream line needs a 'rssi' object: {line.strip()!r}")
    return float(t), {str(k): float(v) for k, v in rssi.items()}


def format_output(t: float, position: LivePosition | None,
                  heard: tuple[str, ...]) -> dict:
    """One output line per input line: the live position, or null
    when the reading was too thin to locate, with the fresh beacons
    listed either way so a dashboard can see the track starving."""
    if position is None:
        return {"t": t, "position": None, "heard": list(heard)}
    return {"t": t, "position": position.to_dict(),
            "heard": list(heard)}


def default_beacons() -> list[Beacon]:
    """Four beacons, one in each corner of the default 20 x 12 m ward."""
    return [
        Beacon("A", 0.0, 0.0),
        Beacon("B", 20.0, 0.0),
        Beacon("C", 20.0, 12.0),
        Beacon("D", 0.0, 12.0),
    ]


def run_demo(stale_after: float, sigma_rssi_db: float,
             output_stream) -> int:
    """Feed a simulated ward walk through the tracker at 1 Hz and
    print one JSON line per step — the mode with no scanner needed."""
    beacons = default_beacons()
    steps = simulate_walk(rectangular_walk(20.0, 12.0, 10), beacons,
                          noise_db=2.0, seed=7)
    tracker = LiveTracker(PositioningEngine(beacons),
                          stale_after=stale_after,
                          sigma_rssi_db=sigma_rssi_db)
    for i, step in enumerate(steps):
        t = float(i)
        position = tracker.feed(step["rssi"], t)
        print(json.dumps(format_output(t, position, tracker.heard)),
              file=output_stream)
    return 0


def main(argv=None, input_stream=None, output_stream=None) -> int:
    """Stream entry point: read JSON-line readings from stdin, print
    one JSON line per reading with the live position attached.

    ``--demo`` runs a simulated walk instead of reading stdin.
    ``--stale-after`` (seconds) and ``--sigma-rssi`` (dB) tune the
    tracker. ``input_stream`` / ``output_stream`` exist so tests can
    drive the stream without touching sys.stdin / sys.stdout.
    """
    parser = argparse.ArgumentParser(
        description="Live BLE indoor-positioning tracker: read RSSI "
        "readings as JSON lines and print a live position per reading.")
    parser.add_argument("--demo", action="store_true",
                        help="run a simulated walk instead of reading stdin")
    parser.add_argument("--stale-after", type=float,
                        default=DEFAULT_STALE_AFTER_S,
                        help="seconds before a reading ages out "
                        f"(default: {DEFAULT_STALE_AFTER_S})")
    parser.add_argument("--sigma-rssi", type=float,
                        default=DEFAULT_SIGMA_RSSI_DB,
                        help="assumed RSSI noise in dB for the confidence "
                        f"ellipse (default: {DEFAULT_SIGMA_RSSI_DB})")
    args = parser.parse_args(argv)
    if args.stale_after <= 0:
        parser.error("--stale-after must be positive")
    if args.sigma_rssi <= 0:
        parser.error("--sigma-rssi must be positive")

    if args.demo:
        return run_demo(args.stale_after, args.sigma_rssi,
                        output_stream or sys.stdout)

    tracker = LiveTracker(PositioningEngine(default_beacons()),
                          stale_after=args.stale_after,
                          sigma_rssi_db=args.sigma_rssi)
    source = input_stream if input_stream is not None else sys.stdin
    sink = output_stream if output_stream is not None else sys.stdout
    for line in source:
        if not line.strip():
            continue
        t, rssi = parse_line(line)
        position = tracker.feed(rssi, t)
        print(json.dumps(format_output(t, position, tracker.heard)),
              file=sink)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
