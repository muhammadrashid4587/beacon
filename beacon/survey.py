"""Survey export/import: share a walked survey as a portable file.

A survey is the raw material everything else in this package is
built from: the beacons (positions plus their fitted calibration
constants) and the walked steps (known positions plus the RSSI
observed from each beacon at each stop). The fingerprint radio map
is built from those steps, so a survey file carries all three —
beacons, steps, and the built map — and importing it restores the
exact same survey, map included, with nothing re-surveyed.

The format is plain JSON, so a survey can be emailed, checked into
version control next to a ward's floor plan, or loaded on another
machine and used straight away:

    from beacon import Survey
    from beacon.simulate import simulate_walk, grid_points

    cells = grid_points(20.0, 12.0, spacing=2.0)
    steps = simulate_walk(cells, beacons, noise_db=2.0, seed=7)
    survey = Survey.build(beacons, steps)   # builds the radio map
    survey.save("ward-a-survey.json")

    loaded = Survey.load("ward-a-survey.json")
    x, y = loaded.fingerprint_map().locate({"A": -62.1, "B": -71.4})

The round trip is lossless: beacons keep their calibrated
``rssi_at_1m`` and path-loss exponent, steps keep their exact
positions and RSSI values, and the radio map keeps its per-cell
sums and sample counts (not just the averaged fingerprints), so
samples added after an import are averaged with the original
weights. JSON floats round-trip exactly in Python, so the
restored survey locates identically to the original.

Pure standard library, no dependencies.
"""

import dataclasses
import json

from .beacon import Beacon
from .calibration import calibrate_survey
from .fingerprint import (FingerprintMap, MultiFloorFingerprintMap,
                          build_fingerprint_map,
                          build_multifloor_fingerprint_map)

FORMAT = "beacon-survey"
FORMAT_VERSION = 1


@dataclasses.dataclass
class Survey:
    """A walked survey: beacons, steps, and the built radio map.

    ``beacons`` is the list of surveyed beacons (with their
    calibration constants); ``steps`` is a list of
    ``{"true": (x, y) [, floor], "rssi": {id: rssi}}`` dicts —
    exactly the shape ``simulate_walk`` returns. ``radio_map`` is
    the fingerprint map built from those steps, or ``None`` when
    the survey was imported without one — ``fingerprint_map()``
    rebuilds it on demand in that case.
    """

    beacons: list[Beacon]
    steps: list[dict]
    radio_map: FingerprintMap | MultiFloorFingerprintMap | None = None

    @classmethod
    def build(cls, beacons: list[Beacon], steps: list[dict]) -> "Survey":
        """Build a survey from beacons and steps, map included.

        Single-floor steps (``"true"`` is ``(x, y)``) build a
        ``FingerprintMap``; multi-floor steps (``(x, y, floor)``)
        build a ``MultiFloorFingerprintMap``.
        """
        survey = cls(list(beacons), list(steps))
        survey.radio_map = survey.fingerprint_map()
        return survey

    def fingerprint_map(self) -> FingerprintMap | MultiFloorFingerprintMap:
        """The survey's radio map, built from the steps if needed.

        Returns the stored map when the survey has one; otherwise
        builds it from the beacons and steps. Raises ValueError
        on an empty survey (no steps to build a map from).
        """
        if self.radio_map is not None:
            return self.radio_map
        if not self.steps:
            raise ValueError(
                "cannot build a fingerprint map from an empty survey")
        if len(self.steps[0]["true"]) == 3:
            return build_multifloor_fingerprint_map(self.beacons,
                                                     self.steps)
        return build_fingerprint_map(self.beacons, self.steps)

    def calibrated_beacons(self) -> list[Beacon]:
        """The beacons with path-loss constants fitted to the survey.

        Single-floor surveys only: calibration fits RSSI against
        2D distance, and a multi-floor survey's cross-floor readings
        (heard through the slab) would corrupt the fit.
        """
        if self.steps and len(self.steps[0]["true"]) == 3:
            raise ValueError(
                "calibration needs a single-floor survey; "
                "multi-floor steps carry a floor in each position")
        return calibrate_survey(self.beacons, self.steps)

    def to_dict(self) -> dict:
        """Serialise the survey to plain JSON-ready data."""
        radio_map = None
        if self.radio_map is not None:
            if isinstance(self.radio_map, MultiFloorFingerprintMap):
                radio_map = {"kind": "multi-floor",
                             "map": self.radio_map.to_dict()}
            else:
                radio_map = {"kind": "single",
                             "map": self.radio_map.to_dict()}
        return {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "beacons": [
                {"id": b.id, "x": b.x, "y": b.y,
                 "rssi_at_1m": b.rssi_at_1m,
                 "path_loss_exponent": b.path_loss_exponent,
                 "floor": b.floor}
                for b in self.beacons
            ],
            "steps": [
                {"true": list(step["true"]),
                 "rssi": dict(step["rssi"])}
                for step in self.steps
            ],
            "radio_map": radio_map,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Survey":
        """Rebuild a survey from ``to_dict`` output. Lossless."""
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise ValueError("not a beacon survey file")
        if data.get("version") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported survey format version: "
                f"{data.get('version')!r} (expected {FORMAT_VERSION})")
        beacons = [Beacon(b["id"], b["x"], b["y"],
                          rssi_at_1m=b.get("rssi_at_1m", -59.0),
                          path_loss_exponent=b.get("path_loss_exponent", 2.0),
                          floor=b.get("floor", 0))
                   for b in data.get("beacons", [])]
        if not beacons:
            raise ValueError("a survey needs at least one beacon")
        steps = [{"true": tuple(step["true"]),
                  "rssi": dict(step["rssi"])}
                 for step in data.get("steps", [])]
        radio_map = None
        raw_map = data.get("radio_map")
        if raw_map is not None:
            if raw_map.get("kind") == "multi-floor":
                radio_map = MultiFloorFingerprintMap.from_dict(
                    raw_map["map"])
            elif raw_map.get("kind") == "single":
                radio_map = FingerprintMap.from_dict(raw_map["map"])
            else:
                raise ValueError(
                    f"unknown radio map kind: {raw_map.get('kind')!r}")
        return cls(beacons, steps, radio_map)

    def save(self, path: str) -> None:
        """Write the survey to ``path`` as JSON."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path: str) -> "Survey":
        """Read a survey back from a file written by ``save``."""
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
