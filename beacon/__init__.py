"""Beacon: a BLE indoor-positioning engine for hospitals.

Pure Python 3 standard library. No BLE hardware is needed to use it:
the positioning engine works on RSSI observations, and a simulator
generates noisy RSSI readings from a true path for testing.
"""

from .beacon import Beacon
from .calibration import calibrate_beacon, calibrate_survey, fit_path_loss
from .engine import PositioningEngine
from .filter import KalmanFilter, MovingAverageFilter
from .drift import DriftAlarm, DriftMonitor
from .health import BeaconHealth, assess_beacon_health, healthy_beacons
from .fingerprint import (FingerprintMap, MultiFloorFingerprintMap,
                          build_fingerprint_map,
                          build_multifloor_fingerprint_map, grid_points)
from .floors import FloorEstimate, Position, estimate_floor
from .placement import (LayoutScore, PlacementAdvice, advise_placement,
                        placement_uncertainty, score_layout)
from .rssi import distance_to_rssi, rssi_to_distance
from .simulate import simulate_multifloor_walk, simulate_walk, two_floor_walk
from .survey import Survey
from .export import (export_track_csv, export_track_json,
                     load_track_csv, load_track_json, track_from_dict,
                     track_rows, track_to_dict)
from .map_overlay import (CoverageCell, HeatmapSummary, confidence_colour,
                        coverage_heatmap, heatmap_summary,
                        load_track_for_overlay, render_map_overlay,
                        write_map_overlay)
from .tracking import (ExcludedBeacon, LivePosition, LiveTracker,
                       TrackReport, evaluate_live_track)
from .trilateration import trilaterate
from .uncertainty import PositionUncertainty, estimate_uncertainty

__version__ = "0.21.0"

__all__ = [
    "Beacon",
    "PositioningEngine",
    "MovingAverageFilter",
    "KalmanFilter",
    "BeaconHealth",
    "assess_beacon_health",
    "healthy_beacons",
    "DriftAlarm",
    "DriftMonitor",
    "FingerprintMap",
    "MultiFloorFingerprintMap",
    "build_fingerprint_map",
    "build_multifloor_fingerprint_map",
    "grid_points",
    "FloorEstimate",
    "Position",
    "estimate_floor",
    "calibrate_beacon",
    "calibrate_survey",
    "fit_path_loss",
    "distance_to_rssi",
    "rssi_to_distance",
    "simulate_walk",
    "simulate_multifloor_walk",
    "two_floor_walk",
    "Survey",
    "ExcludedBeacon",
    "LivePosition",
    "LiveTracker",
    "TrackReport",
    "evaluate_live_track",
    "export_track_csv",
    "export_track_json",
    "load_track_csv",
    "load_track_json",
    "track_from_dict",
    "track_rows",
    "track_to_dict",
    "trilaterate",
    "confidence_colour",
    "coverage_heatmap",
    "CoverageCell",
    "heatmap_summary",
    "HeatmapSummary",
    "load_track_for_overlay",
    "render_map_overlay",
    "write_map_overlay",
    "PositionUncertainty",
    "estimate_uncertainty",
    "LayoutScore",
    "PlacementAdvice",
    "advise_placement",
    "placement_uncertainty",
    "score_layout",
    "__version__",
]
