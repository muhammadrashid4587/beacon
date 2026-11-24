"""Beacon: a BLE indoor-positioning engine for hospitals.

Pure Python 3 standard library. No BLE hardware is needed to use it:
the positioning engine works on RSSI observations, and a simulator
generates noisy RSSI readings from a true path for testing.
"""

from .battery import BatteryModel
from .beacon import Beacon
from .calibration import calibrate_beacon, calibrate_survey, fit_path_loss
from .deployment import (DEFAULT_LOSS_THRESHOLD_DB,
                         DEFAULT_NOISE_THRESHOLD_DB,
                         MIN_BEACONS_TO_TRIANGULATE, BeaconForecast,
                         DeploymentPlan, forecast_beacon, plan_deployment)
from .engine import PositioningEngine
from .filter import KalmanFilter, MovingAverageFilter
from .drift import DriftAlarm, DriftClassification, DriftMonitor
from .health import BeaconHealth, assess_beacon_health, healthy_beacons
from .fingerprint import (FingerprintMap, MultiFloorFingerprintMap,
                          build_fingerprint_map,
                          build_multifloor_fingerprint_map, grid_points)
from .floors import FloorEstimate, Position, estimate_floor
from .placement import (LayoutScore, PlacementAdvice, advise_placement,
                        placement_uncertainty, score_layout)
from .replay import render_track_replay, write_track_replay
from .rssi import distance_to_rssi, rssi_to_distance
from .simulate import simulate_multifloor_walk, simulate_walk, two_floor_walk
from .survey import Survey
from .export import (export_track_csv, export_track_json,
                     load_track_csv, load_track_json, track_from_dict,
                     track_rows, track_to_dict)
from .map_overlay import (CoverageCell, HeatmapSummary, PlannerCell,
                        PlannerHeatmapSummary, confidence_colour,
                        coverage_heatmap, heatmap_summary,
                        load_track_for_overlay, planner_colour,
                        planner_heatmap, planner_heatmap_summary,
                        render_map_overlay, write_map_overlay)
from .multi_device import MultiDeviceTracker
from .schedule import (REASON_DARK, REASON_LOSS, REASON_NEVER,
                       REASON_NOISE, ReplacementSchedule, SwapEntry,
                       replacement_schedule)
from .tracking import (DEFAULT_FORECAST_WINDOW_HOURS, ExcludedBeacon,
                       ForecastFlag, LivePosition, LiveTracker,
                       TrackReport, drift_alerts, evaluate_live_track,
                       forecast_alerts, forecast_flags)
from .trilateration import trilaterate
from .uncertainty import PositionUncertainty, estimate_uncertainty
from .validation import (DEFAULT_MAX_WORST_SEMI_MAJOR_M,
                         FloorLayoutVerdict, LayoutValidation,
                         validate_floor_layouts)

__version__ = "0.38.0"

__all__ = [
    "BatteryModel",
    "Beacon",
    "PositioningEngine",
    "MovingAverageFilter",
    "KalmanFilter",
    "BeaconHealth",
    "assess_beacon_health",
    "healthy_beacons",
    "DriftAlarm",
    "DriftClassification",
    "DriftMonitor",
    "BeaconForecast",
    "DeploymentPlan",
    "DEFAULT_LOSS_THRESHOLD_DB",
    "DEFAULT_NOISE_THRESHOLD_DB",
    "MIN_BEACONS_TO_TRIANGULATE",
    "forecast_beacon",
    "plan_deployment",
    "ReplacementSchedule",
    "SwapEntry",
    "REASON_DARK",
    "REASON_LOSS",
    "REASON_NOISE",
    "REASON_NEVER",
    "replacement_schedule",
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
    "ForecastFlag",
    "DEFAULT_FORECAST_WINDOW_HOURS",
    "LivePosition",
    "LiveTracker",
    "MultiDeviceTracker",
    "TrackReport",
    "evaluate_live_track",
    "drift_alerts",
    "forecast_alerts",
    "forecast_flags",
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
    "PlannerCell",
    "PlannerHeatmapSummary",
    "planner_colour",
    "planner_heatmap",
    "planner_heatmap_summary",
    "load_track_for_overlay",
    "render_map_overlay",
    "render_track_replay",
    "write_map_overlay",
    "write_track_replay",
    "PositionUncertainty",
    "estimate_uncertainty",
    "LayoutScore",
    "PlacementAdvice",
    "FloorLayoutVerdict",
    "LayoutValidation",
    "DEFAULT_MAX_WORST_SEMI_MAJOR_M",
    "validate_floor_layouts",
    "advise_placement",
    "placement_uncertainty",
    "score_layout",
    "__version__",
]
