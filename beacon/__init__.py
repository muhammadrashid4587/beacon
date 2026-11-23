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
from .rssi import distance_to_rssi, rssi_to_distance
from .simulate import simulate_multifloor_walk, simulate_walk, two_floor_walk
from .survey import Survey
from .trilateration import trilaterate

__version__ = "0.12.0"

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
    "trilaterate",
    "__version__",
]
