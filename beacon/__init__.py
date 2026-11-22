"""Beacon: a BLE indoor-positioning engine for hospitals.

Pure Python 3 standard library. No BLE hardware is needed to use it:
the positioning engine works on RSSI observations, and a simulator
generates noisy RSSI readings from a true path for testing.
"""

from .beacon import Beacon
from .calibration import calibrate_beacon, calibrate_survey, fit_path_loss
from .engine import PositioningEngine
from .filter import KalmanFilter, MovingAverageFilter
from .fingerprint import FingerprintMap, build_fingerprint_map, grid_points
from .rssi import distance_to_rssi, rssi_to_distance
from .simulate import simulate_walk
from .trilateration import trilaterate

__version__ = "0.6.0"

__all__ = [
    "Beacon",
    "PositioningEngine",
    "MovingAverageFilter",
    "KalmanFilter",
    "FingerprintMap",
    "build_fingerprint_map",
    "grid_points",
    "calibrate_beacon",
    "calibrate_survey",
    "fit_path_loss",
    "distance_to_rssi",
    "rssi_to_distance",
    "simulate_walk",
    "trilaterate",
    "__version__",
]
