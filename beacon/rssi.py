"""Log-distance path-loss model: RSSI <-> distance.

Model (the standard indoor log-distance model):

    RSSI(d) = rssi_at_1m - 10 * n * log10(d)

where ``n`` is the path-loss exponent (2.0 is free space; indoor
spaces with walls and people are usually 2.0-4.0) and ``rssi_at_1m``
is the beacon's calibrated RSSI at 1 metre. The inverse converts an
observed RSSI back into an estimated distance.

The model is only as good as its calibration: ``n`` and
``rssi_at_1m`` must be fitted to the real space. The simulator uses
n = 2.0 and rssi_at_1m = -59 dBm as defaults.
"""

import math

MIN_DISTANCE_M = 0.1


def distance_to_rssi(distance_m: float, rssi_at_1m: float = -59.0,
                     path_loss_exponent: float = 2.0) -> float:
    """Expected RSSI (dBm) observed at ``distance_m`` from a beacon."""
    if distance_m <= 0:
        raise ValueError("distance must be positive")
    d = max(distance_m, MIN_DISTANCE_M)
    return rssi_at_1m - 10.0 * path_loss_exponent * math.log10(d)


def rssi_to_distance(rssi_dbm: float, rssi_at_1m: float = -59.0,
                     path_loss_exponent: float = 2.0) -> float:
    """Estimated distance (m) from an observed RSSI (dBm)."""
    if path_loss_exponent <= 0:
        raise ValueError("path_loss_exponent must be positive")
    return 10.0 ** ((rssi_at_1m - rssi_dbm) / (10.0 * path_loss_exponent))
