# beacon

A BLE indoor-positioning engine for hospitals. Pure Python 3 standard
library, no dependencies, no BLE hardware needed.

## What it is

- Fixed beacons with known `(x, y)` positions in a building.
- A log-distance path-loss model converting RSSI to distance.
- Trilateration (least squares, 3+ beacons) to estimate position.
- A moving-average filter to smooth jittery estimates.
- A simulator that generates noisy RSSI readings from a true path,
  so the engine can be tested without real BLE hardware.
- A CLI that runs a simulated walk and prints estimated vs true
  positions with error statistics.

## What it is not

- It does not talk to real BLE hardware. There is no scanner, no
  Bluetooth stack, no phone app. The engine consumes RSSI
  observations; in production those would come from a BLE scanner.
- Real-world accuracy depends on calibration. The defaults
  (`rssi_at_1m = -59 dBm`, path-loss exponent `n = 2.0`) are typical
  BLE values, not measurements from any real ward. Walls, people,
  metal beds and multipath all shift RSSI; expect to re-fit `n` and
  `rssi_at_1m` per space, and expect metres of error in a busy ward
  until you do.

## Quickstart

```sh
make test   # run the tests
make demo   # simulated walk through a 20 x 12 m ward
```

Or directly:

```sh
python3 -m beacon.cli --steps 10 --noise 2.0 --seed 7
```

Use it as a library:

```python
from beacon import Beacon, PositioningEngine

beacons = [Beacon("A", 0, 0), Beacon("B", 20, 0),
           Beacon("C", 20, 12), Beacon("D", 0, 12)]
engine = PositioningEngine(beacons)
x, y = engine.estimate({"A": -62.1, "B": -71.4, "C": -68.0, "D": -59.8})
```

## Model

RSSI follows the log-distance model `RSSI(d) = rssi_at_1m - 10*n*log10(d)`.
Distances feed a linearised least-squares trilateration, refined by
Gauss-Newton on the true distance residuals; the result passes through
a moving average (window 5) to reduce jitter.

## Measured accuracy (simulator, not the real world)

Default ward walk, 2 dB RSSI noise: mean error about 2.7 m (2.3-2.9 m
across seeds). The test suite asserts mean error under 4.0 m on a
fixed seed as a regression guard. Real hardware will be worse until
calibrated.

## Roadmap

- Per-beacon calibration from a walked survey (fit `n`, `rssi_at_1m`)
- Kalman filter in place of the moving average
- Fingerprinting mode (RSSI vectors per room/zone)
- Multi-floor support (beacon z coordinate, floor detection)
