# beacon

A BLE indoor-positioning engine for hospitals. Pure Python 3 standard
library, no dependencies, no BLE hardware needed.

## What it is

- Fixed beacons with known `(x, y)` positions in a building.
- A log-distance path-loss model converting RSSI to distance.
- Trilateration (least squares, 3+ beacons) to estimate position,
  refined by Gauss-Newton from several seed points (the linear
  solution, the beacon centroid, each beacon) keeping the lowest
  residual, so a bad initial guess cannot trap the refinement.
- A constant-velocity Kalman filter smooths jittery estimates and
  tracks a walking device with less lag than a moving average; the
  moving average is still available as an opt-out
  (`PositioningEngine(..., filter="moving_average")`).
- A simulator that generates noisy RSSI readings from a true path,
  so the engine can be tested without real BLE hardware.
- Per-beacon calibration: fit each beacon's `rssi_at_1m` and
  path-loss exponent `n` from a walked survey (known positions plus
  observed RSSI), so the engine uses the real space's constants
  instead of the defaults.
- Fingerprinting mode: a radio map of averaged RSSI vectors per
  survey grid cell, located by k-nearest-neighbour matching in RSSI
  space — no path-loss model involved, so it keeps working where
  trilateration's distance inversion explodes.
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

The CLI uses the Kalman filter by default; pass
`--filter moving_average` to compare against the old default.

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
Gauss-Newton on the true distance residuals; the refinement runs from
several seed points (linear solution, beacon centroid, each beacon)
and keeps the lowest-residual result, so a bad initial guess cannot
trap it or let it diverge — on 2,000 noisy walks the worst
single-seed error was ~32,500 m against ~93 m multi-seed. The result
passes through a constant-velocity Kalman filter (state `(x, y, vx,
vy)`, one step per observation), which predicts the next position
from the estimated velocity and so lags less behind a walking device
than the moving average it replaced as the default in v0.5.0; the
moving average (window 5) is still available via
`filter="moving_average"`. Each beacon carries its
own `rssi_at_1m` and `n` (defaults -59 dBm and 2.0); the engine uses
each beacon's own constants unless a single global exponent is passed
to `PositioningEngine`, which overrides them. Passing
`filter="moving_average"` swaps the Kalman filter back for the
moving average.

## Calibration

Walk the space once, recording the RSSI from each beacon at known
positions, then fit each beacon's constants with least squares
(`RSSI` against `log10(d)`):

```python
from beacon import Beacon, PositioningEngine, calibrate_survey
from beacon.simulate import rectangular_walk, simulate_walk

beacons = [Beacon("A", 0, 0), Beacon("B", 20, 0),
           Beacon("C", 20, 12), Beacon("D", 0, 12)]
survey = simulate_walk(rectangular_walk(20, 12, 10), beacons,
                       noise_db=0.0, seed=7)   # a walked survey
calibrated = calibrate_survey(beacons, survey)
engine = PositioningEngine(calibrated)
```

`calibrate_beacon(beacon, samples)` fits one beacon from
`((x, y), rssi)` pairs; `fit_path_loss(samples)` fits
`(rssi_at_1m, n)` from `(distance, rssi)` pairs directly. A beacon
with too little survey data (fewer than two distinct distances) is
left unchanged, and a fit is rejected when RSSI does not decrease
with distance.

## Fingerprinting

Trilateration trusts the path-loss model: when the model is wrong
the distance inversion explodes and the estimate lands kilometres
away. Fingerprinting trusts the survey instead. Walk a grid over
the space once, record the RSSI vector at each cell, and locate by
matching the live vector against the map:

```python
from beacon import build_fingerprint_map, grid_points
from beacon.simulate import simulate_walk

cells = grid_points(20.0, 12.0, spacing=2.0)   # 60 cells, 1 m margin
survey = simulate_walk(cells, beacons, noise_db=0.0, seed=7)
radio_map = build_fingerprint_map(beacons, survey)
x, y = radio_map.locate({"A": -62.1, "B": -71.4, "C": -68.0, "D": -59.8})
```

`locate` is k-nearest-neighbour in RSSI space (root-mean-square
distance over the beacons heard) and returns the
inverse-distance-weighted centroid of the `k` nearest cells
(`k=3` by default); `k=1` returns the single best-matching cell.
An exact fingerprint match returns that cell directly. Repeated
survey passes over a cell are averaged. A beacon unheard in an
observation is left out of the comparison rather than filled with
a made-up value, so a partially-heard vector still locates.
`radio_map.nearest(observations, k)` returns the ranked cells with
their RSSI distances for inspection.

Measured accuracy (simulator, 2 dB RSSI noise, 2 m grid): mean
error about 2.1 m with a noisy survey, comparable to trilateration's
2.2 m on the same walks — and unlike trilateration it needs no
calibrated path-loss constants at all.

## Measured accuracy (simulator, not the real world)

Default ward walk, 2 dB RSSI noise: mean error about 2.2 m (1.8-2.6 m
across seeds) with the Kalman filter, which became the default in
v0.5.0; the moving average it replaced gave 2.3-3.0 m on the same
walks (seed 1: 1.97 m vs 2.77 m; seed 2: 2.57 m vs 2.96 m; seed 3:
2.63 m vs 2.69 m; seed 7: 2.16 m vs 2.73 m; seed 11: 1.83 m vs
2.30 m). The test suite asserts mean error under 4.0 m on a fixed
seed as a regression guard, and under 4.0 m on every seed 1, 2, 3, 7,
11 for the default engine. Real hardware will be worse until
calibrated.

Calibration matters more than the filter: in a ward whose true
constants are `n = 3.0` and `rssi_at_1m = -65 dBm`, the default
assumptions (`n = 2.0`, `-59 dBm`) give a mean error of about
5.7×10⁷ m — the distance inversion explodes and trilateration
diverges, so the uncalibrated engine is unusable there. Calibrating
from one noiseless survey walk recovers the true constants exactly
and brings mean error back to 2.2–2.5 m across seeds 1, 2, 3, 7, 11
(seed 7: 2.46 m mean, 2.37 m median).

## Roadmap

- Kalman filter in place of the moving average — done: the
  filter is implemented (v0.3.0), the multi-seed accuracy
  comparison is done (Kalman beats the moving average on seeds 1,
  2, 3, 7, 11, v0.4.0), trilateration refines from several seed
  points keeping the lowest residual, and the Kalman filter is now
  the default, with the moving average kept as an opt-out
  (`filter="moving_average"`, v0.5.0)
- Fingerprinting mode (RSSI vectors per room/zone) — done:
  `FingerprintMap` with a surveyed radio map, k-NN `locate()` with
  inverse-distance-weighted centroid (`k=3` default, `k=1` returns
  the best cell), `grid_points` survey grids, repeated samples
  averaged per cell, unheard beacons left out of the comparison
  (v0.6.0); noisy-survey mean error ~2.1 m, comparable to
  trilateration, with no path-loss model involved
- Multi-floor support (beacon z coordinate, floor detection)
