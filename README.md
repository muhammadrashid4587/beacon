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
- Multi-floor support: beacons carry a `floor` attribute, the
  engine votes on which floor the device is on (closer beacons vote
  harder; beacons heard through a floor slab vote weakly), then
  localises within that floor's 2D plane — with hysteresis so the
  reported floor does not flicker near a stairwell, and a filter
  reset on every floor change.
- Multi-floor fingerprinting: one radio map per floor, the floor
  classified by which floor's map holds the nearest fingerprint in
  RSSI space, then the k-NN lookup runs within that floor's map
  only — no path-loss model involved at any step.
- Beacon health: a walked survey flags each beacon as ok, drifting
  (its RSSI no longer matches its calibrated constants), or dead
  (never heard), so a broken beacon can be recalibrated or excluded
  from the engine instead of poisoning every estimate.
- Calibration drift alarm: a live monitor watches the observation
  stream and raises an alarm when a beacon's rolling mean RSSI
  residual crosses the 5 dB threshold, so a beacon that goes bad
  mid-shift is caught while the ward is running, not at the next
  re-walk — and the alarm clears on its own when the beacon
  recovers.
- Trajectory smoothing: a recorded walk can be re-run through the
  Rauch-Tung-Striebel smoother, which revises each past position
  with the observations that came after it, so the replayed path
  is steadier and closer to the true path than the live estimates.
- Survey export/import: a walked survey (beacons with their
  calibration, the steps, and the built radio map) saves to a
  portable JSON file and loads back losslessly, so a survey can be
  shared or reused across runs without re-walking the ward.
- Position uncertainty: every raw estimate carries a 95%
  confidence ellipse, so a ward dashboard knows how much to trust
  each position — tight in the middle of the ward, stretched along
  the weak axis near an edge or where the beacons cluster on one
  side.
- Beacon placement advisor: given a room and a beacon count, it
  suggests where to mount the beacons so the 95% ellipse stays
  small across the whole room, instead of mounting them where it
  is convenient and finding out later that one corner is barely
  located.
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

In a multi-floor building, give each beacon its floor and use
`estimate_with_floor`, which returns a `Position` carrying the
floor and the confidence of the floor vote:

```python
from beacon import Beacon, PositioningEngine

beacons = [Beacon("A", 0, 0, floor=0), Beacon("B", 20, 0, floor=0),
           Beacon("C", 20, 12, floor=0), Beacon("D", 0, 12, floor=0),
           Beacon("E", 0, 0, floor=1), Beacon("F", 20, 0, floor=1),
           Beacon("G", 20, 12, floor=1), Beacon("H", 0, 12, floor=1)]
engine = PositioningEngine(beacons)
pos = engine.estimate_with_floor({"A": -80.1, "B": -84.3, "C": -82.0,
                                  "D": -79.5, "E": -62.4, "F": -70.8,
                                  "G": -68.2, "H": -60.1})
# pos.x, pos.y, pos.floor == 1, pos.floor_confidence
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

## Multi-floor

Give each beacon a `floor` (`Beacon("E", 0, 0, floor=1)`; the
default is 0, so single-floor code is unaffected) and the engine
votes on the floor before localising within it. Each heard beacon
votes for its own floor with weight `1 / distance` from the
path-loss model, so near beacons count more than far ones — and
beacons on an adjacent floor are heard through the slab, so their
apparent distance is inflated and they vote weakly. The winning
floor's share of the total weight is its confidence
(`estimate_floor` returns the vote, its confidence, and every
floor's score for inspection).

`estimate_with_floor` returns a `Position(x, y, floor,
floor_confidence)`. The floor only switches when the winning
floor's confidence clears 0.6 (configurable via
`floor_switch_confidence`); a 50/50 vote near a stairwell keeps the
current floor instead of flickering. A floor change resets the
smoothing filter, since the device has moved to a different plane.
When fewer than three beacons are heard on the winning floor, the
engine falls back to all heard beacons rather than failing — the
(x, y) degrades (cross-floor distances are inflated by the slab)
but the floor stays right. A floor with no beacons heard scores
zero and cannot win. `estimate` still returns `(x, y)`; in a
single-floor building the vote is trivial and the result is
identical to before.

The simulator models a floor as 3 m of vertical separation plus
15 dB of slab attenuation for beacons on a different floor from the
device (`simulate_multifloor_walk`, path points are
`(x, y, floor)`; `two_floor_walk` walks the ward downstairs then
upstairs).

Measured accuracy (simulator, 2 dB RSSI noise, two-floor ward,
4 beacons per floor): floor classification is 100% correct on
seeds 1, 2, 3, 7, 11, and within-floor mean error is 2.1–2.4 m —
comparable to the single-floor engine's 1.8–2.6 m on the same walks.
A noiseless vote from mid-ward reports the correct floor at 0.85
confidence.

## Multi-floor fingerprinting

Fingerprinting works per floor too, with no path-loss model
involved at any step. Survey each floor's grid (the same
`grid_points` cells, walked once per floor) and build one radio
map per floor:

```python
from beacon import build_multifloor_fingerprint_map, grid_points
from beacon.simulate import simulate_multifloor_walk

cells = grid_points(20.0, 12.0, spacing=2.0)
path = ([(x, y, 0) for x, y in cells]
        + [(x, y, 1) for x, y in cells])
survey = simulate_multifloor_walk(path, beacons, noise_db=2.0, seed=7)
maps = build_multifloor_fingerprint_map(beacons, survey)
pos = maps.locate_with_floor({"A": -80.1, "B": -84.3, "C": -82.0,
                              "D": -79.5, "E": -62.4, "F": -70.8,
                              "G": -68.2, "H": -60.1})
# pos.x, pos.y, pos.floor == 1, pos.floor_confidence
```

`locate_with_floor` classifies the floor first, then runs the k-NN
lookup within that floor's map only. The floor is classified by
matching the live RSSI vector against every floor's map: the floor
whose map holds the nearest fingerprint in RSSI space wins. That
works because a beacon heard through the slab reads ~15 dB weaker
than the same beacon surveyed on its own floor, so the wrong
floor's map matches badly (about 14 dB away) while the right
floor's matches within the noise (about 1 dB). Each floor's map is
built from that floor's beacons only, the same restriction the
trilateration engine applies when it localises within a floor, so
a cell on the wrong floor can never drag the estimate across
floors. `classify_floor` returns the floor, its confidence, and
every floor's nearest-fingerprint distance for inspection; the
confidence is the runner-up distance's share of the two best
distances, so it is near 1.0 when the winner matches far better
than any other floor and 0.5 when two floors match equally well.
`locate` returns just the `(x, y)` with the floor still classified
first. A floor whose beacons were not heard is skipped rather than
failing the classification, and an observation sharing no beacons
with any floor's map raises instead of returning a made-up
position.

Measured accuracy (simulator, 2 dB RSSI noise, two-floor ward,
4 beacons per floor, noisy survey): floor classification is 100%
correct on seeds 1, 2, 3, 7, 11, and within-floor mean error is
1.95–2.38 m — comparable to single-floor fingerprinting's
1.83–2.30 m on the same walks. A noiseless classification from
mid-ward reports the correct floor at 0.94–0.95 confidence.

## Beacon health

A beacon that has died, been moved, or had its transmit power
changed poisons every estimate it feeds, because its RSSI no longer
matches the path-loss model. Survey the ward once (a walked survey
with known positions, the same shape `simulate_walk` returns) and
check each beacon against the model:

```python
from beacon import assess_beacon_health, healthy_beacons
from beacon.simulate import rectangular_walk, simulate_walk

survey = simulate_walk(rectangular_walk(20, 12, 10), beacons,
                       noise_db=2.0, seed=7)
report = assess_beacon_health(beacons, survey)
# [BeaconHealth(id='A', status='ok', mean_residual_db=0.4, ...), ...]
engine = PositioningEngine(healthy_beacons(beacons, report))
```

Each entry reports a status — `ok`, `drifting` (mean RSSI residual
beyond the 5 dB threshold, so the beacon moved, its power changed,
or something is shielding it), or `dead` (never heard during the
survey) — plus the mean residual in dB and how many steps heard it.
`healthy_beacons` keeps only the `ok` beacons, so the engine is
built from beacons that can be trusted. A beacon whose transmit
power rose 15 dB pulls a noiseless estimate 5.6 m off; dropping
it brings the error back to 0 m. Multi-floor surveys are
rejected: the slab attenuation would look like drift.

## Calibration drift alarm

The health check above is a one-shot verdict on a finished survey.
A ward that is already running needs the same verdict live: a
beacon whose transmit power changed, that was moved, or that a
trolley is now shielding starts poisoning estimates the moment it
goes bad, and nobody re-walks the ward every hour to find out.
`DriftMonitor` watches the observation stream — feed it each
observation as it arrives, together with the device's position
(from a reference tag at a known spot, or any ground truth the
deployment has), and it keeps a rolling window of RSSI residuals
per beacon and alarms when a beacon's rolling mean residual
crosses the 5 dB threshold:

```python
from beacon import DriftMonitor
from beacon.simulate import rectangular_walk, simulate_walk

monitor = DriftMonitor(beacons)   # window=20 readings, 5 dB threshold
for step in simulate_walk(rectangular_walk(20, 12, 10), beacons,
                          noise_db=2.0, seed=7):
    monitor.feed(step)
monitor.alarms()   # [] while every beacon matches its calibration
```

`alarms()` returns the beacons currently alarming, each with its
rolling mean residual in dB and how many readings are in its
window; `alarm(id)` and `is_alarming(id)` check one beacon, and
`status(id)` reports the rolling mean even when it is not alarming.
A beacon needs at least 5 readings in its window before it can
alarm, so a single noisy reading never trips it — only a sustained
shift does. The alarm clears on its own when the rolling mean
comes back inside the threshold, so a beacon that recovers (the
trolley moved away) stops alarming without anyone resetting the
monitor. Unknown beacon ids in an observation are ignored rather
than failing the stream, and multi-floor positions are rejected,
since the slab attenuation would look like drift.

Measured behaviour (simulator, 2 dB RSSI noise): a healthy walk
raises no alarms; a beacon reading 12 dB too loud alarms with a
rolling mean of about +12 dB while the other three stay silent; a
beacon reading 8 dB too quiet alarms the other way. A short drift
that leaves the 5-reading window stops alarming on its own, and
feeding a healthy walk after a drifted one clears the alarm.

## Trajectory smoothing

The live filter is causal: each estimate only uses the observations
up to that step, so it lags behind turns and keeps some jitter. For
a recorded walk the future is already known, so the walk can be
smoothed after the fact. `smooth_walk` runs the whole walk through
the Kalman filter, then runs the Rauch-Tung-Striebel backward
recursion over the recorded forward pass, revising each past
position with the observations that came after it:

```python
from beacon import Beacon, PositioningEngine
from beacon.simulate import rectangular_walk, simulate_walk

beacons = [Beacon("A", 0, 0), Beacon("B", 20, 0),
           Beacon("C", 20, 12), Beacon("D", 0, 12)]
steps = simulate_walk(rectangular_walk(20, 12, 10), beacons,
                      noise_db=2.0, seed=7)
engine = PositioningEngine(beacons)
smoothed = engine.smooth_walk([step["rssi"] for step in steps])
# one (x, y) per step, in order; the last one equals the last live
# estimate, since nothing comes after it to revise it with
```

`KalmanFilter.smooth()` exposes the same recursion directly for a
filter fed by hand, and returns `[]` before any update or after a
reset. Smoothing needs the Kalman filter: the moving-average
opt-out keeps no forward-pass history, so `smooth_walk` raises on
an engine built with `filter="moving_average"`. A walk that changes
floor is rejected too — the floor change resets the filter
mid-walk, so only the final segment's history would survive; split
the walk per floor instead.

Measured accuracy (simulator, 2 dB RSSI noise, default ward walk):
smoothing cuts the mean error on every seed 1, 2, 3, 7, 11 — seed 1:
1.97 m to 1.31 m; seed 2: 2.57 m to 1.71 m; seed 3: 2.63 m to
1.83 m; seed 7: 2.16 m to 1.48 m; seed 11: 1.83 m to 1.18 m. A
stationary device's jitter (3 dB noise, 60 steps) drops from 5.32 m
to 3.15 m of combined axis standard deviation.

## Survey export/import

Walking a ward once is the expensive part of fingerprinting, so
the survey should survive the run that collected it. A `Survey`
bundles the beacons (positions plus their fitted calibration
constants), the walked steps, and the radio map built from them,
and saves the whole thing as one JSON file:

```python
from beacon import Survey
from beacon.simulate import simulate_walk, grid_points

cells = grid_points(20.0, 12.0, spacing=2.0)
steps = simulate_walk(cells, beacons, noise_db=2.0, seed=7)
survey = Survey.build(beacons, steps)   # builds the radio map
survey.save("ward-a-survey.json")

loaded = Survey.load("ward-a-survey.json")
x, y = loaded.fingerprint_map().locate({"A": -62.1, "B": -71.4})
```

The round trip is lossless: beacons keep their calibrated
`rssi_at_1m` and path-loss exponent, steps keep their exact
positions and RSSI values, and the radio map keeps its per-cell
sums and sample counts rather than just the averaged fingerprints
— so samples added after an import are averaged with the original
weights, and the imported map locates identically to the original.
A survey imported without a map (`Survey(beacons, steps)`,
`radio_map=None`) rebuilds it from the steps on demand, and a
map-only survey (no steps) still locates. Multi-floor surveys
round-trip the same way, one map per floor. `Survey.from_dict`
rejects files that are not beacon surveys and versions it does
not understand, instead of loading garbage. `calibrated_beacons()`
fits the beacons from the survey's steps; it refuses multi-floor
surveys, whose cross-floor readings would corrupt the fit.

## Position uncertainty

A position estimate on its own does not say how much to trust it.
A device in the middle of the ward, heard by beacons spread around
it, is pinned down tightly; the same device near the ward's edge,
heard by beacons all on one side, is well located along one axis
and barely located across it. `estimate_with_uncertainty` returns
the raw estimate together with a 95% confidence ellipse for it:

```python
from beacon import Beacon, PositioningEngine

beacons = [Beacon("A", 0, 0), Beacon("B", 20, 0),
           Beacon("C", 20, 12), Beacon("D", 0, 12)]
engine = PositioningEngine(beacons)
x, y, unc = engine.estimate_with_uncertainty(
    {"A": -62.1, "B": -71.4, "C": -68.0, "D": -59.8})
# unc.semi_major, unc.semi_minor (metres), unc.angle_deg
# (orientation of the major axis, degrees from +x)
unc.contains(10.0, 6.0, (x, y))   # is a point inside the ellipse?
```

The ellipse comes from the weighted least-squares covariance of
the trilateration: each Jacobian row is the unit vector from a
beacon to the estimate, weighted by `1 / sigma_d`, where the
distance noise `sigma_d` follows from the RSSI noise
(`sigma_rssi_db`, 2.0 dB by default) through the log-distance
model — a steeper path-loss exponent turns the same RSSI noise
into a smaller distance error, so calibrated beacons get tighter
ellipses. The eigenvalues of the covariance give the axes, scaled
by the 95% chi-square quantile for 2 degrees of freedom. The
ellipse describes one raw trilateration, not the filtered track:
it is the uncertainty of the geometry and the RSSI noise at this
step. Fewer than three beacons, or a geometry whose covariance is
singular (the estimate sitting on the beacon line), raises instead
of returning a made-up ellipse.

Measured behaviour (simulator, 2 dB RSSI noise, 200 trials per
seed): the true position lands inside the 95% ellipse in 90–95%
of trials across seeds 1, 2, 3, 7, 11 — close to the nominal 95%,
slightly under it because the linearisation and the estimated
distances are themselves noisy. A symmetric four-beacon square
gives a near-circular ellipse; beacons clustered on one wall
stretch the ellipse more than 2:1 along the weak axis.

## Beacon placement

Beacons usually end up where it is convenient to mount them —
the corners, the walls — not where the geometry is good, and
geometry is what decides how well a device can be located. The
confidence ellipse measures that directly: at any point in the
room, the ellipse's semi-major axis is the worst-case 95%
positioning error there. `score_layout` averages that axis (and
reports the worst point) over a grid of test points, and
`advise_placement` picks the layout greedily — the first beacon
nearest the room centre, the second as far from it as possible,
each further beacon where it most reduces the mean semi-major
axis:

```python
from beacon import advise_placement, score_layout

advice = advise_placement(20.0, 12.0, 4)   # a 20 x 12 m ward
# advice.beacons: P1 (9, 5), P2 (19, 11), P3 (5, 11), P4 (9, 1)
# advice.score.mean_semi_major: 4.30 m
score_layout(beacons, 20.0, 12.0)          # score any layout
```

The greedy choice optimises the mean, so it trades the worst
point for the average one: on the default ward the four-corner
layout averages 6.30 m with a worst point of 8.33 m, while the
advised layout averages 4.30 m but its worst point (the far
corner, thinly covered) is 11.16 m. If worst-case coverage
matters, add beacons instead of rearranging four: six advised
beacons average 2.78 m with a worst point of 7.20 m, better than
the corners on both measures. Adding a beacon can only add
information to the least-squares system, so the mean uncertainty
never gets worse as beacons are added. The advice assumes the
log-distance model with the beacons' own exponents and the
stated RSSI noise (2.0 dB by default), and it knows nothing
about walls, metal beds, or dead zones in the real ward — treat
the suggested spots as a starting point, then survey and
calibrate.

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
- Multi-floor support (beacon z coordinate, floor detection) —
  done: `Beacon.floor`, a `1 / distance` floor vote with a
  confidence score, `estimate_with_floor` returning a `Position`
  with the floor attached, 0.6-confidence hysteresis against floor
  flicker, a filter reset on floor change, a fallback to all heard
  beacons when fewer than three are heard on the winning floor, and
  a multi-floor simulator (3 m floor height, 15 dB slab
  attenuation) with a two-floor walk fixture (v0.7.0); floor
  classification 100% on seeds 1, 2, 3, 7, 11, within-floor mean
  error 2.1–2.4 m, comparable to single-floor mode.
- Multi-floor fingerprinting (radio map per floor) — done:
  `MultiFloorFingerprintMap` with one `FingerprintMap` per floor,
  `classify_floor` matching the live RSSI vector against every
  floor's map (the floor whose map holds the nearest fingerprint
  wins), `locate_with_floor` returning a `Position` with the floor
  and its confidence, and `build_multifloor_fingerprint_map`
  building the per-floor maps from a multi-floor survey (v0.8.0);
  floor classification 100% on seeds 1, 2, 3, 7, 11, within-floor
  mean error 1.95–2.38 m, comparable to single-floor
  fingerprinting's 1.83–2.30 m, with no path-loss model involved.
- Beacon health / outlier detection (dead or drifting beacons) —
  done: `assess_beacon_health` checks each beacon against a walked
  survey and reports `ok`, `drifting` (mean RSSI residual beyond
  the 5 dB threshold), or `dead` (never heard), and
  `healthy_beacons` builds the engine from the `ok` beacons only
  (v0.9.0); excluding a beacon whose transmit power rose 15 dB
  brings a noiseless estimate from 5.6 m off back to 0 m.
- Trajectory smoothing (replay a recorded walk with the future
  known) — done: `PositioningEngine.smooth_walk` runs a walk
  through the Kalman filter and then the Rauch-Tung-Striebel
  backward recursion over the recorded forward pass, revising each
  past position with the observations that came after it, and
  `KalmanFilter.smooth()` exposes the recursion directly (v0.10.0);
  mean error falls on every seed 1, 2, 3, 7, 11 (1.97→1.31 m,
  2.57→1.71 m, 2.63→1.83 m, 2.16→1.48 m, 1.83→1.18 m), and a
  stationary device's jitter drops from 5.32 m to 3.15 m. A walk
  that changes floor is rejected (split it per floor), and the
  moving-average opt-out raises, since it keeps no forward-pass
  history to smooth over.
- Survey export/import (share a walked survey across runs) —
  done: `Survey` bundles the beacons (positions plus fitted
  calibration), the walked steps, and the built radio map into one
  JSON file (`save`/`load`, `to_dict`/`from_dict`), and the round
  trip is lossless — the map keeps its per-cell sums and sample
  counts, so samples added after an import average with the
  original weights, and the imported map locates identically to
  the original (v0.11.0); a map-less survey rebuilds its map from
  the steps on demand, a map-only survey still locates, and
  multi-floor surveys round-trip one map per floor.
- Calibration drift alarm (catch a beacon going bad while the
  ward is running) — done: `DriftMonitor` watches the live
  observation stream, keeps a rolling window of RSSI residuals
  per beacon (window 20, 5 dB threshold, 5 readings before it
  can alarm), and raises a `DriftAlarm` when a beacon's rolling
  mean residual crosses the threshold, clearing it on its own
  when the beacon recovers (v0.12.0); a healthy walk raises no
  alarms, a beacon 12 dB too loud alarms at about +12 dB while
  the others stay silent, and a short drift that leaves the
  window stops alarming by itself.
- Position uncertainty (confidence ellipse per estimate) —
  done: `estimate_uncertainty` derives a 95% confidence ellipse
  from the trilateration's weighted least-squares covariance,
  with per-beacon distance noise propagated from the RSSI noise
  through the log-distance model, and
  `PositioningEngine.estimate_with_uncertainty` returns the raw
  estimate with its ellipse attached (v0.13.0); the true
  position lands inside the ellipse in 90–95% of trials across
  seeds 1, 2, 3, 7, 11, a symmetric layout gives a near-circular
  ellipse, and beacons clustered on one wall stretch it more
  than 2:1 along the weak axis.
- Beacon placement advisor (where to mount the beacons) —
  done: `placement_uncertainty` gives the 95% ellipse a layout
  would give at any point in the room, `score_layout` averages
  that semi-major axis (and reports the worst point) over a
  test grid, and `advise_placement` picks the layout greedily —
  first beacon nearest the room centre, second farthest from
  it, each further beacon where it most reduces the mean
  semi-major axis (v0.14.0); on the default 20 x 12 m ward the
  advised four-beacon layout averages 4.30 m against the
  corners' 6.30 m, though its worst point is worse (11.16 m
  vs 8.33 m) since the mean-optimal layout leaves the far
  corner thin — six advised beacons beat the corners on both
  measures (2.78 m mean, 7.20 m worst), and adding a beacon
  never worsens the mean.
  The roadmap grows as new needs surface — the placement
  advisor is the latest addition.
