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
- Live tracking mode: a tracker that consumes a stream of RSSI
  readings as they arrive and emits a live position per reading,
  with the 95% confidence ellipse attached — partial scans merge
  across beacons, stale readings age out, and a beacon dropping
  out mid-stream degrades the track instead of breaking it.
- Beacon-failure fallback: a beacon that goes quiet mid-track is
  excluded from the estimate once it has been unheard for longer
  than the dark window, and the fallback is recorded on the live
  position — which beacon went dark and why — so a dashboard can
  see the track running on three beacons instead of four, and a
  beacon heard again rejoins after three consecutive readings,
  so one stray scan cannot flap a dying beacon back into the
  estimate.
- A live track accuracy report: score a recorded walk against
  ground truth — mean, median, 90th percentile and worst error,
  plus coverage (how much of the walk was actually located) and
  floor accuracy on multi-floor walks.
- A live track export: save a recorded track as JSON or CSV for a
  ward dashboard — one row per located position with the
  timestamp, the position, the floor, the 95% confidence ellipse,
  the beacons used, the beacons excluded, and the track's
  coverage — so the track survives the run that produced it.
- A ward-map overlay: plot an exported live track on the ward
  plan as a self-contained SVG — ward outline, beacons at their
  known positions, the track path with start and end markers,
  and one 95% confidence ellipse per point, coloured by how wide
  it is, with a legend — so a dashboard can show where the
  device went and how much to trust each point.
- A per-beacon coverage heatmap on the ward-map overlay: shade
  the ward grid by the positioning quality the beacon layout
  gives at each cell, so a dashboard can see where the ward is
  well covered and where a device would be guessing — before
  any device walks there.
- A live-track replay: play an exported track back as an
  animated SVG — a dot glides along the path, the trail draws
  itself behind it, the 95% confidence ellipse for the current
  step appears with the dot, and a timestamp label steps through
  the track's own times — SMIL animation, no JavaScript, looping
  by default, so a dashboard can watch a shift play back.
- Multi-device tracking: one tracker for the whole ward, with a
  separate live track per device id — a patient tag, a nurse's
  badge, a pump on a trolley — each with its own filter and floor
  state, so the tracks never smear into each other.
- Per-floor layout validation: check every floor's beacons before
  deployment — a floor with fewer than three beacons, or with
  beacons so clustered they cannot triangulate, is flagged
  `cannot_triangulate`, and a floor whose worst 95% error is above
  the 10 m limit is flagged `thin`, with a plain-language warning
  saying which floor to fix and why.
- Battery degradation: a beacon's RSSI weakens and gets noisier
  as its battery drains, and the beacon goes dark below a cutoff
  — so a walk with a draining beacon shows the error growing
  while the beacon is dying, and the battery level rides on the
  live positions and the exported track rows.
- Deployment planner: from each beacon's battery model and its
  current charge, the hours until it goes dark, until its RSSI
  loss crosses 3 dB, and until its extra noise crosses 2 dB —
  plus when the ward drops below three lit beacons and can no
  longer triangulate at all, so replacement cells are scheduled
  before the ward goes blind instead of after.
- Battery-swap replacement schedule: the planner's forecasts as
  a work order — each beacon's swap-by time, soonest first, with
  the milestone that drives it (dark, 3 dB loss, or 2 dB noise)
  and an optional lead time so the cells are ordered ahead of
  the failure, plus the ward's hard deadline for keeping
  triangulation.
- A planner-aware ward-map overlay: the overlay takes a
  deployment plan — each beacon is annotated with its forecast
  horizon and coloured by its forecast status, and the planner
  heatmap shades each cell by when its coverage first degrades,
  so the map shows not just the coverage now but when it will
  thin.
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
make live-demo   # the same walk, streamed through live tracking mode
```

Or directly:

```sh
python3 -m beacon.cli --steps 10 --noise 2.0 --seed 7
python3 -m beacon.tracking --demo   # live tracking on a simulated walk
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

## Classifying drift: battery sag or a moved beacon

An alarm says a beacon drifted, not why — and the fix depends on
the why. A dying battery means send someone with a replacement
cell; a moved beacon means re-survey its position; a noisy beacon
means check the radio. `DriftMonitor.classify(id)` names the
cause, as a `DriftClassification` with a `verdict` — `healthy`,
`battery-sag`, `moved`, `noisy` or `uncertain` — plus the numbers
the verdict rests on (window mean and spread, time slope and
correlation, revisit delta and pair count) and a `reason` in
words. `classifications()` classifies every beacon at once.

The three causes leave different signatures, and the test that
separates a sag from a move is the revisit: the monitor keeps a
history (500 readings per beacon by default) alongside the
rolling window, and pairs each recent reading with the closest
earlier reading at (nearly) the same spot, at least 10 readings
earlier. A draining battery reads weaker at the same spot over
time — the `BatteryModel` loses RSSI with the square of the
charge used and adds noise as the cell drains — while a move
cannot change what a beacon reads like at a fixed spot, so a
falling revisit delta means sag. A moved beacon is the opposite:
its residual is stable at the same spot but wrong by a
position-dependent amount, so the window's residuals spread out
(std above 2.5 dB) around a shifted mean (beyond 1.5 dB). A
noisy beacon's spread is more than twice the expected RSSI noise
(2 dB by default, so above 4 dB) with no steady decline. A
steady offset fits none of these — a battery that sagged and
then stabilised looks exactly like a miscalibration — so the
classifier says `uncertain` instead of guessing.

The revisit test is what makes this work: within a single pass,
time and position are confounded (the device walks from one side
of the ward to the other as the battery drains), so a declining
residual could be either a sag or a move. Walk the loop twice
and the two come apart. Without revisits the classifier falls
back to the within-window trend — a strong monotonic decline, or
a second half much weaker than the first — which catches a fast
sag most of the time but cannot rule a move out, and the reason
says which evidence was used.

```sh
python3 -m beacon.cli --drift-classify --seed 7
python3 -m beacon.cli --drift-classify --seed 7 --battery-degrade D
python3 -m beacon.cli --drift-classify --seed 7 --move-beacon D:5,-5
```

`--drift-classify` walks the loop twice and prints each beacon's
verdict with its mean and spread, revisit delta and pair count,
and the reason. `--battery-degrade D` drains D at 0.3 per hour
(120 s steps, level 1.0 down to 0.20 across the two loops, still
heard throughout); `--move-beacon D:5,-5` simulates D displaced
5 m east and 5 m south of its calibrated corner, at (5, 7).

Measured behaviour (simulator, 2 dB RSSI noise, two-loop ward
walk, beacon D, seeds 1, 2, 3, 7, 11 — the other three beacons
stay `healthy` in every scenario):

| scenario | verdicts |
|---|---|
| healthy | `healthy` 5/5 (mean -0.90 to +0.38 dB) |
| D draining, 0.3/hr | `battery-sag` 5/5 (mean -2.17 to -4.73 dB, revisit delta -1.97 to -5.50 dB) |
| D moved (0, 12) to (5, 7) | `moved` 4/5 (mean +3.12 to +4.21 dB, spread 2.60 to 3.23 dB, revisit delta near zero); seed 11 `uncertain` |
| D moved 1.4 m, to (1, 11) | `uncertain` 3/5, `healthy` 2/5 — never `moved` or `battery-sag` |
| D with 5 dB extra noise | `noisy` 5/5 (spread 4.33 to 6.68 dB) |
| D steady -6 dB offset | `uncertain` 4/5; seed 11 misread as `battery-sag` |

The misses are honest limits, not bugs to tune away. The slow
sag's mean residual (-2.17 to -4.73 dB) never reaches the 5 dB
alarm threshold, so the v0.25.0 alarm misses it entirely — the
classifier catches it through the revisits. A move under about
2 m is invisible at 2 dB noise: its residual pattern is smaller
than the noise, so the classifier calls it `healthy` or
`uncertain` rather than inventing a cause. And a steady offset
on seed 11 is misread as a sag because the noise itself drifts
down -1.84 dB over that walk — the same noise fluke also turns
the moved beacon's seed-11 verdict `uncertain`, since the
monitor cannot rule a decline out. No statistic in the window
can tell a real decline from that fluke; the fix is a longer
history (more loops), not a sharper threshold. On a single pass
with no revisits, the fallback catches the fast 1.2-per-hour sag
on 4/5 seeds (seed 11's decline is buried in the noise: slope
-0.05 dB per reading) and a beacon moved away from the ward
reads like a steady offset, so it comes back `uncertain`.

## Deployment planner

The drift classifier says why a beacon is already misbehaving; a
ward that has to schedule battery replacements needs to know
*when* each beacon will go dark, before it does. `plan_deployment`
forecasts that from the same `BatteryModel` the simulator uses —
no new model, no simulation, just the model's own algebra: the
level falls linearly, so the hours until any target level are
`(level now - level target) / drain rate`, and the RSSI loss and
extra noise at that level follow from the model's square and
linear laws:

```python
from beacon import BatteryModel, plan_deployment

plan = plan_deployment({
    "A": BatteryModel(drain_rate_per_hour=0.05),
    "B": BatteryModel(drain_rate_per_hour=0.05),
    "C": BatteryModel(drain_rate_per_hour=0.05),
    "D": BatteryModel(drain_rate_per_hour=0.05),
}, levels={"D": 0.2})   # D measured at 20% charge, the rest fresh
# plan.forecasts — soonest-dark first, each with hours_to_dark,
# hours_to_rssi_loss_db (3 dB by default), hours_to_extra_noise_db
# (2 dB by default), and a status: ok, degrading or dark
# plan.earliest_dark_hours — 3.0 for D above
# plan.hours_until_fewer_than(3) — when fewer than three beacons
# are still lit, and the ward can no longer triangulate at all
plan.to_dict()   # JSON-friendly; never-reached horizons are null
```

A milestone a beacon never reaches is `math.inf`, not a made-up
number: a beacon that does not drain never goes dark, and a
beacon whose loss at its cutoff level is below the threshold
never reads that weak — it just goes quiet first (at the 0.05
cutoff the loss is 6 × 0.95² = 5.42 dB, so a 6 dB loss threshold
is never reached). A beacon already past a threshold reports 0
hours for it, and a beacon measured below its cutoff is `dark`
with 0 hours to dark; a beacon already past the 3 dB loss
threshold is `degrading`. `levels` takes measured charges — a
beacon cannot be forecast above its model's initial charge, and
a level for an unknown beacon id is rejected rather than
silently ignored. The CLI plans the default ward:

```sh
python3 -m beacon.cli --deployment-plan
python3 -m beacon.cli --deployment-plan --battery-levels A=0.9,B=0.6,C=0.4,D=0.2
python3 -m beacon.cli --deployment-plan --drain-rate 0.1
```

Measured behaviour: four fresh beacons draining at 0.05 per hour
go dark in 19.0 hours each, cross 3 dB of RSSI loss in 14.1 hours
and 2 dB of extra noise in 13.3 hours — the noise and the loss
arrive hours before the dark, which is the point: the beacon
starts lying well before it disappears. With measured levels
0.9, 0.6, 0.4, 0.2 the ward's earliest dark is D at 3.0 hours
(D is already past both thresholds at 20% charge, so it reads
`degrading`), and the ward drops below three lit beacons at 7.0
hours, when C goes dark — four hours of warning between the
first beacon dying and triangulation failing. A drain rate of
zero reports every horizon as `never`.

## Battery-swap replacement schedule

A forecast is not a work order: nobody schedules a battery
change from "dark in 19.0 hours". `replacement_schedule` turns
a `DeploymentPlan` into the list a technician actually works
from — one entry per beacon, soonest swap first, each with the
swap-by time in hours from now, the milestone that drives it,
and the forecast's own status:

```python
from beacon import BatteryModel, plan_deployment, replacement_schedule

plan = plan_deployment({
    "A": BatteryModel(drain_rate_per_hour=0.05),
    "B": BatteryModel(drain_rate_per_hour=0.05),
    "C": BatteryModel(drain_rate_per_hour=0.05),
    "D": BatteryModel(drain_rate_per_hour=0.05),
}, levels={"D": 0.2})   # D measured at 20% charge, the rest fresh
schedule = replacement_schedule(plan)
# schedule.entries — D first, swap by 0.0h, reason "3 dB loss"
# (D is already past both the loss and the noise thresholds, and
# the loss wins the tie); then C, B, A, each on its 2 dB noise
# milestone. schedule.ward_deadline_hours — 7.0, when the ward
# drops below three lit beacons.
schedule.to_dict()   # JSON-friendly; a never-due swap is null
entry.due_at()       # the swap-by time as a clock time, or None
```

The swap is due at the beacon's *first* milestone, not at the
dark: by the time a beacon goes dark it has already been lying
for hours, so the schedule replaces it when it first starts
lying. `lead_time_hours` pulls every swap earlier by the same
margin — order the cells a day ahead and the schedule says so —
floored at zero, so a beacon already past a milestone is due
now, not in the past; a negative lead time is rejected, since a
schedule cannot push a swap later than the beacon's own
forecast. A beacon that never drains never needs a swap
(`swap_by_hours` is `math.inf`, reason `never`), and the
schedule carries the ward deadline from the plan, so the swaps
can be checked against the hard date triangulation fails. The
CLI prints the same list, with clock times, reusing
`--battery-levels` and `--drain-rate` from the planner:

```sh
python3 -m beacon.cli --replacement-schedule
python3 -m beacon.cli --replacement-schedule --battery-levels A=0.9,B=0.6,C=0.4,D=0.2
python3 -m beacon.cli --replacement-schedule --swap-lead-hours 24
```

Measured behaviour: four fresh beacons at 0.05 per hour are all
due in 13.3 hours on their 2 dB noise milestone, and the ward
deadline is 19.0 hours. With measured levels 0.9, 0.6, 0.4, 0.2
the order is D now (3 dB loss), C in 1.3 hours, B in 5.3 hours,
A in 11.3 hours — all on the 2 dB noise milestone except D —
and the ward deadline is 7.0 hours, so the C swap at 1.3 hours
is the one that keeps the ward triangulating. A 24-hour lead
puts every fresh beacon's swap at now, which is the honest
answer when the cells should already have been ordered.

## Planner-aware ward-map overlay

The deployment planner says when each beacon goes dark; the ward
map is where that forecast belongs, next to the coverage it
threatens. `render_map_overlay(..., plan=plan)` annotates every
beacon with its forecast horizon — the soonest of its three
milestones, e.g. `2 dB noise in 13.3h`, `3 dB loss now`, or
`dark now` — and colours its diamond by its forecast status
(blue ok, amber degrading, red dark), with a legend line carrying
the plan's earliest dark and when the ward drops below three lit
beacons. `render_map_overlay(..., plan=plan, planner=True)`
shades the ward grid the same way, but by time instead of by
quality: `planner_heatmap` grids the ward (2 m cells, 1 m off
the walls) and gives each cell the soonest forecast milestone
among its three nearest beacons — the beacons whose RSSI would
dominate the solve there, so the ones the cell depends on — with
the limiting beacon and milestone on the cell, so the map shows
not just the coverage now but when it will thin. The cells are
coloured red inside 4 hours, amber inside 12, green beyond, and
teal where the beacons never drain, so a dashboard can see at a
glance which corner of the ward needs a battery first. The
planner shading and the coverage heatmap cannot be combined —
they would paint the same cells in different colours — and a
beacon in the layout with no forecast in the plan raises instead
of being drawn as if it were stable. The CLI renders it directly,
reusing `--battery-levels` and `--drain-rate` for the plan:

```sh
python3 -m beacon.cli --map-overlay ward-a-track.json --map-output ward-a-track.svg --map-planner
python3 -m beacon.cli --map-overlay ward-a-track.json --map-output ward-a-track.svg --map-planner --battery-levels A=0.9,B=0.6,C=0.4,D=0.2
```

Measured behaviour: on the default ward with four fresh beacons
at 0.05 per hour every cell degrades at 13.3 hours (the 2 dB
noise milestone comes first for all four), so the whole planner
heatmap is green. With measured levels 0.9, 0.6, 0.4, 0.2 the
earliest cell is (1, 1) m at 0.0 hours — D is already past 3 dB
of loss — while the far corner (19, 1) m holds until 1.3 hours
(C's 2 dB noise milestone), and all 60 cells are red: every
cell's three nearest beacons include C or D, so the whole ward
starts degrading within 1.3 hours. The overlay's canvas grows
with its legend, so the plan and planner summary lines are never
clipped below the SVG — the plain overlay's legend already
overflowed its 560 px canvas before this change, and now renders
at 594 px.

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

## Live tracking

Everything above answers one-shot questions: one observation set,
one position. A ward dashboard needs the opposite — a device that
keeps reporting, and a position that keeps updating as readings
arrive. `LiveTracker` wraps a `PositioningEngine` and turns the
observation stream into a live track:

```python
from beacon import Beacon, LiveTracker, PositioningEngine

beacons = [Beacon("A", 0, 0), Beacon("B", 20, 0),
           Beacon("C", 20, 12), Beacon("D", 0, 12)]
tracker = LiveTracker(PositioningEngine(beacons))
live = tracker.feed({"A": -62.1, "B": -71.4, "C": -68.0, "D": -59.8}, t=0.0)
# live.x, live.y, live.floor, live.floor_confidence,
# live.uncertainty (the 95% ellipse), live.heard
```

Readings merge across beacons, so a scan that only hears two
beacons still updates the track from the beacons heard a moment
ago — BLE scans rarely hear every beacon on every scan. A reading
older than `stale_after` seconds (5 s by default) ages out of the
merged set, so a beacon that has gone quiet stops anchoring the
track to where the device was when it was last heard. When fewer
than three fresh beacons remain, `feed` returns `None` instead of
a made-up position, and the last good position stays available as
`tracker.last_position` until a fresh reading revives the track.
Unknown beacon ids are ignored rather than failing the stream,
and timestamps must not go backwards — a replayed or misordered
scan is rejected. The ellipse comes from the same beacon subset
the engine localised on (the winning floor's beacons, or every
heard beacon when fewer than three were heard there), and is
`None` on the rare step whose geometry is degenerate.

The stream format is JSON lines, one reading per line — the shape
a BLE scanner would emit:

```sh
python3 -m beacon.tracking < examples/live-stream.jsonl
```

Each input line produces one output line with the live position
attached (`"position": null` on a reading too thin to locate, with
the fresh beacons listed either way so a dashboard can see the
track starving). `examples/live-stream.jsonl` is twelve readings
from a simulated walk in which beacon D dies after step 7, so the
last four lines arrive without it. `python3 -m beacon.tracking
--demo` runs a full simulated walk through the tracker instead,
with no scanner and no input file.

Measured behaviour (simulator, 2 dB RSSI noise, default ward walk,
seed 7): the live track's mean error is 2.16 m over the 40-step
walk, the same as the one-shot engine's — the tracker adds no
error of its own. With beacon D dropping out at step 20 the mean
error is 2.43 m and the track keeps emitting on three beacons; the
ellipse widens as the geometry thins (mean semi-major axis 6.32 m
with four beacons, 7.42 m after D drops).

## Beacon-failure fallback

Stale readings already age out of the merged set, so a beacon
that goes quiet stops anchoring the track — but nothing said so.
A dashboard watching the track could see it drop from four
beacons to three and not know whether a beacon died, was
shielded, or was simply never installed. `LiveTracker` now
names the fallback: a beacon heard before, then unheard for
longer than `dark_after` (10 s by default), counts as dark, and
the live position carries it as an `ExcludedBeacon` — the beacon
id and the reason, e.g. `"no reading for 12.0s (dark after
10.0s)"`:

```python
from beacon import Beacon, LiveTracker, PositioningEngine

beacons = [Beacon("A", 0, 0), Beacon("B", 20, 0),
           Beacon("C", 20, 12), Beacon("D", 0, 12)]
tracker = LiveTracker(PositioningEngine(beacons))
tracker.feed({"A": -62.1, "B": -71.4, "C": -68.0, "D": -59.8}, t=0.0)
# ... D goes quiet; eleven seconds after its last reading:
live = tracker.feed({"A": -62.1, "B": -71.4, "C": -68.0}, t=11.0)
# live.heard == ("A", "B", "C")
# live.excluded == (ExcludedBeacon(id="D",
#     reason="no reading for 11.0s (dark after 10.0s)"),)
tracker.excluded   # the same list, without a position in hand
```

The dark window is longer than the stale window (`dark_after`
must be at least `stale_after`, 5 s by default), so a beacon is
only called dark once it has already aged out of the estimate —
the two windows never disagree about whether a beacon is in the
position. A beacon never heard is not excluded: there is no
fallback to record for a beacon that was never part of the track.
A beacon heard again is not trusted back at once: it must be
heard on `readmit_after` consecutive readings (3 by default)
before it rejoins the estimate, so a beacon flickering at the
edge of range — heard once, missed, heard again — cannot flap
in and out of the solve. While it is coming back the exclusion
stays on the position with the reason counting its progress
(`"heard again, 1 of 3 fresh readings to re-admit"`), and a
missed reading resets the count to zero; `readmit_after=1`
restores the old single-reading re-admission. The exclusion
travels with the position into its JSON form and into the stream
output (`python -m beacon.tracking`), including on a line whose
position is null, so a dashboard sees why a track is running
thin. `--dark-after` and `--readmit-after` tune the windows from
the command line.

Measured behaviour (simulator, 2 dB RSSI noise, default ward
walk, seed 7, beacon D dying at step 20): D is first recorded
excluded at step 30, eleven seconds after its last reading; the
track keeps emitting on three beacons with coverage 1.0 and a
mean error of 2.43 m — the same as before the fallback was
recorded — and the degradation shows up where it should, in the
uncertainty: the ellipse's semi-major axis widens from 7.65 m
just before D drops to 8.68 m after it is excluded.

## Live track accuracy report

The README's accuracy numbers above were measured by hand, once.
A ward that re-runs its walks — after moving a beacon, after a
recalibration, after a firmware change — needs the same verdict in
code, on every walk. `evaluate_live_track` scores a recorded
track against ground truth, step by step:

```python
from beacon import LiveTracker, PositioningEngine, evaluate_live_track
from beacon.simulate import rectangular_walk, simulate_walk

beacons = [Beacon("A", 0, 0), Beacon("B", 20, 0),
           Beacon("C", 20, 12), Beacon("D", 0, 12)]
steps = simulate_walk(rectangular_walk(20, 12, 10), beacons,
                      noise_db=2.0, seed=7)
tracker = LiveTracker(PositioningEngine(beacons))
positions = [tracker.feed(step["rssi"], float(i))
             for i, step in enumerate(steps)]
report = evaluate_live_track(positions, [step["true"] for step in steps])
# report.mean_error_m, report.median_error_m, report.p90_error_m,
# report.max_error_m, report.coverage, report.floor_accuracy
report.to_dict()   # JSON-friendly, for dashboards and regression logs
```

The report is a `TrackReport`: how many steps were scored, how
many produced a position, and `coverage` — the located share —
so a track starved by dead beacons shows up as low coverage
rather than silently short. The error figures (mean, median,
90th percentile, worst step) are computed over the located steps
only. Ground truth is one `(x, y)` per step, or `(x, y, floor)`
on a multi-floor walk, in which case `floor_accuracy` is the
share of located steps whose floor was right (it is `None` when
the truth carries no floors). A walk with no located step at all
raises instead of reporting zeroes — with nothing located there
is no error to summarise, and a report of zeroes would hide the
failure. Length mismatches, mixed two- and three-tuple truth, and
empty walks raise the same way.

Measured behaviour (simulator, 2 dB RSSI noise, default ward walk,
seed 7): mean error 2.16 m, median 2.08 m, 90th percentile
4.14 m, worst step 6.70 m, coverage 1.0. With beacon D dropping
out at step 20 the coverage stays 1.0 — three beacons still
locate — and the dropout shows up in the error instead (mean
2.43 m, worst step 7.63 m). On the two-floor walk the floor
accuracy is 1.0 with a mean error of 2.53 m. Even a noiseless
walk is not exact through the live tracker: the Kalman filter
lags the walk around the corners, so the mean error is 0.31 m
and the worst step 1.02 m.

## Live track export

A live track is only useful while the dashboard is watching it.
A ward that wants the track afterwards — to replay a shift, to
audit where a device went, to feed a dashboard that was not
running at the time — needs the track as a file. `track_rows`
turns a recorded track into one row per located position, and
`export_track_json` / `export_track_csv` save it:

```python
from beacon import LiveTracker, PositioningEngine, export_track_csv
from beacon.simulate import rectangular_walk, simulate_walk

beacons = [Beacon("A", 0, 0), Beacon("B", 20, 0),
           Beacon("C", 20, 12), Beacon("D", 0, 12)]
steps = simulate_walk(rectangular_walk(20, 12, 10), beacons,
                      noise_db=2.0, seed=7)
tracker = LiveTracker(PositioningEngine(beacons))
positions = [tracker.feed(step["rssi"], float(i))
             for i, step in enumerate(steps)]
export_track_csv(positions, "ward-a-track.csv")
```

Each row carries the timestamp, the (x, y) in metres, the floor
and its confidence, the 95% confidence ellipse (semi-major,
semi-minor, angle — empty when the step's geometry was
degenerate), the beacons the estimate was built from, the
beacons excluded as dark with their reasons, and the track's
overall coverage — the located share of the walk, the same on
every row. Steps that produced no position are not rows, since
there is no position to write, but they are counted in the
coverage, so a starved track never pretends it was complete;
the JSON file's header carries the step counts and the coverage
too. Both formats round-trip: `load_track_json` /
`load_track_csv` read back exactly the rows `track_rows`
produced, so an export can be verified by loading it again.
`examples/live-track-export.json` and
`examples/live-track-export.csv` are the default demo walk
exported both ways. The CLI saves the track the same way:

```sh
python3 -m beacon.tracking --demo --export-csv ward-a-track.csv
python3 -m beacon.tracking < readings.jsonl --export-json track.json
```

Measured behaviour (simulator, 2 dB RSSI noise, default ward
walk, seed 7): the demo export is 40 rows over 40 steps,
coverage 1.0, and the JSON and CSV exports load back to
identical rows. With beacon D dying at step 20 the rows from
step 30 run on three beacons and record D excluded with its
reason, while the coverage stays 1.0.

## Ward-map overlay

An exported track is a table; a ward dashboard wants a picture.
`render_map_overlay` plots the track on the ward plan as a
self-contained SVG — no dependencies, no JavaScript, openable in
a browser or embeddable in a dashboard:

```python
from beacon import load_track_json, render_map_overlay
from beacon.cli import default_beacons

rows = load_track_json("ward-a-track.json")
svg = render_map_overlay(rows, default_beacons())
open("ward-a-track.svg", "w").write(svg)
```

The SVG draws the ward outline with a 5 m grid, the beacons as
labelled diamonds at their known positions, the track as a path
through the located points with S (start) and E (end) markers,
and one 95% confidence ellipse per point — coloured by how wide
it is (green tight, semi-major ≤ 2 m; amber loose, ≤ 5 m; red
very loose, above 5 m; grey where the geometry was degenerate
and the point has no ellipse), with a legend explaining the
colours. The header carries the floor, the point count and the
coverage, so a thin track shows up as thin. A multi-floor track
plots one floor only — the most common floor in the track by
default, or the floor named with `floor=` — so two floors never
land on top of each other. The CLI renders it directly:

```sh
python3 -m beacon.cli --map-overlay ward-a-track.json --map-output ward-a-track.svg
python3 -m beacon.cli --map-overlay ward-a-track.csv --map-floor 1
```

Measured behaviour: the default demo export
(`examples/live-track-export.json`, 40 rows, coverage 1.0)
renders 40 track points, 40 ellipses, all four beacons, and the
same SVG from the CSV export — JSON and CSV are the same track.

## Coverage heatmap

The overlay above shows where a device *went*; it says nothing
about where a device *could* be located well. The heatmap fills
that gap: `coverage_heatmap` grids the ward (2 m cells, 1 m off
the walls) and scores each cell with the 95% ellipse semi-major
axis the beacon layout would give there — the same uncertainty
model behind the placement advisor and the overlay's per-point
ellipses, so the heatmap and the track agree about where the
ward is thin. `render_map_overlay(..., heatmap=True)` shades
each cell in the same confidence colours as the ellipses (green
tight, amber loose, red very loose), under the track, and adds
a summary line to the legend — mean and worst semi-major axis
over the cells, and where the worst cell is. Degenerate cells
are left unshaded rather than coloured as if they were covered.
The CLI renders it directly:

```sh
python3 -m beacon.cli --map-overlay ward-a-track.json --map-output ward-a-track.svg --map-heatmap
```

Measured behaviour: on the default four-corner layout the
heatmap is 60 cells with a mean 95% semi-major axis of 6.26 m
and a worst cell of 7.78 m at (9, 1) m — the corners cover the
ward thinly everywhere, which is exactly what the placement
advisor's advised layout (mean 4.30 m) improves on. The heatmap
is a layout verdict, not a track verdict: it depends only on
the beacons, so it is the same for every track on that ward.

## Live track replay

The static overlay shows where a device went; a replay shows
when it was where. `render_track_replay` turns an exported
live track into a self-contained animated SVG — no JavaScript,
just SMIL, so it plays in a browser or a dashboard embed as-is:

```python
from beacon import load_track_json, render_track_replay
from beacon.cli import default_beacons

rows = load_track_json("ward-a-track.json")
svg = render_track_replay(rows, default_beacons())
open("ward-a-track-replay.svg", "w").write(svg)
```

A dot glides along the track on the track's own key times (the
dot's `cx` / `cy` animate through the projected positions), the
trail draws itself behind it (a polyline whose
`stroke-dashoffset` animates from its full length to zero), the
95% confidence ellipse for the current step appears with the
dot — coloured by the same confidence bands as the static
overlay, and absent where the step's geometry was degenerate —
and a timestamp label steps through the track's own timestamps,
so a pause between readings shows up as a pause in the replay.
Each frame is on screen from the midpoint with the previous
frame to the midpoint with the next, so the ellipse and the
label switch while the dot is gliding between positions. The
replay loops by default (`loop=False` plays it once), at
`seconds_per_step` per step (1 s by default), and it plots one
floor only — the most common floor by default, or the floor
named with `floor=` — with the same optional coverage heatmap
as the static overlay. The CLI renders it directly:

```sh
python3 -m beacon.cli --map-overlay ward-a-track.json --map-output ward-a-track-replay.svg --map-replay
python3 -m beacon.cli --map-overlay ward-a-track.json --map-replay --replay-step-seconds 0.5
```

Measured behaviour: the default demo export
(`examples/live-track-export.json`, 40 rows, t = 0–39 s) replays
as 40 frames in 39 s at the default 1 s per step, with 40
ellipses and 40 timestamp labels, and the same SVG from the CSV
export — JSON and CSV are the same track.
`examples/live-track-replay.svg` is that replay, ready to open
in a browser.

## Multi-device tracking

A ward has more than one tracked device — a patient tag, a
nurse's badge, a pump on a trolley — and one `LiveTracker`
tracks one device: its Kalman filter state, its floor vote,
and its dark-beacon bookkeeping all belong to that device's
track. Feeding two devices through one tracker would smear
them into a single position somewhere between them.
`MultiDeviceTracker` keeps one `LiveTracker` per device id,
each with its own `PositioningEngine`, over the same shared
beacon layout, and routes each reading to the right track:

```python
from beacon import Beacon, MultiDeviceTracker

beacons = [Beacon("A", 0, 0), Beacon("B", 20, 0),
           Beacon("C", 20, 12), Beacon("D", 0, 12)]
tracker = MultiDeviceTracker(beacons)
tag1 = tracker.feed("tag-1", {"A": -62.1, "B": -71.4,
                              "C": -68.0, "D": -59.8}, t=0.0)
tag2 = tracker.feed("tag-2", {"A": -70.0, "B": -65.0,
                              "C": -60.0, "D": -72.0}, t=0.0)
# tag1.x, tag1.y — tag-1's position; tag2.x, tag2.y — tag-2's
tracker.positions()   # {"tag-1": ..., "tag-2": ...} — the
                      # dashboard snapshot of every device
```

Devices are created lazily, on their first reading, so a
deployment never has to declare its tags up front — a new tag
showing up mid-shift just starts tracking. Each device's clock
is its own: timestamps only have to be non-decreasing *per
device*, so readings from different devices can interleave in
any order, and one device going quiet never stalls another's
track — the quiet device's last position just stays where it
was. A device that hears too few beacons returns `None` for
that reading while the other devices keep locating, and it is
left out of `positions()` rather than reported with a made-up
position. `reset(device_id)` forgets one device's track (a tag
handed to a new patient starts fresh) and `remove(device_id)`
drops the device entirely. The stream format is the same JSON
lines as single-device tracking, with the device attached:

```sh
python3 -m beacon.multi_device < readings.jsonl
python3 -m beacon.multi_device --demo   # two simulated walks,
                                        # one forward, one in reverse
```

Each input line produces one output line with the device and
its live position attached (`"position": null` on a reading
too thin to locate). The tracker windows (`--stale-after`,
`--dark-after`, `--readmit-after`, `--sigma-rssi`) are shared
by every device's tracker. One honest limitation: every
device shares one beacon layout, so a tag that walks into a
part of the building the layout does not cover is located
badly, not handed off to a different layout — and estimates
are not clamped to the room, so a filtered position can sit
just outside the ward's walls.

Measured behaviour (simulator, 2 dB RSSI noise, default ward
walk, one device forward and one in reverse): the forward
device's mean error is 2.16 m (the same as the single-device
track on that walk — sharing the tracker adds no error of its
own), the reverse device's is 1.72 m, and the two tracks stay
9.43 m apart on average, so they are genuinely two tracks
rather than one smeared position.

## Per-floor layout validation

The engine localises within one floor's 2D plane, so each floor
needs its own beacons — three at least, spread out enough that
the geometry pins a device down. A floor that fails that test
does not fail loudly today: with fewer than three beacons heard
on the winning floor the engine falls back to beacons from other
floors, whose cross-floor distances are inflated by the slab
attenuation, so positions on that floor degrade silently. A
floor whose beacons are clustered on one wall is just as quiet
about it — the estimates still come out, only with a very wide
confidence ellipse in the thin corners. `validate_floor_layouts`
scores every floor before deployment, with the same 95%
confidence-ellipse model as the placement advisor, and reports a
verdict per floor:

```python
from beacon import Beacon, validate_floor_layouts

beacons = [Beacon("A", 0, 0, floor=0), Beacon("B", 20, 0, floor=0),
           Beacon("C", 20, 12, floor=0), Beacon("D", 0, 12, floor=0),
           Beacon("E", 0, 0, floor=1), Beacon("F", 20, 0, floor=1)]
validation = validate_floor_layouts(beacons, 20.0, 12.0)
validation.ok   # False — floor 1 has only two beacons
validation.by_floor(1).status   # "cannot_triangulate"
validation.warnings()
# ["floor 1 has only 2 beacons — the engine needs 3 to
#  triangulate, so it falls back to beacons on other floors,
#  whose cross-floor distances are inflated by the slab
#  attenuation"]
```

Each verdict is a `FloorLayoutVerdict`: the floor, how many
beacons it carries, a status — `ok`, `thin` (it triangulates,
but the worst point in the room is looser than the limit), or
`cannot_triangulate` (fewer than three beacons, or a geometry —
collinear or coincident beacons — under which no test point
gives a defined uncertainty) — the floor's `LayoutScore`, or
`None` when it cannot triangulate, and the plain-language
warnings behind the status. A floor with no beacons at all
never appears as a verdict: only floors that carry beacons are
validated, so a missing floor is a deployment gap, not a
verdict. The worst-error limit is 10 m by default
(`max_worst_semi_major`), and the test grid uses the placement
advisor's 0.5 m margin, so a beacon never lands exactly on a
test point and is never reported as a coverage hole for merely
coinciding with the grid. The CLI validates the default layout:

```sh
python3 -m beacon.cli --validate-layout
# floor 0: ok (4 beacons, mean 95% error 6.30 m, worst 8.32 m)
```

The check carries the same honest limits as the placement
advisor: it assumes the log-distance path-loss model with the
beacons' own exponents and the stated RSSI noise, and it knows
nothing about walls, metal beds, or dead zones in the real ward
— a floor that validates `ok` still needs a walked survey and
calibration before the numbers are trusted.

Measured behaviour: the default four-corner layout validates
`ok` on the 20 x 12 m ward (mean 6.30 m, worst 8.32 m), and a
two-floor building with four beacons downstairs and two upstairs
validates `ok` downstairs and `cannot_triangulate` upstairs. The
placement advisor's mean-optimal layout validates `thin` — its
worst 95% error is 11.16 m at its far corner, above the 10 m
limit, which is the same trade the advisor's own notes describe
(it trades the worst point for the mean); raising the limit to
12 m accepts it. Three beacons mounted at the same spot validate
`cannot_triangulate`, and three beacons on one line validate
`thin` with test points on the line reported as undefined.

## Battery degradation

A beacon does not die all at once. As its cell drains, its
transmit power sags — the signal gets weaker — and the readings
get noisier, until the cell can no longer power the radio and
the beacon goes dark. That slow decline poisons positioning
long before the beacon disappears: the engine trusts the
path-loss model, so a beacon reading weaker than its calibration
says looks farther away than it is, and the estimate is pulled
towards it. `BatteryModel` describes one beacon's decline — a
linear drain (`drain_rate_per_hour`, a fraction of a full charge
per hour), a loss that grows with the square of the charge used
(up to `max_rssi_loss_db`, 6 dB by default, so the beacon looks
about twice as far away on an empty cell) and extra RSSI noise
that grows linearly with the charge used (up to
`max_extra_noise_db`, 3 dB by default), and a `cutoff` (0.05 by
default) below which the beacon goes dark:

```python
from beacon import BatteryModel
from beacon.simulate import rectangular_walk, simulate_walk

model = BatteryModel(drain_rate_per_hour=1.2)   # ~50 minutes left
steps = simulate_walk(rectangular_walk(20, 12, 10), beacons,
                      noise_db=2.0, seed=7, batteries={"D": model},
                      step_seconds=120.0)
# steps[0]["battery"]["D"] == 1.0; D is unheard from step 24,
# when its level crosses the 0.05 cutoff
```

`batteries` is one model for every beacon, or a dict of models
by beacon id — a beacon with no model runs on a fresh cell and
never degrades — and it works the same way in
`simulate_multifloor_walk`. Each step carries `"battery":
{id: level}`, so the walk's battery state can be fed straight
into the tracker: `LiveTracker.feed(rssi, t,
battery=step["battery"])` attaches the levels to the live
position (`LivePosition.battery`), and a beacon that goes dark
with an exhausted level is excluded with the battery named as
the cause — `"battery exhausted at 0.04 — no reading for 120.0s
(dark after 10.0s)"` — instead of only the silence, so a
dashboard can tell a beacon that died from a drained cell from
one that was shielded or unplugged. The battery rides into the
exported track rows too (`track_rows`, JSON and CSV — files
written before v0.25.0 load back with an empty battery dict),
and the tracking demo drains beacon D on request:

```sh
python3 -m beacon.tracking --demo --battery-drain-per-hour 150
```

The demo walk is 40 seconds, so the drain rate is a
time-compression knob — 150 per hour drains D from full to its
cutoff in about 23 seconds of it.

Measured behaviour (simulator, 2 dB RSSI noise, default ward
walk, seed 7, beacon D draining at 1.2 per hour over 120 s
steps): the live track's mean error is 5.09 m against 2.16 m
with D healthy, and the coverage stays 1.0 — the track never
stops locating. The error grows while D is dying, not after it
is dead: steps 0–5 (D healthy) average 2.19 m, steps 12–19 (D
at 0.52 down to 0.24) average 13.66 m, and once D goes dark at
step 24 the three-beacon track recovers to 3.36 m over the rest
of the walk. Across seeds 1, 2, 3, 7, 11 the degraded mean is
4.51, 4.85, 47.70, 5.09 and 3.24 m against a healthy 1.97,
2.57, 2.63, 2.16 and 1.83 m — seed 3 is the warning: while D is
nearly dead but still heard, the estimate diverges to about
280 m, the same distance-inversion explosion a wrong path-loss
model causes, so a dying beacon is worse for the track than a
dead one, and the drift alarm is the tool that catches it
before the battery does.

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
- Live tracking mode (stream RSSI readings, emit a live position
  per reading) — done: `LiveTracker` wraps the engine, merges
  partial scans across beacons, ages out readings older than
  `stale_after` (5 s by default), returns `None` instead of a
  made-up position when fewer than three fresh beacons remain
  (`last_position` keeps the last good one), and attaches the
  95% confidence ellipse from the same beacon subset the engine
  localised on; `python -m beacon.tracking` reads JSON-line
  readings from stdin and prints one JSON line per reading
  (`--demo` runs a simulated walk, `examples/live-stream.jsonl`
  is twelve readings in which beacon D dies after step 7)
  (v0.15.0); on the default ward walk the live track's mean error
  is 2.16 m, and 2.43 m with beacon D dropping out at step 20 —
  the tracker adds no error of its own.
- Live track accuracy report (score a recorded walk against
  ground truth) — done: `evaluate_live_track` returns a
  `TrackReport` with mean, median, 90th percentile and worst error
  over the located steps, plus `coverage` (the located share of
  the walk) and `floor_accuracy` on multi-floor walks, with a
  `to_dict()` for dashboards and regression logs; empty walks,
  length mismatches, and walks with nothing located raise instead
  of reporting zeroes (v0.16.0); on the default ward walk, seed 7,
  mean 2.16 m, median 2.08 m, p90 4.14 m, worst 6.70 m, coverage
  1.0 — and a beacon dropout shows up in the error (mean 2.43 m)
  rather than the coverage, which stays 1.0 on three beacons.
- Beacon-failure fallback (record which beacon went dark and why)
  — done: `LiveTracker` counts a beacon heard before, then
  unheard for longer than `dark_after` (10 s by default, never
  shorter than the 5 s stale window), as dark, and records it on
  the `LivePosition` as an `ExcludedBeacon` with the reason, in
  the position's JSON form, and in the stream output; a beacon
  never heard is not excluded, and a beacon heard again rejoins
  on its own (v0.17.0); on the default ward walk, seed 7, with
  beacon D dying at step 20, D is first recorded excluded at
  step 30, the track keeps coverage 1.0 at 2.43 m mean error, and
  the ellipse's semi-major axis widens from 7.65 m to 8.68 m.
- Beacon-failure re-admission (a dark beacon must prove it is
  back) — done: a dark beacon heard again stays excluded until
  it has been heard on `readmit_after` consecutive readings (3
  by default, `readmit_after=1` restores single-reading
  re-admission), with the exclusion reason counting its progress
  and a missed reading resetting the count, so a flickering
  beacon cannot flap in and out of the estimate (v0.18.0); on
  the default ward walk, seed 7, with beacon D dying at step 20
  and returning at step 35, D is first recorded excluded at step
  30, stays excluded through its first two readings back, and
  re-admits at step 37 — the track keeps coverage 1.0 at 2.31 m
  mean error (median 2.13 m, 90th percentile 4.14 m, worst step
  6.70 m).
- Live track export (save a recorded track for a ward dashboard)
  — done: `track_rows` turns a recorded track into one row per
  located position (timestamp, x, y, floor, confidence ellipse,
  beacons used, beacons excluded, track coverage), and
  `export_track_json` / `export_track_csv` save it, with
  `load_track_json` / `load_track_csv` reading back exactly the
  same rows; `python -m beacon.tracking` takes `--export-json`
  and `--export-csv` in both demo and stream mode, and
  `examples/live-track-export.json` / `.csv` are the default
  demo walk exported both ways (v0.19.0); on the default ward
  walk, seed 7, the export is 40 rows over 40 steps at coverage
  1.0, and with beacon D dying at step 20 the rows from step 30
  run on three beacons and record D excluded — steps that
  produced no position are counted in the coverage, not hidden.
- Ward-map overlay (plot an exported live track on the ward plan)
  — done: `render_map_overlay` renders an exported track as a
  self-contained SVG — ward outline with a 5 m grid, beacons as
  labelled diamonds, the track as a path with start and end
  markers, and one 95% confidence ellipse per point, coloured by
  its semi-major axis (green ≤ 2 m, amber ≤ 5 m, red above, grey
  with no ellipse), with a legend and the floor, point count and
  coverage in the header; a multi-floor track plots one floor
  only, the most common by default or the floor named with
  `floor=`; `load_track_for_overlay` reads a `.json` or `.csv`
  export, `write_map_overlay` writes the SVG to a file, and
  `python -m beacon.cli --map-overlay track.json
  [--map-output map.svg] [--map-floor N]` renders it from the
  command line (v0.20.0); on the default demo export (40 rows,
  coverage 1.0) the overlay renders 40 points, 40 ellipses and
  all four beacons, and the JSON and CSV exports render the
  identical SVG.
- Per-beacon coverage heatmap on the ward-map overlay — done:
  `coverage_heatmap` grids the ward and scores each cell with
  the 95% ellipse semi-major axis the beacon layout would give
  there, from the same uncertainty model as the placement
  advisor, and `render_map_overlay(..., heatmap=True)` shades
  each cell in the same confidence colours as the per-point
  ellipses, under the track, with a mean/worst summary line in
  the legend; `python -m beacon.cli --map-overlay track.json
  --map-heatmap` renders it from the command line (v0.21.0); on
  the default four-corner layout the heatmap is 60 cells with a
  mean semi-major axis of 6.26 m and a worst cell of 7.78 m at
  (9, 1) m — the corners cover the ward thinly everywhere, which
  the advised layout (mean 4.30 m) improves on.
- Live-track replay animation (replay an exported live track
  as an animated sequence) — done: `render_track_replay`
  renders an exported track as a self-contained animated SVG —
  a dot gliding along the path on the track's key times, the
  trail drawing itself behind it, the 95% confidence ellipse
  for the current step appearing with the dot, and a timestamp
  label stepping through the track's own times — SMIL animation,
  no JavaScript, looping by default at 1 s per step
  (`--replay-step-seconds` to change it, `loop=False` to play
  it once), one floor only, with the same optional coverage
  heatmap as the static overlay; `write_track_replay` writes
  the SVG to a file, and `python -m beacon.cli --map-overlay
  track.json --map-replay` renders it from the command line
  (v0.22.0); on the default demo export (40 rows, t = 0–39 s)
  the replay is 40 frames in 39 s with 40 ellipses and 40
  timestamp labels, and the JSON and CSV exports replay the
  identical SVG.
- Multi-device tracking (track several devices at once on one
  beacon layout) — done: `MultiDeviceTracker` keeps one
  `LiveTracker` (and its own `PositioningEngine`) per device
  id, created lazily on the device's first reading, and routes
  each reading to the right track — `feed(device_id, rssi, t)`,
  `last_position(device_id)`, `positions()` for the dashboard
  snapshot of every device, `reset(device_id)` to forget one
  track, `remove(device_id)` to drop a device; each device's
  clock is its own, so interleaved readings never share a
  clock and a quiet device never stalls another's track, and a
  device too thin to locate returns `None` while the others
  keep locating; `python -m beacon.multi_device` reads JSON
  lines with the device attached and prints one line per
  reading with the device and its live position (`--demo` runs
  two simulated walks, one forward and one in reverse)
  (v0.23.0); on the default ward walk, seed 7 forward and seed
  11 in reverse, the forward device's mean error is 2.16 m —
  the same as the single-device track, so sharing the tracker
  adds no error — the reverse device's is 1.72 m, and the two
  tracks stay 9.43 m apart on average. The roadmap grows as
  new needs surface — multi-device tracking is the latest
  addition.
- Per-floor layout validation (can every floor triangulate?) —
  done: `validate_floor_layouts` scores each floor's beacons
  with the same 95% ellipse model as the placement advisor and
  reports a `FloorLayoutVerdict` per floor — `ok`, `thin` (the
  worst 95% error is above the 10 m default limit), or
  `cannot_triangulate` (fewer than three beacons, or a
  collinear/coincident geometry under which no test point gives
  a defined uncertainty) — with plain-language warnings naming
  the floor to fix and why, and `python -m beacon.cli
  --validate-layout` validates the default layout from the
  command line, exiting non-zero when a floor is thin or cannot
  triangulate (v0.24.0); the default four-corner layout
  validates `ok` (mean 6.30 m, worst 8.32 m), a floor with two
  beacons validates `cannot_triangulate`, the placement
  advisor's mean-optimal layout validates `thin` at its far
  corner (worst 11.16 m), three coincident beacons validate
  `cannot_triangulate`, and three beacons on one line validate
  `thin` with the test points on the line reported as undefined.
- Battery degradation (a beacon's RSSI as its battery drains) —
  done: `BatteryModel` drains a beacon's charge linearly
  (`drain_rate_per_hour`), weakens its RSSI with the square of
  the charge used (up to 6 dB by default) and adds noise linearly
  with the charge used (up to 3 dB by default), and goes dark
  below the 0.05 cutoff; `simulate_walk` and
  `simulate_multifloor_walk` take `batteries` (one model, or one
  per beacon id) and `step_seconds`, report each step's levels,
  and leave a dark beacon out of the readings; `LiveTracker.feed`
  takes the step's battery state, `LivePosition` carries it, a
  beacon that goes dark with an exhausted level is excluded with
  the battery named as the cause, and the battery rides into the
  exported track rows (JSON and CSV, with pre-v0.25.0 files still
  loading); `python -m beacon.tracking --demo
  --battery-drain-per-hour 150` drains beacon D mid-demo
  (v0.25.0); on the default ward walk, seed 7, with D draining
  at 1.2 per hour over 120 s steps, the live track's mean error
  is 5.09 m against 2.16 m healthy — 2.19 m over steps 0–5,
  13.66 m over steps 12–19 while D is dying, 3.36 m over steps
  24–39 once D is dark and the track runs on three beacons — at
  coverage 1.0 throughout.
- Drift classification (battery sag or a moved beacon?) —
  done: `DriftMonitor.classify` names the cause of a drift —
  `healthy`, `battery-sag`, `moved`, `noisy` or `uncertain` —
  from the residual signatures, with the revisit delta as the
  test that separates a sag from a move (same spot, read again
  later: a draining battery reads weaker, a move cannot change
  what a beacon reads like at a fixed spot), a within-window
  trend as the fallback when the device never revisits a spot,
  and `uncertain` instead of a guess when no signature fits;
  `python -m beacon.cli --drift-classify` walks the loop twice
  and prints every beacon's verdict, with `--battery-degrade D`
  and `--move-beacon D:5,-5` to show it a real sag and a real
  move (v0.26.0); measured on two-loop ward walks, seeds 1, 2,
  3, 7, 11: a slow sag is called `battery-sag` 5/5, a 7 m move
  is called `moved` 4/5 (seed 11 `uncertain`), 5 dB of extra
  noise is called `noisy` 5/5, and a steady -6 dB offset is
  called `uncertain` 4/5 and misread as a sag on seed 11, where
  the noise itself drifts down — while a move under about 2 m
  is invisible at 2 dB noise and comes back `healthy` or
  `uncertain`.
- Deployment planner (when does each beacon go dark?) — done:
  `plan_deployment` forecasts each beacon's hours to dark, to 3 dB
  of RSSI loss, and to 2 dB of extra noise from its `BatteryModel`
  and its measured charge — closed-form on the model's own laws
  (linear drain, loss with the square of the charge used, noise
  linear in it), with `math.inf` for a milestone the beacon never
  reaches (no drain, or dark before the loss gets that bad), a
  `dark` / `degrading` / `ok` status per beacon, and
  `hours_until_fewer_than(3)` for when the ward can no longer
  triangulate; `python -m beacon.cli --deployment-plan` plans the
  default ward, with `--battery-levels` for measured charges and
  `--drain-rate` for the burn rate (v0.27.0); four fresh beacons
  at 0.05 per hour go dark in 19.0 hours, cross 3 dB of loss in
  14.1 hours and 2 dB of noise in 13.3 hours — the beacon starts
  lying hours before it disappears — and with measured levels
  0.9, 0.6, 0.4, 0.2 the earliest dark is 3.0 hours and the ward
  drops below three lit beacons at 7.0 hours.
- Planner-aware ward-map overlay (the map shows when coverage
  thins, not just where it is thin) — done: `render_map_overlay`
  takes a `DeploymentPlan` and annotates each beacon with its
  forecast horizon (`2 dB noise in 13.3h`, `3 dB loss now`,
  `dark now`) with its diamond coloured by forecast status, and
  `planner_heatmap` shades each cell by the soonest forecast
  milestone among its three nearest beacons (red inside 4 hours,
  amber inside 12, green beyond, teal where the beacons never
  drain), with the limiting beacon and milestone on the cell and
  the plan's earliest dark and below-three-lit-beacons horizons
  in the legend; `python -m beacon.cli --map-overlay track.json
  --map-planner` renders it, reusing `--battery-levels` and
  `--drain-rate` for the plan (v0.28.0); four fresh beacons give
  an all-green planner heatmap at 13.3 hours, while measured
  levels 0.9, 0.6, 0.4, 0.2 turn all 60 cells red — every cell's
  three nearest include C or D, so the whole ward starts
  degrading within 1.3 hours. The roadmap grows as new needs
  surface — the planner-aware overlay is the latest addition.
