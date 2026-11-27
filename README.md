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
- Per-floor drift windows: each beacon keeps one rolling window
  per floor, and the drift classification reads the beacon's own
  floor — so a verdict taken mid-walk on a two-floor ward is the
  verdict taken at the end, instead of following whichever floor
  the window happens to cover.
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
  coverage — and the planner's forecast flags and alerts per row
  when the track was recorded with a plan, so an exported track
  is self-describing about upcoming battery milestones, and the
  drift monitor's alarms and alerts per row when the track was
  recorded with a monitor, so it is self-describing about a
  beacon going bad mid-shift too, and the monitor's drift
  classification per beacon per row — the verdict with the
  numbers it rests on — so the export also says what the monitor
  thinks is wrong with each beacon, not just which ones are
  alarming.
- A ward-map overlay: plot an exported live track on the ward
  plan as a self-contained SVG — ward outline, beacons at their
  known positions, the track path with start and end markers,
  and one 95% confidence ellipse per point, coloured by how wide
  it is, with a legend — so a dashboard can show where the
  device went and how much to trust each point. With
  `--map-drift` a beacon alarming at the end of the track is
  badged on the map with its drift classification, so the bad
  beacon reads as bad at a glance.
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
- Multi-device replay: the replay plays every device's track
  together — one dot, trail, ellipse series and timestamp label
  per device, each in its own colour — with the per-frame drift
  badges, the timeline strip and the forecast checks computed
  per device from that device's own rows, so one tag's alarm
  never colours another tag's track.
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
- Planner forecast flags on the live track: a tracker given a
  deployment plan carries the planner's forecast on every live
  position — one flag per beacon, the soonest of its dark, 3 dB
  loss and 2 dB noise milestones with the hours remaining — and
  a plain-language alert for every beacon whose milestone falls
  inside the forecast window, so the track warns about a beacon
  about to start lying before it does.
- Planner forecast flags on multi-device tracks: the same
  forecast flags on every device's live position — one per
  beacon, the soonest milestone with its hours remaining —
  counted down from each device's own first reading, so a tag
  that shows up mid-shift warns on its own clock, and the flags
  ride on a null line too, so a device too thin to locate still
  warns about a beacon about to go dark.
- Drift flags on multi-device tracks: the same drift alarms on
  every device's live position — one monitor per device, so a
  tag alarming on a beacon going bad never dilutes or clears
  another tag's alarm, and resetting one device clears only
  its monitor.
- Per-device series on the ward-map overlay: a multi-device
  track renders one series per device on the static overlay —
  each device's own dots, trail and ellipses in its own colour,
  its own start and end markers, and its drift alarms read off
  that device's own rows — grouped and coloured the same way
  as the replay, so the overlay and the replay cannot disagree
  about which rows belong to which device, and a single-device
  track renders exactly as before.
- A combined multi-device track export: the tracker records
  every device's track as it runs and writes all of them as
  one file — `MultiDeviceTracker.export_track_json` /
  `export_track_csv`, or `--export-json` / `--export-csv` on
  `python -m beacon.multi_device` — the per-device exports
  with their rows concatenated, each row carrying its device,
  so a ward dashboard can replay the whole shift from one
  file; the rows are the same `multi_track_rows` rows the
  overlay and the replay group by device, so the file cannot
  disagree with them about which rows belong to which device,
  and a single device exports exactly the single-device file
  for its track.
- A multi-device track accuracy report: score every device's
  recorded track against its own ground truth — one
  `TrackReport` per device plus an overall report over every
  device's located steps pooled together — so a ward can see
  which tag's walk went wrong instead of one blended number
  hiding it.
- A CLI for the multi-device track accuracy report:
  `python -m beacon.multi_device --demo --evaluate` scores the
  two demo walks against their ground truth and prints the
  report — one line per device plus the overall line, in the
  same fields as the single-device demo summary — and
  `--evaluate-json PATH` writes the same report as JSON, so
  the demo's verdict is available from the command line
  instead of only in code.
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
than failing the stream. A multi-floor position is accepted too:
`observe` takes `(x, y, floor)` (a two-tuple still means floor
0), and the predicted RSSI accounts for the 3 m vertical gap per
floor of separation and the 15 dB slab attenuation for a beacon on
another floor — the same model the simulator uses — so a healthy
beacon heard through a slab does not read as drift, and revisit
pairs only match readings on the same floor.

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
says which evidence was used. And the revisit delta alone is not
enough for a sag: a noise excursion in the tail of the walk can
fake a decline at revisited spots, so the sag call also requires
the residual level to have fallen across the walk — the mean of
the last quarter of the floor's long history at least 1.0 dB
below the mean of the first quarter, at least 1.5 standard
errors from zero. A sag is a process, a steady offset is a
level, and the drop is the test that tells them apart: the
offset reads weak from the first reading and stays flat, so its
drop is near zero whatever the tail's noise does. The
classification carries the drop as `level_drop_db`, so the
evidence for or against a sag is visible on every verdict.

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
| D moved (0, 12) to (5, 7) | `moved` 5/5 (mean +2.92 to +4.21 dB, spread 2.60 to 3.38 dB; seed 11 through the longer history — its short-window revisit delta is -1.84 dB, but over 41 pairs it settles to -1.00 dB) |
| D moved 1.4 m, to (1, 11) | `uncertain` 3/5, `healthy` 2/5 — never `moved` or `battery-sag` |
| D with 5 dB extra noise | `noisy` 5/5 (spread 4.33 to 6.68 dB) |
| D steady -6 dB offset | `uncertain` 5/5 (mean -5.62 to -6.90 dB, alarming on D throughout) |

The misses are honest limits, not bugs to tune away. The slow
sag's mean residual (-2.17 to -4.73 dB) never reaches the 5 dB
alarm threshold, so the v0.25.0 alarm misses it entirely — the
classifier catches it through the revisits. A move under about
2 m is invisible at 2 dB noise: its residual pattern is smaller
than the noise, so the classifier calls it `healthy` or
`uncertain` rather than inventing a cause. A steady offset used
to misread as a sag on seed 11, because the noise itself drifts
down -1.84 dB over that walk — the level-drop check fixes that
one (see the v0.63.0 notes at the end): the offset's level only
falls 0.75 dB from the first quarter to the last, against a 1.0
dB gate, so it stays `uncertain`. The same noise fluke used to
turn the moved beacon's seed-11 verdict `uncertain`, since the
monitor could not rule a decline out from the short window
alone — the longer history fixes that one (see the v0.62.0 notes
at the end): over 41 pairs the delta settles to -1.00 dB,
inside the same 1.0 dB gate by 0.004 dB, which is a knife-edge,
not a comfortable margin. No statistic in the window can tell a
real decline from that fluke; the fix is a longer history (more
loops), not a sharper threshold. On a single pass
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

## Drift alarms on the ward-map overlay

The overlay shows where the device went; with `--map-drift`
it also shows which beacon was going bad when the walk ended.
`render_map_overlay(..., drift=True)` reads the drift alarms
and classifications off the track's last plotted row — the
alarm state at the end of the track — and marks every beacon
still alarming with a red ring and a red "!" badge, plus a
label with its rolling mean residual in dB and the drift
classification verdict when the monitor gave one, e.g. `ALARM
+8.5 dB · uncertain`. The legend carries the same state in one
line, `Drift alarms at end of track: A +8.5 dB (uncertain)`,
or `none` when nothing alarms. The marking composes with the
planner-aware overlay: a beacon carries its forecast horizon
on one line and its drift alarm on the next, and the legend
drops a line so the alarm label under a bottom-row beacon
never prints over it.

```sh
python3 -m beacon.cli --map-overlay ward-a-track.json --map-output ward-a-track.svg --map-drift
python3 -m beacon.cli --map-overlay ward-a-track.json --map-output ward-a-track.svg --map-drift --map-planner
```

A beacon that alarmed mid-track and cleared is not marked:
the overlay shows the end state, not the history, so a badge
means the beacon was still alarming when the walk ended. A
track recorded without a drift monitor carries empty alarms,
so nothing is marked — an unmarked beacon means no alarm was
recorded, not that the beacon is healthy. And the badge marks
what the monitor recorded, not a ground-truth diagnosis: the
monitor's reference is the track's own estimate, so a beacon
8 dB too quiet can alarm on the wrong beacon, and the verdict
on the label is the same hint the classifier gives, not a
verdict on the hardware.

Measured behaviour: on the default ward walk, seed 7, with
beacon A reading 12 dB too loud, the overlay badges A only —
`ALARM +8.5 dB · uncertain`, summary `A +8.5 dB (uncertain)`
— the same alarm and verdict the export carries on the last
row. Seeds 1 and 3 badge A only too, at +8.4 dB and +8.0 dB;
the verdict differs by seed (`moved` on seed 1, `uncertain`
on seeds 3 and 7), because the classifier's verdict is a
hint from the same self-referenced monitor, and the overlay
carries it as recorded.

## Drift alarms on the live-track replay

The static overlay shows which beacon was going bad when the
walk ended; the replay shows it too, while the shift plays
back — frame by frame. `render_track_replay(..., drift=True)`
reads the drift alarms and classifications off each plotted
row and marks the beacons alarming on that row with the same
red ring and red "!" badge, plus the same label — its rolling
mean residual in dB and the drift classification verdict when
the monitor gave one, e.g. `ALARM +8.5 dB · uncertain` — but
the marking for a frame is visible only while that frame is
current, so the badge animates with the replay: a beacon that
alarmed mid-track and cleared is badged while it was alarming
and unmarked once it cleared, and a beacon that only starts
alarming late appears part-way through. The legend carries the
alarm history in one line, e.g. `Drift alarms during replay: A
+8.5 dB (uncertain), frames 4–39`, with each beacon's frame
range and the residual and verdict at its last alarming frame
— or `none` when nothing alarms on any frame. The marking
composes with the replay's coverage heatmap, and the legend
drops a line so the alarm label under a bottom-row beacon
never prints over it — and the replay's canvas now grows with
its legend, so no legend line is ever clipped below it (the
plain replay's own "Replay:" line already ran past the 560 px
canvas before this change).

```sh
python3 -m beacon.cli --map-overlay ward-a-track.json --map-output ward-a-track-replay.svg --map-replay --map-drift
python3 -m beacon.cli --map-overlay ward-a-track.json --map-replay --map-drift --map-heatmap
```

The same limits as the overlay apply: a track recorded
without a drift monitor marks nothing — an unmarked beacon
means no alarm was recorded, not that the beacon is healthy —
and the badge marks what the monitor recorded, not a
ground-truth diagnosis: the monitor's reference is the
track's own estimate, so a beacon 8 dB too quiet can alarm
on the wrong beacon, and the verdict on the label is the same
hint the classifier gives, not a verdict on the hardware.
What the per-frame marking adds over the end-of-track badge
is the history: a beacon that alarmed and cleared is no
longer invisible, so a flicker that the static overlay never
showed now plays back as a badge that appears and disappears
— and the frame range in the legend says exactly when.

Measured behaviour: on the default ward walk with beacon A
reading 12 dB too loud, the replay badges A on frames 4–39
— the first alarm is frame 4, as soon as the monitor has its
5 readings, at +9.2 dB on seed 7 (+9.8 dB seed 1, +9.7 dB
seed 3) — and the summary reads `A +8.5 dB (uncertain),
frames 4–39` on seed 7 (+8.4 dB `moved` seed 1, +8.0 dB
`uncertain` seed 3), the same final alarm and verdict the
overlay carries, since both read the same last row. On seed 7
the replay also badges C on frames 25–33, at +5.0 to +5.1 dB
with verdict `moved` — the flicker the live track's own
notes record ("C flickers into alarm briefly mid-walk and
clears on its own") — which the end-of-track badge never
showed, since C is quiet again by the last row. Seeds 1 and 3
badge A only.

Under the replay, a drift-alarm timeline strip shows the same
history as coloured segments — one row per beacon that alarmed,
one segment per contiguous run of alarming frames, coloured by
the drift classification verdict on those frames (amber moved,
blue battery-sag, purple noisy, grey uncertain — red when no
classification was recorded), labelled with the beacon and its
frame ranges, with a legend key for each colour shown — so a
viewer can see at a glance when each alarm started and ended,
and what the monitor thought was wrong, without stepping
through the frames. The strip reuses the same per-frame alarm
data the badges use, so it can never disagree with them: a
beacon that alarmed, cleared and alarmed again gets one segment
per alarm, not one run covering the gap, and a segment splits
when the verdict changes mid-alarm, so each segment carries
exactly the verdict the badges on those frames carry — the row
label still gives the alarm's full span, merged across verdict
changes. With no alarms the strip is an empty labelled track, so
the absence of segments reads as no alarm recorded rather than
a missing strip. `python3 -m beacon.cli --map-overlay
ward-a-track.json --map-output ward-a-track-replay.svg
--map-replay --map-drift` renders it (v0.41.0). Measured
behaviour: on the default ward walk with beacon A reading 12 dB
too loud, the strip shows A 4–39 and C 25–33 on seed 7 — A
amber for frames 4–6, where the monitor calls it moved, then
grey for 7–39, where it calls it uncertain, and C amber
throughout, matching the per-frame badges exactly; seeds 1 and
3 show A only.

## Per-frame forecast check on the live-track replay

The replay shows the drift alarms; it never showed whether
the planner's forecast saw the drift coming. `render_track_replay(..., forecast=True)`
marks each frame's beacons whose planner forecast is due
inside the forecast window on that frame — an amber "F" badge
with a label carrying the milestone, its hours remaining, and
the check's outcome — and the legend carries one line per
beacon with the same check, so the replay says whether the
forecast predicted the drift that actually followed:

- `confirmed` — the beacon alarmed at a later frame, at or
  after the milestone's time: the drift arrived when the
  forecast said it would. A beacon forecast to cross 3 dB loss
  now that alarms later in the track is confirmed.
- `contradicted` — the beacon alarmed before the milestone's
  time: the drift came far earlier than the battery model said
  it would, so the drift is not battery-driven — a power change
  or a move, not a dying cell — and the fix is to check the
  beacon, not to schedule a battery swap.
- `not observed` — no later frame alarms on it: the forecast
  has not been borne out within this track. A short track
  cannot confirm a forecast hours away, and a due forecast the
  monitor never catches is the self-referenced monitor's known
  miss — a quiet beacon drags the estimate with it and alarms
  on the wrong beacon, or never alarms at all.
- `pending` — the forecast was only active on the last frame,
  so there is no later frame to check against.

The check reads the forecast flags and the drift alarms off
the same per-frame rows the badges use, so it can never
disagree with them: the forecast is judged from the first
frame it was active, and the drift that followed is the
beacon's first alarm after that frame. A track recorded
without a plan marks nothing — the legend says no plan was
recorded — and a plan with nothing due inside the window
says no forecast was due. A track recorded without a monitor
can only ever say "not observed", since no alarm was recorded
to confirm or contradict the forecast.

```sh
python3 -m beacon.tracking --demo --forecast --battery-levels A=0.2,B=0.9,C=0.9,D=0.9 --drift --drift-beacon A:12 --export-json ward-a-track.json
python3 -m beacon.cli --map-overlay ward-a-track.json --map-output ward-a-track-replay.svg --map-replay --map-drift --map-forecast
```

The same honest limits as the drift flags apply: the check
compares the forecast's timing against the alarms the monitor
recorded, not against a ground-truth diagnosis — a beacon
alarming early for a power offset contradicts a battery
forecast for the wrong reason in the same way it alarms on
the right beacon for the wrong cause, and the monitor's
reference is the track's own estimate, so a beacon reading
too quiet can alarm on the wrong beacon and confirm or
contradict the wrong beacon's forecast. The check says
whether the drift arrived when the forecast said it would,
not why it arrived.

Measured behaviour (v0.42.0): on the default ward walk, seed
7, with beacon A reading 12 dB too loud and the plan's levels
A=0.2, B=0.9, C=0.9, D=0.9 — A due 3 dB loss now, the rest due
2 dB noise in 11.3 h — the replay marks all four beacons on
every frame and checks A `confirmed, alarm from frame 4`,
C `contradicted, alarm from frame 25` (its 11.3 h forecast
against an alarm 25 seconds in), and B and D `not observed in
this track`.

## Replay of multi-device tracks

The replay played one device's track; a ward runs several tags
at once, and a shift is all of them together. `track_rows(...,
device="tag-1")` tags a track's rows with its device — the tag
rides the JSON export and the CSV export (a new `device`
column, last, empty for a single unnamed device), and
`multi_track_rows({"tag-1": positions_a, "tag-2": positions_b})`
combines several devices' tracks into one row list, each
device's rows keeping that device's own coverage — so a
combined file is just the per-device exports with their `rows`
concatenated. `render_track_replay` on combined rows plays
every device together: one dot, trail, ellipse series and
timestamp label per device, each in its own colour, on its own
timing — a device that starts late or walks fewer steps still
plays at its own pace — and the drift badges, the timeline
strip and the forecast checks are computed per device from
that device's own rows, so one tag's alarm never colours
another tag's track. The legend carries one line per device
for the replay length, the alarm history and the forecast
check. A row with no device is one unnamed device, so a
single-device track renders exactly as before, and old
exports (no `device` key or column) load with the device
defaulting to None.

```sh
python3 -m beacon.cli --map-overlay ward-a-and-b.json --map-output ward-a-and-b-replay.svg --map-replay --map-drift --map-forecast
```

The same honest limits as the drift flags and the forecast
check apply, per device: the badges mark what each device's
own monitor recorded, and the check judges each device's
forecast against that device's own alarms — so a tag that
never alarms can only ever say "not observed", even when
another tag on the same beacons confirms the same forecast.
Two devices alarming on the same beacon at the same time
share that beacon's badge spot, and the strip and legend tell
them apart. The static overlay renders the same per-device
series, so a multi-device file reads the same on the map and
in the replay — the replay adds the motion, the overlay the
still picture.

Measured behaviour (v0.43.0): two simulated walks through one
`MultiDeviceTracker` — tag-1 forward, seed 7, with beacon A
reading 12 dB too loud, tag-2 the reverse walk, seed 11,
healthy — with the plan's levels A=0.2, B=0.9, C=0.9, D=0.9:
the replay draws two dots, two trails, 80 ellipses and 80
timestamp labels; tag-1 alarms on A frames 4–39 (moved, then
uncertain) and C flickers frames 25–33 (moved), tag-2 alarms
on nothing; tag-1's forecast checks read A `confirmed, alarm
from frame 4`, C `contradicted, alarm from frame 25`, B and D
`not observed in this track`, while tag-2 — carrying the same
forecasts, due on the same beacons — reads all four `not
observed in this track`.

## Planner forecast flags on the live track

The planner-aware overlay shows when coverage will thin, but
only when someone opens the map. A dashboard watching the live
track needs the same warning in the stream itself: a beacon
that is about to start lying should flag while the ward is
running, not at the next map render. `LiveTracker` takes an
optional deployment plan, and every live position it emits
carries one `ForecastFlag` per beacon — the soonest of its
dark, 3 dB loss and 2 dB noise milestones, with the hours
remaining and all three horizons riding along — plus a
plain-language alert for every beacon whose soonest milestone
falls inside the forecast window (24 hours by default):

```python
from beacon import (BatteryModel, Beacon, LiveTracker,
                    PositioningEngine, plan_deployment)

beacons = [Beacon("A", 0, 0), Beacon("B", 20, 0),
           Beacon("C", 20, 12), Beacon("D", 0, 12)]
plan = plan_deployment({
    b.id: BatteryModel(drain_rate_per_hour=0.05) for b in beacons},
    levels={"D": 0.2})   # D measured at 20% charge
tracker = LiveTracker(PositioningEngine(beacons), plan=plan)
live = tracker.feed({"A": -62.1, "B": -71.4, "C": -68.0, "D": -59.8}, t=0.0)
# live.forecasts — D first: milestone "3 dB loss", hours 0.0,
# status "degrading"; the rest "2 dB noise" in 13.3 h
# live.forecast_alerts — ("D: 3 dB loss now (degrading)", ...)
```

The milestone names match the replacement schedule's reasons,
with the same tie-breaking (going dark beats reading weak,
reading weak beats reading noisy), so the flag and the work
order never disagree about which milestone drives a beacon. A
beacon that never drains flags `never` with infinite horizons,
which serialise as null, and never alerts. The plan is taken
as made at the first reading's timestamp, so the horizons count
down as the track runs — a milestone that passes mid-track
reads 0.0 hours remaining rather than going negative — and
`reset()` restarts the clock for a new device. A tracker
without a plan is unaffected: both fields are empty. The
stream carries the flags too, including on a line whose
position is null, so a dashboard sees a beacon about to go
dark even on a step that could not be located:

```sh
python3 -m beacon.tracking --demo --forecast
python3 -m beacon.tracking --demo --forecast --battery-levels A=0.9,B=0.6,C=0.4,D=0.2
python3 -m beacon.tracking --demo --forecast --forecast-window-hours 1.0
```

Measured behaviour: on the default ward with four fresh
beacons at 0.05 per hour every flag reads `2 dB noise in
13.3h (ok)` and all four alert inside the 24-hour window,
while a 1-hour window alerts on none of them — the flags are
still on the position, only the warnings go quiet. With
measured levels 0.9, 0.6, 0.4, 0.2 the flags read D `3 dB
loss now (degrading)`, C `2 dB noise in 1.3h (ok)`, B in
5.3 h and A in 11.3 h — the same order and milestones as the
replacement schedule, now riding on the live track.

## Planner forecast flags on multi-device tracks

The forecast flags above ride on one device's track; a ward
dashboard runs several tags at once, and a nurse's badge
should warn about a dying beacon just as a patient tag does.
`MultiDeviceTracker` takes the same optional deployment plan,
and every device's live position carries the same flags — one
per beacon, the soonest milestone with its hours remaining,
with the plain-language alerts inside the forecast window —
because each device's tracker is built with the plan:

```python
from beacon import (BatteryModel, Beacon, MultiDeviceTracker,
                    plan_deployment)

beacons = [Beacon("A", 0, 0), Beacon("B", 20, 0),
           Beacon("C", 20, 12), Beacon("D", 0, 12)]
plan = plan_deployment({
    b.id: BatteryModel(drain_rate_per_hour=0.05) for b in beacons},
    levels={"D": 0.2})   # D measured at 20% charge
tracker = MultiDeviceTracker(beacons, plan=plan)
tag1 = tracker.feed("tag-1", {"A": -62.1, "B": -71.4,
                              "C": -68.0, "D": -59.8}, t=0.0)
# tag1.forecasts — D first: "3 dB loss", 0.0 hours, "degrading"
# tag1.forecast_alerts — ("D: 3 dB loss now (degrading)", ...)
```

The plan is taken as made at each device's own first reading,
not at the first tag's: each device's clock is its own, so a
tag that shows up an hour into the shift counts its horizons
down from when it started, and resetting one device restarts
only its clock. A tracker without a plan is unaffected: both
fields are empty on every device. The stream carries the flags
too, including on a null line, so a device too thin to locate
still warns about a beacon about to go dark:

```sh
python3 -m beacon.multi_device --demo --forecast
python3 -m beacon.multi_device --demo --forecast --battery-levels A=0.9,B=0.6,C=0.4,D=0.2
python3 -m beacon.multi_device --demo --forecast --forecast-window-hours 1.0
```

Measured behaviour: on the default ward with four fresh
beacons at 0.05 per hour both demo devices flag `2 dB noise
in 13.3h (ok)` and all four alert inside the 24-hour window.
With measured levels 0.9, 0.6, 0.4, 0.2 both devices flag D
`3 dB loss now`, C in 1.3 h, B in 5.3 h and A in 11.3 h —
the same flags as the single-device track, on every device.
A tag starting one hour after the first and running five more
hours flags 8.3 h remaining while the second tag, four hours
in, flags 9.3 h — the two clocks really are separate.

## Drift flags on the live track

The drift monitor watches the observation stream for a beacon
going bad — but it needs a position to compare each beacon's
RSSI against, and a live tracker has no ground truth: the only
position it has is its own estimate. `LiveTracker` takes an
optional `DriftMonitor` and feeds it each located step's fresh
readings together with the live position just computed, so the
position carries the beacons currently alarming — each with its
rolling mean RSSI residual in dB and how many readings are in
its window — plus a plain-language alert per alarming beacon:

```python
from beacon import Beacon, DriftMonitor, LiveTracker, PositioningEngine

beacons = [Beacon("A", 0, 0), Beacon("B", 20, 0),
           Beacon("C", 20, 12), Beacon("D", 0, 12)]
tracker = LiveTracker(PositioningEngine(beacons),
                      drift_monitor=DriftMonitor(beacons))
live = tracker.feed({"A": -62.1, "B": -71.4, "C": -68.0, "D": -59.8}, t=0.0)
# live.drift_alarms — () while every beacon matches its calibration
# live.drift_alerts — () likewise
```

The monitor is only fed on steps that locate — a step too thin
to produce a position has no reference to check against — and
only with the beacons heard in that reading that sit on the
track's current floor, since a beacon heard through a slab
would read as drift. The alarms ride on a null line too, so a
dashboard sees a beacon going bad even on a step that could
not be located, and `reset()` clears the monitor along with the
track, so a new device starts with a clean window. The alert
states only what the alarm establishes — `A: RSSI +8.5 dB off
calibration` — and the classification verdict is deliberately
not in it: with the track's own estimate as the reference the
verdict is a hint, not a diagnosis, so it stays available
through `tracker.drift_classifications()` for inspection
instead of riding on the position. The stream carries the
flags with `--drift`, and `--drift-beacon A:12` (demo only)
offsets one beacon's RSSI by that many dB for the whole demo
walk, so the demo shows a beacon going bad mid-track:

```sh
python3 -m beacon.tracking --demo --drift
python3 -m beacon.tracking --demo --drift --drift-beacon A:12
python3 -m beacon.tracking --drift < readings.jsonl
```

The reference is the track's own estimate, not ground truth,
so the monitor is a consistency check on the live track — each
beacon checked against the position the consensus of the
beacons produced — and the measured behaviour shows both sides
of that. A beacon reading too loud is caught: it alarms on the
right beacon, because the estimate is only pulled part-way
towards it. A beacon reading too quiet is not: the quiet beacon
drags the estimate with it, the distance inversion blows the
track up, and the monitor blames a healthy beacon. A slow sag
and a move never alarm at all through the live track — their
residuals stay under the threshold against the dragged
estimate — even though the same monitor fed a reference
position classifies both. The reliable per-beacon verdict still
comes from feeding the monitor a reference position (a
reference tag at a known spot), as `DriftMonitor` documents;
the live-track flags say the track is inconsistent and name
the prime suspect, nothing more.

Measured behaviour (simulator, 2 dB RSSI noise, default ward
walk, seeds 1, 3, 7 — the alarms below are the final state
after the 40-step walk):

- Healthy walk: no alarms on any seed. Seed 11 classifies D
  `noisy` with no alarm — a false verdict, not a false alarm.
- A reading 12 dB too loud alarms on A from step 4 — as soon
  as the monitor has its 5 readings — at a rolling mean of
  +8.0 to +8.5 dB, attenuated from +12 dB because the estimate
  is pulled towards the loud beacon, and only A alarms at the
  end of the walk. On seed 7 C flickers into alarm briefly
  mid-walk and clears on its own.
- A reading 6 dB too loud never alarms at the end of the walk:
  the pull attenuates the mean below the 5 dB threshold (on
  seed 1 it flickers into alarm for a step, then clears).
- C reading 8 dB too quiet — the battery-sag direction —
  alarms on D, not on C, at +5.3 to +7.6 dB, while B flickers
  into alarm early on and clears. The track's mean error in
  that scenario is 11–59 m, so the track is already broken by
  the time the alarm fires.
- A beacon draining at 0.3 per hour over two loops never
  alarms and classifies `noisy`, not `battery-sag`. A beacon
  moved 7 m never alarms either, and classifies `noisy`.

## Drift flags on multi-device tracks

The drift flags above ride on one device's track; a ward
dashboard runs several tags at once, and a nurse's badge
should warn about a dying beacon just as a patient tag does.
`MultiDeviceTracker` takes the same optional drift monitor,
and every device's live position carries the beacons
currently alarming on that device's own track — each with its
rolling mean RSSI residual in dB and how many readings are in
its window — plus a plain-language alert per alarming beacon:

```python
from beacon import Beacon, DriftMonitor, MultiDeviceTracker

beacons = [Beacon("A", 0, 0), Beacon("B", 20, 0),
           Beacon("C", 20, 12), Beacon("D", 0, 12)]
tracker = MultiDeviceTracker(beacons,
                             drift_monitor=DriftMonitor(beacons))
tag1 = tracker.feed("tag-1", {"A": -62.1, "B": -71.4,
                              "C": -68.0, "D": -59.8}, t=0.0)
# tag1.drift_alarms — () while every beacon matches its calibration
# tag1.drift_alerts — () likewise
```

Each device gets its own fresh monitor with the same tuning
(`DriftMonitor.fresh()`), not one monitor shared across the
ward: a shared window would mix two devices' readings, so one
tag's healthy readings would dilute another tag's alarm, and
resetting one device would clear the monitor for every device.
The monitor passed in is a template for the tuning — it is
never fed itself — and `reset("tag-1")` clears only tag-1's
monitor, so a tag handed to a new patient starts with a clean
window while the other tags keep alarming. A tracker without a
monitor is unaffected: both fields are empty on every device.
The alarms ride on a null line too, so a device too thin to
locate still warns about a beacon going bad, and
`tracker.drift_alarms(device_id)`, `drift_alerts(device_id)`
and `drift_classifications(device_id)` expose one device's
monitor without a position in hand. The stream carries the
flags with `--drift`, and `--drift-beacon A:12` (demo only)
offsets one beacon's RSSI on tag-1's walk only, so the demo
shows one device alarming while the other stays quiet:

```sh
python3 -m beacon.multi_device --demo --drift
python3 -m beacon.multi_device --demo --drift --drift-beacon A:12
python3 -m beacon.multi_device --drift < readings.jsonl
```

The same honest limit as the single-device track applies per
device: the reference is each track's own estimate, not ground
truth, so a beacon reading too quiet can drag the estimate
with it and alarm on the wrong beacon — the alarm says that
device's track is inconsistent and names the prime suspect,
nothing more.

Measured behaviour (simulator, 2 dB RSSI noise, default ward
walk, seeds 1, 3, 7 — the alarms below are the final state
after each device's 40-step walk):

- Healthy walks: no alarms on either device, on any seed.
- A reading 12 dB too loud on tag-1's walk only alarms on A
  for tag-1 at a rolling mean of +8.0 to +8.5 dB, while tag-2
  stays quiet — the monitors are separate, so tag-2's healthy
  readings never dilute tag-1's alarm. The same offset on
  both devices' walks alarms on A for both.
- C reading 8 dB too quiet on tag-1's walk alarms on D, not
  on C, at +5.3 to +7.6 dB, while tag-2 stays quiet — the
  mis-alarm is per device, not shared across them.
- Resetting tag-1 after both devices alarm clears only
  tag-1's monitor; tag-2 keeps alarming.

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
beacons excluded as dark with their reasons, the battery level
reported for each beacon at that step, and the track's
overall coverage — the located share of the walk, the same on
every row. When the track was recorded with a deployment plan
(`--forecast` on the CLI), each row also carries the planner's
forecast flags — one per beacon, the soonest milestone with its
hours remaining — and the plain-language alerts for the beacons
due inside the forecast window, so the exported track warns
about a beacon about to start lying without the plan alongside
it; a track recorded without a plan exports empty flags and
alerts. When the track was recorded with a drift monitor
(`--drift` on the CLI), each row also carries the drift alarms
— the beacons alarming at that step, each with its rolling mean
RSSI residual in dB and how many readings are in its window —
and the plain-language alert per alarming beacon, so the
exported track warns about a beacon going bad mid-shift without
the monitor alongside it; a track recorded without a monitor
exports empty alarms and alerts. The alarms say only what the
monitor establishes: the monitor's reference is the track's own
estimate, not ground truth, so a beacon 8 dB too quiet can
alarm on the wrong beacon — the export carries the alarms as
recorded, without claiming a ground-truth attribution. Steps
that produced no position are not rows, since
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
as the static overlay, and with `--map-drift` the same drift
alarm marking as the static overlay: a red ring and a red "!"
badge on every beacon still alarming at the end of the track.
The CLI renders it directly:

```sh
python3 -m beacon.cli --map-overlay ward-a-track.json --map-output ward-a-track-replay.svg --map-replay
python3 -m beacon.cli --map-overlay ward-a-track.json --map-replay --replay-step-seconds 0.5
python3 -m beacon.cli --map-overlay ward-a-track.json --map-replay --map-drift
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
  called `uncertain` 5/5 (v0.63.0: the sag call now also requires
  the residual level to have fallen across the walk, so the
  seed-11 noise fluke no longer reads as a decline), while a move
  under about 2 m is invisible at 2 dB noise and comes back
  `healthy` or `uncertain`.
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
  degrading within 1.3 hours.
- Planner forecast flags on the live track — done: `LiveTracker`
  takes an optional `DeploymentPlan` and every live position
  carries one `ForecastFlag` per beacon — the soonest of its
  dark, 3 dB loss and 2 dB noise milestones with the hours
  remaining, all three horizons riding along — plus a
  plain-language alert (`D: 3 dB loss now (degrading)`) for
  every beacon whose milestone falls inside the forecast window
  (24 hours by default); `python -m beacon.tracking --demo
  --forecast` attaches the plan to the stream, reusing
  `--battery-levels` and `--drain-rate`, and the flags ride on
  null lines too (v0.30.0); four fresh beacons flag `2 dB noise
  in 13.3h (ok)` and all alert inside 24 hours but none inside
  1 hour, while measured levels 0.9, 0.6, 0.4, 0.2 flag D `3 dB
  loss now`, C in 1.3 h, B in 5.3 h, A in 11.3 h — the same
  order and milestones as the replacement schedule. The roadmap
  grows as new needs surface — the forecast flags are the latest
  addition.
- Planner forecast flags on multi-device tracks — done:
  `MultiDeviceTracker` takes the same optional `DeploymentPlan`
  and passes it to every device's `LiveTracker`, so every
  device's live position carries the same forecast flags — one
  per beacon, soonest milestone with its hours remaining, with
  the plain-language alerts inside the forecast window (24
  hours by default) — counted down from each device's own first
  reading, so a tag that shows up mid-shift warns on its own
  clock, and resetting one device restarts only its clock; the
  flags ride on null lines too, and `python -m
  beacon.multi_device --demo --forecast` attaches the plan,
  reusing `--battery-levels` and `--drain-rate` (v0.31.0); four
  fresh beacons flag `2 dB noise in 13.3h (ok)` on both demo
  devices, and measured levels 0.9, 0.6, 0.4, 0.2 flag D `3 dB
  loss now`, C in 1.3 h, B in 5.3 h, A in 11.3 h — the same
  flags as the single-device track, on every device.
- Forecast flags in exported track rows — done: `track_rows`
  carries the same per-beacon forecast flags and alerts the live
  track already exposes — the soonest milestone with its hours
  remaining, all three horizons riding along — on every exported
  row, in both JSON and CSV, so an exported track is
  self-describing about upcoming battery milestones without the
  plan alongside it; a track recorded without a plan exports
  empty flags and alerts, and exports written before this change
  still load, with the forecasts defaulting to empty (v0.32.0);
  measured levels 0.9, 0.6, 0.4, 0.2 flag D `3 dB loss now`, C
  in 1.3 h, B in 5.3 h, A in 11.3 h on every row, and the JSON
  and CSV exports load back to identical rows. The roadmap
  grows as new needs surface — the exported forecast flags are
  the latest addition.
- Drift flags on the live track — done: `LiveTracker` takes an
  optional `DriftMonitor` and feeds it each located step's
  fresh readings together with the live position just computed,
  so every live position carries the beacons currently alarming
  — each with its rolling mean RSSI residual in dB and how many
  readings are in its window — plus a plain-language alert per
  alarming beacon (`A: RSSI +8.5 dB off calibration`); the
  alarms ride on null lines too, only the beacons on the
  track's current floor are fed to the monitor, `reset()`
  clears the monitor with the track, and `python -m
  beacon.tracking --drift` attaches the monitor to the stream,
  with `--drift-beacon A:12` (demo only) offsetting one
  beacon's RSSI so the demo shows a beacon going bad (v0.33.0);
  measured on the default ward walk, seeds 1, 3, 7: a beacon
  12 dB too loud alarms on the right beacon from step 4 at +8.0
  to +8.5 dB, a beacon 6 dB too loud never alarms, and a beacon
  8 dB too quiet alarms on the wrong beacon — the
  self-referenced monitor is a consistency check on the track,
  not a diagnosis, and the reliable verdict still comes from
  feeding the monitor a reference position.
- Drift flags on multi-device tracks — done:
  `MultiDeviceTracker` takes the same optional `DriftMonitor`
  and gives each device's tracker its own fresh monitor with
  the same tuning, so every device's live position carries the
  beacons currently alarming on its own track, with a
  plain-language alert per alarming beacon; the alarms ride on
  null lines too, `reset("tag-1")` clears only tag-1's
  monitor, and `python -m beacon.multi_device --drift`
  attaches the monitor to the stream, with `--drift-beacon
  A:12` (demo only) offsetting one beacon's RSSI on tag-1's
  walk so the demo shows one device alarming while the other
  stays quiet (v0.34.0); measured on the default ward walk,
  seeds 1, 3, 7: a beacon 12 dB too loud on one device's walk
  alarms on the right beacon for that device at +8.0 to
  +8.5 dB while the other device stays quiet, and a beacon
  8 dB too quiet alarms on the wrong beacon for that device
  only — the same self-referenced limit as the single-device
  track, per device.
- Drift alarms in exported track rows — done: `track_rows`
  carries the same drift alarms and alerts the live track
  already exposes — the beacons alarming at that step, each
  with its rolling mean RSSI residual in dB and how many
  readings are in its window, plus the plain-language alert
  per alarming beacon — on every exported row, in both JSON
  and CSV, so an exported track is self-describing about a
  beacon going bad mid-shift without the monitor alongside it;
  a track recorded without a monitor exports empty alarms and
  alerts, and exports written before this change still load,
  with the drift alarms defaulting to empty (v0.35.0);
  measured on the default ward walk, seed 7, with beacon A
  reading 12 dB too loud: the first exported row carrying an
  alarm is row 4, `A: RSSI +9.2 dB off calibration` on 5
  readings, and the last row still alarms on A only at
  +8.5 dB on 20 readings — the same alarm the live track
  carries, now on the file. The alarms say only what the
  monitor establishes: the monitor's reference is the track's
  own estimate, not ground truth, so a beacon 8 dB too quiet
  can alarm on the wrong beacon — the export carries the
  alarms as recorded, without claiming a ground-truth
  attribution. The roadmap grows as new needs surface — the
  exported drift alarms are the latest addition.
- Drift classifications in exported track rows — done:
  `track_rows` carries the same per-beacon drift
  classifications the live track already exposes — one
  classification per beacon per row, the verdict (healthy,
  battery-sag, moved, noisy, uncertain) with the numbers it
  rests on — on every exported row, in both JSON and CSV, so an
  exported track says what the monitor thinks is wrong with
  each beacon, not just which ones are alarming, without the
  monitor alongside it; a track recorded without a monitor
  exports empty classifications, and exports written before
  this change still load, with the classifications defaulting
  to empty (v0.36.0); measured on the default ward walk, seed
  7, with beacon A reading 12 dB too loud: the first rows
  classify every beacon `uncertain` — one reading is not
  enough to classify — from row 4 A classifies `moved` at
  +9.2 dB on 5 readings while the other three stay `healthy`,
  and on the last row A is `uncertain` at +8.5 dB on 20
  readings with a null revisit delta (too few revisits to
  measure one), while C inherits a false `moved` verdict at
  +2.9 dB — the estimate dragged towards loud A spreads C's
  residuals around a shifted mean. The verdicts are a hint
  from the same self-referenced monitor as the alarms, not a
  diagnosis, and the export carries them as recorded. On a
  multi-device track each device's rows carry only its own
  monitor's verdicts: with A 12 dB too loud on tag-1's walk
  only, tag-1's last row classifies A `uncertain` at +8.5 dB
  while tag-2's last row classifies every beacon `healthy`.
- Drift alarms on the ward-map overlay — done:
  `render_map_overlay(..., drift=True)` marks every beacon
  alarming on the track's last plotted row with a red ring
  and a red "!" badge, plus a label with its rolling mean
  residual in dB and the drift classification verdict, e.g.
  `ALARM +8.5 dB · uncertain`, and a legend line carrying the
  alarm state at the end of the track; the marking composes
  with the planner-aware overlay, and the legend drops a line
  so the alarm label under a bottom-row beacon never prints
  over it; `python -m beacon.cli --map-overlay track.json
  --map-drift` renders it (v0.37.0); measured on the default ward walk
  with beacon A reading 12 dB too loud: the overlay badges A
  only, at +8.5 dB on seed 7 (+8.4 dB seed 1, +8.0 dB seed 3),
  with the verdict the monitor gave (`moved` on seed 1,
  `uncertain` on seeds 3 and 7). The badge marks what the
  monitor recorded, not a ground-truth diagnosis: a beacon
  8 dB too quiet can alarm on the wrong beacon, a beacon that
  alarmed and cleared is not marked, and a track recorded
  without a monitor marks nothing — an unmarked beacon means
  no alarm was recorded, not that it is healthy.
- Drift alarms on the live-track replay — done:
  `render_track_replay(..., drift=True)` marks every beacon
  alarming on the track's last plotted row with the same red
  ring, red "!" badge and label as the static overlay, and the
  legend carries the same summary line; the badge is static —
  it marks the end state, not the alarm history — and the
  marking composes with the replay's coverage heatmap, while
  the replay's canvas now grows with its legend so no legend
  line is ever clipped (the plain replay's own "Replay:" line
  already ran past the 560 px canvas); `python -m beacon.cli
  --map-overlay track.json --map-replay --map-drift` renders
  it (v0.38.0); measured on the default ward walk with beacon
  A reading 12 dB too loud: the replay badges A only, at
  +8.5 dB on seed 7 (+8.4 dB seed 1, +8.0 dB seed 3), with the
  same verdicts as the overlay. The same limits as the overlay
  apply: cleared alarms unmarked, no-monitor tracks mark
  nothing, and the badge marks what the monitor recorded, not
  a ground-truth diagnosis.
- Drift alarms animated per frame on the live-track replay —
  done: `render_track_replay(..., drift=True)` now reads the
  drift alarms and classifications off each plotted row
  instead of the last row only, and marks the beacons alarming
  on that row with the same red ring, red "!" badge and label
  as the static overlay — one group per alarming frame,
  visible only while that frame is current, so the badge
  animates with the replay: a beacon that alarmed mid-track
  and cleared is badged while it was alarming and unmarked
  once it cleared, and the legend carries the alarm history
  in one line, with each beacon's frame range and the residual
  and verdict at its last alarming frame; `python -m
  beacon.cli --map-overlay track.json --map-replay --map-drift`
  renders it (v0.39.0); measured on the default ward walk with
  beacon A reading 12 dB too loud: the replay badges A on
  frames 4–39 — the first alarm is frame 4, at +9.2 dB on seed
  7 — and on seed 7 it also badges C on frames 25–33 at +5.0
  to +5.1 dB, the flicker the end-of-track badge never showed,
  since C is quiet again by the last row; seeds 1 and 3 badge
  A only. The same limits as the overlay apply: no-monitor
  tracks mark nothing, and the badge marks what the monitor
  recorded, not a ground-truth diagnosis.
- Drift-alarm timeline strip under the live-track replay — done:
  `render_track_replay(..., drift=True)` draws a horizontal timeline
  strip under the replay canvas — one row per beacon that alarmed,
  with a coloured segment per contiguous run of alarming frames,
  labelled with the beacon and its frame ranges, so a viewer can see
  at a glance when each alarm started and ended without stepping
  through the frames; the strip reuses the same per-frame alarm data
  the badges use, so a beacon that alarmed, cleared and alarmed again
  gets one segment per alarm, and with no alarms the strip is an empty
  labelled track; `python -m beacon.cli --map-overlay track.json
  --map-replay --map-drift` renders it (v0.40.0); measured on the
  default ward walk with beacon A reading 12 dB too loud: seed 7 shows
  A 4–39 and C 25–33, matching the per-frame badges exactly, and seeds
  1 and 3 show A only.
- Verdict-coloured timeline strip segments — done: each strip segment
  is coloured by the drift classification verdict on its frames
  (amber moved, blue battery-sag, purple noisy, grey uncertain, red
  when no classification was recorded), with a legend key for each
  colour shown, and a segment splits when the verdict changes
  mid-alarm, so the strip's colours are the badges' verdicts, not a
  second opinion — the row label still gives the alarm's full span;
  `python -m beacon.cli --map-overlay track.json --map-replay
  --map-drift` renders it (v0.41.0); measured on the default ward
  walk with beacon A reading 12 dB too loud: seed 7 shows A amber
  for frames 4–6 (moved) then grey for 7–39 (uncertain), and C amber
  throughout 25–33 (moved). The roadmap grows as new needs surface
  — the forecast check is the latest addition.
- Per-frame forecast check on the live-track replay — done:
  `render_track_replay(..., forecast=True)` marks each frame's
  beacons whose planner forecast is due inside the forecast window
  on that frame — an amber "F" badge with a label carrying the
  milestone, its hours remaining, and the check's outcome — and the
  legend carries one line per beacon with the same check: whether
  the forecast predicted the drift that actually followed —
  `confirmed` (alarmed at or after the milestone's time),
  `contradicted` (alarmed before it — the drift is not
  battery-driven), `not observed` (no later alarm in this track),
  or `pending` (only live on the last frame); the check reads the
  forecast flags and the drift alarms off the same per-frame rows
  the badges use, so it can never disagree with them; `python -m
  beacon.cli --map-overlay track.json --map-replay --map-forecast`
  renders it (v0.42.0); measured on the default ward walk, seed 7,
  with beacon A reading 12 dB too loud and the plan's levels A=0.2,
  B=0.9, C=0.9, D=0.9: A is confirmed from frame 4, C is
  contradicted from frame 25 against an 11.3 h forecast, and B and D
  are not observed.
- Replay of multi-device tracks — done: `track_rows(...,
  device="tag-1")` tags a track's rows with its device, the tag
  riding the JSON export and a new `device` column on the CSV
  export, and `multi_track_rows` combines several devices'
  tracks into one row list; `render_track_replay` on combined
  rows plays every device together — one dot, trail, ellipse
  series and timestamp label per device, each in its own colour
  and on its own timing — with the drift badges, the timeline
  strip and the forecast checks computed per device from that
  device's own rows, so one tag's alarm never colours another
  tag's track; a row with no device is one unnamed device, so a
  single-device track renders exactly as before, and old exports
  load with the device defaulting to None; `python -m beacon.cli
  --map-overlay tracks.json --map-replay --map-drift --map-forecast`
  renders it (v0.43.0); measured on two simulated walks through
  one `MultiDeviceTracker` — tag-1 forward, seed 7, with beacon
  A reading 12 dB too loud, tag-2 the reverse walk, seed 11,
  healthy, plan levels A=0.2, B=0.9, C=0.9, D=0.9: two dots, two
  trails, 80 ellipses and 80 timestamp labels; tag-1 alarms on A
  frames 4–39 and C frames 25–33, tag-2 on nothing; tag-1's
  forecast checks read A confirmed from frame 4 and C
  contradicted from frame 25, while tag-2 reads all four not
  observed. The static overlay renders the same two walks as
  two series (v0.44.0): two trails, 80 ellipses and 80 dots,
  two start and two end markers, tag-1 pink and tag-2 blue —
  the same colours as the replay's dots — and the drift badge
  on A names tag-1 only, `tag-1: A +8.5 dB (uncertain);
  tag-2: none`, with a single-device overlay byte-identical
  to v0.43.0.
- Steady-offset misread as a moved beacon — done (v0.64.0): on
  the tripled two-floor walk at the default 20-reading window, a
  steady -6 dB offset read `moved` on seed 1 — the short window's
  spread is 2.70 dB, 0.20 dB over the moved gate, around a -5.96
  dB mean with a stable revisit delta — a noise-inflated spread,
  not a position-dependent pattern. The level-drop check (v0.63.0)
  fixed the same offset misreading as a sag; the moved call now
  has its own discriminator, in the same design: when a longer
  history exists, the short-window moved call needs it to agree
  that the spread is real — the long window's spread must clear
  the 2 dB expected noise by at least 1.5 standard errors. The
  offset's 120-reading history spreads 2.09 dB, 0.70 standard
  errors above noise, so it reads `uncertain` on 5/5 seeds, while
  the moved beacon's long spread is 2.57 to 2.95 dB, 4.4 to 7.4
  standard errors above noise, so it stays `moved` on 5/5. A
  residual-vs-distance correlation was measured and rejected: the
  moved beacon reads louder almost everywhere, since it moved
  towards the ward interior, so its correlation with distance is
  near zero (-0.05 over the long history on seed 1) and the
  offset's is the same size. The margin is honest but thin — the
  offset's long spread clears the noise test by only 0.10 dB on
  seed 1, and on two loops seed 3's short spread sits 0.01 dB
  under the moved gate with a long spread that would pass the
  test, so a slightly unluckier noise draw could still misread
  it (see the v0.64.0 notes at the end).

## Combined multi-device track export

The replay and the overlay could already read a combined
file, but nobody wrote one: a ward dashboard had to collect
each device's positions itself and concatenate the per-device
exports by hand. `MultiDeviceTracker` now records every
device's track as it runs — one position per reading, None
for a reading too thin to locate, so the coverage stays
honest — and `export_track_json(path)` /
`export_track_csv(path)` write all of them as one combined
file directly, in the same JSON and CSV formats as the
single-device export, with the same header (total steps,
located rows, overall coverage) and the device in the last
CSV column; `python -m beacon.multi_device --demo
--export-json ward-a-and-b.json --export-csv
ward-a-and-b.csv` does the same at the end of a run, and the
stream mode exports when the stream ends. The rows are the
same `multi_track_rows` rows the overlay and the replay group
by device, so the exported file replays and overlays exactly
like the per-device exports it was combined from — the JSON
and CSV files load back to identical rows, the replay draws
the same dots and timestamp labels from either, and the
overlay renders the same two series. A device's recorded
track goes with it when the device is reset or removed, so a
tag handed to a new patient stops appearing in the export
until its next reading, and exporting a tracker that has
tracked nothing raises instead of writing an empty file.

```sh
python3 -m beacon.multi_device --demo --export-json ward-a-and-b.json
python3 -m beacon.multi_device --demo --export-json ward-a-and-b.json --export-csv ward-a-and-b.csv
```

Measured behaviour (v0.45.0): two simulated walks through
one `MultiDeviceTracker` — tag-1 forward, seed 7, tag-2 the
reverse walk, seed 11 — export to 80 rows, 40 per device,
header steps 80, located 80, coverage 1.0, and the JSON and
CSV files load back to the same rows `multi_track_rows`
produces; a one-device tracker exports a file identical to
the single-device export for that device's track; a tag fed
one thin reading before its walk exports 80 rows from 81
steps, header coverage 80/81, with that tag's rows carrying
its own 40/41 coverage while the other tag's stay at 1.0.
The roadmap grows as new needs surface — the combined
multi-device export is the latest addition.

## Multi-device track accuracy report

The single-device accuracy report scores one walk; a ward
runs several tags at once, and one blended number hides which
tag's walk went wrong — a pump that spent the shift behind a
metal bed and a nurse's badge that tracked cleanly do not
deserve the same verdict. `evaluate_multi_device_track`
scores every device's recorded track against its own ground
truth and returns a `MultiDeviceTrackReport` — one
`TrackReport` per device, each with the same mean, median,
90th percentile, worst error, coverage and floor accuracy as
the single-device report, plus an `overall` report over every
device's located steps pooled together:

```python
from beacon import MultiDeviceTracker, evaluate_multi_device_track
from beacon.simulate import rectangular_walk, simulate_walk

beacons = [Beacon("A", 0, 0), Beacon("B", 20, 0),
           Beacon("C", 20, 12), Beacon("D", 0, 12)]
forward = simulate_walk(rectangular_walk(20.0, 12.0, 10), beacons,
                        noise_db=2.0, seed=7)
reverse = simulate_walk(
    list(reversed(rectangular_walk(20.0, 12.0, 10))), beacons,
    noise_db=2.0, seed=11)
tracker = MultiDeviceTracker(beacons)
for i, step in enumerate(forward):
    tracker.feed("tag-1", step["rssi"], float(i))
for i, step in enumerate(reverse):
    tracker.feed("tag-2", step["rssi"], float(i))
report = evaluate_multi_device_track(tracker.tracks(), {
    "tag-1": [step["true"] for step in forward],
    "tag-2": [step["true"] for step in reverse],
})
# report.by_device["tag-1"].mean_error_m — 2.16 m
# report.by_device["tag-2"].mean_error_m — 1.72 m
# report.overall.mean_error_m — 1.94 m over 80 steps
report.to_dict()   # JSON-friendly, for dashboards and regression logs
```

The tracks are the lists `MultiDeviceTracker.tracks()` returns
— one `LivePosition` (or None, for a reading too thin to
locate) per reading, in feed order — so the report scores the
track the tracker actually recorded, including its thin steps:
a device's coverage is its own located share, and the overall
coverage counts every device's thin steps. Each device is
scored with the same rules as `evaluate_live_track` — the two
lists for a device must be the same length, a device's truth
must not mix two- and three-tuples, and a device with nothing
located raises instead of reporting zeroes — and the tracks
and the truths must name the same devices, so a missing tag's
truth is a mistake, not a silent omission. The overall report
pools the errors, so its 90th percentile and worst step are
computed over the pooled error list, not averaged from the
per-device figures — the overall worst step is the worst
device's worst step — and its mean is the located-weighted mean
of the per-device means. The overall floor accuracy is only set
when every device's truth carries floors; a device whose truth
has no floors still gets its own report, with its floor
accuracy None. A single device through the multi-device scorer
is the same report the single-device scorer gives, overall
included.

Measured behaviour (v0.46.0): two simulated walks through one
`MultiDeviceTracker` — tag-1 forward, seed 7, tag-2 the reverse
walk, seed 11 — score tag-1 at mean 2.16 m, median 2.08 m, 90th
percentile 4.14 m, worst step 6.70 m, and tag-2 at mean 1.72 m,
median 1.69 m, 90th percentile 2.94 m, worst step 3.25 m, both
at coverage 1.0; the overall report is 80 steps, 80 located,
mean 1.94 m, median 1.91 m, 90th percentile 3.10 m, worst step
6.70 m, coverage 1.0 — the same per-device means the
multi-device tracking section measured by hand, now in code.
The roadmap grows as new needs surface — the multi-device
accuracy report is the latest addition.

## CLI wiring for the multi-device track accuracy report

The report lived in code only: a ward that wanted the demo's
verdict had to write the scoring call itself. `python -m
beacon.multi_device --demo --evaluate` prints it at the end
of the demo run — after the 80 JSON stream lines, one line
per device plus the overall line, in the same fields as the
single-device demo summary (steps, located, coverage, mean,
median, 90th percentile, worst error, and floor accuracy when
the truth carries floors):

```sh
python3 -m beacon.multi_device --demo --evaluate
python3 -m beacon.multi_device --demo --evaluate-json ward-report.json
```

`--evaluate-json PATH` writes the same report as JSON —
`MultiDeviceTrackReport.to_dict()`, one `TrackReport` per
device plus the overall report — for dashboards and regression
logs, and composes with `--export-json` / `--export-csv`, so
one run produces the combined track file and its verdict
together. `format_multi_device_report(report)` is the same
text for library use, and `run_demo(..., return_truths=True)`
returns the demo's ground truth alongside the tracker, so the
report can be scored without re-running the walks. Both flags
are demo-only: a stream has no ground truth to score against,
so `--evaluate` or `--evaluate-json` without `--demo` is
rejected instead of reporting zeroes. The evaluation scores
the tracks the tracker actually recorded — including any
`--drift-beacon` offset on tag-1's walk, so the report shows
the drifted device's error growing while the other tag stays
put. `--evaluate-smoothed` / `--evaluate-smoothed-json PATH`
score the same walks again after smoothing each device's
track, in the same report shape — see "Smoothed-track
accuracy report" below.

Measured behaviour (v0.47.0): the healthy demo prints tag-1
at mean 2.16 m, median 2.08 m, 90th percentile 4.14 m, worst
step 6.70 m, tag-2 at mean 1.72 m, median 1.69 m, 90th
percentile 2.94 m, worst step 3.25 m, and the overall line at
80 steps, 80 located, mean 1.94 m, median 1.91 m, 90th
percentile 3.10 m, worst step 6.70 m, coverage 1.0 — the same
numbers the library report gives. With beacon A reading 12 dB
too loud on tag-1's walk only (`--drift --drift-beacon A:12`),
tag-1's mean rises to 4.14 m (median 4.50 m, worst step
7.51 m) while tag-2 stays at 1.72 m, and the overall mean
rises to 2.93 m — the per-device lines show which tag went
wrong instead of one blended number hiding it.
The roadmap grows as new needs surface — the CLI wiring for
the multi-device accuracy report is the latest addition.

## Smoothed-track export

The exported track is the live filter's output: causal, so it
lags behind turns and keeps some jitter. For a recorded walk
the future is already known, so the walk can be smoothed after
the fact and exported smoothed alongside the raw track —
`smooth_track_rows` re-runs each device's located positions
through the Kalman filter and the Rauch-Tung-Striebel backward
recursion (the same smoothing `PositioningEngine.smooth_walk`
performs on a walk), revising each past position with the
observations that came after it, and the smoothed rows export
exactly like the raw ones:

```python
from beacon import MultiDeviceTracker, smooth_track_rows
from beacon.export import (export_smoothed_multi_track_json,
                           export_smoothed_track_json)
from beacon.simulate import rectangular_walk, simulate_walk

beacons = [Beacon("A", 0, 0), Beacon("B", 20, 0),
           Beacon("C", 20, 12), Beacon("D", 0, 12)]
forward = simulate_walk(rectangular_walk(20.0, 12.0, 10), beacons,
                        noise_db=2.0, seed=7)
reverse = simulate_walk(
    list(reversed(rectangular_walk(20.0, 12.0, 10))), beacons,
    noise_db=2.0, seed=11)
tracker = MultiDeviceTracker(beacons)
for i, step in enumerate(forward):
    tracker.feed("tag-1", step["rssi"], float(i))
for i, step in enumerate(reverse):
    tracker.feed("tag-2", step["rssi"], float(i))
export_smoothed_multi_track_json(tracker.tracks(),
                                 "ward-a-and-b-smoothed.json")
# or per device, from rows already exported:
# smooth_track_rows(load_track_json("ward-a-track.json"))
```

`smooth_track_rows(rows)` takes a row list as `track_rows`
produces — or as `load_track_json` / `load_track_csv` loads
back — and returns new rows with only the (x, y) revised:
the timestamp, the floor, the confidence ellipse, the beacons
used and excluded, the battery, the forecasts, the drift state
and the coverage are the recorded step's own, carried
unchanged. The ellipse is the live estimate's, not the
smoothed position's, so it still reads as wide as the live
geometry was. `smoothed_track_to_dict` /
`smoothed_multi_track_to_dict` and
`export_smoothed_track_json` / `export_smoothed_track_csv` /
`export_smoothed_multi_track_json` /
`export_smoothed_multi_track_csv` build and write the smoothed
export directly from recorded tracks, and
`MultiDeviceTracker.export_smoothed_track_json` /
`export_smoothed_track_csv` write the combined smoothed file
from the tracker — the same file shape, format and version as
the raw export, so the smoothed file loads back with
`load_track_json` / `load_track_csv` and renders in the
overlay and the replay exactly like the raw one, with a
`smoothed` flag and the `smoothing_method` in the JSON header
so the file says it is smoothed instead of looking like a raw
export. Rows from several devices are smoothed per device, so
one tag's walk never pulls another tag's positions, and a
series that changes floor is split at the change and each run
smoothed on its own — the live filter resets on a floor
change, so the smoother does too — instead of smoothing
across two planes. `method="moving_average"` runs the
moving-average filter over the series instead, the same
opt-out the engine offers; on a track that is already
live-filtered it adds lag and measures worse than the raw
track, so the Rauch-Tung-Striebel method is the default. The
smoother's measurement noise is a parameter too —
`measurement_noise_m`, the standard deviation in metres of
the estimates being smoothed, defaulting to the measured
2 m (see the calibrated noise model below) — and it is
ignored by the moving-average method, which has no noise
model. The
CLIs export the smoothed file alongside the raw one:

```sh
python3 -m beacon.multi_device --demo --export-json ward-a-and-b.json --export-smoothed-json ward-a-and-b-smoothed.json
python3 -m beacon.tracking --demo --export-smoothed-json ward-a-track-smoothed.json
```

Measured behaviour (v0.48.0, re-measured in v0.56.0): on the
two-device demo walk the smoothed export cuts tag-1's mean
error from 2.16 m to 1.29 m and tag-2's from 1.72 m to 0.98 m
— overall 1.94 m to 1.14 m over the 80 located steps — and on
the single-device demo walk it cuts the mean on every seed 1,
2, 3, 7, 11 (seed 1: 1.97 m to 1.15 m; seed 2: 2.57 m to
1.60 m; seed 3: 2.63 m to 1.72 m; seed 7: 2.16 m to 1.29 m;
seed 11: 1.83 m to 1.04 m). The moving-average opt-out on the
same single-device walk measures 2.93 m against the raw
2.16 m — worse than doing nothing, which is why it stays the
opt-out.

## Smoothed track on the ward-map overlay and replay

The smoothed export is a file; a dashboard still has to see
the smoothed path next to the raw one. `render_map_overlay`
and `render_track_replay` both take the smoothed export's
rows — `smoothed_rows`, as `load_track_json` loads a smoothed
export file — and draw each device's smoothed track alongside
its raw one: on the overlay a dashed trail in that device's
own colour, the same colour as its raw trail, with a hollow
dot per smoothed point; in the replay a hollow dot gliding
along the smoothed positions on the same key times as the raw
dot, over a dashed trail in the device's own colour, with the
legend keying each device's smoothed series. The renders never
smooth the raw rows themselves — they draw the export's own
(x, y), so the overlay and the replay cannot disagree with the
export about where the smoothed track went, and a smoothed
file from a different walk (a device the track never had, a
device missing, a different step count, or different
timestamps) is refused instead of drawn over the wrong track.
The smoothed trail in the replay is drawn whole from the
start, so it reads as where the smoothed path ends up while
the raw trail draws itself; the ellipses stay on the raw
frames only, since the ellipse is the live estimate's, not
the smoothed position's. With no smoothed rows both renders
are exactly as before — a single-device track is byte-identical
to the render without smoothing.

```sh
python3 -m beacon.cli --map-overlay ward-a-and-b.json --map-smoothed ward-a-and-b-smoothed.json --map-output ward-a-and-b-map.svg
python3 -m beacon.cli --map-overlay ward-a-and-b.json --map-smoothed ward-a-and-b-smoothed.json --map-replay --map-output ward-a-and-b-replay.svg
```

Measured behaviour (v0.49.0, re-measured in v0.56.0): on the
two-device demo walk the dashed trails on the overlay,
unprojected back to metres, score exactly the smoothed
export's numbers — tag-1 1.29 m against the raw 2.16 m, tag-2
0.98 m against 1.72 m, overall 1.14 m against 1.94 m over the
80 located steps — and on the single-device demo walk the
rendered smoothed trail scores 1.29 m against the raw 2.16 m
on seed 7. The improvement the map shows is the improvement
the export measured, not a second opinion.
The roadmap grows as new needs surface — the smoothed-track
rendering on the overlay and the replay is the latest addition.

## Smoothed-track accuracy report

The overlay and the replay draw the smoothed track, and the
export measures it — but until now nothing scored it the way
the raw track is scored, so the improvement the map showed
had no report of its own. `evaluate_smoothed_multi_device_track`
scores every device's recorded track against its own ground
truth after smoothing, in the same report shape as
`evaluate_multi_device_track`: one `TrackReport` per device
plus the overall report over every device's located steps
pooled together — mean, median, 90th percentile and worst
error, plus coverage, and floor accuracy when the truth
carries floors — so the smoothed and raw reports read the
same way and a ward can compare them line for line. The
smoothing is the same `smooth_track_rows` the smoothed export
uses — each device's located steps re-run through the
Rauch-Tung-Striebel smoother (or the moving-average opt-out,
`method="moving_average"`, the same opt-out the engine
offers) and written back onto the track's located steps in
order — so the report scores exactly the positions the
overlay and the replay draw from the smoothed export, and the
scoring is the shared `evaluate_multi_device_track` itself,
so the raw and smoothed reports cannot disagree about what
counts as an error. A step too thin to locate stays
unlocated, so the coverage is the raw track's and the smoothed
report never claims a step the tracker never located. The CLI
scores the demo walks the same way, alongside the raw report:

```sh
python3 -m beacon.multi_device --demo --evaluate --evaluate-smoothed
python3 -m beacon.multi_device --demo --evaluate-smoothed-json smoothed-report.json
```

`--evaluate-smoothed` prints the smoothed report under a
"Smoothed track (Rauch-Tung-Striebel):" heading after the raw
one, and `--evaluate-smoothed-json PATH` writes it as JSON
with a `smoothed` flag and the smoothing method, so the
verdict is available from the command line instead of only in
code. Both need `--demo`, like `--evaluate`: a stream has no
ground truth to score against.

Measured behaviour (v0.50.0, re-measured in v0.56.0): on the
two-device demo walk the smoothed report cuts tag-1's mean
error from 2.16 m to 1.29 m (median 1.01 m, p90 2.73 m, worst
3.37 m) and tag-2's from 1.72 m to 0.98 m (median 0.80 m, p90
1.93 m, worst 2.85 m) — overall 1.94 m to 1.14 m (median
0.90 m, p90 2.40 m, worst 3.37 m) over the 80 located steps —
the same numbers the smoothed export and the rendered trails
measured, so the report, the export and the map now all
agree. The moving-average opt-out on the same walk measures
2.74 m against the raw 1.94 m — worse than doing nothing,
which is why it stays the opt-out.
The roadmap grows as new needs surface — the smoothed-track
accuracy report is the latest addition.

## Scoring a smoothed track file

The smoothed report scores the demo walks, but a ward that
recorded a real walk has a file, not a demo — a smoothed
export from a shift, and a ground truth from wherever the
walk's true positions are known. `--evaluate-smoothed-file`
scores that file from the command line, with no demo
involved: it loads the smoothed export (JSON or CSV, as
`--export-smoothed-json` / `--export-smoothed-csv` write it),
aligns its rows to the ground truth, and prints the same
per-device + overall report as `--evaluate-smoothed` — one
line per device plus the overall line over every device's
located steps pooled together — and
`--evaluate-smoothed-file-json PATH` writes the same report
as JSON, with a `smoothed` flag and the smoothing method,
so the verdict is available from the command line instead of
only in code.

The ground truth is a JSON file mapping each device id to
its true positions, one point per step of the walk, in step
order, each point carrying its timestamp on the same clock
as the track file's rows:

```json
{"tag-1": [{"t": 0.0, "x": 1.0, "y": 2.0},
           {"t": 1.0, "x": 1.5, "y": 2.5, "floor": 0}],
 "tag-2": [...]}
```

Rows are matched to truth steps by timestamp, not by
position in the file, so the alignment survives a step too
thin to locate: a truth step with no row stays unlocated and
counts against the coverage, exactly as the live track would.
The scoring itself is the shared `evaluate_multi_device_track`
over the file's own positions — the file is already smoothed,
so it is scored as it stands, not smoothed a second time —
which is what makes the file report and the demo's smoothed
report the same report: the same scorer, the same fields,
the same numbers for the same walk. As a library,
`load_ground_truth` reads the truth file,
`tracks_from_export_rows` rebuilds the per-device tracks
aligned to it, and `load_smoothed_track_file` loads the
export with its smoothing method.

```sh
python3 -m beacon.multi_device --demo --export-smoothed-json ward-a-smoothed.json
python3 -m beacon.multi_device --evaluate-smoothed-file ward-a-smoothed.json --ground-truth ward-a-truth.json
python3 -m beacon.multi_device --evaluate-smoothed-file ward-a-smoothed.json --ground-truth ward-a-truth.json --evaluate-smoothed-file-json ward-a-smoothed-report.json
```

A JSON file whose header does not say `smoothed` is refused
rather than scored — a raw export scored as a smoothed one
would report the live filter's numbers under the smoother's
name — and a file and a truth that disagree about the walk
are refused too: a row whose timestamp has no truth step, a
device the truth names but the track never had, or rows with
no device when the truth names more than one. A single-device
export with no device on its rows scores against the truth's
single device. A CSV export carries no smoothing marker in
its columns, so it loads unchecked and the report records
the default RTS method — the file is assumed to be the
smoothed export the flag asks for. The rebuilt positions
carry the row's timestamp, (x, y), floor, ellipse, beacons
used and excluded, and battery; the forecast flags and drift
state are not rebuilt, since the report scores positions,
not the monitor.

Measured behaviour (v0.51.0, re-measured in v0.56.0): scoring
the demo's own smoothed export file reproduces the demo's
smoothed report exactly — tag-1 1.29 m, tag-2 0.98 m, overall
1.14 m over the 80 located steps, in JSON and in CSV — and a
track starved of one step scores 39 of 40 located on that
device, coverage 0.98, instead of silently dropping the step
from the denominator.
The roadmap grows as new needs surface — scoring a smoothed
track file is the latest addition.

## Smoothed-track accuracy in the demo summary

The multi-device demo prints its smoothed report, but the
single-device demo summary scored only the raw walk — the
smoothed track the overlay and the replay draw had no number
next to it. The demo now prints both: the raw summary as
before, then the smoothed summary under a
"Smoothed track (Rauch-Tung-Striebel):" heading, in the same
fields as the multi-device smoothed report — steps, located
steps, coverage, mean, median, 90th percentile and worst
error — so the single-device and multi-device views read the
same way. `evaluate_smoothed_live_track` scores one track
the same way `evaluate_smoothed_multi_device_track` scores
several: each device's located steps re-run through the same
`smooth_track_rows` the smoothed export uses, then scored by
the shared `evaluate_live_track`, so the raw and smoothed
summaries cannot disagree about what counts as an error, and
the demo's smoothed line is printed with the shared
`format_report_line`, so it cannot disagree with the library
report either. A step too thin to locate stays unlocated, so
the coverage is the raw walk's.

```sh
python3 -m beacon.cli --steps 10 --noise 2.0 --seed 7
```

Measured behaviour (v0.52.0, re-measured in v0.56.0): on the
default demo walk the smoothed summary cuts the mean error
from 2.16 m to 1.29 m (median 1.01 m, p90 2.73 m, worst
3.37 m) over the 40 located steps — the same numbers tag-1
scores in the multi-device smoothed report, since it is the
same forward walk, seed 7 — and smoothing helps on every
seed (1, 2, 3, 7, 11: raw 1.83–2.63 m, smoothed 1.04–1.72 m).
The moving-average opt-out smoothed over a Kalman track
measures 2.93 m against the raw 2.16 m — worse than doing
nothing, which is why it stays the opt-out — and smoothing a
moving-average walk helps a little (2.73 m to 2.54 m) but
stays worse than the Kalman default.
The roadmap grows as new needs surface — the smoothed-track
accuracy in the demo summary is the latest addition.

## Smoothed-track uncertainty ellipses on the overlay and replay

The smoothed track on the map and in the replay showed where
the smoothing put the device, but not how much the smoothing
trusted each revised position: the ellipses drawn were the
live estimate's, on the raw frames only. Each smoothed export
row now carries its own 95% ellipse — the
Rauch-Tung-Striebel smoother's posterior covariance at that
step (see `beacon.uncertainty.ellipse_from_covariance`), so
the ellipse is the smoother's own, tightened by the
observations that came after, instead of the live geometry's
reused at the new point — and the overlay draws one dashed
ellipse per smoothed point, in that device's own colour, while
the replay animates one per frame with the smoothed dot, with
a legend entry for it. The moving-average opt-out has no
covariance, so its smoothed rows carry no smoothed ellipse
and draw none — and a step whose smoothed covariance was
degenerate draws none either, rather than a made-up one. The
ellipses are the export's own data, like the smoothed
positions: the overlay and the replay draw what the smoothed
file says, and a smoothed file from a different walk is
refused instead of drawn.

```sh
python3 -m beacon.multi_device --demo --export-smoothed-json ward-a-smoothed.json
python3 -m beacon.cli --map-overlay ward-a-track.json --map-smoothed ward-a-track-smoothed.json --map-output ward-a-track.svg
python3 -m beacon.cli --map-overlay ward-a-track.json --map-smoothed ward-a-track-smoothed.json --map-replay --map-output ward-a-track-replay.svg
```

Measured behaviour (v0.53.0, re-measured in v0.56.0): on the
default demo walk the smoothed ellipse averages 1.82 m
semi-major against the raw ellipse's 6.32 m — the smoother
has seen the observations that came after, so it trusts each
revised position far more than the live geometry did — and
the overlay draws all 40 and the replay animates all 40,
from the JSON export and from the CSV export alike. The
smoothed ellipse is still near-circular on every seed
(semi-minor 1.82 m too) — the per-step noise is isotropic at
each step, so it still carries almost no orientation, unlike
the raw geometry ellipse, which stretches along the weak axis —
but its 95% claim is now an honest one: with the per-step
geometry noise (v0.56.0) 29 of the 40 true positions fall
inside the smoothed ellipse on the demo walk (25–36 across
seeds 1, 2, 3, 7, 11), against 33 of 40 inside the raw
ellipse. Before the calibration it was 20 of 40 on seed 7
(6–20 across seeds) at 1.14 m semi-major: the filter assumed
1 m measurement noise while the live estimates err by about
2 m, so the ellipse read tighter than the track's actual
accuracy. It still does not cover a full 95% on seeds 2 and
3 (26 and 25 of 40) — the per-step noise varies in size with
the geometry but stays isotropic at each step, so the ellipse
still cannot stretch along the weak axis, and the interval
stays an approximation, just a per-step one now.
The roadmap grows as new needs surface — the smoothed-track
uncertainty ellipse on the overlay and replay is the latest
addition.

## Calibrated smoother noise model

A 95% ellipse is a promise, and the smoother was breaking
it: its filter assumed 1 m measurement noise while the live
estimates it smooths err by about 2 m, so the smoothed
ellipse covered far fewer than 95% of the true positions —
20 of 40 on the demo walk, 6–20 across seeds. The fix is the
noise model, not the positions: the smoothing path now takes
a `measurement_noise_m` parameter — the standard deviation,
in metres, of the position estimates being smoothed — and
defaults it to the measured value
`MEASURED_MEASUREMENT_NOISE_M` (2.0 m, from
`beacon.filter`), squared into the Kalman filter's
measurement-noise variance. The parameter is plumbed through
the whole smoothing path — `smooth_track_rows`,
`smoothed_track_to_dict` / `smoothed_multi_track_to_dict`,
the four `export_smoothed_*` writers,
`MultiDeviceTracker.export_smoothed_track_json` /
`export_smoothed_track_csv`, and
`evaluate_smoothed_live_track` /
`evaluate_smoothed_multi_device_track` — so the overlay, the
replay, the export and the reports all draw and score the
same calibrated track, and passing 1.0 reproduces the old
numbers exactly. A non-positive value raises instead of
smoothing on a made-up noise, and the moving-average
opt-out ignores the parameter, since it has no noise model.
The live filter keeps its 1 m default: its positions carry no
covariance-based ellipse of their own, so the overconfidence
never showed there, and `PositioningEngine.smooth_walk` still
runs through the live filter and returns positions only.

```python
from beacon import smooth_track_rows
from beacon.export import load_track_json

smoothed = smooth_track_rows(load_track_json("ward-a-track.json"))
# or, on a space whose estimates err differently:
# smooth_track_rows(rows, measurement_noise_m=1.5)
```

Measured behaviour (v0.54.0): on the default demo walk the
smoothed ellipse widens from 1.14 m to 1.90 m mean semi-major
and now covers 30 of 40 true positions on seed 7 (21–37
across seeds 1, 2, 3, 7, 11), against 33 of 40 inside the raw
ellipse. The positions move too — a filter that trusts its
measurements less leans harder on its velocity model — and
the move is a small improvement, not a retune: the smoothed
mean error falls from 1.45 m to 1.34 m on seed 7 (1.26 m,
1.64 m, 1.92 m, 1.34 m, 1.11 m across seeds 1, 2, 3, 7, 11),
and the two-device report from 1.25 m to 1.13 m overall. The
honest limit stays honest: seeds 2 and 3 still cover only 26
and 21 of 40, because one isotropic number cannot capture
geometry-dependent error — the ellipse is calibrated to the
average error, not to each step's — so the interval is an
honest approximation, not a guarantee.
The roadmap grows as new needs surface — the calibrated
smoother noise model is the latest addition.

## The smoother noise on the command line

The noise value is a library parameter, but a ward dashboard
that only runs the CLIs had no way to set it — the smoothing
path always used the calibrated 2 m. `--smoothing-noise-m`
exposes it on every CLI that smooths: the single-device demo
summary (`python -m beacon.cli`), the live tracker's smoothed
exports (`python -m beacon.tracking --demo
--export-smoothed-json`), and the multi-device smoothed
exports and smoothed demo reports (`python -m beacon.multi_device
--demo --evaluate-smoothed`, `--export-smoothed-json` /
`--export-smoothed-csv`) — the same parameter the library
already takes, so the overlay, the replay, the export and the
reports all honour one value: smooth a track with the flag and
the file the overlay and the replay draw is the track the
report scores. It defaults to the calibrated
`MEASURED_MEASUREMENT_NOISE_M` (2.0 m), and a non-positive
value is rejected by the parser instead of smoothing on a
made-up noise. A track file scored with
`--evaluate-smoothed-file` is already smoothed, so the flag
does not re-smooth it — re-export the track with the flag
instead.

```sh
python3 -m beacon.cli --smoothing-noise-m 1.0
python3 -m beacon.tracking --demo --export-smoothed-json ward-a-track-smoothed.json --smoothing-noise-m 1.5
python3 -m beacon.multi_device --demo --evaluate-smoothed --smoothing-noise-m 1.0
```

Measured behaviour (v0.55.0): `--smoothing-noise-m 1.0`
reproduces the pre-0.54.0 numbers exactly — the single-device
smoothed mean error goes back from 1.34 m to 1.45 m on the
demo walk, the two-device overall mean from 1.13 m to 1.25 m,
and the smoothed ellipse from 1.90 m to 1.14 m mean
semi-major. The flag does not fix the honest limit from
v0.54.0: one isotropic number still cannot capture
geometry-dependent error, so seeds 2 and 3 still cover only 26
and 21 of 40 inside the smoothed ellipse — the flag lets a
space whose estimates err differently calibrate to its own
average error, it does not make the interval per-step.
The roadmap grows as new needs surface — the smoother noise
on the command line is the latest addition.

## Per-step geometry-aware smoother noise

One isotropic number cannot capture geometry-dependent error:
the smoother's noise was calibrated to the measured 2 m, but
seeds 2 and 3 still covered only 26 and 21 of 40 inside the
smoothed ellipse, because a step located by beacons clustered
on one side errs more than a step pinned down from all sides.
The smoothing path now defaults to a per-step noise: each step
is fed with the RMS sigma of its own live confidence ellipse
(`beacon.uncertainty.ellipse_rms_sigma` — the root-mean-square
of the standard deviations along the ellipse's axes, so a
stretched ellipse means a larger sigma), and the Kalman
filter's `update` takes that step's variance as an override
for that update only. A step with no live ellipse (degenerate
geometry) falls back to the measured 2 m for that step, so a
track that carries no ellipses at all smooths exactly like
the scalar 2 m. Passing an explicit `measurement_noise_m`
selects one scalar noise for every step instead — 1.0
reproduces the pre-0.54.0 numbers exactly, 2.0 the
v0.54.0/v0.55.0 calibrated ones — and the moving-average
opt-out ignores the parameter, since it has no noise model.
The single-device demo summary now attaches each step's
raw-geometry ellipse to its positions, so the demo itself
smooths on the per-step noise instead of falling back to the
scalar; the live tracker already carried the ellipse, so its
exports needed no change. The flag follows the same rule:
omit `--smoothing-noise-m` for the per-step noise, pass a
value for one scalar noise.

```sh
python3 -m beacon.cli
python3 -m beacon.cli --smoothing-noise-m 1.0
python3 -m beacon.multi_device --demo --evaluate-smoothed
```

Measured behaviour (v0.56.0): on the default demo walk the
smoothed mean error falls from 1.34 m to 1.29 m (median
1.01 m, p90 2.73 m, worst 3.37 m) and the smoothed ellipse
from 1.90 m to 1.82 m mean semi-major, covering 29 of 40 true
positions; the per-step sigmas run from 1.2 m on
geometrically strong steps to 3.7 m on weak ones, averaging
about 2.0 m — the same average the scalar was calibrated to,
so the change varies the noise per step instead of changing
it. The mean error improves on every seed (1, 2, 3, 7, 11:
1.26 m to 1.15 m, 1.64 m to 1.60 m, 1.92 m to 1.72 m,
1.34 m to 1.29 m, 1.11 m to 1.04 m) and total coverage rises
from 147 to 152 of 200 — seed 3's 21 of 40 becomes 25 —
while seed 2 stays at 26 of 40. The honest limit is the
reverse walk: the two-device overall mean is flat (1.13 m to
1.14 m), because tag-2's reverse walk gets slightly worse,
0.92 m to 0.98 m, while tag-1 improves 1.34 m to 1.29 m —
the per-step noise is a better default on the forward walks,
not a universal win. The full anisotropic version was tried
and abandoned: feeding each step its whole ellipse covariance
made most smoothed covariances degenerate (the backward
recursion's subtraction drives the minor eigenvalue
non-positive when the filter is overconfident along one
axis), so the per-step noise stays isotropic and the smoothed
ellipse stays near-circular — its size now follows the step's
geometry, its shape still does not.
The roadmap grows as new needs surface — the per-step
geometry-aware smoother noise is the latest addition.

## A numerically stable anisotropic smoother

The full anisotropic version failed on numerics, not on the
idea: the forward pass updated its covariance with the short
form `(I - K H) P`, which is not symmetric by construction,
so rounding drifted it off symmetry and off
positive-definiteness, and the smoother's backward subtraction
finished the job. This version fixes the numerics instead of
abandoning the idea. The filter's covariance update is now
the Joseph form, `(I - K H) P (I - K H)^T + K R K^T`, which
keeps the covariance symmetric and positive semi-definite by
construction; every stored covariance is symmetrised after
each step; and the smoother's backward pass symmetrises each
smoothed covariance and floors its position block's
eigenvalues at a small positive value (1 mm standard
deviation), so a degenerate smoothed covariance can never
leave a step without an ellipse. `KalmanFilter.update` takes
the full per-step covariance as `measurement_covariance` — a
symmetric positive-definite 2x2 matrix, so a step can be
trusted more along its strong axis than its weak one — and
`ellipse_covariance(ellipse)` builds that matrix from a live
confidence ellipse, the inverse of `ellipse_from_covariance`.

```python
from beacon import KalmanFilter, ellipse_covariance

kalman = KalmanFilter(measurement_noise=4.0)
for position, ellipse in zip(positions, ellipses):
    kalman.update(position,
                  measurement_covariance=ellipse_covariance(ellipse))
smoothed = kalman.smooth_with_covariance()
```

Measured behaviour (v0.57.0): the numerics are fixed — zero
degenerate smoothed covariances on all five demo seeds, where
the v0.56.0 attempt degenerated 33–37 of 40 per walk — and
the smoothed ellipses are genuinely anisotropic now, with a
mean axis ratio of 1.51 on the demo walk against 1.00 for the
near-circular isotropic ones (mean semi-major 1.75 m, mean
semi-minor 1.17 m). The eigenvalue floor never fires on the
demo walks: the Joseph form and the symmetrisation alone keep
every covariance healthy, so the floor is a backstop for
other walks, not a correction to these numbers. But the
anisotropic weighting still measures worse than the per-step
isotropic noise it was meant to replace, on both figures that
matter:

| seed | anisotropic mean | isotropic mean | anisotropic inside | isotropic inside |
|---|---|---|---|---|
| 1 | 1.37 m | 1.15 m | 21/40 | 36/40 |
| 2 | 1.59 m | 1.60 m | 19/40 | 26/40 |
| 3 | 2.05 m | 1.72 m | 15/40 | 25/40 |
| 7 | 1.85 m | 1.29 m | 19/40 | 29/40 |
| 11 | 1.14 m | 1.04 m | 29/40 | 36/40 |
| total | 1.60 m | 1.36 m | 103/200 | 152/200 |

Blending the anisotropic covariance halfway toward its own
isotropic RMS gives 1.38 m and 143 of 200; a quarter blend
gives 1.36 m and 150 of 200 — closer, still not beating the
isotropic default. The reason is what the smoother is fed:
live-filtered positions, whose error is dominated by the
filter's own dynamics and lag, not by the instantaneous
trilateration geometry — so the raw ellipse's shape misweights
the steps, trusting the "strong" axis too much in a direction
the filtered error does not follow, and the stretched
ellipses end up overconfident in the wrong direction. The
default therefore stays the per-step isotropic noise, and the
smoothed ellipses the overlay and the replay draw stay
near-circular. The anisotropic path is available at the
filter level, numerically stable, for a space where the
geometry does follow the filtered error — and the numbers
above are pinned in the tests, so a future change cannot
silently claim the anisotropic default wins.
The roadmap grows as new needs surface — the numerically
stable anisotropic smoother is the latest addition.

## Drift classification on multi-floor walks

The drift monitor used to reject multi-floor positions: a beacon
heard through a floor slab reads ~15 dB weaker than the path-loss
model predicts, so the slab looked like drift, and the live
tracker worked around it by feeding the monitor only the current
floor's beacons. The monitor is now floor-aware instead.
`DriftMonitor.observe` takes `(x, y, floor)` — a two-tuple still
means floor 0 — and the predicted RSSI is the 3D distance (3 m
of vertical separation per floor, `floor_height_m`) minus the
slab attenuation for a beacon on a different floor
(`floor_attenuation_db`, 15 dB), the same model the simulator
uses, so a healthy beacon heard through a slab reads as healthy.
Revisit pairs only match readings on the same floor, so a reading
upstairs never counts as revisiting the same spot downstairs. The
tracker passes the live position with its floor now, so a
floor-1 beacon is no longer read as 15 dB of drift the moment the
device goes upstairs.

```python
from beacon import DriftMonitor
from beacon.simulate import simulate_multifloor_walk, two_floor_walk

monitor = DriftMonitor(beacons)   # beacons on floors 0 and 1
for step in simulate_multifloor_walk(
        two_floor_walk(20, 12, 10), beacons, noise_db=2.0, seed=7):
    monitor.feed(step)   # step["true"] is (x, y, floor)
monitor.classifications()   # one verdict per beacon
```

```sh
python3 -m beacon.cli --drift-classify --multifloor --seed 7
python3 -m beacon.cli --drift-classify --multifloor --seed 7 --battery-degrade D
python3 -m beacon.cli --drift-classify --multifloor --seed 7 --move-beacon D:5,-5
```

`--multifloor` walks the two-floor ward twice — each floor's loop
twice, so revisits exist per floor — and classifies all eight
beacons. The window matters as much as the floor model: the
classifier's evidence is the rolling window, and the default
20-reading window only ever sees one floor's half-loop, where the
slab compresses the moved signature below its threshold — a moved
beacon reads `uncertain` on 4 of 5 seeds. The demo's window spans
a full two-floor cycle instead (80 readings at 10 steps per side),
and then the verdicts match the single-floor table. The verdict
follows the window, not the beacon: classified halfway through
the walk, at the end of the floor-0 pass with the window on D's
own floor, the moved beacon reads `moved` on 4 of 5 seeds; by the
end of the walk the window has moved upstairs and the same beacon
reads `uncertain` on 4 of 5.

Measured behaviour (v0.58.0): on the doubled two-floor walk,
2 dB RSSI noise, beacon D, seeds 1, 2, 3, 7, 11 — the other
seven beacons stay `healthy` in every scenario:

| scenario | verdicts |
|---|---|
| healthy | `healthy` 5/5 (mean -0.11 to +0.30 dB, revisit -0.03 to +0.48 dB, 80 pairs) |
| D draining, 0.3/hr | `battery-sag` 5/5 (mean -1.65 to -2.46 dB, revisit delta -1.14 to -2.02 dB); no alarm, the mean never reaches 5 dB |
| D moved (0, 12) to (5, 7) | `moved` 4/5 (mean +3.71 to +4.13 dB, spread 2.55 to 2.89 dB); seed 2 `uncertain` (spread 2.32 dB, under the 2.5 dB threshold) |
| D with 5 dB extra noise | `noisy` 5/5 (spread 4.76 to 5.92 dB) |
| D steady -6 dB offset | `uncertain` 5/5 (mean -5.70 to -6.11 dB), alarming on D throughout |

The misses are the same honest limits as the single-floor table,
not new bugs: a move's spread is compressed by the slab on the
other floor, so seed 2 falls just under the moved threshold, and
the slow sag's mean never reaches the 5 dB alarm threshold, so
the alarm misses it and only the classifier catches it. The
steady offset is actually cleaner here than on the single-floor
walk — `uncertain` on 5/5, where the single-floor walk misreads
one seed as a sag — because the longer window averages the noise
fluke out. What the longer window did not fix was a verdict taken
mid-walk on the wrong floor: the per-floor window below is that
step — the classifier always sees each beacon on its own floor
now, not a sharper threshold.
The roadmap grows as new needs surface — drift classification
on multi-floor walks is the latest addition.

## Per-floor classification window

The drift monitor's window used to be one window per beacon,
mixing readings from every floor the device visited. The
classifier's evidence is the window, so on a multi-floor walk the
verdict followed whichever floor the window happened to cover:
classified halfway through the walk, at the end of the floor-0
pass with the window on D's own floor, the moved beacon read
`moved` on 4 of 5 seeds; by the end of the walk the window had
moved upstairs and the same beacon read `uncertain` on 4 of 5.
The window is per floor now. Each beacon keeps one rolling
window per floor it was heard on, and `classify` — and the alarm
— read the window for the beacon's own floor by default, the
floor the beacon is mounted on, so the verdict always rests on
the beacon's own geometry. `classify(id, floor=...)`,
`status(id, floor=...)` and `alarm(id, floor=...)` read another
floor's window instead, for inspection, and the classification
carries the floor it was taken on (`DriftClassification.floor`,
in the exported rows too — files written before per-floor windows
load with the floor defaulting to 0). A beacon never heard on its
own floor is `uncertain` — no readings to classify — not
healthy, and the reason names the floor.

```python
monitor.classify("D")          # D's own floor (floor 0)
monitor.classify("D", floor=1) # the same beacon, seen through the slab
```

```sh
python3 -m beacon.cli --drift-classify --multifloor --seed 7
python3 -m beacon.cli --drift-classify --multifloor --seed 7 --move-beacon D:5,-5
```

Measured behaviour (v0.59.0): on the doubled two-floor walk,
2 dB RSSI noise, beacon D, seeds 1, 2, 3, 7, 11 — the other
seven beacons stay `healthy` in every scenario:

| scenario | verdicts |
|---|---|
| healthy | `healthy` 5/5 (mean -0.01 to +0.12 dB, revisit -0.14 to +0.79 dB, 41 pairs) |
| D draining, 0.3/hr | `healthy` 5/5 — the slow sag is missed on this walk (see below) |
| D moved (0, 12) to (5, 7) | `moved` 3/5 (mean +4.10 to +4.24 dB, spread 2.47 to 3.07 dB); seeds 1 and 2 `uncertain` (spread 2.47 and 2.50 dB, just under the 2.5 dB threshold) |
| D with 5 dB extra noise | `noisy` 5/5 (spread 4.48 to 6.11 dB) |
| D steady -6 dB offset | `uncertain` 5/5 (mean -5.88 to -6.01 dB), alarming on D throughout |

The verdict no longer follows the window: classified at the end
of the floor-0 pass and at the end of the walk, the moved beacon
reads the same on every seed — `moved` on 1, 2, 7, 11 and
`uncertain` on 3 at the default 20-reading window, `uncertain`
on 1 and 2 and `moved` on 3, 7, 11 at the demo's 80-reading
window — because the own-floor window does not change while the
device is upstairs. And the default window now catches the move
it used to miss: the default 20 readings are the last 20 on the
beacon's own floor — the old window held the last 20 readings of
the walk, which were all taken upstairs, on the wrong floor for
D — so the moved beacon reads `moved` on 4 of 5 seeds (seed 3
`uncertain`: its revisit delta is +1.20 dB, so the moved call,
which needs a stable revisit, does not fire).

The slow sag is the honest cost. The old mixed window borrowed
the more-drained cross-floor tail — readings taken later in the
walk, at a lower battery — which pulled the window's mean past
the -1.5 dB sag gate. The own-floor window averages the fresh
first loop with the drained second loop, so the mean stays at
-0.60 to -0.79 dB and the gate blocks the verdict, even though
the revisit delta still measures the decline (-1.50 dB on seeds
1 and 11). The walk ends upstairs, so D's own-floor window never
sees the most-drained readings at all — the last quarter of the
walk is on the wrong floor for D. The fix is a longer history
(more loops on the beacon's own floor), not a sharper threshold:
as the window fills with drained readings, its mean crosses the
gate on its own.
The roadmap grows as new needs surface — the per-floor
classification window is the latest addition.

## Long-history drift classification

The per-floor window fixed the verdict following the window, and
left the slow sag behind: the own-floor window averages a fresh
loop with a drained one, so its mean never reaches the -1.5 dB
sag gate. The fix is a longer history, not a sharper threshold.
Each beacon now keeps a second window per floor — the long
window, `long_window` readings (the full history by default,
configurable per monitor, at least the short window and at most
the history) — and the sag check consults it. The long window is
consulted for the sag only, never for a move: a move's signature
is positional, not temporal, and the short window already carries
it. And it uses the same revisit thresholds as the short-window
test — a revisit delta below -1.0 dB, at least 1.5 standard errors
from zero, over at least 5 pairs — plus one guard: the beacon must
read weaker overall across the long history, its mean at least
1.5 standard errors below zero. A noise fluke that fakes a
revisit decline leaves the mean near zero, so it cannot borrow
this route: the single-floor healthy walk's seed 11 flukes a
-1.00 dB long revisit delta but its long mean is only -0.15 dB,
so it stays healthy. The noisy check runs first, so a noisy
beacon whose revisit delta flukes negative stays noisy, and a
moved beacon's revisit delta is near zero, so it cannot borrow
the route either. The alarm, the moved and noisy verdicts, and
the fast-sag fallback all still read the short window exactly as
before — fast detection does not wait for the long history.

```python
monitor = DriftMonitor(beacons, window=20, long_window=240)
```

```sh
python3 -m beacon.cli --drift-classify --multifloor --seed 7
python3 -m beacon.cli --drift-classify --multifloor --seed 1 --battery-degrade D --long-window 120
```

Measured behaviour (v0.60.0): on the doubled two-floor walk,
2 dB RSSI noise, beacon D, seeds 1, 2, 3, 7, 11 — the other
seven beacons stay `healthy` in every scenario, and the
single-floor table is unchanged (the slow sag still reads
`battery-sag` 5/5 there, healthy 5/5, the steady offset still
misread as a sag on seed 11 through the short window):

| scenario | verdicts |
|---|---|
| healthy | `healthy` 5/5 (mean -0.01 to +0.12 dB, revisit -0.14 to +0.79 dB, 41 pairs) |
| D draining, 0.3/hr | `battery-sag` 2/5 (seeds 1 and 11; mean -0.67 to -0.79 dB, revisit delta -1.50 dB); seeds 2, 3 and 7 `healthy` (revisit -0.87, -0.19 and -0.48 dB) — no alarm, the mean never reaches 5 dB |
| D moved (0, 12) to (5, 7) | `moved` 3/5 (mean +4.10 to +4.24 dB, spread 2.47 to 3.07 dB); seeds 1 and 2 `uncertain` (spread 2.47 and 2.50 dB, just under the 2.5 dB threshold) |
| D with 5 dB extra noise | `noisy` 5/5 (spread 4.48 to 6.11 dB) |
| D steady -6 dB offset | `uncertain` 5/5 (mean -5.88 to -6.01 dB), alarming on D throughout |

The three misses are honest limits, not a threshold to sharpen.
The inter-loop decline is only about 1.5 dB while each reading
carries 2 dB of noise plus up to 2.4 dB of extra battery noise by
the end of the walk, so on seeds 2, 3 and 7 the revisit pairs
genuinely do not show a decline — seed 3's is -0.19 dB — and no
statistic can separate that from a healthy beacon without more
history. More history is what fixes it: walk a third loop and
the window fills with drained readings until its mean crosses
the gate on its own, and the slow sag reads `battery-sag` on
5/5 seeds (mean -2.00 to -2.82 dB, revisit delta -1.71 to
-3.11 dB, 72 pairs). The long route's own guard is thin on seed
11 — its long mean of -0.67 dB clears the 1.5-standard-error gate
of -0.61 dB by 0.06 dB — which is the price of catching a slow
decline from two loops of data: the verdict rests on a decline
that is real but barely significant, and the reason says so,
citing the long history it was taken from.
The roadmap grows as new needs surface — the long-history
drift classification is the latest addition.

## A third loop for drift classification

The long-history fix caught the slow sag on two of five seeds
from two loops of data, and the v0.60.0 notes pointed at the
rest: walk a third loop and the window fills with drained
readings until its mean crosses the gate on its own. The
classifier could already do that — the monitor takes any
number of loops — but the demo could not: `--drift-classify`
hardcoded two loops. It takes `--loops N` now, walking the
loop N times instead of twice (the default is still 2, so the
existing demo is unchanged), on the single-floor loop and the
two-floor ward alike. One loop has no revisits, so the
classifier falls back to the within-window trend and cannot
rule a move out; fewer than one loop is rejected.

```sh
python3 -m beacon.cli --drift-classify --multifloor --seed 7 --battery-degrade D --loops 3
python3 -m beacon.cli --drift-classify --seed 7 --battery-degrade D --loops 3
python3 -m beacon.cli --drift-classify --seed 7 --loops 1
```

Measured behaviour (v0.61.0): on the tripled two-floor walk,
2 dB RSSI noise, beacon D, seeds 1, 2, 3, 7, 11 — the other
seven beacons stay `healthy` in every scenario:

| scenario | verdicts |
|---|---|
| healthy | `healthy` 5/5 (8/8 beacons on every seed) |
| D draining, 0.3/hr | `battery-sag` 5/5 (mean -2.00 to -2.82 dB, revisit delta -1.71 to -3.11 dB, 72 pairs) |
| D moved (0, 12) to (5, 7) | `moved` 4/5 (mean +3.98 to +4.48 dB, spread 2.59 to 3.07 dB); seed 2 `uncertain` (spread 2.48 dB, just under the 2.5 dB threshold) |

The third loop does what the notes said it would: the slow
sag reads `battery-sag` on 5/5 seeds, where two loops caught
only 2/5 — the extra loop's drained readings pull the
long-window mean to -2.00 to -2.82 dB, past the -1.5 dB gate,
and the revisit delta measures the decline at -1.71 to
-3.11 dB over 72 pairs. The moved beacon reads `moved` on
one more seed than the two-loop walk (4/5 instead of 3/5 at
the demo's 80-reading window), and seed 2 stays `uncertain`
for the same honest reason as before — its spread is 2.48 dB,
0.02 dB under the moved threshold. On the single-floor walk
a third loop keeps the sag at `battery-sag` 5/5 (mean -3.75
to -5.12 dB, revisit delta -2.72 to -5.16 dB, 20 pairs — the
20-reading window limits the pairs) and the healthy walk at
`healthy` 4/4.
The roadmap grows as new needs surface — the third loop for
drift classification is the latest addition.

## A longer history for the moved-beacon call

The third loop left one near-miss on the table: the moved
beacon on the tripled two-floor walk reads `moved` on 4/5
seeds, and seed 2 stays `uncertain` because its spread is
2.48 dB — 0.02 dB under the 2.5 dB moved threshold. The
threshold is not the problem: the spread in an 80-reading
window is an estimate, and it wobbles around the true value
by about that much from seed to seed. The fix is the same one
that caught the slow sag — consult the longer history the
monitor already keeps. `classify` now reads the moved gates
off the long window too, when it holds more readings than the
short one: the same spread above 2.5 dB, the same mean beyond
1.5 dB, the same stable revisit (delta within 1.0 dB over at
least 5 pairs) — a longer history, not a sharper threshold.
The noisy check still runs first, so a noisy beacon whose
long spread is wide stays `noisy`, and a sagging beacon's
revisit delta is strongly negative, so it cannot borrow the
moved route. The verdict's reason says which history it rests
on, and the reported mean, spread, delta and pair count stay
the short window's — the same convention as the long-history
sag verdict.

```sh
python3 -m beacon.cli --drift-classify --multifloor --seed 2 --move-beacon D:5,-5 --loops 3
python3 -m beacon.cli --drift-classify --seed 11 --move-beacon D:5,-5
```

Measured behaviour (v0.62.0): 2 dB RSSI noise, beacon D,
seeds 1, 2, 3, 7, 11 — the other beacons stay `healthy` in
every scenario:

| walk | verdicts |
|---|---|
| single-floor, D moved (0, 12) to (5, 7) | `moved` 5/5 (was 4/5 — seed 11 now `moved` from the longer history: short-window delta -1.84 dB over 20 pairs, long-window delta -1.00 dB over 41 pairs, inside the same 1.0 dB gate by 0.004 dB) |
| two-floor, D moved, `--loops 3` | `moved` 5/5 (was 4/5 — seed 2 now `moved` from the longer history: short-window spread 2.48 dB, long-window spread 2.57 dB over 120 readings, revisit delta +0.38 dB over 81 pairs) |
| two-floor, D moved, two loops | `moved` 3/5, unchanged — seeds 1 and 2 stay `uncertain` (spreads 2.47 and 2.50 dB), because the short window already holds the whole per-floor history there, so no longer history exists to consult; the fix is a third loop, not a sharper threshold |
| two-floor, D moved, default 20-reading window | `moved` 5/5 (was 4/5 — seed 3 now `moved` from the longer history: short-window delta +1.20 dB over 20 pairs, long-window delta +0.80 dB over 41 pairs) |
| two-floor, D draining 0.3/hr, `--loops 3` | `battery-sag` 5/5, unchanged |
| two-floor, healthy, `--loops 3` | `healthy` 5/5, unchanged |

The seed-11 single-floor call is honest about its margin:
the long-window delta clears the gate by 0.004 dB, so the
verdict is right but barely so — a different noise draw could
put it back to `uncertain`, and the reason cites the longer
history it was taken from, so the thinness is visible. The
two-loop two-floor miss is the remaining honest limit: with
only two loops the long window and the short window hold the
same 80 readings per floor, so there is no more history to
consult, and the verdict stays `uncertain` rather than
inventing a cause.
The roadmap grows as new needs surface — the longer history
for the moved-beacon call is the latest addition.
## A level-drop check for the sag call

The longer history fixed the moved beacon's seed-11 verdict, but
the steady offset's seed-11 misread was still untouched: a
steady -6 dB offset read as `battery-sag` on one seed of five,
because the noise itself drifts down -1.84 dB over that walk and
the revisit test reads that as a real decline. A longer history
does not fix this one — over 41 pairs the long revisit delta is
-1.00 dB, still past the same 1.0 dB gate by 0.005 dB — so the
fix is a second piece of evidence, not a sharper threshold on
the first. A sag is a process: the beacon's residual level falls
steadily from the start of the walk to the end, as the battery
drains. A steady offset is a level: weak from the first reading,
flat throughout. `classify` now requires, on both revisit-based
sag paths (the short window and the long history), that the
level actually fell across the walk — the mean of the last
quarter of the floor's long history at least 1.0 dB below the
mean of the first quarter, at least 1.5 standard errors from
zero, the same magnitude-plus-significance convention as the
revisit-delta gate. The classification carries the drop as
`level_drop_db`, and the sag reason cites it, so the evidence is
visible on every verdict; the export round-trips the new field,
and a file written before v0.63.0 loads with it null.

Two candidate discriminators were measured and rejected first.
A gap-dependence test — the decline should grow with the time
between the two visits of a revisit pair — has nothing to work
with on these walks: every pair spans exactly one loop (gap
~40 readings), so there is no gap variation to regress on. And
a within-half slope test — the decline should be sustained in
both halves of the walk — fails on the real sag itself: the
battery loses RSSI with the square of the charge used, so the
decline is concentrated in the second half, and the first half's
slope is near zero (+0.04 to -0.04 dB per reading) on every
seed. The quarter-level drop is the statistic that matches the
physics: compare the freshest readings against the most drained.

```sh
python3 -m beacon.cli --drift-classify --seed 11 --battery-degrade D
python3 -m beacon.cli --drift-classify --multifloor --seed 1 --battery-degrade D --loops 3
```

Measured behaviour (v0.63.0): 2 dB RSSI noise, beacon D,
seeds 1, 2, 3, 7, 11 — the other beacons stay `healthy` in
every scenario:

| walk | verdicts |
|---|---|
| single-floor, D steady -6 dB offset | `uncertain` 5/5 (was 4/5 — seed 11 now `uncertain`: short-window revisit delta still -1.84 dB, but the level only falls 0.75 dB from the first quarter to the last, against the 1.0 dB gate) |
| single-floor, D draining 0.3/hr | `battery-sag` 5/5, unchanged (level drops -1.68 to -4.48 dB, 2.05 to 4.93 standard errors) |
| single-floor, healthy / moved / noisy | unchanged: `healthy` 5/5, `moved` 5/5, `noisy` 5/5 |
| two-floor, D draining 0.3/hr, `--loops 3` | `battery-sag` 5/5, unchanged (level drops -3.50 to -5.37 dB) |
| two-floor, D draining 0.3/hr, two loops | `battery-sag` 2/5 (seeds 1 and 11), unchanged (level drops -2.03 and -3.11 dB) |
| two-floor, D steady -6 dB offset | `uncertain` 5/5, unchanged |

The seed-11 fix is honest about its margin: the offset's level
drop is -0.75 dB against a 1.0 dB gate — 0.25 dB of margin — and
its t is -1.46 against a -1.5 gate, so both gates exclude it but
neither by much. A noise excursion that happened to span the
whole walk, rather than just the tail, would still fake a sag,
and only more history — a third loop — would settle it. The
two-floor offset's seed 11 is the same lesson from the other
side: its level falls 1.18 dB across the walk, past the drop
gate, and it stays `uncertain` only because its long revisit
delta (-0.14 dB) never clears the sag gate — the two pieces of
evidence check different things, and a verdict needs both.
The adjacent misread on the moved side — the steady offset
reading `moved` on the tripled walk at the default window — is
fixed in v0.64.0 (see the notes at the end): the moved call now
needs the longer history to agree that the spread is real.
The roadmap grows as new needs surface — the level-drop check
for the sag call is the latest addition.

## A longer history must agree the spread is real

The level-drop check fixed the steady offset misreading as a
sag, but the same offset misread as a moved beacon on the other
side: on the tripled two-floor walk at the default 20-reading
window it reads `moved` on seed 1, because the short window's
spread is 2.70 dB — 0.20 dB over the moved gate — around a
-5.96 dB mean with a stable revisit delta (+0.17 dB over 20
pairs). That is a noise-inflated spread, not a position
pattern, and the gate is not the problem: a spread estimated
from 20 readings carries a ~0.3 dB standard error, so pure
noise clears a 2.5 dB gate about one time in twenty. The fix
is the same design as the longer history for the moved call —
more history, not a sharper threshold — applied the other way
round. When a longer history exists, the short-window moved
call now needs it to agree that the spread is real: the long
window's spread must clear the expected 2 dB noise by at least
1.5 standard errors, the same magnitude-plus-significance
convention as the sag gates, where the standard error of a
spread estimated from n readings under pure noise is
noise / sqrt(2n). A moved beacon's residuals carry a position
pattern on top of the noise, so its settled spread clears it;
a steady offset's residuals are pure noise, so its settled
spread sits at the noise level and the verdict falls through to
`uncertain`. The verdict's reason cites the longer history it
rests on, and the reported mean, spread, delta and pair count
stay the short window's — the same convention as the
long-history sag and moved verdicts. When the long window holds
no more readings than the short one, it says nothing the short
window has not already said, and the call reads the short
window alone.

A residual-vs-distance correlation was measured and rejected
as the discriminator first. The moved beacon reads louder
almost everywhere, since it moved from the corner (0, 12)
towards the ward interior (5, 7), so its correlation with
distance to the calibrated position is near zero (-0.05 over
the 120-reading history on seed 1) — the pattern is a
localised dip near the old corner, not a monotone trend — and
the offset's correlation is the same size (+0.08 on the short
window, -0.06 over the long history), so no correlation gate
separates them. The settled spread is the statistic that
matches the physics: a move adds position variance on top of
the noise, an offset does not.

```sh
python3 -m beacon.cli --drift-classify --multifloor --seed 1 --loops 3
python3 -m beacon.cli --drift-classify --multifloor --seed 1 --move-beacon D:5,-5 --loops 3
```

Measured behaviour (v0.64.0): 2 dB RSSI noise, beacon D,
seeds 1, 2, 3, 7, 11 — the other beacons stay `healthy` in
every scenario:

| walk | verdicts |
|---|---|
| two-floor, D steady -6 dB offset, `--loops 3`, default 20-reading window | `uncertain` 5/5 (was 4/5 — seed 1 now `uncertain`: short-window spread still 2.70 dB, but the 120-reading history spreads only 2.09 dB, 0.70 standard errors above the 2 dB noise) |
| two-floor, D moved (0, 12) to (5, 7), `--loops 3`, default 20-reading window | `moved` 5/5, unchanged (long spread 2.57 to 2.95 dB, 4.4 to 7.4 standard errors above noise; the reason cites the longer history) |
| two-floor, D moved, two loops, default 20-reading window | `moved` 5/5, unchanged (long spread 2.47 to 3.07 dB — seed 1's 2.47 dB is 2.97 standard errors above noise) |
| two-floor, D steady -6 dB offset, two loops, default 20-reading window | `uncertain` 5/5, unchanged |
| single-floor, D moved / offset | unchanged: `moved` 5/5, `uncertain` 5/5 (long spread 2.64 to 2.92 dB for the move) |
| two-floor, D moved / offset, demo 80-reading window | unchanged: `moved` 5/5 / `uncertain` 5/5 at three loops; `moved` 3/5 / `uncertain` 5/5 at two loops |

The fix is honest about its margin, and it is thin. The
offset's long spread clears the noise test by only 0.10 dB on
seed 1 (2.09 dB against a 2.19 dB gate), and on two loops seed
3's short spread sits 0.01 dB under the moved gate with a long
spread of 2.34 dB — 2.15 standard errors above noise, past the
test — so a slightly unluckier noise draw, one that pushed the
short spread 0.02 dB higher, would still misread it as `moved`.
The two gates check different things — the short spread against
2.5 dB, the long spread against the noise — and a verdict needs
both, but both are estimates from the same noise, so a fluke
that spans the whole walk rather than just the short window
would still fake a move, and only more history would settle it.
Extra seeds 5, 13, 17 and 23, measured after the fix, agree:
the offset reads `uncertain` on 4/4 and the move reads `moved`
on 4/4 at the default window on the tripled walk.
The roadmap grows as new needs surface — the longer history
agreeing the spread is real is the latest addition.
