# Neuralized High-Stacking Continuation

Date: 2026-06-18

This note records the first online use of modular small networks inside the
structured MuJoCo dry-stacking loop. The goal was not to declare success. The
goal was to reduce brute-force candidate search while still letting MuJoCo
verify the final candidate placements.

## Code Change

`StoneFitNet` is now integrated into the online literature/statics placement
path.

New CLI options:

```powershell
--stone-fit-ranker-dir <model_dir>
--stone-fit-top-k <k>
--candidate-pose-ranker-dir <model_dir>
--candidate-pose-top-k <k>
```

Current online policy:

1. For each target slot, score unused stones with `stone_fit_net`.
2. Blend the network probability with the existing dry-stacking role prior.
3. Keep only the top-K stones for MuJoCo pose evaluation.
4. For each kept stone, score generated poses with `candidate_pose_rank_net`.
5. Keep only the top-K poses.
6. Run MuJoCo settle/hold simulation on the remaining candidates.
7. Choose the physically verified candidate with the best local support/target
   score.

The old non-network behavior remains available by omitting the new ranker
arguments.

## Smoke Test

Run:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_neural_stonefit_pose_smoke_wall_high_earth_v2`

Configuration:

- target: `single_face_wall_high_v1`
- strategy: `statics_wall`
- gravity: Earth
- stone top-K: 4
- pose top-K: 3
- candidates requested: 6

Result:

| Metric | Value |
| --- | ---: |
| Placed stones | 31 / 31 |
| Stable stones | 20 / 31 |
| Visible courses | 8 |
| Height | 0.248 m |
| Target RMSE | 0.203 m |
| Max drift | 0.231 m |
| Velocity inf norm | 14.333 |
| Candidate-pose rows | 372 |
| Strict success | 0 |

Interpretation:

- The online neural path works and writes network metadata to
  `placement_log.csv` and `candidate_pose_log.csv`.
- Stone top-K 4 is too aggressive for high-wall work; it can make the planner
  choose locally plausible but globally poor stones.

## High Wall, Earth And Moon

Run:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_neural_stonefit_pose_high_wall_top8_earth_moon_v1`

Configuration:

- target: `single_face_wall_high_v1`
- strategy: `statics_wall`
- target slots: 31
- stone top-K: 8
- pose top-K: 3
- gravities: Earth and Moon

Results:

| Gravity | Placed | Stable | Visible courses | Height | RMSE | Max drift | Velocity | Success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 28 / 31 | 16 / 28 | 8 | 0.234 m | 0.219 m | 0.379 m | 8.057 | 0 |
| Moon | 31 / 31 | 21 / 31 | 8 | 0.283 m | 0.193 m | 0.141 m | 2.933 | 0 |

Interpretation:

- Relaxing stone top-K from 4/5 to 8 improved coverage, especially on Moon.
- The Moon wall reached all 31 slots and kept max drift under 0.15 m, but it
  still failed strict success because residual velocity and target outliers
  remain too high.
- This is the best current neuralized Moon high-wall attempt, but it is still a
  structured failure rather than a stable dry-stone wall.

Captures:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_neural_stonefit_pose_high_wall_top8_earth_moon_v1\captures_960x720`

The capture directory contains front, wall-front, side, top, and wall-top RGB
and depth views for both Earth and Moon failure cases.

## Multi-Stone Column

Run:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_neural_stonefit_pose_multi_column_lit_earth_moon_v1`

Configuration:

- target: `multi_stone_column_v3`
- strategy: `literature_column`
- target slots: 40
- stone top-K: 6
- pose top-K: 3
- gravities: Earth and Moon

Results:

| Gravity | Placed | Stable | Visible courses | Height | RMSE | Max drift | Velocity | Success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 40 / 40 | 18 / 40 | 8 | 0.268 m | 0.271 m | 0.365 m | 12.558 | 0 |
| Moon | 40 / 40 | 23 / 40 | 8 | 0.303 m | 0.439 m | 0.158 m | 1.843 | 0 |

Interpretation:

- The neuralized column reaches greater height than the neuralized high wall in
  Moon gravity.
- The Moon column is height-promising, but the target error is large. It is a
  tall partial column, not yet a compact stone pillar.
- Ring closure and course-level compactness should be explicit objectives
  before making the column taller.

Captures:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_neural_stonefit_pose_multi_column_lit_earth_moon_v1\captures_960x720`

## Scientific Lessons

1. `StoneFitNet` is useful as a sorter, but hard pruning is risky. Top-K 4 or 5
   caused missed feasible placements in high-wall runs.
2. A wider stone pool plus pose top-K pruning is the better near-term compromise:
   keep more stones, reduce pose simulations.
3. The current labels imitate the existing local search and success/failure
   logs. They do not yet encode whole-structure stability well enough.
4. Moon gravity can preserve height and reduce immediate drift in some cases,
   but low-gravity structures still suffer from target spread and residual
   motion.
5. Height is not success. The best current Moon neuralized high wall reached
   0.283 m and 8 visible courses, but it is still a strict failure.
6. The best current Moon neuralized column reached 0.303 m and 8 visible
   courses, but it is too spread out to count as a reliable pillar.

## Next Required Model Work

The next neuralization step should add stability objectives, not just imitate
the existing local search:

1. Train a course-level closure classifier for wall/column course compactness.
2. Train `MoonDriftRiskNet` on more high-layer Moon negatives and use it as a
   rejection gate.
3. Add compact stack-state features to `WorldDeltaNet`: local support height,
   support centroid, support span, course occupancy, and wall/column spread.
4. Change labels from `selected_by_pose_search` to a delayed target such as
   "placement remains stable after N later stones".
5. Keep MuJoCo in the loop for verification until the learned models predict
   post-hold stability and target error reliably.
