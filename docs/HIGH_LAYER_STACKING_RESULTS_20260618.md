# High-Layer Stacking Results

Date: 2026-06-18

This note records the first continuation run after the four-course milestone.
The goal was to push higher structures while keeping the evaluation strict:
visible courses alone do not count as success if the wall or column is unstable,
spread out, or no longer matches the target structure.

## Runs

### High Single-Face Wall, Earth

Run:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_high_single_face_wall_v1_earth_fast`

Configuration:

- target: `single_face_wall_high_v1`
- target slots: 31
- strategy: `statics_wall`
- gravity: Earth
- candidates: 2
- settle/hold steps: 90 / 220

Result:

| Metric | Value |
| --- | ---: |
| Placed stones | 31 / 31 |
| Stable stones | 21 / 31 |
| Visible courses | 8 |
| Height | 0.237 m |
| Target RMSE | 0.173 m |
| Max target error | 0.309 m |
| Max drift | 0.136 m |
| Velocity inf norm | 7.494 |
| Strict success | 0 |

Interpretation:

- It reaches 8 visible courses, so higher layer sequencing is possible.
- It is not stable enough; residual velocity is high and 10 stones fail the
  final stability criteria.

### High Single-Face Wall, Moon

Run:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_high_single_face_wall_v1_moon_fast`

Result:

| Metric | Value |
| --- | ---: |
| Placed stones | 29 / 31 |
| Stable stones | 14 / 29 |
| Visible courses | 8 |
| Height | 0.204 m |
| Target RMSE | 0.576 m |
| Max target error | 2.786 m |
| Max drift | 0.186 m |
| Strict success | 0 |

Interpretation:

- The low-gravity run also reaches 8 visible courses.
- The geometry is not wall-like: lateral/target spread dominates.
- Moon high walls need gravity-specific lateral risk rejection and support
  repair before adding more layers.

### High Single-Face Wall, Earth, Ranker-Guided Top-3

Run:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_high_single_face_wall_v1_earth_ranker_top3`

Configuration:

- target: `single_face_wall_high_v1`
- strategy: `statics_wall`
- gravity: Earth
- pose candidates requested: 8
- learned pose top-K simulated: 3
- ranker: `20260618_modular_small_networks_v2_high_and_ranker`

Result:

| Metric | Value |
| --- | ---: |
| Placed stones | 31 / 31 |
| Stable stones | 20 / 31 |
| Visible courses | 8 |
| Height | 0.300 m |
| Target RMSE | 0.253 m |
| Max target error | 0.980 m |
| Max drift | 0.257 m |
| Strict success | 0 |

Interpretation:

- The ranker-guided run is taller than the non-ranker fast Earth wall.
- It is less stable and less target-accurate.
- The current ranker optimizes local pose imitation, not whole-structure
  stability. It must be combined with `MoonDriftRiskNet`, support features, and
  a stability/risk objective before being trusted for high layers.

### Ten-Course Single-Stone Column

Run:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_single_column_v4_10course_smoke`

Target:

- `single_column_v4`
- 10 one-stone courses.

Results:

| Gravity | Stable | Visible courses | Height | RMSE | Max drift | Success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 2 / 10 | 10 | 0.237 m | 0.280 m | 0.012 m | 0 |
| Moon | 2 / 10 | 10 | 0.212 m | 0.232 m | 0.233 m | 0 |

Interpretation:

- The target can place 10 one-stone courses, but only 2 stones remain stable.
- A pure one-stone vertical column is too contact-sensitive for the current
  irregular rock set.
- This should not be the main route for high lunar landmarks unless the planner
  learns very accurate local support and contact normals.

### Multi-Stone Column

Runs:

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_multi_stone_column_v3_earth_fast`
- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_multi_stone_column_v3_moon_fast`

Target:

- `multi_stone_column_v3`
- 40 slots.

Results:

| Gravity | Stable | Visible courses | Height | RMSE | Max drift | Success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 17 / 40 | 8 | 0.267 m | 0.543 m | 0.235 m | 0 |
| Moon | 15 / 40 | 8 | 0.325 m | 0.578 m | 0.226 m | 0 |

Interpretation:

- Multi-stone columns are more promising than one-stone columns for height.
- The Moon run reached the greatest height in this batch, 0.325 m, but target
  spread is still too large.
- The next column planner should prioritize compact ring closure and local
  support before advancing upward.

### Neuralized Stone-Fit + Pose-Ranker Continuation

Detailed note:

`D:\MoonStack\experiments\moon_rock_stack\docs\NEURALIZED_HIGH_STACKING_RESULTS_20260618.md`

The structured loop now supports online `StoneFitNet` stone ranking and
`CandidatePoseRankNet` pose top-K pruning before MuJoCo verification.

High wall run:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_neural_stonefit_pose_high_wall_top8_earth_moon_v1`

| Gravity | Placed | Stable | Visible courses | Height | RMSE | Max drift | Velocity | Success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 28 / 31 | 16 / 28 | 8 | 0.234 m | 0.219 m | 0.379 m | 8.057 | 0 |
| Moon | 31 / 31 | 21 / 31 | 8 | 0.283 m | 0.193 m | 0.141 m | 2.933 | 0 |

Multi-stone column run:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_neural_stonefit_pose_multi_column_lit_earth_moon_v1`

| Gravity | Placed | Stable | Visible courses | Height | RMSE | Max drift | Velocity | Success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 40 / 40 | 18 / 40 | 8 | 0.268 m | 0.271 m | 0.365 m | 12.558 | 0 |
| Moon | 40 / 40 | 23 / 40 | 8 | 0.303 m | 0.439 m | 0.158 m | 1.843 | 0 |

Interpretation:

- Neural stone ranking is useful as a soft sorter, but narrow top-K pruning is
  too brittle for high-layer structures.
- Moon high-wall coverage improved when stone top-K was widened to 8.
- The Moon multi-stone column reached greater height than the neuralized wall,
  but target spread remains too large.
- These are structured failures with useful data, not strict dry-stack
  successes.

## Captures

Typical failure captures were saved under:

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_high_single_face_wall_v1_earth_fast\captures_960x720`
- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_high_single_face_wall_v1_moon_fast\captures_960x720`
- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_multi_stone_column_v3_moon_fast\captures_960x720`
- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_high_single_face_wall_v1_earth_ranker_top3\captures_960x720`
- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_neural_stonefit_pose_high_wall_top8_earth_moon_v1\captures_960x720`
- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_neural_stonefit_pose_multi_column_lit_earth_moon_v1\captures_960x720`

## Lessons

1. Eight visible courses are now attainable in both wall and column targets.
2. Height alone is misleading; all high-layer successes in this batch are
   strict failures due to spread, drift, or instability.
3. The neural ranker can produce taller structures, but it currently worsens
   target accuracy and drift in high-wall Earth tests.
4. `StoneFitNet` helps as a sorter, but hard pruning too early can remove
   feasible high-layer stones.
5. A pure one-stone column is too fragile; multi-stone rings/cores are a better
   high-layer direction.
6. The next higher-wall attempt should use:
   - `StoneFitNet` as a wide sorter, not a narrow hard gate;
   - `CandidatePoseRankNet` only as a pose pre-filter;
   - `MoonDriftRiskNet` as a hard rejection signal;
   - a support-continuity gate before moving to the next course.
