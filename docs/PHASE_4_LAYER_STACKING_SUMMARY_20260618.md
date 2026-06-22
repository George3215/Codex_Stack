# Phase Summary: Four-Course Angular Rock Stacking

Date: 2026-06-18

This note records the current milestone: a four-course target-locked dry-stacked
rock structure can now be produced in MuJoCo with angular, non-slab,
non-spike-like synthetic stones. The result is not yet a mature lunar wall
construction method. It is a useful stage because the pipeline now has enough
structure, logging, and failure modes to support learning-based placement.

## Goal

The near-term objective is lunar landmark stacking: build a small but deliberate
dry-stacked structure from irregular angular rocks. The current target is
`single_face_wall_4course_v1`, a single-face wall segment with four visible
courses and 24 intended target slots.

This is explicitly not a loose pile. A useful result must have:

- visible courses;
- controlled wall-depth spread;
- low drift after settling;
- enough target occupancy to preserve the intended wall geometry;
- stable residual dynamics after hold time;
- recorded failures, not only successful images.

## Environment

Simulation environment:

- Conda environment: `moon-rock-stack`
- Physics engine: MuJoCo through Python
- Earth gravity: `9.80665 m/s^2`
- Moon gravity: `1.624 m/s^2`
- Main MuJoCo contact friction coefficient: `1.15`
- Equivalent friction angle: `atan(1.15) = 48.99 deg`

Physics validation run:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_physics_validation_v1`

Validation results:

- Earth and Moon freefall acceleration are numerically consistent with the
  configured values.
- Tilt-equivalent friction tests behave consistently around the expected
  Coulomb threshold near 49 degrees.

Important limitation: this is still a rigid-body contact proxy. It does not
model regolith cohesion, dust, crushing, abrasion, contact aging, or granular
interlock at particle scale.

NASA context checked for this milestone: the Moon surface is dominated by rocky
debris and regolith, and surface gravity is about one-sixth of Earth's gravity.
The simulator uses the standard numeric value `1.624 m/s^2` for Moon gravity.

## Stone Geometry

The generator now creates angular multifaceted polyhedral stones instead of
smooth blobs, flat plates, or spiky shapes. The current source categories are:

- `equant_clast`
- `subangular_block`
- `wedge_clast`
- `fractured_clast`
- `elongated_clast`
- `upright_block_clast`
- `compact_block_clast`
- `wall_block_clast`
- `buttress_clast`
- `keystone_clast`
- `angular_boulder_clast`
- `notched_block_clast`
- `bearing_block_clast`
- `course_block_clast`
- `tie_bridge_clast`
- `chock_clast`
- `interlock_block_clast`
- `cap_block_clast`

The current best catalog for the four-course wall is:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_targeted_rock_catalog_single_face_wall_v2`

Catalog summary:

| Item | Value |
| --- | ---: |
| Generated rocks | 360 |
| Accepted by geometry screen | 337 |
| Rejected by geometry screen | 23 |
| Spike rejects | 23 |
| Slab rejects | 0 |
| Clusters | 10 |
| Profile | `single_face_wall` |

Screening rules reject candidates that violate the current rock prior:

- `spike_score > 0.16`
- `flatness > 1.62`
- `short_to_mid < 0.62`
- `elongation > 1.85`
- `compactness < 0.22`

The accepted stones are clustered by measured geometry and then scored for
structural roles such as base, middle, cap, chock, and tie.

## Current Stacking Method

The current four-course pipeline is deterministic structure planning plus local
MuJoCo pose search:

1. Generate a target-conditioned angular rock catalog.
2. Extract geometry features and cluster the stones.
3. Reject spikes, slabs, stringers, and low-compactness shapes.
4. Build `assignment_plan_single_face_wall_4course_v1.csv`, which assigns
   screened stones to target slots by role.
5. For each target slot, generate a small set of candidate poses around the
   slot center.
6. Settle each candidate in MuJoCo.
7. Score candidates by target error, wall-depth error, support overlap, residual
   velocity, height gain, and drift.
8. Use the assignment gate to reject infeasible placements instead of leaving
   outliers in the scene.
9. If enabled, try same-role fallback stones from `role_screening.csv`.
10. Record placements, skips, failures, final state, RGB/depth captures, and
    per-run metrics.

This is better than random placement, but it is still search-based. It is not
yet a deployable robotic method because each real placement would require too
many failed physical trials.

## Best Runs

### Best Moon Four-Course Partial

Run:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_single_face_wall_4course_v1_assignment_plan_wall_bonded_gated_smoke`

Method:

- target: `single_face_wall_4course_v1`
- strategy: `wall_bonded`
- gravity: Moon, `1.624 m/s^2`
- assignment plan enabled
- assignment gate enabled
- no same-role fallback

Result:

| Metric | Value |
| --- | ---: |
| Placed stones | 14 / 24 |
| Stable stones | 14 / 14 placed |
| Skipped slots | 10 |
| Visible courses | 4 |
| Height | 0.345 m |
| Target RMSE | 0.045 m |
| Max target error | 0.097 m |
| Wall x/y/aspect | 0.732 / 0.077 / 9.47 |
| Max horizontal drift | 0.0015 m |

Interpretation:

- This is the cleanest Moon wall-line partial so far.
- It reaches four visible courses and stays narrow in wall depth.
- It is not a strict wall success because 10 target slots are skipped.
- The bottleneck is missing support/repair, not basic rock generation.

### First Current-Gate Earth Four-Course Success

Run:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_single_face_wall_4course_v1_assignment_fallback5_gated_smoke`

Method:

- target: `single_face_wall_4course_v1`
- strategy: `wall_bonded`
- gravity: Earth, `9.80665 m/s^2`
- assignment plan enabled
- assignment gate enabled
- same-role fallback pool: 5 per slot

Result:

| Metric | Value |
| --- | ---: |
| Success | 1 |
| Shape success | 1 |
| Placed stones | 17 / 24 |
| Stable stones | 15 / 17 placed |
| Skipped slots | 7 |
| Visible courses | 4 |
| Height | 0.378 m |
| Target RMSE | 0.099 m |
| Max target error | 0.221 m |
| Wall x/y/aspect | 0.805 / 0.227 / 3.54 |
| Max horizontal drift | 0.019 m |

Interpretation:

- This is the first current-gate Earth success for the four-course single-face
  wall target.
- Fallback helps Earth because it can find feasible same-role substitutions.
- The wall is still short and partial, not a long masonry wall.

### Same Fallback Under Moon Gravity

Run:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_single_face_wall_4course_v1_assignment_fallback5_gated_smoke`

Result:

| Metric | Value |
| --- | ---: |
| Success | 0 |
| Shape success | 0 |
| Placed stones | 13 / 24 |
| Stable stones | 10 / 13 placed |
| Visible courses | 4 |
| Height | 0.378 m |
| Target RMSE | 0.143 m |
| Wall x/y/aspect | 0.622 / 0.396 / 1.57 |
| Max horizontal drift | 0.364 m |

Interpretation:

- The Earth fallback rule does not transfer directly to Moon gravity.
- Moon failures are dominated by lateral drift, side outliers, and post-hold
  instability.
- The next Moon planner needs stricter lateral prediction and support repair.

## Captures

Useful RGB/depth captures are stored under:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_single_face_wall_4course_v1_assignment_fallback5_gated_smoke\captures_960x720`

The capture set includes:

- front RGB/depth;
- wall-front RGB/depth;
- side views;
- top RGB/depth;
- wall-top depth.

For future learning, captures should be produced at every placement step, not
only after final success or failure.

## Lessons

1. Angular rock generation is now usable. The current issue is no longer smooth
   stones, slabs, or spikes.
2. Geometry screening is necessary but not sufficient. Candidate stones still
   need physics-aware placement and role-aware sequencing.
3. Assignment plans are useful only with a gate. Blindly forcing planned stones
   creates outliers and destroys the wall shape.
4. Same-role fallback helped Earth but hurt Moon. Gravity-aware stability
   prediction is required.
5. The four-course milestone is enough to start data collection for learning,
   but not enough to claim a robust dry-stone construction algorithm.
6. Current search is acceptable for simulation label generation, not for real
   deployment.

## Reproduction Commands

Generate the current target-conditioned catalog:

```powershell
conda run -n moon-rock-stack python scripts\screen_target_rocks.py `
  --output batch_runs\targeted_rock_catalog_local `
  --rocks 360 `
  --clusters 10 `
  --seed 91 `
  --profile single_face_wall `
  --target single_face_wall_4course_v1
```

Run the gated assignment experiment:

```powershell
conda run -n moon-rock-stack python -m moon_rock_stack.run_structured_experiment `
  --rocks 360 `
  --rock-profile single_face_wall `
  --seed 91 `
  --clusters 10 `
  --trials 1 `
  --targets single_face_wall_4course_v1 `
  --strategies wall_bonded `
  --candidates 4 `
  --steps-per-rock 520 `
  --hold-steps 3000 `
  --workers 2 `
  --assignment-plan batch_runs\targeted_rock_catalog_local\assignment_plan_single_face_wall_4course_v1.csv `
  --assignment-gate `
  --role-screening batch_runs\targeted_rock_catalog_local\role_screening.csv `
  --assignment-fallbacks 5 `
  --output batch_runs\structured_assignment_fallback5_local
```

## Next Direction

The next stage should collect supervised and ranking data for an end-to-end
placement model. The planner should move from random or small local search to a
network that proposes top-K stone poses directly from:

- current partial stack geometry;
- candidate stone geometry;
- target slot, role, and course;
- gravity and contact parameters;
- predicted future stability.

The detailed plan is in:

`D:\MoonStack\experiments\moon_rock_stack\docs\STACKNET_DATA_AND_MODEL_PLAN.md`

## References Checked

- NASA Moon facts: `https://science.nasa.gov/moon/facts/`
- GraspNet-1Billion / GraspNet v2: `https://arxiv.org/abs/1912.13470`
- Efficient-WAM: `https://arxiv.org/abs/2606.10040`
