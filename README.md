# Moon Rock Stack

MuJoCo experiments for angular lunar-rock dry stacking under Earth and Moon gravity.

The near-term target is lunar landmark stacking: build small but recognizable dry-stacked structures such as single-stone columns and single-face wall segments. The long-term target is dry-stone construction on the Moon using geometry-aware stone selection, support reasoning, and data-driven repair.

This is a scientific experiment repository, not a visual demo. Failed runs are part of the dataset.

## Status

Current stage summary:

`docs/PHASE_4_LAYER_STACKING_SUMMARY_20260618.md`

High-layer continuation summary:

`docs/HIGH_LAYER_STACKING_RESULTS_20260618.md`

Neuralized high-stacking continuation:

`docs/NEURALIZED_HIGH_STACKING_RESULTS_20260618.md`

Current best target-locked moon wall-line partial:

`batch_runs/20260618_single_face_wall_4course_v1_assignment_plan_wall_bonded_gated_smoke`

- target: `single_face_wall_4course_v1`
- gravity: Moon, `1.624 m/s^2`
- placed stones: 14 / 24
- stable stones: 14 / 14 placed
- visible courses: 4
- height: 0.345 m
- target RMSE: 0.045 m
- wall x/y/aspect: 0.732 / 0.077 / 9.47
- max drift: 0.0015 m
- verdict: useful partial, still not a strict wall success because 10 target slots were skipped

First current-gate Earth single-face wall success:

`batch_runs/20260618_single_face_wall_4course_v1_assignment_fallback5_gated_smoke`

- target: `single_face_wall_4course_v1`
- gravity: Earth, `9.80665 m/s^2`
- placed stones: 17 / 24
- stable stones: 15 / 17 placed
- visible courses: 4
- height: 0.378 m
- target RMSE: 0.099 m
- wall x/y/aspect: 0.805 / 0.227 / 3.54
- max drift: 0.019 m
- verdict: current-gate success, but still a short high wall segment rather than a long masonry wall

The next bottleneck is Moon-specific fallback/repair, not rock generation.

The next research direction is an end-to-end learned placement model that
proposes rock placement poses directly, using simulation search only as a data
generator. See:

`docs/STACKNET_DATA_AND_MODEL_PLAN.md`

The near-term learning route is modular rather than fully end-to-end: train
small networks for stone-slot fit, pose acceptance, Moon drift risk, compact
world-delta prediction, and candidate-pose ranking. See:

`docs/MODULAR_SMALL_NETWORK_STACKING_PLAN.md`

The first online modular-network route is now implemented: `StoneFitNet` ranks
candidate stones for each target slot and `CandidatePoseRankNet` prunes pose
candidates before MuJoCo verification. The best current neuralized Moon high
wall placed 31 / 31 target stones with 8 visible courses and 0.283 m height, but
it is still a strict failure because residual motion and target outliers remain
too high. The best current neuralized Moon multi-stone column reached 0.303 m
with 8 visible courses, but target spread is still too large.

Latest 2026-06-20 perception-ranker update:

`docs/NETWORK_DESIGN_RATIONALE_AND_LOGGING_20260620.md`

`docs/HIGHCOURSE_4COURSE_RESULTS_20260620.md`

2026-06-20 strict wall-line and WallStateCritic V2 update:

`docs/STRICT_WALL_LINE_AND_WALLCRITIC_V2_20260620.md`

2026-06-21 3-course wall data-flywheel update:

`docs/DATA_FLYWHEEL_3COURSE_20260621.md`

- Added `scripts/run_wall_data_flywheel.py` as the repeatable collect -> dataset -> MuJoCo depth export -> modular training -> closed-loop eval -> capture scheduler.
- Focused 3-course run produced 276 placement examples, 12,972 candidate-pose examples, and 2,751 MuJoCo depth rows after target filtering.
- Added synthetic stone-slot candidate generation from `candidate_pose_log.csv`; the focused 3-course dataset now has 2,484 synthesized assignment-candidate examples.
- Trained a 3-course `PoseRankNet` with test top1 hit `0.408` and top3 hit `1.000`; the top candidate is not reliable enough yet, but the network can keep the best pose inside a small top-3 shortlist.
- 3-course Moon closed-loop eval reached 3 visible courses in all 3 trials, mean height `0.377 m`, mean drift `0.0011 m`, and mean RMSE `0.0497 m`, but strict success remains `0/3` due to skipped target slots and strict shape checks.
- Trained a 3-course synthetic-data `StoneSlotNet` and reran closed loop with `stone_fit_top_k=15`; Moon strict success reached `1/3` and shape success `2/3`. The successful case is `batch_runs/20260621_wall_flywheel_3course_stoneslot_v2_eval/captures_960x720/00_single_face_wall_3course_v1_success_statics_wall_moon_trial_01`.
- Trained `PoseRiskNet V1` as a pre-simulation candidate-pose risk penalty using only target, gravity, candidate pose, and rock geometry inputs. With `pose_risk_weight=0.35`, the first 3-course Moon closed-loop eval reached strict success `3/3`; after a 2-trial increment, the combined validation is Moon `5/5` strict success and Earth `0/5`. Combined Moon means: height `0.3046 m`, RMSE `0.0287 m`, max drift `0.0028 m`; Earth still needs separate impact/drift treatment. Result paths: `batch_runs/20260621_wall_flywheel_3course_pose_risk_w035_eval` and `batch_runs/20260621_wall_flywheel_3course_pose_risk_w035_eval_n4_increment`.
- Built an augmented 3-course dataset with 38 run examples, 606 placement examples, 26,472 candidate-pose examples, and 6,984 assignment candidates. `PoseRiskNet V2` trained on this larger dataset did not replace V1: offline top3 safe stayed `1.000`, but top1 safe fell to `0.333`, and the closed-loop smoke stayed `0/1` on both Earth and Moon. Current best online risk model remains `batch_runs/20260621_pose_risk_net_3course_v1`.
- PoseRiskNet captures are under `batch_runs/20260621_wall_flywheel_3course_pose_risk_w035_eval/captures_960x720`; the best wall evidence files are `wall_front_rgb.png`, `wall_front_object_depth.png`, and `wall_top_object_depth.png`.
- Added process replay rendering with `scripts/render_process_video.py`. A typical Moon 3-course success replay is saved at `batch_runs/20260621_wall_flywheel_3course_pose_risk_w035_eval/process_videos_release_replay/single_face_wall_3course_v1_statics_wall_moon_trial_01_process/process.gif`. This is a MuJoCo release-and-settle replay from selected placement poses, not a robot manipulation trajectory.
- Non-top depth captures now include `*_object_depth` masks; use `wall_front_object_depth.png` and `wall_top_object_depth.png` for wall inspection instead of raw far-plane-dominated depth PNGs.

- Trained `StoneSlotNet` for stone-slot candidate filtering:
  `batch_runs/20260620_torch_stone_slot_net_4course_assignment_v1_20260620_213352`.
- Ran strict single-face-wall control variants V1-V4. The best current result is V4:
  `batch_runs/20260620_highcourse_4course_stoneslot_structure_physicsgate_seed97011_v4`.
- V4 Moon result: strict success `0/1`, shape `0/1`, but it reached 4 visible courses, 12 stable stones, 2 failed stones, 0.340 m height, 0.105 m RMSE, 0.199 m max drift, and wall aspect 3.40.
- Main remaining failure: wall thickness and a few outliers; this is no longer just a loose stone pile, but it is not yet a valid dry-stacked single-face wall.
- Captures for the V4 Moon near-success/failure case are under:
  `batch_runs/20260620_highcourse_4course_stoneslot_structure_physicsgate_seed97011_v4/captures_960x720`.
- Built a new 4-course wall learning dataset:
  `batch_runs/20260620_learning_dataset_4course_wallcritic_v2`.
- Exported MuJoCo front/top depth observations:
  `batch_runs/20260620_mujoco_depth_observation_maps_4course_wallcritic_v2`.
- Trained `WallStateCritic V2`:
  `batch_runs/20260620_wall_state_critic_mujoco_depth_4course_wallcritic_v2`.
- Current conclusion: pose ranking alone is not enough. The next useful step is to wire `WallStateCritic` into closed-loop placement as a second-stage risk filter.

- Added MuJoCo-rendered front/top depth tensor export: `scripts/export_mujoco_depth_observation_maps.py`.
- Rendered depth dataset: `batch_runs/20260620_mujoco_depth_observation_maps_4course_groups256_v1`.
- Trained small CNN ranker: `batch_runs/20260620_torch_mujoco_depth_cnn_4course_groups256_score_v1`.
- Closed-loop 4-course wall test: `batch_runs/20260620_highcourse_4course_mujoco_depth_proxyonline_top3_seed97005_v1`.
- Result: Earth 0/2 and Moon 0/2 strict success; Moon reached 15/16 stable stones in one trial, but shape success stayed 0.
- Added `risk_adjusted` and `structure_aware` target modes for the PyTorch pose ranker.
- Follow-up runs:
  - `batch_runs/20260620_highcourse_4course_mujoco_depth_risk_proxyonline_top3_seed97006_v1`
  - `batch_runs/20260620_highcourse_4course_mujoco_depth_structure_proxyonline_top3_seed97007_v1`
- Main failure mode: the models still make a thick local wall/column-like structure instead of a thin laterally controlled wall. Pose ranking alone is not enough; next work needs stone-slot selection and a wall-state critic using front/top depth.
- RGB/depth failure captures are saved under `captures_960x720` in the closed-loop run.

## Physical Calibration

New runs use:

- Earth gravity: `9.80665 m/s^2`
- Moon gravity: `1.624 m/s^2`
- MuJoCo contact friction: `1.15 0.025 0.002`
- equivalent sliding friction angle: `atan(1.15) = 48.99 deg`

Validation run:

`batch_runs/20260618_physics_validation_v1`

Run again:

```powershell
conda run -n moon-rock-stack python scripts\validate_physics.py --output batch_runs\physics_validation_local
```

The validation performs:

- freefall acceleration fitting under Earth and Moon gravity;
- tilted-gravity static-friction tests across 35, 40, 45, 48, 49, 50, and 55 degrees.

Important limitation: MuJoCo Coulomb friction is a contact proxy. It does not model regolith cohesion, dust, particle crushing, or contact aging. Use friction sensitivity batches before making broad scientific claims.

## Environment

```powershell
conda env create -f environment.yml
conda activate moon-rock-stack
```

For editable package use:

```powershell
pip install -e .
```

## Generate And Screen Rocks

Generate target-conditioned angular rock catalogs:

```powershell
conda run -n moon-rock-stack python scripts\screen_target_rocks.py `
  --output batch_runs\targeted_rock_catalog_local `
  --rocks 360 `
  --clusters 10 `
  --seed 91 `
  --profile single_face_wall `
  --target single_face_wall_4course_v1
```

Outputs:

- `features.csv`
- `cluster_summary.csv`
- `role_screening.csv`
- `assignment_plan_<target>.csv`
- `README.md`

The geometry screen rejects spike-like, slab-like, over-elongated, or low-compactness candidates before physics simulation.

## Run Structured Dry-Stacking

Run a structured wall experiment with a pre-screened assignment plan and feasibility gate:

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
  --output batch_runs\structured_assignment_gated_local
```

The assignment plan chooses which stone should go to each target slot. The gate rejects physically bad placements instead of letting a bad stone contaminate later courses.

Run with same-role fallback candidates:

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

Fallback helped Earth in the first test, but hurt Moon because low-gravity lateral drift needs stricter acceptance.

## Capture RGB And Depth Views

```powershell
conda run -n moon-rock-stack python scripts\capture_cases.py `
  --output batch_runs\structured_assignment_gated_local `
  --max-success 0 `
  --max-failure 2 `
  --width 960 `
  --height 720 `
  --capture-dir-name captures_960x720
```

Captured views include:

- front RGB/depth;
- wall-front RGB/depth;
- side views;
- top RGB/depth;
- wall-top depth;
- object-only depth images such as `wall_front_object_depth.png`, `right_object_depth.png`, and `wall_top_object_depth.png`.

The regular depth PNG keeps floor/environment context and masks MuJoCo far-plane background before color normalization. For wall geometry inspection and later vision-model inputs, prefer `*_object_depth.png` / `*_object_depth.npy`; these are rendered from MuJoCo segmentation and keep rock geometry while masking the ground plane.

## Main Modules

- `moon_rock_stack/fractal_rocks.py`: angular polyhedral rock generation and target-conditioned profiles.
- `moon_rock_stack/features.py`: geometric feature extraction.
- `moon_rock_stack/clustering.py`: feature clustering.
- `moon_rock_stack/mjcf.py`: MuJoCo world writer and contact parameters.
- `moon_rock_stack/simulate.py`: basic stacking simulation.
- `moon_rock_stack/structured.py`: target slots, structured placement, support scoring, assignment gate.
- `moon_rock_stack/run_structured_experiment.py`: assignment plans, same-role fallback pools, structured experiment CLI.
- `scripts/screen_target_rocks.py`: catalog generation, screening, role scoring, assignment plan output.
- `scripts/validate_physics.py`: gravity and friction validation.
- `scripts/capture_cases.py`: RGB/depth capture for success and failure cases.
- `scripts/build_learning_dataset.py`: converts run logs into V0 learning tables.
- `scripts/train_modular_stack_models.py`: trains pure-NumPy modular small-network baselines.
- `scripts/evaluate_candidate_pose_ranker.py`: evaluates pose-ranker top-K recovery against logged pose-search winners.
- `scripts/export_rock_pointclouds.py`: exports generated OBJ rocks as fixed-size point-cloud tensors.
- `scripts/export_local_support_maps.py`: exports local height/support/candidate raster tensors for neural placement models.
- `scripts/train_candidate_pose_group_ranker.py`: trains a tabular groupwise pose-ranker baseline.
- `scripts/train_support_map_group_ranker.py`: trains a pooled support-map groupwise baseline without extra deep-learning dependencies.
- `scripts/train_torch_support_map_ranker.py`: trains a PyTorch CNN + MLP groupwise candidate-pose ranker.
- `scripts/train_torch_pointnet_rock_encoder.py`: trains a PyTorch PointNet-style rock point-cloud encoder.

## Data Policy

Large experiment outputs are kept locally and ignored by Git by default. Do not delete them during normal iteration.

See:

`docs/data_management.md`

Commit source, scripts, environment files, and curated summaries. Do not accidentally push local papers, third-party repositories, raw states, bulk meshes, or full image batches.

## GitHub Preparation

This directory is intended to be the Git repository root:

`D:\MoonStack\experiments\moon_rock_stack`

Initial setup:

```powershell
git init
git add README.md environment.yml pyproject.toml .gitignore .github docs moon_rock_stack scripts
git status
```

When ready to connect a remote:

```powershell
git remote add origin https://github.com/<owner>/<repo>.git
git push -u origin main
```

Do not run those remote commands until the GitHub repository name and visibility are decided.
