# StackNet Data And Model Plan

Date: 2026-06-18

The next technical target is to replace deployment-time random or brute-force
pose search with learned placement models. The near-term route is modular:
train several small networks for stone-slot fit, pose acceptance, Moon drift
risk, and compact future prediction before attempting one large end-to-end
policy. Search can remain in simulation as a label generator, but a real robot
on the Moon cannot afford many failed physical placements.

## Papers Read For This Direction

### GraspNet-1Billion, arXiv:1912.13470v2

The important lesson from GraspNet is not the gripper itself. The lesson is the
dataset structure: large-scale RGB-D scenes, object geometry, dense 6DoF action
annotations, and a standardized evaluation benchmark. For MoonStack, the
analogous action is not a parallel-jaw grasp, but a dry-stacking placement pose.
The arXiv v2 abstract reports `97,280` RGB-D images, `88` objects, `190`
cluttered scenes, and more than `1.1 billion` annotated 6DoF grasp poses, plus
analytic evaluation of grasp success.

MoonStack translation:

- scene: current partial rock structure;
- object: candidate angular rock;
- action: 6DoF placement pose;
- label: whether the placement settles into a useful structural state;
- benchmark: success, shape success, wall-depth spread, drift, support, and
  number of simulation trials needed per accepted placement.

### Efficient-WAM, arXiv:2606.10040v2

The important lesson from Efficient-WAM is that action planning does not always
need full high-fidelity future video generation. A compact future representation
can be enough if it predicts the outcome needed for decision making.
The 2026 paper frames this as a 1B-parameter world-action model with low-cost
future imagination; it uses compact/coarse future latents as guidance for action
generation and reports around `100 ms` per action chunk in deployment with a
large speedup over heavier WAM baselines.

MoonStack translation:

- do not start with photorealistic future video prediction;
- encode the current stack and candidate placement into a compact latent state;
- predict future support, drift, target error, stability, and failure risk;
- use that prediction to rank or generate placement actions.

### Related Physical Stability Literature

Physical stability datasets such as ShapeStacks show that visual and geometric
representations can learn stack stability from synthetic 3D structures. For this
project, that idea must be extended from simple block towers to irregular,
angular, multi-contact dry-stone structures under Earth and Moon gravity.
The ShapeStacks v2 abstract reports `20,000` simulated stack configurations with
stability annotations.

## Problem Formulation

Near-term modular formulation:

```text
StoneFitNet + PoseAcceptNet + MoonDriftRiskNet + WorldDeltaNet + RepairNet
  -> top-K placement actions for MuJoCo or robot verification
```

Longer-term end-to-end formulation:

Train a model that maps:

```text
current stack state + candidate rock geometry + target structure context
  -> top-K 6DoF placement poses + stability/confidence scores
```

The output should be useful as a robot policy proposal, not only as an offline
classifier. The practical deployment loop should be:

1. observe current stack with RGB-D;
2. segment or reconstruct candidate rock geometry;
3. ask the network for top-K placement poses;
4. optionally verify the top 1 to 3 poses in a fast simulator or lightweight
   learned dynamics model;
5. execute the best placement;
6. observe the settled state and update the model dataset.

## Inputs

Core input tensors:

- scene depth or point cloud before placement;
- candidate rock mesh or point cloud;
- target slot position and local target volume;
- course index and role: base, middle, cap, chock, tie, repair;
- already placed rock ids and local support context;
- gravity id and numeric gravity value;
- contact/friction parameters;
- optional RGB images for visual-domain adaptation;
- optional occupancy grid, height map, and support map.

Useful engineered features:

- rock volume, mass, bounding box, elongation, flatness, compactness;
- roughness, angularity, spike score, stability score;
- cluster label and source kind;
- target x/y/z, wall-depth tolerance, course height;
- local gap/support density near the target slot.

## Outputs

The model should produce:

- top-K placement poses: `(x, y, z, qw, qx, qy, qz)`;
- confidence for each pose;
- predicted stable probability;
- predicted target error after settling;
- predicted wall-depth error after settling;
- predicted horizontal drift after hold time;
- predicted support overlap/contact count;
- optional role compatibility score for the candidate rock.

The first version can be a ranker over sampled candidate poses. The stronger
version should generate placement poses directly.

## Dataset Levels

### Dataset V0: Existing Logs

Existing files already support a first supervised dataset:

- `results.csv`: run-level outcome labels;
- `placement_log.csv`: one row per accepted, skipped, or best-rejected slot;
- `failure_cases.csv`: failed placements with failure reasons;
- `features.csv`: rock geometry features and clusters;
- `target_slots_*.csv`: target course and role geometry;
- `assignment_candidates_*.csv`: same-role fallback candidates when available;
- `states/*.npz`: final MuJoCo states;
- `captures_*/`: final RGB/depth views for selected cases.

V0 can train:

- stone-to-slot compatibility;
- imitation of accepted search results;
- success/failure classification;
- gravity-specific failure prediction;
- role and cluster statistics.

V0 cannot yet train a true GraspNet-style dense pose proposal model because the
simulator does not record every sampled candidate pose and its outcome.

### Dataset V1: Dense Candidate Pose Logging

Add a `candidate_pose_log.csv` per run. Every sampled pose should be recorded,
not only the selected or final skipped candidate.

Required columns:

- `episode_id`, `run_name`, `target_name`, `gravity`, `trial`, `strategy`;
- `step`, `slot_id`, `course`, `role`;
- `candidate_rock_index`, `source_kind`, `cluster_label`;
- `candidate_rank`, `candidate_seed`, `candidate_strategy`;
- `pose_x`, `pose_y`, `pose_z`, `pose_qw`, `pose_qx`, `pose_qy`, `pose_qz`;
- `settled_x`, `settled_y`, `settled_z`;
- `target_error_xy_m`, `target_y_error_m`;
- `support_overlap`, `support_contact_count`, `support_balance_error_m`;
- `height_gain_m`, `horizontal_drift_m`, `velocity_inf_norm_after_place`;
- `accepted_by_gate`, `selected_for_commit`, `failure_reason`;
- `pre_state_path`, `post_candidate_state_path`, `post_commit_state_path`;
- `scene_depth_path`, `candidate_depth_path`, `scene_pointcloud_path`.

This turns the simulator into an offline label factory. The cost is acceptable
because failures happen in simulation, not on hardware.

### Dataset V2: Active Learning

Once a first model exists:

1. train on V0/V1;
2. use the model to propose top-K placements;
3. verify those placements in MuJoCo;
4. log failures as hard negatives;
5. retrain with a larger emphasis on Moon failures, top-course drift, and
   difficult geometry clusters.

This is a DAgger-style loop for dry stacking: the learned policy replaces most
search, while simulation continues to supervise edge cases.

## Model Options

### Option A: Pose Ranker

This is the fastest useful model.

Input:

- scene features;
- rock features or rock point cloud embedding;
- target slot features;
- a proposed pose.

Output:

- scalar score;
- stable probability;
- expected target error and drift.

Use it to rank sampled poses. This still samples poses, but can reduce the
number of MuJoCo checks by selecting only the best few.

### Option B: GraspNet-Style Placement Proposal Network

Input:

- scene point cloud;
- candidate rock point cloud;
- target/role conditioning.

Output:

- top-K 6DoF placement poses;
- confidence and stability estimates.

This is the long-term policy candidate. It should learn direct placement
proposals instead of depending on random search.

### Option C: WAM-Style Compact Dynamics Model

Input:

- encoded current stack latent;
- candidate rock latent;
- proposed action latent.

Output:

- future compact stack latent;
- predicted support, drift, target error, and failure risk.

Use it as a planner/ranker. This is especially useful when full MuJoCo
verification is too slow or when the robot must evaluate multiple future
placement sequences.

### Recommended Architecture

Start with a hybrid:

1. PointNet or Point Transformer encoder for candidate rock point clouds.
2. Sparse point cloud or height-map encoder for the current stack.
3. MLP or transformer fusion with target slot, role, and gravity tokens.
4. Pose proposal head for top-K placement.
5. Outcome head for stability, target error, drift, and support.
6. Optional WAM head that predicts compact future support/height-map features.

This gives both direct action generation and outcome prediction.

## Training Losses

Use multiple labels instead of a single success flag:

- pose regression loss for accepted placements;
- quaternion/geodesic orientation loss;
- binary cross entropy for stable/accepted labels;
- pairwise ranking loss between better and worse candidate poses;
- drift regression loss;
- target error regression loss;
- support overlap regression loss;
- gravity-domain classification or conditioning loss;
- calibration loss for confidence reliability.

For early experiments, a candidate ranker can be trained with:

```text
score = + stable
        + shape_success
        - target_error_xy_m
        - wall_depth_error_m
        - horizontal_drift_m
        - velocity_after_place
        + support_overlap
```

The exact coefficients should be learned or tuned per target. Do not freeze
them as a scientific claim.

## Evaluation Metrics

Report metrics that match the real task:

- top-1 placement success;
- top-3 and top-5 placement success;
- mean MuJoCo checks per accepted placement;
- strict wall success rate;
- shape success rate;
- stable stones / placed stones;
- placed stones / target slots;
- visible courses;
- wall x/y/aspect;
- max drift;
- target RMSE and max target error;
- Moon-to-Earth and Earth-to-Moon transfer gap;
- hard-negative failure rate by source kind and cluster.

The key comparison is:

```text
random/local search vs learned top-K proposal vs learned proposal + WAM ranker
```

## Data Collection Experiments

Immediate experiments:

1. Convert existing runs into V0 learning tables.
2. Add candidate-pose logging to the structured simulator.
3. Run balanced Earth/Moon batches with the same rock catalogs.
4. Oversample Moon failures and high-course placements.
5. Capture per-step depth/point cloud for accepted and rejected candidates.
6. Train a baseline tabular/MLP ranker before training a point-cloud model.

Suggested first batch design:

- targets: `single_face_wall_4course_v1`, high single-stone pillar;
- gravities: Earth and Moon;
- rock profiles: `single_face_wall`, `high_wall`, `wall_statics`;
- seeds: at least 20 per target/profile/gravity;
- candidates per slot: 16 to 64 for label generation;
- store all candidates, not only selected candidates.

## What Must Change In Code

1. Add `candidate_pose_log.csv` in `moon_rock_stack/structured.py`.
2. Save pre-placement and post-candidate compact states.
3. Add systematic per-step depth rendering for scene and candidate rock.
4. Add a dataset builder that creates flat CSV and JSONL examples.
5. Add a baseline trainer for tabular pose ranking.
6. Add point-cloud export for candidate rocks and partial stacks.
7. Add an evaluation command that runs learned top-K proposals through MuJoCo.

The first dataset builder is:

`D:\MoonStack\experiments\moon_rock_stack\scripts\build_learning_dataset.py`

The modular small-network plan is:

`D:\MoonStack\experiments\moon_rock_stack\docs\MODULAR_SMALL_NETWORK_STACKING_PLAN.md`

The first modular baseline trainer is:

`D:\MoonStack\experiments\moon_rock_stack\scripts\train_modular_stack_models.py`

## Scientific Guardrails

- Do not report pile height as wall success.
- Separate Earth and Moon results.
- Keep rejected and failed cases.
- Report skipped slots explicitly.
- Avoid claiming real lunar validity until friction, cohesion, granular
  effects, sensing noise, and manipulation uncertainty are tested.
- Treat simulation search as a data generator, not as the final policy.

## Milestone Definition

The next milestone is not "more random trials." It is:

- V1 candidate-pose dataset exists;
- a baseline ranker reduces MuJoCo checks per placement by at least 3x;
- learned top-3 proposals match or exceed the current search/gate success rate;
- performance is reported separately for Earth and Moon;
- failure cases are saved with depth/RGB views and structured labels.

## References Checked

- GraspNet v2, arXiv:1912.13470: `https://arxiv.org/abs/1912.13470`
- Efficient-WAM, arXiv:2606.10040: `https://arxiv.org/abs/2606.10040`
- ShapeStacks v2, arXiv:1804.08018: `https://arxiv.org/abs/1804.08018`
