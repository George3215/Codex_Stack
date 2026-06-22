# Modular Small-Network Stacking Plan

Date: 2026-06-18

This is the near-term learning route. Do not start with a single large
end-to-end model. First build a modular planner that combines several small
networks with MuJoCo verification and dry-stacking rules.

## Why Modular First

The current simulator can already generate useful placements, failures, and
four-course wall partials. The immediate problem is not representation size; it
is data structure and action selection. A large end-to-end model would hide too
many failure modes. Small networks let us measure which part fails:

- stone selection;
- slot-role matching;
- pose proposal;
- Moon lateral drift;
- post-settle stability;
- repair/fallback choice.

This matches the GraspNet lesson at a smaller scale: generate structured labels
for 6DoF actions, train networks to score or propose actions, and evaluate with
physics and task metrics.

## Network Modules

### StoneFitNet

Purpose: decide whether a candidate rock is suitable for a target slot before
pose search.

Inputs:

- rock geometry features;
- source kind and cluster label;
- course, role, target slot position;
- gravity and target name.

Outputs:

- stone-slot fit probability;
- role compatibility score.

Use:

- rank fallback pools;
- avoid trying poor cap stones in base slots or wide wall stones in thin wall
  slots.

### PoseAcceptNet

Purpose: evaluate a proposed 6DoF pose before spending expensive simulation or
robot attempts.

Inputs:

- StoneFitNet features;
- proposed pose `(x, y, z, qw, qx, qy, qz)`;
- candidate id and assignment fallback metadata.

Outputs:

- probability that the placement will be committed and remain useful.

Use:

- rank candidate poses;
- keep only top 1 to 3 poses for MuJoCo verification.

### MoonDriftRiskNet

Purpose: catch low-gravity lateral drift and side-outlier risks.

Inputs:

- same as PoseAcceptNet;
- gravity conditioning.

Outputs:

- risk that `target_y_error_m`, drift, or post-hold failure will exceed the
  wall gate.

Use:

- stricter Moon gating;
- block high-course placements when lateral support is weak.

### WorldDeltaNet

Purpose: predict compact future outcome instead of a full future video.

Inputs:

- current compact state features;
- rock and pose features.

Outputs:

- target error;
- wall-depth error;
- residual velocity;
- height gain.

Use:

- WAM-style action ranking;
- sequence lookahead without full MuJoCo for every candidate.

### CandidatePoseRankNet

Purpose: learn local candidate-pose ordering from `candidate_pose_log.csv`.

Inputs:

- all sampled pose features;
- rock geometry;
- target slot and role.

Outputs:

- probability that this candidate is the best local pose among sampled poses.

Use:

- replace hand-coded candidate ordering;
- bootstrap a future direct pose proposal network.

### RepairNet

Purpose: choose fallback or chock actions after a slot fails.

Inputs:

- failed slot;
- best rejected pose metrics;
- local support context;
- remaining role-compatible stones.

Outputs:

- skip, retry, fallback, chock, or block-upper-course decision.

Use:

- reduce cascaded failures in Moon wall stacking.

## Planner Composition

For each target slot:

1. StoneFitNet ranks candidate stones for the slot.
2. CandidatePoseRankNet or a rule sampler proposes candidate poses.
3. PoseAcceptNet filters poor poses.
4. MoonDriftRiskNet applies gravity-specific lateral-risk filtering.
5. WorldDeltaNet predicts compact future state and ranks surviving actions.
6. MuJoCo verifies only the top few actions.
7. RepairNet decides whether to retry, use fallback, place chock, or block the
   upper slot.

The baseline keeps MuJoCo in the loop. The goal is to reduce random or brute
force trials, not to remove physics validation immediately.

## Current Baseline Script

The first baseline is pure NumPy because the current conda environment has
`numpy` and MuJoCo but not PyTorch or scikit-learn.

Script:

`D:\MoonStack\experiments\moon_rock_stack\scripts\train_modular_stack_models.py`

First smoke-test results:

`D:\MoonStack\experiments\moon_rock_stack\docs\MODULAR_SMALL_NETWORK_RESULTS_20260618.md`

The first candidate-pose ranker evaluation recovered the hand-coded selected
pose in top 3 for about two thirds of candidate groups, which is enough to use
the model as a candidate-pruning module but not enough to remove MuJoCo
verification.

Outputs:

- one `.npz` model file per small network;
- one JSON schema per model;
- `metrics.json`;
- `README.md`.

## Data Requirements

V0 placement logs are enough for:

- StoneFitNet;
- PoseAcceptNet;
- MoonDriftRiskNet;
- WorldDeltaNet.

V1 candidate-pose logs are required for:

- CandidatePoseRankNet;
- dense GraspNet-style pose proposal training.

The simulator now writes:

`candidate_pose_log.csv`

for new structured runs. Old runs do not contain this file, so
CandidatePoseRankNet will be weak until new dense candidate batches are
generated.

## Next Experiments

1. Generate larger candidate-pose logs with `--candidates 16` or higher.
2. Train modular NumPy baselines from tabular features.
3. Compare model-ranked top-1/top-3 candidates against the current hand-coded
   pose score.
4. If the small networks reduce MuJoCo checks per accepted placement, replace
   tabular rock features with learned mesh or point-cloud encoders.
5. Only after the modules work should we merge them into an end-to-end StackNet.
