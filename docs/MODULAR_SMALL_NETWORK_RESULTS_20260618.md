# Modular Small-Network Results

Date: 2026-06-18

This note records the first modular learning smoke test. It is not yet a final
policy. The goal was to verify that the project can train several small
networks from simulation logs instead of relying only on random or hand-coded
candidate search.

## Dataset

Latest dataset directory:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_learning_dataset_v2_modular_candidate_pose`

Summary:

| Item | Count |
| --- | ---: |
| Run directories scanned | 58 |
| Run examples | 473 |
| Placement examples | 7,909 |
| Candidate-pose examples | 306 |
| Assignment-candidate examples | 264 |

Important limitation:

- `placement_examples.csv` is large enough for first tabular networks.
- `candidate_pose_examples.csv` is only a smoke test because only new runs write
  `candidate_pose_log.csv`; old runs do not have dense candidate-pose labels.

## Trained Models

Output directory:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_modular_small_networks_v1_candidate_pose`

All models are pure NumPy two-layer MLPs. The current environment does not have
PyTorch or scikit-learn installed.

| Model | Rows | Metric |
| --- | ---: | --- |
| `stone_fit_net` | 7,909 | accuracy 0.789, precision 0.824, recall 0.896, F1 0.858 |
| `pose_accept_net` | 7,909 | accuracy 0.788, precision 0.833, recall 0.885, F1 0.858 |
| `moon_drift_risk_net` | 7,909 | accuracy 0.771, precision 0.640, recall 0.487, F1 0.553 |
| `world_delta_net` | 1,396 | target error MAE 0.313 m, wall-depth error MAE 0.198 m |
| `candidate_pose_rank_net` | 306 | accuracy 0.770, precision 1.000, recall 0.067, F1 0.125 |

## Interpretation

`StoneFitNet` and `PoseAcceptNet` already learn a moderate signal from geometry,
role, gravity, and proposed pose metadata. They are good enough to justify a
next experiment where they reduce candidate attempts before MuJoCo verification.

`MoonDriftRiskNet` has poor recall. This matches the current physical problem:
Moon failures are relatively sparse but important, and lateral drift needs more
targeted negative examples from high-course Moon wall runs.

`WorldDeltaNet` is too weak. It tries to predict target error, wall-depth error,
velocity, and height gain without a proper pre-placement stack state encoding.
It should be retrained after we export compact support maps, local height maps,
or point-cloud features.

`CandidatePoseRankNet` is no longer just a file-path smoke test, but it is still
too conservative: precision is high on the tiny test split, but recall is only
0.067. The next data collection run must produce thousands of candidate-pose
rows with more selected positives and harder near-miss negatives.

## Candidate-Pose Data Collected

Two small candidate-pose batches were added:

- `20260618_candidate_pose_single_face_wall4c_c8_minibatch`: 192 Earth
  candidate poses. The command timed out before Moon finished, but the Earth
  files were written and retained.
- `20260618_candidate_pose_single_face_wall4c_moon_c4_minibatch`: 96 Moon
  candidate poses. This produced a failed but useful low-gravity drift example.

## CandidatePoseRankNet Evaluation

Latest evaluation output:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_candidate_pose_ranker_eval_v2_high`

The evaluator groups candidate poses by run, gravity, slot, and candidate rock,
then asks whether the learned network recovers the current hand-coded
pose-search winner.

| Metric | Value |
| --- | ---: |
| Candidate-pose rows | 1,193 |
| Candidate groups | 476 |
| Network top-1 hit rate | 0.529 |
| Network top-3 hit rate | 0.964 |
| Earth top-1 / top-3 | 0.474 / 0.943 |
| Moon top-1 / top-3 | 0.590 / 0.987 |

Interpretation:

- The ranker is still not ready to replace search directly.
- After adding high-layer and ranker-guided logs, it can recover the current
  hand-coded selected pose in top 3 for most candidate groups.
- This suggests a realistic short-term use: use the network to reduce candidate
  poses from 8 or more down to 3, then run MuJoCo verification on those 3.
- The labels are still imitation labels from the current hand-coded scorer, not
  independent physical ground truth.

## Ranker-Guided Simulation Smoke

The ranker is now integrated into `run_structured_experiment.py`:

```powershell
--candidate-pose-ranker-dir <model_dir>
--candidate-pose-top-k 3
```

Smoke run:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_ranker_top3_wall_segment_smoke`

Result:

- target: `wall_segment_v1`
- candidates requested: 8
- candidate-pose top-K simulated: 3
- candidate-pose rows written: 27
- expected without pruning for 9 slots: 72

This confirms the network can prune candidate poses before MuJoCo verification.

High-wall ranker-guided run:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_high_single_face_wall_v1_earth_ranker_top3`

Result:

- target: `single_face_wall_high_v1`
- gravity: Earth
- visible courses: 8
- height: 0.300 m
- stable stones: 20 / 31
- RMSE: 0.253 m
- max drift: 0.257 m
- verdict: higher than the non-ranker fast run, but less stable and less
  wall-like.

Interpretation:

- `CandidatePoseRankNet` helps select taller candidate poses.
- It does not yet understand global wall stability.
- In `statics_wall`, there is still a stone-pool search around each slot. Pose
  top-K reduces pose trials per candidate stone, but does not yet reduce the
  stone pool.

## Online StoneFitNet Integration

`StoneFitNet` has now been integrated before the stone-pool loop in
`statics_wall`, `literature_wall`, and `literature_column`.

New CLI options:

```powershell
--stone-fit-ranker-dir <model_dir>
--stone-fit-top-k <k>
```

The online planner now supports two-stage neural pruning:

- `StoneFitNet`: rank candidate stones for a slot using geometry, target role,
  gravity, and dry-stacking role priors.
- `CandidatePoseRankNet`: rank generated poses for a candidate stone.
- MuJoCo: still performs the final physical settle/hold verification.

First neuralized high-wall/column result summary:

`D:\MoonStack\experiments\moon_rock_stack\docs\NEURALIZED_HIGH_STACKING_RESULTS_20260618.md`

Key observation:

- `StoneFitNet` should be used as a sorter, not a narrow hard gate. Stone top-K
  4 or 5 missed feasible high-wall placements. Stone top-K 8 restored better
  coverage while pose top-K 3 still reduced pose simulations.

## Next Experiment

1. Run a data-collection batch for `single_face_wall_4course_v1` under Earth and
   Moon with `--candidates 16` or `--candidates 32`.
2. Use the same rock catalog and assignment plan so that differences are from
   placement policy, not rock distribution.
3. Retrain `CandidatePoseRankNet` on the dense candidate-pose table.
4. Add an evaluator that compares:
   - hand-coded candidate score;
   - `PoseAcceptNet` ranking;
   - `PoseAcceptNet + MoonDriftRiskNet`;
   - `PoseAcceptNet + MoonDriftRiskNet + WorldDeltaNet`.
5. Report top-1/top-3 candidate success and MuJoCo checks saved per accepted
   placement.
