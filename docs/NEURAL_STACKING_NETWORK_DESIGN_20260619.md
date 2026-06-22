# Neural Network Design for Lunar Dry-Stone Wall Stacking

Date: 2026-06-19

This note shifts the project from search-heavy dry stacking toward learnable
stone selection and pose prediction. The immediate goal is not to train a
large end-to-end model blindly. The current data are mostly structured tabular
logs with some RGB/depth captures, so the right next step is a modular neural
pipeline plus targeted data collection.

## Current Data State

Latest dataset:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_learning_dataset_wall_network_design_v1`

Summary:

| Item | Count |
| --- | ---: |
| Run directories scanned | 72 |
| Run-level examples | 514 |
| Placement examples | 8,665 |
| Candidate-pose examples | 10,986 |
| Assignment-candidate examples | 480 |

Placement split:

| Gravity | Examples | Success placements | Failure placements | Skipped |
| --- | ---: | ---: | ---: | ---: |
| Earth | 4,416 | 3,158 | 887 | 371 |
| Moon | 4,249 | 3,011 | 854 | 384 |

Important limitation:

- These are not yet image-first data.
- Dense candidate-pose labels exist only for runs with `candidate_pose_log.csv`.
- Current labels mostly imitate the hand-coded search/scorer, not independent
  physical ground truth.
- The current Python environment has no `torch`, no `sklearn`, and no `cv2`.
  Existing training is therefore NumPy MLP only.

## Current Modular Baseline

Latest model output:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_modular_networks_wall_design_v1`

Metrics:

| Model | Rows | Main metric | Interpretation |
| --- | ---: | --- | --- |
| `stone_fit_net` | 8,665 | F1 0.854 | Useful stone/slot compatibility signal. |
| `pose_accept_net` | 8,665 | F1 0.859 | Useful placement-acceptance signal. |
| `moon_drift_risk_net` | 8,665 | F1 0.578 | Has signal, but recall is too low for lunar safety. |
| `world_delta_net` | 2,152 | target-error MAE 0.219 m | Too weak; lacks local stack-state encoding. |
| `candidate_pose_rank_net` | 10,986 | binary F1 0.077 | Binary objective is wrong; use groupwise ranking. |

Groupwise candidate-pose evaluation:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_candidate_pose_ranker_eval_wall_design_v1`

| Metric | Value |
| --- | ---: |
| Candidate-pose rows | 10,986 |
| Candidate groups | 3,197 |
| Top-1 recovery | 0.358 |
| Top-3 recovery | 0.907 |
| Earth Top-3 | 0.903 |
| Moon Top-3 | 0.913 |

Target-specific warning:

| Target | Gravity | Groups | Top-1 | Top-3 |
| --- | --- | ---: | ---: | ---: |
| `single_face_wall_2course_v1` | Earth | 363 | 0.270 | 0.667 |
| `single_face_wall_2course_v1` | Moon | 363 | 0.251 | 0.664 |
| `single_face_wall_4course_v1` | Earth | 113 | 0.186 | 0.602 |
| `single_face_wall_4course_v1` | Moon | 24 | 0.292 | 0.792 |
| `single_face_wall_high_v1` | Earth | 930 | 0.362 | 1.000 |
| `single_face_wall_high_v1` | Moon | 806 | 0.427 | 1.000 |

The high-wall top-3 result is misleadingly easy. The urgent data gap is dense
2-course and 4-course wall candidate data, because those are the stages that
must become reliable before higher walls are meaningful.

## Literature-Aligned Model Routes

Relevant architecture families and why they matter:

| Route | Relevant references | What to borrow | Fit to current MoonStack data |
| --- | --- | --- | --- |
| 6DoF grasp ranking | GraspNet-1Billion, AnyGrasp-style grasp perception | Rank many grasp/pose hypotheses before physics verification | Good analogy for candidate-pose ranking; use groupwise top-K objective. |
| Pick-place spatial maps | Transporter Networks, CLIPort | Predict where to place an object from top-down observations | Good for top-view wall/slot maps once local height maps are exported. |
| 3D voxel transformer | PerAct | Tokenize 3D scene and predict 6DoF action | Good later, after dense RGB-D/voxel data exist. |
| Action sequence transformer | ACT, RT-1-style sequence models | Predict multi-step manipulation actions | Useful after we have many successful trajectories, not just one-step candidate logs. |
| Diffusion policy / DiT | Diffusion Policy, DiT | Generate continuous action distributions with multimodal uncertainty | Not appropriate yet; needs much larger, balanced trajectory data. |
| CNN / ResNet / U-Net | ResNet, U-Net | Encode depth/height/support maps | Good once per-slot local maps are exported. |
| Point cloud / mesh encoders | PointNet, PointNet++, MeshGraphNets | Encode irregular rock geometry directly | Strong fit; better than only hand-crafted rock features. |
| World model | graph/transformer dynamics models | Predict settled support, drift, and disturbance before MuJoCo | Best near-term "large model" target, but should predict compact physics features, not raw video. |

Primary paper links for follow-up:

- GraspNet-1Billion: <https://arxiv.org/abs/1912.13470>
- Transporter Networks: <https://arxiv.org/abs/2010.14406>
- CLIPort: <https://arxiv.org/abs/2109.12098>
- PerAct: <https://arxiv.org/abs/2209.05451>
- Diffusion Policy: <https://arxiv.org/abs/2303.04137>
- ACT: <https://arxiv.org/abs/2304.13705>
- RT-1: <https://arxiv.org/abs/2212.06817>
- ResNet: <https://arxiv.org/abs/1512.03385>
- U-Net: <https://arxiv.org/abs/1505.04597>
- Vision Transformer: <https://arxiv.org/abs/2010.11929>
- Diffusion Transformer: <https://arxiv.org/abs/2212.09748>
- PointNet: <https://arxiv.org/abs/1612.00593>
- PointNet++: <https://arxiv.org/abs/1706.02413>
- MeshGraphNets: <https://arxiv.org/abs/2010.03409>

## Recommended Architecture: Do Modular First

The near-term system should be a composed model, not a single end-to-end policy:

1. `StoneRoleNet`
   - Input: rock geometry features, mesh/point-cloud embedding, slot role,
     course, gravity, target type.
   - Output: compatibility score for base/middle/cap/tie/chock slots.
   - Current baseline: `stone_fit_net`.

2. `PoseRankNet`
   - Input: candidate pose, rock features, slot features, gravity, target type,
     pre-placement support summary.
   - Output: groupwise score among candidate poses.
   - Replace binary classification with pairwise/listwise ranking loss.
   - Operational use: reduce 8-16 pose candidates to top 3, then MuJoCo verifies.

3. `StabilityRiskNet`
   - Input: same as `PoseRankNet`, plus local support features.
   - Output: probability of drift, high residual velocity, support failure.
   - Needs focal/class-balanced loss because moon drift is safety-critical and
     relatively sparse.

4. `WorldDeltaNet-v2`
   - Input: compact stack state around the slot, not just scalar rock features.
   - Output: predicted target error, y-error, height gain, disturbance,
     velocity, and support-contact outcomes.
   - Architecture: graph neural network or small transformer over placed stones
     plus candidate stone. Do not start with raw video world modeling.

5. `RepairPolicyNet`
   - Input: failed slot, local hole/support map, available fallback stones.
   - Output: retry same slot, pick fallback, repair lower course, or skip.
   - This is important because high walls fail when the planner keeps building
     over a bad lower-course support state.

## When to Use Larger Models

### CNN / ResNet / U-Net

Use after exporting local height/depth maps:

- 64 x 64 or 96 x 96 top-down local height map centered on the target slot.
- Separate channels for current stack height, target-slot mask, occupied stones,
  support candidates, and candidate stone footprint.
- ResNet is enough for classification/ranking; U-Net is useful if predicting a
  dense placement-quality heatmap.

### PointNet / PointNet++ / Mesh Encoder

Use as soon as possible:

- Sample 512-2048 points per rock mesh.
- Encode geometry into learned embeddings.
- Concatenate with scalar features and slot tokens.

This directly addresses the current limitation that hand-crafted features miss
contact patch quality and local concavities.

### Transformer / Perceiver / Set Transformer

Use when each decision has many stones and a stack context:

- Tokens: candidate stone, placed stones within support radius, target slot,
  course tokens, gravity token.
- Output: stone score, pose score, risk score.
- This is the best "medium-size" model for the project because stacking is a
  set/sequence problem.

### Diffusion / DiT

Do not use immediately.

Diffusion becomes attractive only after:

- at least 50k-100k dense candidate-pose examples for wall targets;
- balanced positives, near misses, and hard failures;
- local depth/height maps or point-cloud state encodings;
- a reliable reward/acceptance definition independent of the old hand-coded
  scorer.

Then the policy could generate continuous 6DoF pose distributions conditioned
on slot and stack state.

### One Big End-to-End Model

Not recommended now. The dataset is too small and too heterogeneous. A large
model would learn shortcuts from target/strategy labels and fail when the wall
target changes.

Recommended future unified model:

`RockStackFormer`

- Rock point-cloud encoder.
- Local top-down support-map CNN/U-Net encoder.
- Placed-stone graph/transformer context encoder.
- Slot/target/gravity tokens.
- Multi-head outputs: stone role score, pose distribution, drift risk,
  world-delta prediction, and repair action.

## Data Collection Priorities

The next simulation batches should be designed for learning, not just success:

1. Dense candidate-pose logs for 2-course and 4-course walls.
   - `--candidates 12-16`
   - Earth and Moon
   - `statics_wall`, `risk_aware`, `support_first`
   - Keep all candidate logs, including rejected candidates.

2. Hard negatives.
   - Near target but high drift.
   - Good support overlap but high residual velocity.
   - Good local fit but destroys lower-course stones.

3. Local stack state exports.
   - Before/after top-down height map.
   - Target-slot mask.
   - Candidate footprint mask.
   - Support-contact count and support-balance error.

4. Rock geometry tensors.
   - Per-rock sampled point cloud.
   - Mesh face normals / bounding box / mass / volume.
   - Link to `rock_index` in every placement and candidate row.

5. Better splits.
   - Split by run directory, seed, and rock catalog.
   - Never mix the same rock catalog into train and test for final claims.

6. Gravity and friction sweeps.
   - Earth, Moon.
   - Friction values such as 0.75, 0.95, 1.15.
   - This prevents a model from only learning the current optimistic friction.

## Immediate Experiments

### Experiment A: Dense 2-course candidate collection

Purpose: make the foundation-wall stage reliable.

Target:

- `single_face_wall_2course_v1`

Suggested command shape:

```powershell
conda run -n moon-rock-stack python -m moon_rock_stack.run_structured_experiment `
  --rocks 240 --rock-profile single_face_wall --seed <seed> --clusters 10 `
  --trials 3 --targets single_face_wall_2course_v1 `
  --strategies statics_wall,risk_aware,support_first `
  --gravities earth,moon --candidates 12 `
  --steps-per-rock 360 --hold-steps 1600 --workers 4 `
  --output <batch_runs>\20260619_candidate_dense_2course_v1
```

### Experiment B: Dense 4-course candidate collection

Purpose: improve the weak `single_face_wall_4course_v1` pose ranker.

Target:

- `single_face_wall_4course_v1`

Use assignment plan and role fallback, but collect more candidate-pose rows.
Do not judge the run only by success. Candidate logs are the objective.

### Experiment C: Local map export

Add a post-processing script that reads final states and writes:

- `local_height_map_before.npy`
- `local_height_map_after.npy`
- `target_mask.npy`
- `candidate_footprint.npy`
- `support_graph.json`

This unlocks CNN/U-Net/Transformer training.

## Current Decision

Use modular small networks now:

- `StoneRoleNet`
- `PoseRankNet`
- `StabilityRiskNet`
- `WorldDeltaNet-v2`
- `RepairPolicyNet`

Do not train ResNet/U-Net/ViT/Diffusion yet. First collect local maps and point
cloud tensors. After that, the first deep architecture should be a hybrid:

`PointNet rock encoder + local height-map CNN + placed-stone transformer + ranking/risk heads`.

## 2026-06-19 Update: Dense Candidate And Groupwise Ranker

After the first version of this note, a dense 2-course candidate collection was
started:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_candidate_dense_2course_v1`

The command timed out, but it preserved useful data:

| Item | Count |
| --- | ---: |
| Completed result rows | 2 |
| Placement rows | 22 |
| Candidate-pose rows | 2,112 |
| Captured cases | 2 |

The dataset was rebuilt as:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_learning_dataset_wall_network_design_v2`

Updated size:

| Item | Count |
| --- | ---: |
| Run directories scanned | 73 |
| Run-level examples | 516 |
| Placement examples | 8,687 |
| Candidate-pose examples | 13,098 |
| Assignment-candidate examples | 480 |

The old binary `candidate_pose_rank_net` got worse on harder dense data:

| Metric | Value |
| --- | ---: |
| Candidate rows | 13,098 |
| Groups | 3,373 |
| Top-1 recovery | 0.332 |
| Top-3 recovery | 0.862 |

This confirmed that binary classification is not the right loss for candidate
pose selection.

A new NumPy groupwise softmax ranker was added:

`D:\MoonStack\experiments\moon_rock_stack\scripts\train_candidate_pose_group_ranker.py`

Best regularized output:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_candidate_pose_group_ranker_v1_regularized`

Metrics:

| Metric | Value |
| --- | ---: |
| Rankable groups | 3,373 |
| Test top-1 recovery | 0.404 |
| Test top-3 recovery | 0.881 |
| Train top-1 recovery | 0.602 |
| Train top-3 recovery | 0.944 |

Target details:

| Target | Gravity | Test groups | Top-1 | Top-3 |
| --- | --- | ---: | ---: | ---: |
| `single_face_wall_2course_v1` | Earth | 101 | 0.238 | 0.564 |
| `single_face_wall_2course_v1` | Moon | 76 | 0.316 | 0.697 |
| `single_face_wall_4course_v1` | Earth | 27 | 0.259 | 0.630 |
| `single_face_wall_4course_v1` | Moon | 7 | 0.571 | 0.714 |

Interpretation:

- Groupwise ranking is the correct direction.
- The regularized smaller ranker generalizes better than the wider ranker.
- 2-course Earth remains weak because the new dense data came from only one
  catalog/seed and made the test groups harder.
- The next data batch should add more independent 2-course and 4-course
  catalogs, not just more candidates from the same catalog.
- The model still uses only tabular features. Local support maps and rock point
  clouds are needed before a ResNet/U-Net/PointNet/Transformer model can make a
  meaningful jump.

## 2026-06-19 Update: Tensor Exports And PyTorch Baselines

The project now has mainstream deep-learning dependencies installed in the
`moon-rock-stack` conda environment:

| Package | Version |
| --- | --- |
| PyTorch | `2.11.0+cu128` |
| torchvision | `0.26.0+cu128` |
| torchaudio | `2.11.0+cu128` |
| scikit-learn | `1.9.0` |
| pandas | `3.0.3` |
| tensorboard | `2.20.0` |

CUDA validation:

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2080 Ti |
| VRAM | 11 GB class |
| PyTorch CUDA | 12.8 |
| Device used | `cuda` |

### Tensor datasets

Rock point clouds:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_rock_pointcloud_tensors_v1`

| Item | Value |
| --- | ---: |
| Run directories | 77 |
| Rocks | 11,910 |
| Points per rock | 512 |
| Arrays | `points[N,P,3]`, `normals[N,P,3]`, `rock_index[N]` |

Wall local support maps:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_local_support_maps_wall_candidate_v1`

| Item | Value |
| --- | ---: |
| Candidate examples | 11,418 |
| Shards | 6 |
| Map shape | `[8,64,64]` |
| Numeric features | 31 values + 31 missing masks |

Channels:

1. `height_before_m`
2. `support_occupancy`
3. `support_count_clipped`
4. `target_gaussian`
5. `candidate_footprint`
6. `candidate_height_m`
7. `gravity_ratio`
8. `course_ratio`

Support-state assumption:

The support map is reconstructed from earlier successful committed placements in
the same run/target/strategy/gravity/trial group. This is a deterministic
learning proxy, not a MuJoCo state rewind. It is good enough for first neural
rankers, but final policy validation must run in MuJoCo.

### PyTorch support-map CNN ranker

Script:

`D:\MoonStack\experiments\moon_rock_stack\scripts\train_torch_support_map_ranker.py`

Output:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_torch_support_map_cnn_ranker_wall_v2`

Architecture:

- CNN encoder over the 8-channel local support map.
- MLP over candidate pose, rock geometry, support metrics, and missing-value
  masks.
- Groupwise softmax imitation loss over candidates sharing run, target,
  strategy, gravity, trial, slot, and candidate rock.

Metrics:

| Metric | Value |
| --- | ---: |
| Candidate rows | 11,418 |
| Rankable groups | 2,793 |
| Train groups | 2,234 |
| Test groups | 559 |
| Test top-1 recovery | 0.639 |
| Test top-3 recovery | 0.948 |
| Train top-1 recovery | 0.722 |
| Train top-3 recovery | 0.966 |

By target/gravity:

| Target | Gravity | Groups | Top-1 | Top-3 |
| --- | --- | ---: | ---: | ---: |
| `single_face_wall_2course_v1` | Earth | 116 | 0.483 | 0.810 |
| `single_face_wall_2course_v1` | Moon | 63 | 0.556 | 0.921 |
| `single_face_wall_4course_v1` | Earth | 15 | 0.600 | 0.867 |
| `single_face_wall_4course_v1` | Moon | 3 | 0.333 | 1.000 |
| `single_face_wall_high_v1` | Earth | 191 | 0.702 | 1.000 |
| `single_face_wall_high_v1` | Moon | 169 | 0.710 | 1.000 |

Interpretation:

- The PyTorch CNN is now the main deep-learning pose-ranker baseline.
- It is slightly below the overfit NumPy pooled-map baseline on imitation
  top-k, but the train/test gap is smaller and the architecture can be extended
  to ResNet/U-Net/Transformer blocks.
- The next meaningful metric is closed-loop MuJoCo wall success when this model
  chooses candidate poses. Imitation top-k is only a proxy.

### PyTorch PointNet rock encoder

Script:

`D:\MoonStack\experiments\moon_rock_stack\scripts\train_torch_pointnet_rock_encoder.py`

Strict cluster-label output:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_torch_pointnet_rock_encoder_v2`

Family-label output:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_torch_pointnet_rock_encoder_family_v1`

Architecture:

- PointNet-style shared 1x1 MLP over point samples.
- Global max pooling for permutation-invariant rock embedding.
- Multi-task heads for generated `source_kind` and geometry cluster family.
- Input is xyz + sampled normals, shape `[512,6]`.

Family-label metrics:

| Metric | Value |
| --- | ---: |
| Rocks | 11,910 |
| Train rocks | 10,068 |
| Test rocks | 1,842 |
| Point count | 512 |
| Embedding dim | 256 |
| Source-kind test accuracy | 0.498 |
| Cluster-family test accuracy | 0.754 |

Interpretation:

- `cluster_family` is the more useful supervision target because suffixes such
  as `_2`/`_3` split one physical family into small generated subclusters.
- `source_kind` remains noisy because it is a generator provenance label, not a
  directly observable geometry label.
- The exported embeddings should become the rock-shape input for the next
  placement network, replacing or augmenting hand-written geometry scalars.
- External pretrained PointNet++/Point Transformer weights should be evaluated
  later only if license and domain transfer are clear. In-domain synthetic rock
  pretraining is currently more reliable than using ModelNet-style furniture or
  CAD priors directly.

### Hybrid support-map + PointNet ranker

The PointNet family embedding was joined into the support-map ranker by
`(run_name, candidate_rock_index)`.

Output:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_torch_support_map_pointnet_hybrid_ranker_wall_v1`

Metrics:

| Metric | Pure support-map CNN | Hybrid with PointNet embedding |
| --- | ---: | ---: |
| Candidate rows | 11,418 | 11,418 |
| Rankable groups | 2,793 | 2,793 |
| Numeric dim | 62 | 319 |
| Test top-1 recovery | 0.639 | 0.581 |
| Test top-3 recovery | 0.948 | 0.930 |
| Train top-1 recovery | 0.722 | 0.663 |
| Train top-3 recovery | 0.966 | 0.953 |

Interpretation:

- Naively concatenating a 256-D PointNet embedding reduced imitation ranking
  quality.
- This does not mean point clouds are useless. It means the current embedding
  was trained on generator/cluster labels, not on stacking affordance.
- The next PointNet use should be one of:
  - pretrain on geometric family, then fine-tune jointly with placement loss;
  - use a small gated fusion layer so the ranker can down-weight uncertain
    shape channels;
  - train an affordance-specific PointNet head to predict support role,
    contact patch quality, and drift risk.

## Immediate Next Experiments

1. Add a closed-loop MuJoCo evaluator that lets
   `support_map_cnn_ranker.pt` choose candidate poses, then compare against the
   hand-coded heuristic on Earth and lunar gravity.
2. Add an affordance-specific PointNet fine-tuning objective instead of direct
   frozen-embedding concatenation.
3. Collect more dense candidates specifically for
   `single_face_wall_4course_v1`, because current 4-course test groups are too
   few for reliable claims.
4. After closed-loop validation, try a stronger backbone: small ResNet over
   maps, U-Net support-field predictor, or a slot/placed-stone transformer.
