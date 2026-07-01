# MoonStack 数据生成与模型训练 Runbook

更新时间：2026-07-01  
适用仓库：`D:\MoonStack\experiments\moon_rock_stack`  
目标：让另一台拥有相同 `batch_runs` 数据的电脑可以继续进行月面干式堆叠数据生成、清洗、训练和评估。

## 0. 当前阶段目标

当前研究不是随机堆石头，而是构建“数据飞轮”：

1. 用 MuJoCo 自动生成 3 层、4 层单面墙 dry stacking 数据。
2. 记录成功和失败样本，尤其是候选位姿、支撑关系、释放扰动和失败机制。
3. 用清洗后的数据训练小网络，逐步替代启发式搜索。
4. 只把有指标证据的模型接回采样流程。
5. 用更高效的采样策略制造更多成功样本，再回流训练。

当前短期目标：

- 稳定提高 `single_face_wall_3course_v1` 成功率。
- 用 3 层高质量数据训练 PoseRiskNet / SupportMapCNNRanker。
- 继续推进 `single_face_wall_4course_v1`，比较旧 PoseRiskNet 与 clean PoseRiskNet。

当前长期目标：

- 从 3 层到 4-5 层，再逐步推进 8-10 层高墙。
- 训练“石头几何 + 墙体观测”的网络，而不是只靠石头几何或随机搜索。

## 1. 目录和环境约定

默认路径：

```powershell
$ROOT = "D:\MoonStack\experiments\moon_rock_stack"
$BATCH = "$ROOT\batch_runs"
$PY = "C:\Users\all\miniconda3\envs\moon-rock-stack\python.exe"
Set-Location $ROOT
```

如果另一台电脑路径不同，优先保持相同目录结构；否则把上面的 `$ROOT/$BATCH/$PY` 改成本机路径。

当前本机验证过的环境：

- Python: 3.11.15
- PyTorch: 2.11.0+cu128
- CUDA available: true
- GPU: NVIDIA GeForce RTX 2080 Ti
- MuJoCo: 3.3.6
- NumPy: 2.4.6

最小环境检查：

```powershell
@'
import sys
print("python", sys.version)
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import mujoco
print("mujoco", getattr(mujoco, "__version__", "unknown"))
import numpy as np
print("numpy", np.__version__)
'@ | $PY -
```

Git 根目录是：

```powershell
D:\MoonStack\experiments\moon_rock_stack
```

不是 `D:\MoonStack`。做 git 操作时要在 `$ROOT` 下。

## 2. 强约束

严禁删除实验数据。

执行原则：

- 只创建新目录。
- 不覆盖旧 `batch_runs`。
- 不清理、不移动、不重命名旧实验。
- 新任务必须使用新的 `session` 或新的输出目录。
- 如果输出目录已存在，训练/清洗脚本通常会自动加时间戳；仍建议手动换新名字。
- NASA/NASAlike 数据只作为 held-out 测试集，不进入训练。

## 3. 数据表含义

raw run 目录中常见文件：

- `results.csv`: trial 级结果，一个 trial 一行。
- `placement_log.csv`: 已提交放置、跳过槽位、失败槽位记录。
- `candidate_pose_log.csv`: 候选位姿级别数据，训练 PoseRiskNet / pose ranker 的核心。
- `failure_cases.csv`: 失败案例摘要。
- `structured_progress.csv`: 运行中的进度事件。
- `features.csv`: 石头几何特征。
- `target_slots_*.csv`: 目标墙体槽位。

learning dataset 目录中常见文件：

- `run_examples.csv`: trial 级训练/统计样本。
- `placement_examples.csv`: placement 级训练/统计样本。
- `candidate_pose_examples.csv`: 候选位姿训练样本。
- `assignment_candidate_examples.csv`: 石头-槽位候选样本，训练 StoneSlotNet。
- `dataset_summary.json`: 数据量、成功率、来源统计。

当前最有用的数据粒度：

- PoseRiskNet: `candidate_pose_examples.csv`
- SupportMapCNNRanker: MuJoCo/top/front depth tensor，由 `candidate_pose_examples.csv` 导出。
- StoneSlotNet: `assignment_candidate_examples.csv`，但当前仅靠石头几何 + 槽位信息效果不足。

## 4. 当前关键数据集

### 4.1 clean 3-4 月面墙体数据

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_clean_policy_moon_wall_3to4_strict_v1
```

规模：

- run_examples: 57
- placement_examples: 920
- candidate_pose_examples: 35779
- assignment_candidate_examples: 20109

用途：

- 当前最可信的 3-4 层月面墙体 clean slice。
- 比 full-history 混训更可靠。
- 训练得到的 clean PoseRiskNet 仍是当前 4 层线上推荐版本。

### 4.2 controller 3 层高成功率快照

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260701_controller_next_3course_snapshot_learning_dataset_v1
```

规模：

- run_examples: 8
- placement_examples: 120
- candidate_pose_examples: 8640
- assignment_candidate_examples: 2880
- strict success: 4/8
- shape success: 5/8

清洗后：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260701_controller_next_3course_clean_policy_v1
```

规模：

- run_examples: 8
- placement_examples: 95
- candidate_pose_examples: 6378
- assignment_candidate_examples: 2880

用途：

- 当前 3 层成功样本效率最高的数据源。
- 单独训练模型会过拟合 controller 分布，不能直接推广。
- 更适合做 support-map / 墙体观测模型增量训练。

### 4.3 clean 3-4 + controller 合并数据

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260701_clean_policy_moon_wall_3to4_plus_controller_v1
```

规模：

- run_examples: 65
- placement_examples: 1015
- candidate_pose_examples: 42157
- assignment_candidate_examples: 22989

用途：

- 用于对比“加入 controller 高成功率数据后，模型是否真的提升”。
- 已训练 PoseRiskNet，但当前指标未超过上一版 clean 3-4 PoseRiskNet。

## 5. 当前模型结论

### 5.1 PoseRiskNet

作用：

- 输入：重力、目标名、策略、层级、角色、候选位姿、候选编号、石头几何、石头类别。
- 输出：候选位姿在提交前的安全概率。
- 不使用后验字段作为输入。
- 后验指标只用于监督标签。

当前线上推荐版本：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_clean_policy_moon_wall_3to4_pose_risk_net_v1
```

指标：

- row_count: 35779
- input_dim: 126
- hidden layers: 768 / 384 / 192 / 96
- parameters: 485377
- test F1: 0.8020
- group top1 safe: 0.6579
- group top3 safe: 0.9786

controller-only 版本：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260701_controller_next_3course_pose_risk_net_v1
```

指标：

- row_count: 6378
- input_dim: 110
- hidden layers: 512 / 256 / 128
- parameters: 221185
- test F1: 0.6685
- group top1 safe: 0.6998
- group top3 safe: 1.0

判断：

- controller-only 指标高，但只有 3 个 run，泛化风险高。
- 不建议直接替换线上模型。

clean 3-4 + controller 版本：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260701_clean_policy_3to4_plus_controller_pose_risk_net_v1
```

指标：

- row_count: 42157
- input_dim: 127
- hidden layers: 768 / 384 / 192 / 96
- parameters: 486145
- test F1: 0.7345
- group top1 safe: 0.6232
- group top3 safe: 0.9567

判断：

- 训练成功，但未超过上一版 clean 3-4 PoseRiskNet。
- 暂不提升为线上默认。

### 5.2 SupportMapCNNRanker

作用：

- 输入：局部 top/front depth tensor + 数值特征。
- map shape: `[10, 64, 64]`
- 输出：候选位姿组内排序分数。
- 目标是用墙体观测替代一部分启发式 pose search。

已有 baseline：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_data_flywheel_controller_v1_support_map_ranker
```

指标：

- tensor row_count: 6744
- rankable_group_count: 2048
- hidden: 384
- parameters: 181537
- target_mode: structure_aware
- exclude_postsim_features: true
- test top1: 0.4144
- test top3: 1.0

正在运行的增量任务：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260701_cmd_controller_supportmap_increment_v1
```

输出：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260701_controller_next_3course_support_maps_v1
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260701_controller_plus_supportmap_ranker_v1
```

当前状态：

- 正在导出 controller support maps。
- 导出完成后自动训练增量 support-map ranker。

### 5.3 StoneSlotNet

作用：

- 输入：目标槽位、层级、角色、石头几何、石头类别。
- 输出：候选石头适合该槽位的概率。

当前判断：

- 只靠石头几何 + 槽位，泛化不足。
- clean retrain 后没有提升。
- 暂时不作为主替换方向。
- 后续要改成“石头几何 + 当前墙体观测”的匹配网络。

## 6. 数据生成：3 层高效采样线

当前最有效的 3 层数据生成策略是 controller support-map 采样线。

参考已运行 session：

```text
20260630_data_flywheel_controller_v1_next_3course_collect
```

已完成快照：

- 8 trial
- strict success: 4
- shape success: 5
- candidate_pose_examples: 8640

成功效率：

- deepmlp exploit v2: 约 11340 candidate poses / strict success
- controller next 3course: 约 2160 candidate poses / strict success
- controller clean policy: 约 1594.5 candidate poses / strict success

结论：

- controller 线比 deepmlp exploit v2 的成功数据生成效率高约 5 倍。
- 继续采 3 层成功数据时，优先用 controller/support-map 线。

新机器启动一个新的 3 层采样 session 示例：

```powershell
$ROOT = "D:\MoonStack\experiments\moon_rock_stack"
$BATCH = "$ROOT\batch_runs"
$PY = "C:\Users\all\miniconda3\envs\moon-rock-stack\python.exe"
Set-Location $ROOT

& $PY -m scripts.run_wall_data_flywheel `
  --session 20260701_remote_controller_3course_collect_v1 `
  --seed 207011960 `
  --collect-batches 3 `
  --collect-mode exploit `
  --parallel-data-jobs 2 `
  --mujoco-workers 1 `
  --rocks 260 `
  --rock-profile high_wall `
  --clusters 10 `
  --trials 3 `
  --targets single_face_wall_3course_v1 `
  --gravities moon `
  --candidates 16 `
  --steps-per-rock 420 `
  --hold-steps 1800 `
  --candidate-probe-steps 40 `
  --low-release-search `
  --release-search-step-m 0.003 `
  --release-extra-clearance-m 0.002 `
  --base-support-prior `
  --base-support-prior-weight 1.6 `
  --base-continuity-prior `
  --base-continuity-prior-weight 0.55 `
  --stone-fit-top-k 24 `
  --candidate-pose-top-k 3 `
  --pose-risk-weight 0.25 `
  --collect-commit-best-rejected `
  --skip-training `
  --skip-capture `
  --require-new-data `
  --dataset-target-contains single_face_wall `
  --exploit-stone-ranker-dir "$BATCH\20260630_positive_mining_3course_moon_v1_deep_mlp_train_stone_slot_net_deep" `
  --exploit-pose-ranker-dir "$BATCH\20260630_data_flywheel_controller_v1_support_map_ranker" `
  --exploit-pose-risk-ranker-dir "$BATCH\20260630_positive_mining_3course_moon_v1_deep_mlp_train_pose_risk_net_deep"
```

注意：

- 新机器要换新 `--session` 和 `--seed`，避免和本机目录冲突。
- 这条线用已有 support-map ranker 做 pose ranking。
- 如果另一台机器显存较小，可以把 `--parallel-data-jobs 2` 改成 `1`。

## 7. 数据生成：4 层对照线

当前 4 层线还在运行，尚未完成 `results.csv`。

旧 PoseRiskNet gated：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_fourcourse_lowrelease_deepmlp_probe_v1_collect_exploit_00_seed206305400
```

clean PoseRiskNet all courses：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_fourcourse_clean_pose_risk_probe_v1_collect_exploit_00_seed206305900
```

比较监控器：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260701_cmd_4course_compare_when_ready_v1
```

比较输出：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260701_4course_pose_risk_comparison_v1
```

另一台机器如果要继续跑 4 层，不建议复制正在跑的同名 session。新建 session：

```powershell
$ROOT = "D:\MoonStack\experiments\moon_rock_stack"
$BATCH = "$ROOT\batch_runs"
$PY = "C:\Users\all\miniconda3\envs\moon-rock-stack\python.exe"
Set-Location $ROOT

& $PY -m scripts.run_wall_data_flywheel `
  --session 20260701_remote_fourcourse_clean_pose_risk_v1 `
  --seed 207015900 `
  --collect-batches 2 `
  --collect-mode exploit `
  --parallel-data-jobs 1 `
  --mujoco-workers 1 `
  --rocks 520 `
  --rock-profile high_wall `
  --clusters 12 `
  --trials 2 `
  --targets single_face_wall_4course_v1 `
  --gravities moon `
  --candidates 32 `
  --steps-per-rock 460 `
  --hold-steps 2200 `
  --candidate-probe-steps 48 `
  --low-release-search `
  --release-search-step-m 0.0025 `
  --release-extra-clearance-m 0.0015 `
  --base-support-prior `
  --base-support-prior-weight 2.25 `
  --base-continuity-prior `
  --base-continuity-prior-weight 0.85 `
  --stone-fit-top-k 48 `
  --candidate-pose-top-k 5 `
  --pose-risk-weight 0.50 `
  --collect-commit-best-rejected `
  --skip-training `
  --skip-capture `
  --require-new-data `
  --dataset-target-contains single_face_wall `
  --stone-fit-ranker-max-course 2 `
  --candidate-pose-ranker-max-course 2 `
  --pose-risk-ranker-max-course -1 `
  --exploit-stone-ranker-dir "$BATCH\20260630_positive_mining_3course_moon_v1_deep_mlp_train_stone_slot_net_deep" `
  --exploit-pose-ranker-dir "$BATCH\20260629_highwall_return_profile_v2_pose_ranker_structure" `
  --exploit-pose-risk-ranker-dir "$BATCH\20260630_clean_policy_moon_wall_3to4_pose_risk_net_v1"
```

4 层重点观察：

- `success`
- `shape_success`
- `stable_count`
- `failure_count`
- `stack_height_m`
- `target_rmse_xy_m`
- `target_max_xy_error_m`
- 失败机制是否集中于 `no_feasible_pose`、`upper_contact_too_few`、`release_disturbance_excessive`

## 8. 从 raw run 构建 learning dataset

用具体 run 目录构建：

```powershell
$ROOT = "D:\MoonStack\experiments\moon_rock_stack"
$BATCH = "$ROOT\batch_runs"
$PY = "C:\Users\all\miniconda3\envs\moon-rock-stack\python.exe"
Set-Location $ROOT

& $PY -m scripts.build_learning_dataset `
  --batch-root $BATCH `
  --output "$BATCH\20260701_remote_controller_snapshot_learning_dataset_v1" `
  --run "$BATCH\20260701_remote_controller_3course_collect_v1_collect_exploit_00_seed207011960" `
  --run "$BATCH\20260701_remote_controller_3course_collect_v1_collect_exploit_01_seed207012097"
```

如果某些 worker 还没完成，已经有 `results.csv` 的 run 可以先构建快照版；未完成的 run 后续补 v2。

## 9. 清洗 policy dataset

当前可信清洗规则：

- target: `single_face_wall_3course_v1` / `single_face_wall_4course_v1`
- strategy: `statics_wall`
- gravity: `moon`
- role: `base/middle/cap`
- `target_error_xy_m <= 0.80`
- `abs(target_y_error_m) <= 0.35`
- `placed_disturbance_xy_m <= 0.50`
- `velocity_inf_norm_after_place <= 5.0`
- `rock_volume >= 0.00005`

3 层清洗示例：

```powershell
& $PY -m scripts.clean_policy_dataset `
  --dataset "$BATCH\20260701_remote_controller_snapshot_learning_dataset_v1" `
  --output "$BATCH\20260701_remote_controller_clean_policy_v1" `
  --target single_face_wall_3course_v1 `
  --strategy statics_wall `
  --gravity moon `
  --role base `
  --role middle `
  --role cap `
  --max-target-error-m 0.80 `
  --max-abs-y-error-m 0.35 `
  --max-disturbance-m 0.50 `
  --max-velocity 5.0 `
  --min-rock-volume 0.00005
```

3-4 层清洗示例：

```powershell
& $PY -m scripts.clean_policy_dataset `
  --dataset "$BATCH\20260701_some_merged_learning_dataset_v1" `
  --output "$BATCH\20260701_remote_clean_policy_moon_wall_3to4_v1" `
  --target single_face_wall_3course_v1 `
  --target single_face_wall_4course_v1 `
  --strategy statics_wall `
  --gravity moon `
  --role base `
  --role middle `
  --role cap `
  --max-target-error-m 0.80 `
  --max-abs-y-error-m 0.35 `
  --max-disturbance-m 0.50 `
  --max-velocity 5.0 `
  --min-rock-volume 0.00005
```

## 10. 合并清洗数据集

新增脚本：

```text
D:\MoonStack\experiments\moon_rock_stack\scripts\merge_policy_datasets.py
```

作用：

- 合并多个 clean policy dataset。
- 按 `example_id` 或 run/slot/candidate key 去重。
- 保留 `merged_source_dataset`。
- 只写新目录，不修改旧数据。

示例：

```powershell
& $PY -m scripts.merge_policy_datasets `
  --dataset "$BATCH\20260630_clean_policy_moon_wall_3to4_strict_v1" `
  --dataset "$BATCH\20260701_remote_controller_clean_policy_v1" `
  --output "$BATCH\20260701_remote_clean_policy_3to4_plus_controller_v1"
```

## 11. 训练 PoseRiskNet

推荐训练命令：

```powershell
& $PY -m scripts.train_torch_pose_risk_net `
  --dataset "$BATCH\20260701_remote_clean_policy_3to4_plus_controller_v1" `
  --output "$BATCH\20260701_remote_pose_risk_net_v1" `
  --target-contains single_face_wall `
  --epochs 220 `
  --batch-size 512 `
  --hidden 768 `
  --hidden-layers 768,384,192,96 `
  --activation silu `
  --dropout 0.15 `
  --lr 0.0008 `
  --weight-decay 0.0003 `
  --split-by-run `
  --device cuda `
  --candidate-metric-labels
```

训练后看：

```powershell
Get-Content -Raw "$BATCH\20260701_remote_pose_risk_net_v1\pose_risk_net_metrics.json"
```

是否提升，主要看：

- `test_group.top1_safe_rate`
- `test_group.top3_safe_rate`
- `test.f1`
- 是否在 held-out run 上提升，而不是只在 train 上提升。

当前准入标准：

- 若不能超过 `20260630_clean_policy_moon_wall_3to4_pose_risk_net_v1` 的 top1/top3，暂不替换线上模型。
- 当前参考 top1/top3：0.6579 / 0.9786。

## 12. 导出 support maps

导出 MuJoCo top/front depth tensor：

```powershell
& $PY -m scripts.export_mujoco_depth_observation_maps `
  --dataset "$BATCH\20260701_remote_controller_clean_policy_v1" `
  --output "$BATCH\20260701_remote_controller_support_maps_v1" `
  --source candidate `
  --grid-size 64 `
  --window-m 0.9 `
  --front-height-m 0.6 `
  --shard-size 1000 `
  --max-groups 1800 `
  --sample-mode candidate-groups `
  --sample-seed 207010930 `
  --dtype float16 `
  --target-contains single_face_wall `
  --strategy-contains statics_wall
```

输出目录会包含：

- `support_maps_*.npz`
- `support_maps_*.csv`
- `examples_index.csv`
- `summary.json`

## 13. 训练 SupportMapCNNRanker

推荐训练命令：

```powershell
& $PY -m scripts.train_torch_support_map_ranker `
  --tensor-dir "$BATCH\20260630_data_flywheel_controller_v1_support_maps" `
  --extra-tensor-dir "$BATCH\20260701_remote_controller_support_maps_v1" `
  --output "$BATCH\20260701_remote_controller_plus_supportmap_ranker_v1" `
  --epochs 150 `
  --batch-size 160 `
  --hidden 448 `
  --dropout 0.18 `
  --lr 0.0007 `
  --weight-decay 0.0004 `
  --split-by-run `
  --device cuda `
  --target-mode structure_aware `
  --quality-temperature 35.0 `
  --exclude-postsim-features `
  --group-role-weight middle=1.15 `
  --group-role-weight cap=1.25
```

训练后看：

```powershell
Get-Content -Raw "$BATCH\20260701_remote_controller_plus_supportmap_ranker_v1\metrics.json"
```

重点指标：

- `test_top1_hit_rate`
- `test_top3_hit_rate`
- `test_mean_top1_quality_regret`
- `test_mean_top3_quality_regret`

当前 baseline：

- top1: 0.4144
- top3: 1.0
- top1 quality regret: 10.4989
- top3 quality regret: 0.0

如果新 ranker top1 提高，且 regret 下降，可以接入后续 controller 采样。

## 14. 失败机制标注

全历史机制标注脚本：

```text
D:\MoonStack\experiments\moon_rock_stack\scripts\enrich_failure_mechanisms.py
```

输出过的全历史机制数据：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_all_history_enriched_failure_mechanisms_v1
```

机制标签：

- `mechanism_bottom_support_insufficient`
- `mechanism_upper_contact_too_few`
- `mechanism_release_disturbance_excessive`
- `mechanism_geometry_mismatch`
- `mechanism_neighbor_gap_too_large`
- `mechanism_target_miss`
- `mechanism_no_feasible_pose`
- `mechanism_post_hold_drift`
- `mechanism_low_or_fallen`
- `failure_mechanism_primary`

重要原则：

- 这些是后验弱标签。
- 可以做训练监督、统计、辅助 loss。
- 不能作为推理输入。

4 层主要失败机制：

- `no_feasible_pose`
- `upper_contact_too_few`
- `release_disturbance_excessive`

## 15. 监控活跃任务

查看正在跑的 MuJoCo / 训练 / 导出任务：

```powershell
Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -eq "python.exe" -and
    ($_.CommandLine -match "run_structured_experiment|run_wall_data_flywheel|export_mujoco_depth|train_torch|4course_compare")
  } |
  Select-Object ProcessId,@{N="MB";E={[math]::Round($_.WorkingSetSize/1MB,0)}},CommandLine |
  Sort-Object MB -Descending |
  Format-List
```

查看 GPU：

```powershell
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
```

查看某个 run 是否完成：

```powershell
$RUN = "$BATCH\20260701_remote_controller_3course_collect_v1_collect_exploit_00_seed207011960"
Test-Path "$RUN\results.csv"
if (Test-Path "$RUN\results.csv") {
  Import-Csv "$RUN\results.csv" |
    Select-Object target_name,gravity,trial,success,shape_success,stable_count,failure_count,stack_height_m,target_rmse_xy_m |
    Format-Table -AutoSize
}
```

查看进行到哪个 slot：

```powershell
Get-Content -Tail 20 "$RUN\structured_progress.csv"
```

## 16. 成功效率统计方法

一个策略线的成功效率建议统计：

- `trials`
- `success_trials`
- `shape_success_trials`
- `placement_rows`
- `candidate_pose_rows`
- `failure_rows`
- `success_rate = success_trials / trials`
- `shape_success_rate = shape_success_trials / trials`
- `candidate_pose_per_success = candidate_pose_rows / success_trials`
- `placement_per_success = placement_rows / success_trials`

当前效率表：

```text
D:\MoonStack\experiments\moon_rock_stack\docs\progress_reports\20260701_controller_dataflywheel_update_v1\efficiency_summary.csv
```

当前关键结论：

- controller 线比 deepmlp exploit v2 更适合制造 3 层成功样本。
- deepmlp exploit v2 更适合制造负样本。
- 4 层还没有完整结果，不能提前判断 clean PoseRiskNet 是否真实提升。

## 17. 当前正在运行的本机任务

截至本文档创建时，本机仍有以下任务运行：

1. 4 层旧 PoseRiskNet 对照线。
2. 4 层 clean PoseRiskNet 对照线。
3. 4 层自动比较监控器。
4. controller 第三条 3 层 worker 剩余 trial。
5. success harvest worker。
6. controller support-map 增量导出/训练流水线。

另一台机器可以并行工作，但要使用新的 session 名和 seed，避免输出目录冲突。

## 18. 推荐下一步

优先级从高到低：

1. 等 `20260701_4course_pose_risk_comparison_v1` 自动生成最终对比结果。
2. 等 `20260701_controller_plus_supportmap_ranker_v1` 输出 support-map 增量训练指标。
3. 如果 support-map ranker 有提升，用它启动新一轮 3 层 controller 采样。
4. controller 第三条 worker 完成后，重新构建 `v2` 快照数据集。
5. 新样本回流：build learning dataset -> clean policy -> merge -> train PoseRiskNet / SupportMapCNNRanker。
6. 只有模型在 held-out run 上提升，才接入 4 层或更高层采样。

## 19. 最短可执行流程

如果另一台机器只想马上开始工作，按这个顺序：

```powershell
$ROOT = "D:\MoonStack\experiments\moon_rock_stack"
$BATCH = "$ROOT\batch_runs"
$PY = "C:\Users\all\miniconda3\envs\moon-rock-stack\python.exe"
Set-Location $ROOT
```

1. 先确认环境：

```powershell
& $PY -c "import torch,mujoco; print(torch.__version__, torch.cuda.is_available(), mujoco.__version__)"
```

2. 启动新的 3 层 controller 采样，使用第 6 节命令，换新 `session/seed`。

3. 等至少一个 run 写出 `results.csv` 后，构建 learning dataset，使用第 8 节命令。

4. 清洗数据，使用第 9 节命令。

5. 导出 support maps，使用第 12 节命令。

6. 训练 support-map ranker，使用第 13 节命令。

7. 如果 ranker 指标比 baseline 提升，再接回下一轮数据生成。

## 20. 不建议做的事

- 不要把 full-history 混训结果当作默认模型。
- 不要继续只加宽 StoneSlotNet 来解决石头选择问题。
- 不要把 `mechanism_*` 后验标签作为推理输入。
- 不要在 4 层未完成 `results.csv` 时宣称成功或失败。
- 不要删除旧 run、旧 dataset、旧 tensor、旧模型。
