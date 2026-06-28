# 20260628 网络替代启发式 wall4 训练记录

日期: 2026-06-28  
目标: 提高 `single_face_wall_4course_v1` 月面严格成功率，并让小网络逐步替代启发式搜索。  
安全边界: 全部任务只新增目录、数据集、模型和日志，不删除、不覆盖历史实验。

## 1. 当前判断

4 层失败的主要瓶颈仍然不是“完全堆不起来”，而是:

```text
1. high_wall strict 任务在第 4 层 cap 前后仍有 no_feasible_pose。
2. encyclopedic_poly_train curriculum 可以达到 visible_courses=4/5，但 target RMSE 和 max error 太大。
3. 说明后续重点应是网络化候选排序、风险过滤、支撑连续性建模，而不是单纯继续增大随机尝试次数。
```

## 2. 新启动任务 A: 快速网络替代验证

用途: 快速验证更强网络参与是否能改善 4 层闭环。

脚本:

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260628_cmd_network_replacement_wall4_fast_v1\run.cmd
```

流水线输出:

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_network_replacement_wall4_fast_v1
```

数据:

```text
dataset: D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_policy_replacement_dataset_v1
tensor_dir: D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_policy_replacement_train_eval_v1_mujoco_depth_maps
candidate_pose_examples: 141,988
assignment_candidate_examples: 28,075
placement_examples: 2,515
```

网络替代设置:

```text
candidate-pose-top-k: 2
stone-fit-top-k: 8
candidate-pose-ranker-max-course: -1
stone-fit-ranker-max-course: -1
pose-risk-ranker-max-course: -1
pose-risk-weight: 0.65
```

解释:

- 旧设置常用 top-k=8/10 或只在前 3 层启用网络，本轮把候选位姿压缩到 top2。
- `max-course=-1` 表示网络覆盖所有层，包括第 4 层 cap。
- 启发式仍保留为候选生成器，但最终候选过滤更依赖网络。

评估:

```text
target: single_face_wall_4course_v1
gravity: moon
eval_trials: 2
eval_commit_best_rejected: false
strict success 可统计: yes
```

当前状态:

```text
已启动。
GPU 显存已上升到约 5.8-6.0 GB，说明 PyTorch 训练已经进入运行状态。
```

## 3. 新启动任务 B: 全量非 NASA-like 数据训练

用途: 用最新数据做更完整的 v2 训练，作为 fast v1 后的主力模型。

脚本:

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260628_cmd_network_replacement_wall4_v2\run.cmd
```

流水线输出:

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_network_replacement_wall4_v2
```

新数据集:

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_policy_replacement_dataset_v2_all_non_nasa
```

数据规模:

```text
run_dir_count: 294
excluded_nasa_like_runs: 9
placement_examples: 21,866
candidate_pose_examples: 833,417
assignment_candidate_examples: 140,496
```

NASA-like 处理:

```text
build_learning_dataset 默认排除 nasa_like* profile。
本轮确认排除了 9 个 nasa_like 测试 run。
NASA-like 仍然作为 held-out 泛化测试集，不进入训练。
```

网络替代设置:

```text
candidate-pose-top-k: 2
stone-fit-top-k: 8
candidate-pose-ranker-max-course: -1
stone-fit-ranker-max-course: -1
pose-risk-ranker-max-course: -1
pose-risk-weight: 0.65
```

当前状态:

```text
数据集已构建完成。
正在导出 MuJoCo depth/support map，之后会训练 StoneSlotNet、PoseRiskNet、SupportMapRanker、WallStateCritic。
```

## 4. 与上一轮的差别

| 项目 | 上一轮 c11 评估 | 本轮 fast/v2 |
|---|---:|---:|
| candidate-pose-top-k | 8 | 2 |
| stone-fit-top-k | 18 | 8 |
| 网络覆盖层级 | 主要到 course 3 | `-1`，全层 |
| pose-risk-weight | 0.45 | 0.65 |
| 目标 | 探索/严格混合 | 4 层 strict 成功率 |
| 数据集规模 | c11 局部数据 | fast: 141,988 candidates；v2: 833,417 candidates |

## 5. 后续判据

任务完成后必须比较:

```text
baseline/c11 4course strict success_rate
fast_v1 4course strict success_rate
v2 4course strict success_rate
shape_success_rate
visible_courses
stable_count
target_rmse_xy_m
target_max_xy_error_m
max_horizontal_drift_m
velocity_inf_norm
no_feasible_pose 按 role/course 的分布
```

如果 fast/v2 的 strict success 没提升，但 visible_courses 或 no_feasible_pose 改善，也仍有科学价值，因为说明网络在减少无效搜索，但还需要 wall-state critic 或 support continuity loss。

## 6. 当前资源状态

启动后观察:

```text
GPU memory: 约 5.8-6.0 GB
CPU: 已接近满载
available RAM: 约 34-53 GB，仍安全
```

因此本轮不再增加新任务，避免互相拖慢。
