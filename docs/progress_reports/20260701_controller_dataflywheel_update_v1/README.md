# 2026-07-01 Controller 数据飞轮更新

本轮执行目标有三项：

1. 等4层两条对照线写出 `results.csv` 后自动比较旧 PoseRiskNet 与 clean PoseRiskNet。
2. 把 controller 这批成功率较高的3层数据整理进清洗数据集。
3. 用新增数据继续训练 PoseRiskNet / support-map ranker，并跟踪数据规模和成功数据生成效率。

## 当前结论

1. controller 3层采样是当前最有效的成功样本来源：8个已完成 trial 中严格成功4个，结构成功5个。
2. controller clean 数据单独训练出的 PoseRiskNet 在本分布上 top1 safe = 0.6998、top3 safe = 1.0，但数据只有3个 run，不能直接推广。
3. 将 controller clean 数据合入原 clean 3-4 数据后，增强版 PoseRiskNet 的 top1/top3 = 0.6232/0.9567，未超过上一版 clean 3-4 PoseRiskNet 的 0.6579/0.9786。
4. 因此 PoseRiskNet 暂时不替换线上4层 clean 版本；controller 数据更适合用于“成功经验统计”和 support-map/墙体观测网络增量训练。
5. support-map 增量导出和训练已启动，等待 `20260701_controller_plus_supportmap_ranker_v1` 输出指标。

## 数据规模

昨晚已完成 raw trial 汇总：

- 完整 trial: 18
- 严格成功 trial: 5
- 结构成功 trial: 6
- placement rows: 270
- candidate pose rows: 23280
- failure case rows: 42

controller 快照数据集：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260701_controller_next_3course_snapshot_learning_dataset_v1`

- run_examples: 8
- placement_examples: 120
- candidate_pose_examples: 8640
- assignment_candidate_examples: 2880
- strict success: 4/8
- shape success: 5/8

controller 清洗数据集：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260701_controller_next_3course_clean_policy_v1`

- run_examples: 8
- placement_examples: 95
- candidate_pose_examples: 6378
- assignment_candidate_examples: 2880
- 被过滤 placement outlier: 25
- 被过滤 candidate pose outlier: 2262

clean 3-4 + controller 合并数据集：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260701_clean_policy_moon_wall_3to4_plus_controller_v1`

- run_examples: 65
- placement_examples: 1015
- candidate_pose_examples: 42157
- assignment_candidate_examples: 22989

## 成功生成效率

完整表格：

`D:\MoonStack\experiments\moon_rock_stack\docs\progress_reports\20260701_controller_dataflywheel_update_v1\efficiency_summary.csv`

关键值：

| 数据线 | trial | strict success | shape success | candidate pose / strict success | placement / strict success |
|---|---:|---:|---:|---:|---:|
| deepmlp exploit v2 | 9 | 1 | 1 | 11340 | 135 |
| controller next 3course | 8 | 4 | 5 | 2160 | 30 |
| overnight raw total | 18 | 5 | 6 | 4656 | 54 |
| controller clean policy | 8 | 4 | 5 | 1594.5 | 23.75 |

结论：controller 线的正样本效率明显更高。它每产生一个严格成功 trial 约需 2160 条候选位姿，而 deepmlp exploit v2 约需 11340 条，是约 5.25 倍的差距。

## PoseRiskNet 训练

controller-only PoseRiskNet：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260701_controller_next_3course_pose_risk_net_v1`

- row_count: 6378
- input_dim: 110
- hidden layers: 512 / 256 / 128
- parameter_count: 221185
- test F1: 0.6685
- group top1 safe: 0.6998
- group top3 safe: 1.0

clean 3-4 + controller PoseRiskNet：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260701_clean_policy_3to4_plus_controller_pose_risk_net_v1`

- row_count: 42157
- input_dim: 127
- hidden layers: 768 / 384 / 192 / 96
- parameter_count: 486145
- test F1: 0.7345
- group top1 safe: 0.6232
- group top3 safe: 0.9567

判断：增强版训练成功，但还没有超过上一版 clean 3-4 PoseRiskNet。因此当前4层对照线仍然保留原 clean 3-4 PoseRiskNet，等真实4层结果再决定是否替换。

## Support-Map 增量任务

异步任务：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260701_cmd_controller_supportmap_increment_v1`

将输出：

- 新 controller support maps: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260701_controller_next_3course_support_maps_v1`
- 增量 support-map ranker: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260701_controller_plus_supportmap_ranker_v1`

策略：

- 使用 controller clean 数据导出 MuJoCo-rendered top/front depth tensor。
- 与已有 `20260630_data_flywheel_controller_v1_support_maps` 合并训练。
- 使用 `--exclude-postsim-features`，保证后验仿真指标不进入推理输入。
- 使用 `structure_aware` 目标，强调墙体结构质量而不是简单复刻候选选择。

## 4层对照线

自动比较监控器：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260701_cmd_4course_compare_when_ready_v1`

比较报告输出：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260701_4course_pose_risk_comparison_v1`

当前状态：

- old PoseRiskNet gated: 还没有 `results.csv`，已推进到 slot 18 / course 2。
- clean PoseRiskNet all courses: 还没有 `results.csv`，已推进到 slot 15 / course 2。

监控器会每60秒检查一次；两边都有结果后自动写 `comparison_summary.json` 和 `README.md`。

## 下一步

1. 等4层两条线完成第一个 trial 后，读取自动比较报告。
2. 等 support-map 增量任务完成后，比较新旧 support-map ranker 的 top1/top3 和 quality regret。
3. 如果 support-map 增量 ranker 提升明显，用它替换 controller 采样线的 pose ranker。
4. controller 第三条 worker 完成后，重新构建 v2 数据快照，补齐当前缺失的第3个 trial。
