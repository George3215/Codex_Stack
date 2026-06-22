# 2026-06-22 方案成功率统计

## 统计口径

- strict success：使用 `results.csv` 中的 `success` 字段。
- shape success：使用 `shape_success` 字段。
- 评估对象：月面 `single_face_wall_3course_v1` 和 `single_face_wall_4course_v1`，不是石头堆。
- 主要辅助指标：`skipped_slot_count`、`stable_count`、`failure_count`、`target_rmse_xy_m`、`max_horizontal_drift_m`、`visible_courses`。
- 注意：单次 seed 成功不能视为稳定成功率，必须用多 trial 统计验证。

## 方案表

| 方案 | 训练数据集 | 评估数据集 | 参数 | 3 层成功率 | 4 层成功率 | 主要结论 |
|---|---|---|---|---:|---:|---|
| Candidate-metric PoseRiskNet hardnegative | `20260622_resume_3to4_hardnegative_20260622_100955_learning_dataset` | `20260622_newposerisk_top5_w035_stats_3to4_moon_trials4_20260622_110140` | top5, risk=0.35, rocks=96, clusters=10, trials=4 | 0/4 = 0% | 0/4 = 0% | 单次 seed 表现好，但 4-trial 不稳定；主要失败是 post-hold drift。 |
| Candidate-metric PoseRiskNet hardnegative | 同上 | `20260622_newposerisk_hardnegative_eval_3to4_moon_top5_w035_seed206233003` | top5, risk=0.35, rocks=64, clusters=8, trials=1 | 1/1 = 100% | 0/1 = 0% | 阶段性好结果：3 层 strict success；4 层 visible_courses=4、stable=16/16、failure=0，但 skipped slots 导致 strict=0。 |
| Strictdrift PoseRiskNet | `20260622_resume_3to4_hardnegative_20260622_100955_learning_dataset` | `20260622_strictdrift_top5_w035_eval_3to4_moon_trials2_seed206234200` | top5, risk=0.35, rocks=96, clusters=10, trials=2，前台 run 超时前完成 3 层 | 2/2 = 100% | 未完整写完 | 3 层在更大数据集上显著改善；4 层尚需后台完整统计。 |
| Strictdrift PoseRiskNet | 同上 | `20260622_strictdrift_top5_w035_stats_3to4_moon_trials4_20260622_140938` | top5, risk=0.35, rocks=96, clusters=10, trials=4 | 1/4 = 25% strict；2/4 = 50% shape | 0/4 = 0% | 相比 candidate-metric 的 3 层 0/4 有提升，但仍不稳定；4 层仍主要被 missed_target、post-hold drift 和 velocity 残余卡住。 |
| Strictdrift long-settle | 同上 | `20260622_strictdrift_top5_w035_longsettle_3to4_moon_trials2_20260622_162145` | top5, risk=0.35, steps=420, hold=1800, rocks=96, clusters=10, trials=2 | 1/2 = 50% strict；1/2 = 50% shape | 0/2 = 0% | 3 层有改善信号，速度残余大幅降低；但 4 层仍失败，并且 `no_feasible_pose`/skipped 增多，说明单纯加长 settle 不是 4 层解法。 |

## 已完成多 trial 统计：candidate-metric hardnegative top5+w0.35

评估数据集：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_newposerisk_top5_w035_stats_3to4_moon_trials4_20260622_110140`

### 3 层

- trials：4
- strict success：0/4 = 0%
- shape success：0/4 = 0%
- mean stable count：9.25
- mean failure count：2.00
- mean RMSE：0.103 m
- mean max drift：0.240 m
- mean visible courses：3.0
- 主要失败：`post_hold_drift` 与 `missed_target+post_hold_drift`

### 4 层

- trials：4
- strict success：0/4 = 0%
- shape success：0/4 = 0%
- mean stable count：12.75
- mean failure count：3.75
- mean RMSE：0.103 m
- mean max drift：0.232 m
- mean visible courses：3.75
- 主要失败：`missed_target+post_hold_drift` 8 次，`unstable_structure` 4 次，`post_hold_drift` 3 次。

结论：这个方案能把墙线 RMSE 控制到 0.10 m 量级，但不能稳定控制月面 post-hold drift，因此 strict success 为 0。

## 已完成 partial 统计：strictdrift top5+w0.35

评估数据集：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top5_w035_eval_3to4_moon_trials2_seed206234200`

说明：该前台 run 在 15 分钟工具超时处停止，`results.csv` 已完整写入两个 3 层 trial；4 层仍在进行时被打断，因此这里只统计 3 层。

### 3 层

- trials：2
- strict success：2/2 = 100%
- shape success：2/2 = 100%
- stable count：13/13, 13/13
- skipped slots：2, 2
- failure count：0, 0
- RMSE：0.040 m, 0.015 m；平均 0.027 m
- max drift：0.0009 m, 0.0023 m；平均 0.0016 m

结论：相比 candidate-metric hardnegative 的 3 层 0/4，strictdrift 在已完成的 3 层 partial 统计中提升到 2/2。这个提升仍需后台 4-trial 完整验证。

## 已完成多 trial 统计：strictdrift top5+w0.35

评估数据集：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top5_w035_stats_3to4_moon_trials4_20260622_140938`

### 3 层

- trials：4
- strict success：1/4 = 25%
- shape success：2/4 = 50%
- mean stable count：12.25
- mean failure count：0.50
- mean RMSE：0.054 m
- mean max drift：0.138 m
- mean visible courses：3.0
- mean velocity inf norm：0.767
- 代表性成功：trial 3，success=1，shape=1，stable=13/13，RMSE=0.012 m，drift=0.009 m，velocity=0.067。
- 代表性边界失败：trial 2，shape=1，stable=13/13，RMSE=0.021 m，drift=0.035 m，但 velocity=0.766，说明几何形状已经接近成功，但动态残余仍会让 strict success 失败。

### 4 层

- trials：4
- strict success：0/4 = 0%
- shape success：0/4 = 0%
- mean stable count：10.75
- mean failure count：7.00
- mean RMSE：0.164 m
- mean max drift：0.367 m
- mean visible courses：4.0
- mean velocity inf norm：0.688
- 主要失败：`missed_target+post_hold_drift` 19 次，`unstable_structure` 6 次，`post_hold_drift` 3 次。

结论：strictdrift 的方向是有效的，因为 3 层从 0/4 提升到 1/4 strict 和 2/4 shape；但它还不能稳定替代启发式搜索。当前瓶颈不是“放不上去”，而是放上去以后在月面低重力下持续漂移、速度残余未衰减，以及 4 层时支撑连续性被放大。下一轮应优先验证更长 settle、更强速度约束或候选预筛，而不是直接追求 10 层。

## 已完成 small-batch 统计：strictdrift long-settle top5+w0.35

评估数据集：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top5_w035_longsettle_3to4_moon_trials2_20260622_162145`

参数变化：上一轮短窗口为 `steps_per_rock=260`、`hold_steps=1000`；本轮为 `steps_per_rock=420`、`hold_steps=1800`。其它核心参数保持不变：moon，3/4 层单面墙，rocks=96，clusters=10，candidate top5，pose risk weight=0.35。

### 3 层

- trials：2
- strict success：1/2 = 50%
- shape success：1/2 = 50%
- mean stable count：11.50
- mean failure count：1.50
- mean RMSE：0.077 m
- mean max drift：0.155 m
- mean velocity inf norm：0.022
- 成功 trial：trial 1，stable=12/12，skipped=3，RMSE=0.031 m，drift=0.0007 m，velocity=0.030。
- 失败 trial：trial 0，stable=11/14，skipped=1，RMSE=0.124 m，drift=0.310 m，velocity=0.013。

### 4 层

- trials：2
- strict success：0/2 = 0%
- shape success：0/2 = 0%
- mean stable count：11.50
- mean failure count：6.50
- mean RMSE：0.203 m
- mean max drift：0.577 m
- mean velocity inf norm：0.051
- 主要失败：`missed_target+post_hold_drift` 8 次，`post_hold_drift` 3 次，`unstable_structure` 2 次。

结论：long-settle 对 3 层有价值，尤其是成功样例的速度和漂移非常低；但 4 层没有改善，反而由于更严格的物理暴露导致 skipped/no-feasible 增多。下一轮不应继续单纯加长 settle，而应使用通用 candidate probe，在候选提交前短时筛掉慢漂移候选，同时控制额外仿真成本。

## 当前经验

- `top5 + risk=0.35` 比 `top3 + risk=0.35` 好：top3 过于保守，容易 skipped slots；top5 提供更多可选姿态。
- `risk=0.25` 不好：候选更容易放上去，但 post-hold drift 明显增大。
- 单纯降低风险权重不是正确方向；应保持较强风险约束，同时训练更严格的漂移风险标签。
- 3 层与 4 层的差异不是简单多放一层。3 层主要看落点和小漂移；4 层会放大 support continuity、cap/middle 耦合和后期漂移。
- 4-trial 已确认 strictdrift 不是完全稳定解；它能改善 3 层 shape，但还需要把速度残余和漂移纳入候选选择或训练标签。
- long-settle 对 3 层有帮助，但对 4 层不足；更长 settle 会暴露不可行候选并增加 skipped slots，因此不能靠“无限加仿真时间”解决。
- 下一步实验优先级：使用 `candidate-probe-steps` 做短时提交前筛选，目标是在不显著扩大总步数的情况下提前发现慢漂移候选。

## 候选模型离线排序指标

| 模型 | 训练数据集 | 标签/阈值 | test top1 safe | test top3 safe | 结论 |
|---|---|---|---:|---:|---|
| Strictdrift PoseRiskNet 100ep | `20260622_resume_3to4_hardnegative_20260622_100955_learning_dataset` | target=0.10, y=0.04, drift=0.035, velocity=0.15 | 0.596 | 1.000 | 当前闭环使用版本；能改善 3 层 shape，但仍不足以稳定 4 层。 |
| Velocity-strict PoseRiskNet 90ep | 同上 | target=0.08, y=0.035, drift=0.025, velocity=0.08 | 0.517 | 0.950 | 更严格标签没有带来更好排序，暂不替换闭环模型；后续应考虑 ranking loss 或连续风险回归。 |

## small-batch 统计：candidate-probe80 hard-gate

评估数据集：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top5_w035_candidateprobe80_3to4_moon_trials2_20260622_164000`

参数变化：在 strictdrift top5+w0.35 的基础上增加 `candidate_probe_steps=80`，并在本批次中以 hard-gate 方式使用 probe。由于实际内部会按仿真步长取整，`results.csv` 记录的有效 probe 为 192 steps。其它核心参数保持一致：moon，3/4 层单面墙，rocks=96，clusters=10，trials=2，`steps_per_rock=260`，`hold_steps=1000`。

### 3 层

- trials：2
- strict success：0/2 = 0%
- shape success：0/2 = 0%
- mean stable count：10.50
- mean failure count：1.50
- mean RMSE：0.084 m
- mean max drift：0.131 m
- mean velocity inf norm：1.112
- 典型边界失败：trial 0 几何和漂移都很好，stable=11/11、failure=0、RMSE=0.031 m、drift=0.0019 m，但 skipped=4，墙体缺块导致 shape/strict 均失败。

### 4 层

- trials：2
- strict success：0/2 = 0%
- shape success：0/2 = 0%
- mean stable count：13.50
- mean failure count：3.50
- mean RMSE：0.109 m
- mean max drift：0.174 m
- mean velocity inf norm：0.340
- 典型边界失败：trial 1 stable=17/17、failure=0、visible_courses=4、height=0.356 m、RMSE=0.044 m、drift=0.033 m，但 skipped=7，说明硬门控虽然过滤了坏候选，也显著破坏槽位完整性。

主要失败统计：

- 4 层 `missed_target+post_hold_drift`：4 次
- 4 层 `post_hold_drift`：3 次
- 3 层 `missed_target+post_hold_drift`：2 次
- 3 层 `unstable_structure`：1 次

结论：candidate probe 的方向有价值，因为它能提前暴露部分慢漂移候选；但 `probe=80 + hard-gate` 过于保守，会把本来几何可行的槽位直接跳过，导致完整墙体失败。因此下一轮不应继续加硬门控强度，而应使用 soft probe：把 probe 漂移、扰动和速度作为评分惩罚，让候选排序更谨慎，但保留 fallback，避免 skipped slots 主导失败。

## small-batch 统计：candidate-probe40 soft

评估数据集：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top5_w035_candidateprobe40soft_fix_3to4_moon_trials2_20260622_174900`

参数变化：在 strictdrift top5+w0.35 的基础上增加 `candidate_probe_steps=40`，不开启 hard-gate。probe 只作为候选评分惩罚，不直接拒绝候选。`results.csv` 中有效 probe steps 为 96，`candidate_probe_hard_gate=0`。

### 3 层

- trials：2
- strict success：1/2 = 50%
- shape success：1/2 = 50%
- mean stable count：11.00
- mean failure count：1.50
- mean RMSE：0.064 m
- mean max drift：0.140 m
- mean velocity inf norm：0.258
- 成功 trial：trial 1，rock_count=12，stable=12/12，failure=0，skipped=3，RMSE=0.017 m，max drift=0.0028 m，velocity=0.022，height=0.287 m。
- 失败 trial：trial 0，stable=10/13，failure=3，skipped=2，RMSE=0.112 m，max drift=0.278 m，velocity=0.494。

### 4 层

- trials：2
- strict success：0/2 = 0%
- shape success：0/2 = 0%
- mean stable count：10.00
- mean failure count：5.00
- mean RMSE：0.157 m
- mean max drift：0.342 m
- mean velocity inf norm：1.757
- trial 0：rock_count=13，stable=9/13，failure=4，skipped=11，visible_courses=3，height=0.193 m。
- trial 1：rock_count=17，stable=11/17，failure=6，skipped=7，visible_courses=4，height=0.309 m。

主要失败统计：

- 4 层 `missed_target+post_hold_drift`：5 次
- 4 层 `unstable_structure`：4 次
- 3 层 `post_hold_drift`：2 次
- 4 层 `post_hold_drift`：1 次
- 3 层 `missed_target+post_hold_drift`：1 次

图像目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top5_w035_candidateprobe40soft_fix_3to4_moon_trials2_20260622_174900\capwf_soft_probe40`

结论：`probe=40 soft` 恢复了 3 层成功样例，且成功样例质量高；相比 `probe=80 hard-gate`，它没有因为硬拒绝直接把 3 层成功率压到 0。但 4 层仍然没有突破，主要问题不是单块石头能否稳定，而是 cap/middle 阶段的连续支撑、槽位完整性和后期漂移共同失控。下一步如果继续 probe 方向，应更轻量地使用 probe，例如只在 middle/cap 或高风险候选上触发，而不是所有候选都加 probe；同时应训练连续风险回归或 ranking loss，让网络学习“低漂移且不导致 skipped”的排序，而不是单纯二分类安全。

## small-batch 统计：candidate-probe40 soft + top8

评估数据集：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top8_w035_candidateprobe40soft_3to4_moon_trials2_20260622_183600`

参数变化：在 `probe=40 soft` 的基础上，把 `candidate_pose_top_k` 从 5 扩到 8。其它核心条件一致：moon，3/4 层单面墙，rocks=96，clusters=10，trials=2，`pose_risk_weight=0.35`，`candidate_probe_hard_gate=0`。

### 3 层

- trials：2
- strict success：1/2 = 50%
- shape success：2/2 = 100%
- mean stable count：12.50
- mean failure count：1.00
- mean RMSE：0.053 m
- mean max drift：0.115 m
- mean velocity inf norm：0.028
- trial 0：shape=1 但 strict=0，stable=11/13，skipped=2，RMSE=0.088 m，drift=0.227 m，velocity=0.028。
- trial 1：strict=1，shape=1，stable=14/14，skipped=1，RMSE=0.018 m，drift=0.0036 m，velocity=0.027。

### 4 层

- trials：2
- strict success：0/2 = 0%
- shape success：0/2 = 0%
- mean stable count：14.50
- mean failure count：6.00
- mean RMSE：0.163 m
- mean max drift：0.433 m
- mean velocity inf norm：0.157
- trial 0：rock_count=23，stable=16/23，failure=7，skipped=1，visible_courses=4，height=0.294 m。
- trial 1：rock_count=18，stable=13/18，failure=5，skipped=6，visible_courses=4，height=0.275 m。

主要失败统计：

- 4 层 `missed_target+post_hold_drift`：8 次
- 4 层 `unstable_structure`：2 次
- 4 层 `post_hold_drift`：2 次
- 3 层 `post_hold_drift`：1 次
- 3 层 `missed_target+post_hold_drift`：1 次

图像目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top8_w035_candidateprobe40soft_3to4_moon_trials2_20260622_183600\capwf_top8_probe40`

结论：扩大候选池是有效方向。相对 top5 soft，3 层 shape 从 1/2 提升到 2/2，mean stable count 从 11.0 提升到 12.5；4 层 skipped 明显改善，特别是 trial 0 只 skipped=1，并且 visible_courses=4。但是 4 层仍然 0/2，失败从“缺槽/不可行”转移为“放得更完整但后期漂移和局部失稳”。下一步不应再只扩大 top-k，而应把 drift、support continuity、course-level load path 和 cap/middle 锁定写入候选评分或网络标签。

## 多 trial 统计：candidate-probe40 soft + top8

评估数据集：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top8_w035_candidateprobe40soft_stats_3to4_moon_trials4_20260622_191500`

参数：moon，3/4 层单面墙，rocks=96，clusters=10，trials=4，`candidate_probe_steps=40`，`candidate_pose_top_k=8`，`pose_risk_weight=0.35`，网络只参与到 `course<=2`。

### 3 层

- trials：4
- strict success：4/4 = 100%
- shape success：4/4 = 100%
- mean stable count：13.25
- mean failure count：0.00
- mean RMSE：0.022 m
- mean max drift：0.0056 m
- mean velocity inf norm：0.051
- trial 明细：4 个 trial 全部 strict success；`stable_count` 分别为 13/13、14/14、14/14、12/12，均无 failure。

### 4 层

- trials：4
- strict success：0/4 = 0%
- shape success：0/4 = 0%
- mean stable count：12.25
- mean failure count：4.50
- mean RMSE：0.125 m
- mean max drift：0.281 m
- mean velocity inf norm：0.185
- visible courses：均值 3.5，说明 4 层经常能显现，但不能保持结构稳定。
- 代表失败：trial 1 rock_count=20、skipped=4、visible_courses=4，但 drift=0.386 m；trial 0 rock_count=18、skipped=6、visible_courses=4，但 drift=0.280 m。

主要失败统计：

- 4 层 `missed_target+post_hold_drift`：14 次
- 4 层 `unstable_structure`：2 次
- 4 层 `post_hold_drift`：2 次

图像目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top8_w035_candidateprobe40soft_stats_3to4_moon_trials4_20260622_191500\c4`

结论：这是当前最强的 3 层结果，已经达到 4-trial strict 100%。4 层仍为 0%，但失败重点非常明确：不是 3 层阶段的问题，也不是完全没有候选，而是第 4 层/高层结构在 post-hold 中漂移。下一步需要把 `course=3` 顶层也纳入网络，并训练 support-continuity/ranking 模型，而不是继续只调 top-k 或 probe。
