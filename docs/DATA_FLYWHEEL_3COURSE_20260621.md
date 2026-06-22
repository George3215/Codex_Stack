# 3层墙数据飞轮实验记录 2026-06-21

本轮目标不是继续盲目堆更高，而是先把 `single_face_wall_3course_v1` 做成可重复的数据飞轮：采集 explore/exploit 数据，训练观测驱动的候选位姿网络，再闭环评估。这样后续可以把启发式逐步压到数据生成和损失先验里，而不是在真实任务中大量随机试错。

## 实验目的

1. 优先提高 3层/4层墙的可靠性，暂时不追求 8-10 层高度。
2. 保留干叠先验：目标槽位、交错课程、横向薄墙约束、重心/支撑/接触质量、漂移惩罚。
3. 将先验写进数据和损失函数：`structure_aware` 候选位姿目标会惩罚目标误差、横向厚度、漂移和结构偏离。
4. 让网络输入包含已知几何和观测，而不是测试后验：
   - 石头几何特征：体积、包围盒、扁平度、细长比、粗糙度、棱角度、主面数、支撑面质量等。
   - 当前堆叠观测：MuJoCo 渲染的前视/俯视深度图、目标高斯图、候选轮廓/足迹图、重力和层数通道。
   - 候选位姿参数：位置、四元数、候选编号和候选启发式分数。
5. 输出不是直接执行完整轨迹，而是先做小网络组合：
   - `StoneSlotNet`: 给石头-槽位匹配打分，减少石头选择搜索。
   - `PoseRankNet`: 给同一槽位/同一候选石头的多个候选位姿排序。
   - `WallStateCritic`: 预测放置后的局部结构风险，用于下一步在线风险过滤。

## 新增调度系统

脚本：

`scripts/run_wall_data_flywheel.py`

本轮补充了几个关键能力：

- `--no-default-prior-runs`: 可以关闭旧 4层默认先验数据，做 3层专注训练。
- `--dataset-target-contains`: 控制训练/深度导出的目标过滤，例如只取 `single_face_wall_3course`。
- `--exploit-stone-ranker-dir` / `--exploit-pose-ranker-dir`: exploit 采集阶段可以显式使用上一轮模型。
- `--eval-stone-ranker-dir` / `--eval-pose-ranker-dir`: 评估阶段可以显式指定模型。
- 如果数据集里没有 `assignment_candidate` 样本，会记录并跳过 `StoneSlotNet` 训练，复用已有 StoneSlotNet，继续训练 PoseRankNet 和 WallStateCritic，避免整轮数据浪费。
- `scripts/build_learning_dataset.py` 现在可以从 `candidate_pose_log.csv` 合成石头-槽位候选样本：按目标、重力、trial、slot、candidate rock 聚合，标签来自是否出现 committed success，输入仍只用槽位/角色/石头几何等可观测先验。

调度顺序：

1. 并行采集 explore 和 exploit 结构墙实验。
2. 重建 learning dataset。
3. 导出 MuJoCo 前视/俯视深度观测张量。
4. 训练 `StoneSlotNet`、`PoseRankNet`、`WallStateCritic`。
5. 用最新模型闭环评估。
6. 对成功或典型失败案例保存 RGB、普通 depth、`object_depth`。

## 深度图修正

用户指出非顶视角 depth 基本全黄，没有分析价值。原因是普通 depth PNG 被远平面/背景深度主导。

已修改：

`scripts/capture_cases.py`

现在每个相机都保存：

- `*_depth.png/.npy`: 原始深度；
- `*_object_depth.png/.npy`: 只保留石头分割后的目标深度；
- `wall_front_object_depth.*` 和 `wall_top_object_depth.*`: 后续汇报和训练优先看这两个。

结论：普通 depth 仍可保留作原始记录，但墙体分析应该优先用 object-only depth。

## 数据采集

### Starter 数据飞轮

会话：

`batch_runs/20260620_wall_flywheel_starter_v1`

产物：

- 数据集：`batch_runs/20260620_wall_flywheel_starter_v1_learning_dataset`
- 深度张量：`batch_runs/20260620_wall_flywheel_starter_v1_mujoco_depth_maps`
- StoneSlotNet：`batch_runs/20260620_wall_flywheel_starter_v1_stone_slot_net`
- PoseRankNet：`batch_runs/20260620_wall_flywheel_starter_v1_pose_ranker_structure`
- WallStateCritic：`batch_runs/20260620_wall_flywheel_starter_v1_wall_state_critic`
- 闭环评估：`batch_runs/20260620_wall_flywheel_starter_v1_closed_loop_eval`

数据规模：

| 项目 | 数量 |
|---|---:|
| run examples | 40 |
| placement examples | 924 |
| candidate pose examples | 25,131 |
| assignment candidate examples | 288 |
| MuJoCo depth rows | 893 |
| selected candidate groups | 256 |

### 3层专注数据飞轮

采集会话：

`batch_runs/20260621_wall_flywheel_3course_focus_v1`

恢复训练/评估会话：

`batch_runs/20260621_wall_flywheel_3course_focus_v1_resume_train`

产物：

- 数据集：`batch_runs/20260621_wall_flywheel_3course_focus_v1_learning_dataset`
- 深度张量：`batch_runs/20260621_wall_flywheel_3course_focus_v1_mujoco_depth_maps`
- PoseRankNet：`batch_runs/20260621_wall_flywheel_3course_focus_v1_resume_train_pose_ranker_structure`
- WallStateCritic：`batch_runs/20260621_wall_flywheel_3course_focus_v1_resume_train_wall_state_critic`
- 闭环评估：`batch_runs/20260621_wall_flywheel_3course_focus_v1_resume_train_closed_loop_eval`

数据规模：

| 项目 | 数量 |
|---|---:|
| run examples | 16 |
| placement examples | 276 |
| candidate pose examples | 12,972 |
| assignment candidate examples | 0 |
| candidate rows after 3层过滤 | 8,460 |
| MuJoCo depth rows | 2,751 |
| selected candidate groups | 512 |

重要问题：3层专注采集没有生成 `assignment_candidates_*.csv`，所以本轮不能重新训练 StoneSlotNet，只能复用 starter 的 StoneSlotNet。下一步必须补 3层 assignment candidate 生成，否则石头选择网络会跟不上位姿网络。

### 3层 v2 合成 StoneSlotNet 数据

补丁后重建数据集：

`batch_runs/20260621_wall_flywheel_3course_focus_v2_learning_dataset`

| 项目 | 数量 |
|---|---:|
| placement examples | 276 |
| candidate pose examples | 12,972 |
| synthesized assignment candidate examples | 2,484 |
| StoneSlotNet rows after 3层过滤 | 1,620 |
| StoneSlotNet positives | 117 |

## 网络结果

### StoneSlotNet

Starter 训练结果：

| 指标 | 测试集 |
|---|---:|
| accuracy | 0.7569 |
| precision | 0.2750 |
| recall | 0.6471 |
| f1 | 0.3860 |
| group top1 hit | 0.5882 |
| group top3 hit | 0.7059 |

解释：StoneSlotNet 已经有一定筛石能力，但数据少、正例少，precision 偏低。当前更像“不要漏掉可能有用的石头”的召回型筛选器，还不是精确选择器。

3层 v2 合成数据 StoneSlotNet：

`batch_runs/20260621_wall_flywheel_3course_focus_v2_stone_slot_net_20260621_014556`

| 指标 | 测试集 |
|---|---:|
| rows | 1,620 |
| positives | 117 |
| accuracy | 0.7167 |
| precision | 0.0941 |
| recall | 0.5000 |
| f1 | 0.1584 |
| group top1 hit | 0.1000 |
| group top3 hit | 0.4000 |

解释：离线指标还弱，尤其是候选石头排序 top-k 不够好。但它提供了一个可训练的石头选择入口，并且闭环测试显示它并非完全无效。

### 3层 PoseRankNet

路径：

`batch_runs/20260621_wall_flywheel_3course_focus_v1_resume_train_pose_ranker_structure`

输入：

- 10通道 `64x64` MuJoCo 观测图；
- 74维数值特征；
- 不使用 post-simulation 后验特征；
- 目标模式：`structure_aware`。

测试指标：

| 指标 | 数值 |
|---|---:|
| test top1 hit | 0.4076 |
| test top3 hit | 1.0000 |
| Earth top1/top3 | 0.4205 / 1.0000 |
| Moon top1/top3 | 0.3958 / 1.0000 |
| mean top1 quality regret | 12.9277 |
| mean top3 quality regret | 0.0000 |

解释：网络第一名还不够可靠，但前三名总能覆盖当前组里的最优候选。这说明当前阶段 `top_k=3` 是合理过渡：网络已经能大幅压缩候选空间，但还不能完全替代小规模物理验证。

### 3层 WallStateCritic

路径：

`batch_runs/20260621_wall_flywheel_3course_focus_v1_resume_train_wall_state_critic`

测试指标：

| 指标 | 数值 |
|---|---:|
| classification accuracy | 0.6725 |
| precision | 0.9941 |
| recall | 0.6731 |
| f1 | 0.8027 |
| group top1 hit | 0.2278 |
| group top3 hit | 0.5316 |

解释：critic 能识别一部分风险，但排序能力仍弱。它目前更适合作为辅助风险特征，暂时不应单独接管在线决策。

## 闭环评估结果

### Starter 3层评估

路径：

`batch_runs/20260620_wall_flywheel_starter_v1_closed_loop_eval`

| 重力 | trials | strict success | visible courses | stable/failure | height m | RMSE m | max error m | max drift m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Earth | 1 | 0/1 | 2 | 5 / 1 | 0.225 | 0.019 | 0.037 | 0.0033 |
| Moon | 1 | 0/1 | 3 | 9 / 0 | 0.340 | 0.049 | 0.108 | 0.0052 |

Starter 的月球案例已经接近阶段性成功，但严格判据仍失败。

### 3层专注评估

路径：

`batch_runs/20260621_wall_flywheel_3course_focus_v1_resume_train_closed_loop_eval`

| 重力 | trials | strict success | shape success | visible courses | mean stable | mean failure | height m | RMSE m | max error m | max drift m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Earth | 3 | 0/3 | 0/3 | 2.0 | 5.0 | 1.0 | 0.185 | 0.069 | 0.144 | 0.0694 |
| Moon | 3 | 0/3 | 0/3 | 3.0 | 9.67 | 0.33 | 0.377 | 0.0497 | 0.1028 | 0.0011 |

最好看的月球 trial 2：

- stable/failure: `9 / 0`
- visible courses: `3`
- height: `0.414 m`
- RMSE: `0.0358 m`
- max target error: `0.0831 m`
- max drift: `0.0019 m`
- 仍失败的主要原因：跳过了 6 个目标槽位，严格结构完整性不足。

### 3层 v2 StoneSlotNet 闭环评估

路径：

`batch_runs/20260621_wall_flywheel_3course_stoneslot_v2_eval`

配置：

- StoneSlotNet：`batch_runs/20260621_wall_flywheel_3course_focus_v2_stone_slot_net_20260621_014556`
- PoseRankNet：`batch_runs/20260621_wall_flywheel_3course_focus_v1_resume_train_pose_ranker_structure`
- `stone_fit_top_k=15`
- `candidate_pose_top_k=3`
- 目标：`single_face_wall_3course_v1`
- 重力：Earth / Moon
- 每个重力 3 trials

| 重力 | trials | strict success | shape success | visible courses | mean stable | mean failure | height m | RMSE m | max error m | max drift m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Earth | 3 | 0/3 | 1/3 | 2.67 | 11.33 | 1.33 | 0.235 | 0.0918 | 0.253 | 0.243 |
| Moon | 3 | 1/3 | 2/3 | 3.00 | 13.33 | 1.00 | 0.285 | 0.0626 | 0.164 | 0.148 |

成功案例：

`moon trial 01`

- strict success: `1`
- shape success: `1`
- stable/failure: `14 / 0`
- visible courses: `3`
- skipped slots: `1`
- height: `0.294 m`
- target RMSE: `0.0187 m`
- max target error: `0.0327 m`
- max drift: `0.00044 m`

这个结果说明：即使 StoneSlotNet 离线排序还弱，合成候选数据已经足以让筛石网络参与闭环并产生严格成功案例。问题是平均漂移和速度也上升，说明石头选择网络会把一些高风险石头放进候选集合，下一步必须接入 WallStateCritic 或物理门控压制不稳定候选。

### 3层 PoseRiskNet 风险惩罚闭环评估

路径：

`batch_runs/20260621_wall_flywheel_3course_pose_risk_w035_eval`

新增网络：

`batch_runs/20260621_pose_risk_net_3course_v1`

设计目的：

- `PoseRankNet` 负责把好位姿保留在 `top_k` 小候选集里；
- `PoseRiskNet` 负责在仿真前识别“可能扰动、漂移、速度过大、目标误差过大”的候选位姿；
- 在线打分采用轻量惩罚：`rank_score = pose_rank_score - 0.35 * pose_risk_prob`，最终候选评分也加入同样风险惩罚；
- 输入仍只包含已知/可观测先验：目标槽位、重力、候选位姿、石头几何特征、角色和类别标签；不把测试后统计出来的成功率当作输入；
- 标签来自仿真后的监督信号：是否 committed success、目标误差、横向误差、扰动、速度和失败原因。

PoseRiskNet 离线指标：

| 指标 | 数值 |
|---|---:|
| candidate pose rows | 8,460 |
| train / test rows | 7,560 / 900 |
| risky positives | 8,380 |
| test accuracy | 0.950 |
| test precision | 0.986 |
| test recall | 0.963 |
| test f1 | 0.974 |
| group top1 safe rate | 0.417 |
| group top3 safe rate | 1.000 |

解释：风险标签非常严格，类别严重不平衡，所以这个网络不能单独做强决策。但它能把安全候选保留在前三名，适合作为轻量惩罚项，而不是完全替代物理验证。

闭环配置：

- StoneSlotNet：`batch_runs/20260621_wall_flywheel_3course_focus_v2_stone_slot_net_20260621_014556`
- PoseRankNet：`batch_runs/20260621_wall_flywheel_3course_focus_v1_resume_train_pose_ranker_structure`
- PoseRiskNet：`batch_runs/20260621_pose_risk_net_3course_v1`
- `stone_fit_top_k=15`
- `candidate_pose_top_k=3`
- `pose_risk_weight=0.35`
- 目标：`single_face_wall_3course_v1`
- 重力：Earth / Moon
- 每个重力 3 trials

| 重力 | trials | strict success | shape success | visible courses | mean stable | mean failure | height m | RMSE m | max error m | max drift m | velocity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Earth | 3 | 0/3 | 0/3 | 2.67 | 9.00 | 1.33 | 0.187 | 0.1140 | 0.3125 | 0.2919 | 3.9027 |
| Moon | 3 | 3/3 | 3/3 | 3.00 | 13.00 | 0.00 | 0.309 | 0.0319 | 0.0787 | 0.0014 | 0.0289 |

与上一轮 StoneSlotNet v2 对比：

- Moon strict success 从 `1/3` 提升到 `3/3`；
- Moon shape success 从 `2/3` 提升到 `3/3`；
- Moon 平均漂移从 `0.148 m` 降到 `0.0014 m`；
- Moon 平均速度从接近失稳的量级降到 `0.0289`；
- Earth 仍然失败，主要是 `missed_target + post_hold_drift`，说明地球重力下接触冲击和放置高度策略还需要单独优化。

三个成功月球案例：

| trial | stable/failure | skipped slots | height m | RMSE m | max error m | max drift m |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 13 / 0 | 2 | 0.318 | 0.0142 | 0.0294 | 0.00051 |
| 1 | 14 / 0 | 1 | 0.325 | 0.0289 | 0.0810 | 0.00250 |
| 2 | 12 / 0 | 3 | 0.283 | 0.0525 | 0.1258 | 0.00122 |

阶段性结论：在 3 层月面单面墙上，小网络组合已经开始产生实际价值。`StoneSlotNet + PoseRankNet + PoseRiskNet` 不能说完全替代启发式，但已经把启发式候选集压缩到小范围，并用风险网络显著降低月球闭环漂移。下一步不应立即追求 8-10 层，而应扩大 3层/4层数据，把这个成功率变成统计稳定结果。

### PoseRiskNet 增量验证

增量批次：

`batch_runs/20260621_wall_flywheel_3course_pose_risk_w035_eval_n4_increment`

配置与上一批相同，只改变随机种子和石头库规模：

- `rocks=160`
- `trials=2`
- `workers=3`
- `pose_risk_weight=0.35`
- 每个重力 2 trials

增量结果：

| 重力 | trials | strict success | shape success | visible courses | mean stable | mean failure | height m | RMSE m | max error m | max drift m | velocity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Earth | 2 | 0/2 | 0/2 | 3.00 | 11.00 | 3.00 | 0.209 | 0.1218 | 0.3065 | 0.3039 | 3.3892 |
| Moon | 2 | 2/2 | 2/2 | 3.00 | 15.00 | 0.00 | 0.298 | 0.0239 | 0.0517 | 0.0050 | 0.0310 |

合并 `w035_eval` 与 `w035_eval_n4_increment` 后的当前统计：

| 重力 | trials | strict success | shape success | visible courses | mean stable | mean failure | height m | RMSE m | max error m | max drift m | velocity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Earth | 5 | 0/5 | 0/5 | 2.80 | 9.80 | 2.00 | 0.1956 | 0.1172 | 0.3101 | 0.2967 | 3.6973 |
| Moon | 5 | 5/5 | 5/5 | 3.00 | 13.80 | 0.00 | 0.3046 | 0.0287 | 0.0679 | 0.0028 | 0.0297 |

解释：

- 月球三层单面墙已经从偶然成功变成小样本稳定成功，当前 `5/5` 严格成功可以作为阶段性成果；
- 地球三层单面墙仍是 `0/5`，主要瓶颈不是是否可见三层，而是放置冲击、后保持漂移和目标槽位偏差；
- 这支持下一步分域训练：Moon 可以继续扩大成功数据并迁移到 4 层；Earth 应单独训练风险头或降低放置冲击，而不是直接复用月面成功策略。

### 增强数据集与 PoseRiskNet V2 反例

增强数据集：

`batch_runs/20260621_wall_flywheel_3course_pose_risk_augmented_learning_dataset`

合并的 run：

- `20260620_wall_flywheel_starter_v1_collect_explore_00_seed98001`
- `20260620_wall_flywheel_starter_v1_collect_exploit_00_seed107002`
- `20260621_wall_flywheel_3course_focus_v1_collect_explore_00_seed98001`
- `20260621_wall_flywheel_3course_focus_v1_collect_exploit_00_seed107002`
- `20260621_wall_flywheel_3course_focus_v1_resume_train_closed_loop_eval`
- `20260621_wall_flywheel_3course_stoneslot_v2_eval`
- `20260621_wall_flywheel_3course_pose_risk_w035_eval`
- `20260621_wall_flywheel_3course_pose_risk_w035_eval_n4_increment`

数据规模：

| 项目 | 数量 |
|---|---:|
| run examples | 38 |
| placement examples | 606 |
| candidate pose examples | 26,472 |
| assignment candidate examples | 6,984 |

placement 分布：

| 重力 | examples | committed success | failure | skipped |
|---|---:|---:|---:|---:|
| Earth | 303 | 176 | 27 | 100 |
| Moon | 303 | 221 | 18 | 64 |

训练的候选模型：

`batch_runs/20260621_pose_risk_net_3course_augmented_v2`

V2 离线指标：

| 指标 | 数值 |
|---|---:|
| candidate pose rows after target filter | 21,960 |
| train / test rows | 16,110 / 5,850 |
| risky positives | 21,711 |
| test accuracy | 0.889 |
| test precision | 0.991 |
| test recall | 0.896 |
| test f1 | 0.941 |
| group top1 safe rate | 0.333 |
| group top3 safe rate | 1.000 |

与 V1 对比：V2 数据更多，但 top1 safe 从 `0.417` 降到 `0.333`，top3 safe 仍为 `1.000`。这说明简单追加数据没有解决风险标签极度不平衡问题；当前风险标签把绝大多数候选都判成 risky，网络更像“保守排除器”，还不是精确选择器。

V2 闭环 smoke：

`batch_runs/20260621_wall_flywheel_3course_pose_risk_augmented_v2_smoke_eval`

| 重力 | trials | strict success | shape success | visible courses | stable/failure | height m | RMSE m | max error m | max drift m | velocity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Earth | 1 | 0/1 | 0/1 | 3 | 10 / 2 | 0.199 | 0.1130 | 0.2837 | 0.2624 | 0.0503 |
| Moon | 1 | 0/1 | 0/1 | 3 | 11 / 0 | 0.303 | 0.0176 | 0.0403 | 0.0056 | 0.0324 |

V2 smoke 解释：

- Moon 并没有倒塌，且 RMSE 很低，但 strict/shape 都失败，说明它仍可能漏槽或破坏结构完整性；
- Earth 继续在 cap 层出现 `post_hold_drift` 和 `missed_target + post_hold_drift`；
- 因此当前在线主模型继续使用 V1，V2 作为“更多数据但损失/标签设计仍不足”的反例保留；
- 下一版应减少类别不平衡影响，并把槽位完整性、是否跳槽、课程覆盖率纳入风险标签或多任务输出。

## 图片输出

3层专注评估拍照目录：

`batch_runs/20260621_wall_flywheel_3course_focus_v1_resume_train_closed_loop_eval/captures_960x720`

代表月球近成功/失败案例：

`captures_960x720/03_single_face_wall_3course_v1_failure_statics_wall_moon_trial_01`

关键文件：

- `wall_front_rgb.png`
- `wall_front_object_depth.png`
- `wall_top_object_depth.png`
- `front_rgb.png`
- `top_rgb.png`

v2 严格成功案例拍照目录：

`batch_runs/20260621_wall_flywheel_3course_stoneslot_v2_eval/captures_960x720/00_single_face_wall_3course_v1_success_statics_wall_moon_trial_01`

关键文件：

- `wall_front_rgb.png`
- `wall_front_object_depth.png`
- `wall_top_object_depth.png`

PoseRiskNet 风险惩罚闭环成功/失败拍照目录：

`batch_runs/20260621_wall_flywheel_3course_pose_risk_w035_eval/captures_960x720`

PoseRiskNet 增量验证拍照目录：

`batch_runs/20260621_wall_flywheel_3course_pose_risk_w035_eval_n4_increment/captures_960x720`

PoseRiskNet V2 smoke 失败拍照目录：

`batch_runs/20260621_wall_flywheel_3course_pose_risk_augmented_v2_smoke_eval/captures_960x720`

过程视频输出：

`batch_runs/20260621_wall_flywheel_3course_pose_risk_w035_eval/process_videos_release_replay/single_face_wall_3course_v1_statics_wall_moon_trial_01_process/process.gif`

新增脚本：

`scripts/render_process_video.py`

当前默认视频模式是从 `placement_log.csv` 读取已选石头、已选位姿和四元数，按真实放置顺序在 MuJoCo 里逐块释放并沉降，输出 PNG 帧和 `process.gif`。这个 GIF 用前视 + 俯视并排显示，适合快速观察“堆高过程”和“墙体厚度/线形演化”。

该案例信息：

| 项目 | 数值 |
|---|---:|
| gravity | Moon |
| trial | 1 |
| target | `single_face_wall_3course_v1` |
| strategy | `statics_wall` |
| strict success | 1 |
| shape success | 1 |
| replayed stones | 14 |
| GIF frames | 135 |
| view | front + top |

复现命令：

```powershell
conda run -n moon-rock-stack python -m scripts.render_process_video `
  --output batch_runs\20260621_wall_flywheel_3course_pose_risk_w035_eval `
  --gravity moon `
  --trial 1 `
  --width 480 `
  --height 360 `
  --fps 8 `
  --frame-stride 96 `
  --final-hold-steps 720 `
  --video-dir-name process_videos_release_replay `
  --save-frames
```

注意：原始 `state.npz` 只保存最终 `qpos/qvel`，不保存完整中间轨迹。因此默认 GIF 是“选中位姿释放重放”，不是机器人抓取轨迹，也不是原始 trial 的逐时间步录像。脚本里已经保留 `--algorithm-keyframes` 模式，可重跑算法并保存每块石头放置后的 keyframe；但这个模式会重新执行候选搜索，当前对 `stone_fit_top_k=15` 的试验较慢，暂不作为常规输出路径。后续如果需要严格记录原始过程，应在正式闭环实验时直接开启状态快照或过程录像。

原始 3-trial 批次成功案例：

- `00_single_face_wall_3course_v1_success_statics_wall_moon_trial_01`
- `01_single_face_wall_3course_v1_success_statics_wall_moon_trial_00`
- `02_single_face_wall_3course_v1_success_statics_wall_moon_trial_02`

原始 3-trial 批次失败案例：

- `03_single_face_wall_3course_v1_failure_statics_wall_earth_trial_02`
- `04_single_face_wall_3course_v1_failure_statics_wall_earth_trial_00`
- `05_single_face_wall_3course_v1_failure_statics_wall_earth_trial_01`

增量 2-trial 批次成功案例：

- `00_single_face_wall_3course_v1_success_statics_wall_moon_trial_01`
- `01_single_face_wall_3course_v1_success_statics_wall_moon_trial_00`

增量 2-trial 批次失败案例：

- `02_single_face_wall_3course_v1_failure_statics_wall_earth_trial_01`
- `03_single_face_wall_3course_v1_failure_statics_wall_earth_trial_00`

数值核查：`wall_front_object_depth.npy` 在成功案例 00 中有约 57,543 个有效石头像素，深度范围约 `1.09-5.78 m`；`wall_top_object_depth.npy` 有约 60,566 个有效石头像素，深度范围约 `0.89-1.19 m`。这说明 object-only depth 不是全背景或全黄图，可以作为后续网络输入和汇报证据。

注意：后续汇报优先展示 `wall_front_rgb.png`、`wall_front_object_depth.png`、`wall_top_object_depth.png`，不要再用全黄的非顶视角普通 depth 图作为主要证据。

## 失败案例统计

本轮 3层专注闭环评估中主要失败原因：

| 场景 | 典型失败 |
|---|---|
| Earth | base 石头 low_or_fallen；中层出现 missed_target + post_hold_drift |
| Moon | 个别 base low_or_fallen；更多是槽位缺失，不是整体倒塌 |

可观察到的趋势：

- 月球重力下漂移显著小于地球重力，3层结构更容易保持高而窄。
- exploit 采集比纯 explore 更容易形成薄墙：例如 3层月球 exploit trial 1 达到 10/10 稳定、3层可见、0.331 m 高、最大漂移 0.0005 m。
- 当前失败不是“石头堆成土堆”，而是结构槽位覆盖不完整、局部 target error 超出严格判据。

## 当前结论

1. 继续追求更高层数暂时收益不大。现在的瓶颈是 3层/4层的结构完整率和槽位覆盖率。
2. PoseRankNet 已经有阶段性意义：`top3 hit = 1.0`，说明网络能把好候选保留在前三名，已经可以减少启发式/物理搜索。
3. 网络还不能完全替代启发式：`top1 hit` 只有约 0.41，需要继续积累数据，并把损失函数从“候选内最优”推进到“闭环结构完整”。
4. StoneSlotNet 已经能通过合成 assignment candidates 训练并参与闭环，且产生了 3层月球严格成功；但离线 top-k 指标弱，平均稳定性风险升高，仍是当前数据飞轮短板。
5. PoseRiskNet 作为轻量风险惩罚项显著提升了月球三层墙闭环可靠性，合并验证达到 `5/5` strict success；但地球重力仍为 `0/5`，说明重力/冲击条件需要分域建模。
6. WallStateCritic 目前是辅助模型，不能直接作为强决策器；下一步应在线接入为风险惩罚，而不是单独排序。

## 下一步

1. 继续改进合成 assignment candidates：加入更多负例分层、按 role/course 平衡采样，并把 group key 扩展到 gravity/trial，避免不同试验混在同一个槽位组里。
2. 继续跑 3层专注数据飞轮，但保证每轮同时生成：
   - 石头-槽位候选数据；
   - 候选位姿观测数据；
   - 闭环失败案例；
   - object-only depth 图片。
3. 继续调 PoseRiskNet：降低标签不平衡影响，分别训练 Earth/Moon 风险头，避免月球收益掩盖地球失败。
4. 把 WallStateCritic 接入在线打分：`final_score = PoseRankNet + support prior - pose risk penalty - critic risk penalty`。
5. 对 3层先建立两个判据：
   - strict success：完整槽位、目标误差、厚度、稳定性全过；
   - stage success：可见3层、稳定、漂移低、不是石堆。
6. 当 3层 stage success 达到 80-90%，再迁移到 4层；当 4层稳定后再考虑 5-6层和更丰富石头族。
