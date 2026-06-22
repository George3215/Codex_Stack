# 严格单面墙线与 WallStateCritic V2 实验记录

日期：2026-06-20

目标：把 4 层单面墙实验从“能堆起若干石头”推进到“更像线形干砌墙结构”。本轮不追求盲目加高，而是验证小网络和物理门控是否能减少散堆、厚堆和高层漂移。

## 本轮方法

本轮沿用多凸、棱角化、非尖刺、非扁片的石头生成规则，目标结构仍为：

- `single_face_wall_4course_v1`
- strategy：`statics_wall`
- Earth gravity：`9.80665 m/s^2`
- Moon gravity：`1.624 m/s^2`
- MuJoCo friction：`1.15 0.025 0.002`

新增和使用的模块：

- `StoneSlotNet`：根据石头几何先验和 slot/course/role 选择候选石头，不输入测试后验成功率。
- `PoseRankNet`：使用 MuJoCo 渲染 front/top 深度图和石头几何，缩小候选 pose 到 Top3。
- 严格墙线控制 V2：收紧单面墙 y 方向容差，降低把局部稳定石堆误判为墙的概率。
- 可行候选优先 V3：候选先过硬约束，再按分数排序。
- 物理门控 V4：中高层候选如果放置时扰动已有结构、残余速度过大或支撑重心偏差过大，则拒绝。
- `WallStateCritic V2`：用 front/top 深度观测和石头几何预测局部 wall-state risk，目前仍是离线模型，尚未接入闭环控制。

## StoneSlotNet

训练数据：

- `batch_runs/20260620_learning_dataset_4course_curriculum_v1/assignment_candidate_examples.csv`
- 样本数：552
- 正样本：92
- 输入：石头几何先验、slot 坐标、course、role、石头类别等。
- 明确排除：`candidate_rank`、`is_primary_assignment`，避免网络直接复制旧启发式排序。

模型路径：

- `batch_runs/20260620_torch_stone_slot_net_4course_assignment_v1_20260620_213352`

指标：

| 指标 | 数值 |
| --- | ---: |
| test accuracy | 0.812 |
| test precision | 0.333 |
| test recall | 0.588 |
| test f1 | 0.426 |
| grouped top1 hit | 0.529 |
| grouped top3 hit | 0.765 |

解释：

- Top1 不能信任，Top3/TopK 有一定价值。
- 适合作为候选池缩小器，不适合作为唯一选石策略。

## 闭环对照实验

共同配置：

- rocks：140
- rock profile：`high_wall`
- clusters：10
- candidate pose ranker：`batch_runs/20260620_torch_mujoco_depth_cnn_4course_groups256_structure_v1`
- pose TopK：3
- stone-fit ranker：`batch_runs/20260620_torch_stone_slot_net_4course_assignment_v1_20260620_213352`

### V1：StoneSlotNet + structure-aware PoseRankNet

路径：

- `batch_runs/20260620_highcourse_4course_stoneslot_structure_top3_seed97008_v1`

| gravity | success | shape | stable | failure | visible courses | height m | RMSE m | max drift m | y span m | aspect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 0/1 | 0/1 | 10 | 8 | 4 | 0.233 | 0.346 | 1.175 | 0.515 | 3.62 |
| Moon | 0/1 | 0/1 | 13 | 6 | 4 | 0.297 | 0.168 | 0.279 | 0.591 | 1.36 |

结论：

- 可见层数够，但形态仍偏石堆。
- 主要问题是 wall y span 过大，局部稳定不等于单面墙。

图片：

- `batch_runs/20260620_highcourse_4course_stoneslot_structure_top3_seed97008_v1/captures_960x720`

### V2：严格墙线约束

代码改动：

- `single_face_wall_4course_v1` 的稳定 target error 收紧为 `0.160 m`。
- wall y limit 收紧为 `0.115 m`。
- wall max y span 收紧为 `0.190 m`。
- wall aspect 最小值提高为 `2.80`。
- 候选 pose 生成时，高层 y 方向更多贴近目标墙线。
- 候选评分加入直接 y-line penalty。

路径：

- `batch_runs/20260620_highcourse_4course_stoneslot_structure_strictline_seed97009_v2`

| gravity | success | shape | stable | failure | visible courses | height m | RMSE m | max drift m | y span m | aspect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 0/1 | 0/1 | 10 | 6 | 4 | 0.227 | 0.136 | 0.283 | 0.479 | 1.57 |
| Moon | 0/1 | 0/1 | 8 | 6 | 3 | 0.312 | 0.192 | 0.344 | 0.447 | 1.63 |

结论：

- Earth 的 RMSE 和漂移显著改善，说明墙线约束有效。
- Moon 变差，说明低重力下不能直接套同一套严格释放/保持策略。
- 仍没有形成合格单面墙。

图片：

- `batch_runs/20260620_highcourse_4course_stoneslot_structure_strictline_seed97009_v2/captures_960x720`

### V3：可行候选优先

代码改动：

- 原逻辑：先选分数最低候选，再检查可行性；如果不可行就跳过。
- 新逻辑：先筛可行候选，再从可行候选中选最优；没有可行候选才跳过。

路径：

- `batch_runs/20260620_highcourse_4course_stoneslot_structure_feasiblefirst_seed97010_v3`

| gravity | success | shape | stable | failure | visible courses | height m | RMSE m | max drift m | y span m | aspect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 0/1 | 0/1 | 9 | 8 | 4 | 0.271 | 0.206 | 0.529 | 0.724 | 1.22 |
| Moon | 0/1 | 0/1 | 9 | 6 | 4 | 0.294 | 0.177 | 0.535 | 0.540 | 1.57 |

结论：

- 可行候选优先本身不是充分条件。
- 很多候选在短时落定时几何可行，但会在后保持阶段拖动结构。
- 需要增加扰动、残余速度、支撑重心偏差的硬门控或 WallStateCritic。

### V4：物理门控

代码改动：

中高层候选必须满足：

- `placed_disturbance_xy_m` 不超过约 `0.07-0.09 m`。
- `velocity_inf_norm_after_place <= 0.55`。
- `support_balance_error_m <= 0.135`。
- scoring 中也对扰动、速度和支撑重心偏差增加惩罚。

路径：

- `batch_runs/20260620_highcourse_4course_stoneslot_structure_physicsgate_seed97011_v4`

| gravity | success | shape | stable | failure | visible courses | height m | RMSE m | max drift m | y span m | aspect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 0/1 | 0/1 | 7 | 1 | 2 | 0.215 | 0.093 | 0.213 | 0.250 | 3.00 |
| Moon | 0/1 | 0/1 | 12 | 2 | 4 | 0.340 | 0.105 | 0.199 | 0.271 | 3.40 |

结论：

- V4 是本轮最有价值的闭环结果。
- Moon 下达到 4 层可见、12 块稳定、2 块失败、高度 `0.340 m`。
- 严格 shape 仍失败，主要因为墙厚 `0.271 m` 超过严格阈值 `0.190 m`，且有少数离群块。
- Earth 下门控太保守，只保留到 2 个可见层，但误差低，说明门控有效但需要地球/月球分开调参。

图片：

- `batch_runs/20260620_highcourse_4course_stoneslot_structure_physicsgate_seed97011_v4/captures_960x720`
- Moon 近成功案例：
  - `captures_960x720/00_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00/wall_front_rgb.png`
  - `captures_960x720/00_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00/wall_front_depth.png`
  - `captures_960x720/00_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00/wall_front_object_depth.png`
  - `captures_960x720/00_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00/wall_top_rgb.png`
  - `captures_960x720/00_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00/wall_top_depth.png`
  - `captures_960x720/00_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00/wall_top_object_depth.png`

补充说明：

- 旧的正视/侧视 depth PNG 会被 MuJoCo 远裁剪背景和地面深度主导，看起来几乎全黄。
- 已修正 `scripts/capture_cases.py`：普通 depth PNG 会先去掉远裁剪背景再归一化。
- 新增 `*_object_depth.png` 和 `*_object_depth.npy`，用 MuJoCo segmentation 保留 `objid > 0` 的石头几何并去掉地面。
- 后续看墙体结构、做汇报图和训练局部视觉 critic 时，优先使用 `wall_front_object_depth.png`、`right_object_depth.png`、`wall_top_object_depth.png`。

## 关键失败机理

V4 Moon 的高层 cap `rock_index=71` 在放置瞬间指标很好：

| 指标 | 数值 |
| --- | ---: |
| slot_id | 21 |
| course | 3 |
| target_y_error_m | 0.006 |
| support_overlap | 0.859 |
| support_contact_count | 4 |
| support_balance_error_m | 0.010 |
| placed_disturbance_xy_m | 0.003 |
| velocity_inf_norm_after_place | 0.079 |

但是最终保持后它产生了约 `0.199 m` 的漂移，导致严格失败。

解释：

- 候选瞬时几何和短时动力学不够。
- 当前墙体整体已经接近临界，后保持阶段会暴露累积不稳定。
- 这正是 `WallStateCritic` 应该学习的内容：给定当前墙体观测和候选石，预测后续保持风险。

## WallStateCritic V2

新学习表：

- `batch_runs/20260620_learning_dataset_4course_wallcritic_v2`

数据规模：

- placement examples：864
- candidate pose examples：19,588

MuJoCo 深度观测导出：

- `batch_runs/20260620_mujoco_depth_observation_maps_4course_wallcritic_v2`
- 样本：1536
- 候选组：512
- skipped render rows：0
- 输入通道：
  - `render_front_depth_norm`
  - `render_front_valid`
  - `render_top_depth_norm`
  - `render_top_valid`
  - `top_target_gaussian`
  - `top_candidate_footprint`
  - `front_target_gaussian`
  - `front_candidate_silhouette`
  - `gravity_ratio`
  - `course_ratio`

模型路径：

- `batch_runs/20260620_wall_state_critic_mujoco_depth_4course_wallcritic_v2`

测试指标：

| 指标 | 数值 |
| --- | ---: |
| test rows | 102 |
| test accuracy | 0.814 |
| test f1 | 0.895 |
| group top1 hit | 0.294 |
| group top3 hit | 1.000 |

注意：

- 正样本比例约 `0.98`，所以分类 accuracy/f1 不能单独说明模型强。
- Top3 很稳定，Top1 弱。
- 当前更适合作为候选过滤器或二级 critic，而不是直接输出唯一 placement。
- 该模型还没有接入闭环控制。

## 本轮科学结论

1. 只做 PoseRankNet 不够。它能减少搜索，但会把局部稳定结构推向厚堆或柱状堆。
2. StoneSlotNet 有用，但当前只能做 TopK 候选池缩小，不能独立选石。
3. 严格墙线约束能显著减少散堆，但会暴露月球低重力下的释放和保持问题。
4. 可行候选优先不是充分条件，因为候选短时可行不代表后保持稳定。
5. 物理门控有效，尤其是 Moon V4，把失败数从 V1 的 6 降到 2，并把结构推向线形墙段。
6. 后续真正应该神经网络化的是“当前墙体观测 + 候选石 + 目标槽”的风险评估，而不是只根据单石外观拟合 pose。

## 下一步

短期：

- 把 `WallStateCritic V2` 接入闭环，作为候选二级过滤器。
- Earth/Moon 使用不同扰动和速度阈值，避免 Earth 过保守、Moon 过松。
- 增加近成功 V4 的同分布实验次数，统计 V4 在 Moon 下 4 层墙的成功率、近成功率和失败类型。
- 把 shape label 拆成连续指标：`wall_y_span`、`wall_aspect`、`outlier_count`、`visible_courses`、`post_hold_drift`。

中期：

- 训练 `StoneSlotNet V2`，加入当前墙体局部观测，不只看石头和 slot。
- 训练 `WallStateCritic` 的 pairwise/ranking 版本，使它输出候选之间的相对风险。
- 从 2-3 层墙把网络成功率刷到 80-90%，再扩展到 4-5 层。

长期：

- 将小网络组合升级为端到端或半端到端策略：输入深度/点云、候选石几何和目标结构，输出石头选择、位姿和是否拒绝/修复。
- 目标仍是月面路标堆叠，最终扩展到干砌墙和干砌房屋结构。
