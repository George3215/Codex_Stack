# 月面干叠石墙的多小网络执行计划与阶段结果

日期：2026-06-19

目标：先用多个可解释、可独立验证的小网络替代启发式搜索中的关键环节，而不是直接训练一个难以诊断的大端到端模型。最终目标仍然是堆叠更高、更稳定的单面干叠石墙，并逐步过渡到月面路标和更大尺度干叠结构。

## 1. 当前判断

早期阶段不适合直接做一个大网络端到端输出整面墙的所有石头位置，原因是：

- 当前数据量仍然偏小，尤其是跨随机种子、跨石头目录的泛化样本不够。
- 高墙失败原因多样，包括石头几何不合适、局部支撑不足、候选位姿不稳、上层扰动下层、月球低重力下的慢漂移等。
- 如果直接做大模型，失败后很难判断是石头编码、位姿选择、风险预测还是全局规划出了问题。

因此当前路线是模块化小网络：

1. 石头几何编码器：学习石头的形状、棱角、支撑面、厚度、可互锁性。
2. 候选位姿 imitation ranker：模仿当前搜索器选中的候选姿态，减少 MuJoCo 真实尝试数量。
3. 候选质量 / 风险 ranker：不模仿旧搜索器，而是学习候选姿态仿真后的质量分数。
4. 失败预测器：预测滑移、扰动、目标误差、速度未收敛等失败风险。
5. 全局墙体 planner：决定某一层、某一槽位优先用什么角色的石头，以及何时需要修正或换石头。

当前已经完成并接入闭环的是第 2 和第 3 个模块的早期版本。

## 2. 已实现的两个候选位姿小网络

代码：

- `scripts/train_torch_support_map_ranker.py`

这次新增了 `--target-mode`：

- `--target-mode selected`：旧模式，训练 imitation ranker，标签是 `label_selected_by_pose_search`。
- `--target-mode score`：新模式，训练 candidate quality / risk ranker，标签是仿真后的 `candidate_score`。

两个模式的输入保持一致，并且在闭环可用模型中使用 `--exclude-postsim-features`，即不把仿真后才知道的字段作为输入。

### 2.1 输入

每个候选位姿输入由两部分组成：

1. 局部支撑图：`[8, 64, 64]`
   - `height_before_m`
   - `support_occupancy`
   - `support_count_clipped`
   - `target_gaussian`
   - `candidate_footprint`
   - `candidate_height_m`
   - `gravity_ratio`
   - `course_ratio`

2. 数值特征：26 个 pre-sim 特征 + 26 个 present mask，总维度 52。
   - 槽位和候选姿态：`course, target_x, target_y, pose_x/y/z, pose_qw/qx/qy/qz`
   - 候选编号：`candidate_id, candidate_count`
   - 石头几何：`rock_volume, rock_surface_area, rock_bbox_x/y/z, rock_elongation, rock_flatness, rock_sphericity, rock_roughness, rock_angularity, rock_spike_score, rock_compactness, rock_stability_score, rock_mass`

明确排除的仿真后输入：

- `candidate_score`
- `support_overlap`
- `support_contact_count`
- `support_balance_error_m`
- `bearing_pressure_proxy`

这些字段可以作为监督信号，但不能作为闭环推理输入。

### 2.2 输出

两个网络都输出每个候选位姿的一个标量分数。闭环时，在同一个石头、同一个槽位的候选集合内排序，取 top-k 送进 MuJoCo。

当前闭环参数：

- `candidate_count = 5`
- `candidate_pose_top_k = 3`

## 3. 新质量网络训练结果

训练数据：

- 张量目录：`batch_runs/20260619_local_support_maps_multicatalog_wall_plus93013_v1`
- 行数：15395 个候选位姿
- rankable groups：3269
- 显式 holdout：`20260619_newrocks_highwall_seed93013_fullcandidates_highonly_fg_v1`

模型目录：

- `batch_runs/20260619_torch_support_map_cnn_quality_ranker_multicatalog_presim_holdout93013_v1`

训练命令核心参数：

```powershell
python scripts/train_torch_support_map_ranker.py `
  --tensor-dir batch_runs\20260619_local_support_maps_multicatalog_wall_plus93013_v1 `
  --output batch_runs\20260619_torch_support_map_cnn_quality_ranker_multicatalog_presim_holdout93013_v1 `
  --target-mode score `
  --quality-temperature 35 `
  --exclude-postsim-features `
  --test-run-name 20260619_newrocks_highwall_seed93013_fullcandidates_highonly_fg_v1 `
  --device cuda --amp
```

93013 整批新石头 holdout 结果：

| 指标 | 数值 |
| --- | ---: |
| test top-1 hit | 0.278 |
| test top-3 hit | 0.704 |
| train top-1 hit | 0.485 |
| train top-3 hit | 0.832 |
| test mean top-1 quality regret | 18.402 |
| test mean top-3 quality regret | 4.474 |

分重力：

| 条件 | groups | top-1 | top-3 | top-1 regret | top-3 regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| 高墙 / 地球 | 248 | 0.266 | 0.685 | 22.190 | 5.267 |
| 高墙 / 月球 | 248 | 0.290 | 0.722 | 14.614 | 3.680 |

解释：

- 新质量网络在 top-3 上能覆盖约 70% 的最高质量候选，说明它学到了一部分几何和局部支撑规律。
- top-1 仍然偏低，说明它还不能单独做唯一决策，现阶段更适合作为 top-k 筛选或 rerank 模块。
- 训练集和测试集仍有差距，需要继续增加新石头目录，尤其是高墙专用全候选数据。

## 4. 同 seed 高墙闭环对照

为了验证新小网络是否真的对高墙有用，用同一个新石头种子 `94014` 做闭环对照。

共同参数：

- rocks：200
- profile：`high_wall`
- target：`single_face_wall_high_v1`
- strategy：`statics_wall`
- gravities：`earth,moon`
- candidates：5
- top-k：3
- steps-per-rock：360
- hold-steps：900

### 4.1 Imitation ranker

模型：

- `batch_runs/20260619_torch_support_map_cnn_ranker_multicatalog_presim_holdout93013_v1`

闭环结果目录：

- `batch_runs/20260619_closed_loop_highwall_newrocks_seed94014_mixed_holdout93013_top3_v1`

| 重力 | 严格成功 | 可见层 | 稳定石头 | 失败数 | 高度 m | RMSE m | 结构分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 地球 | 0 | 6 | 14 | 6 | 0.190 | 0.167 | 0.566 |
| 月球 | 0 | 5 | 8 | 2 | 0.303 | 0.136 | 1.942 |

### 4.2 Quality / risk ranker

模型：

- `batch_runs/20260619_torch_support_map_cnn_quality_ranker_multicatalog_presim_holdout93013_v1`

闭环结果目录：

- `batch_runs/20260619_closed_loop_highwall_newrocks_seed94014_quality_holdout93013_top3_v1`

| 重力 | 严格成功 | 可见层 | 稳定石头 | 失败数 | 高度 m | RMSE m | 结构分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 地球 | 0 | 8 | 17 | 7 | 0.287 | 0.165 | 1.106 |
| 月球 | 0 | 6 | 13 | 4 | 0.316 | 0.188 | 1.032 |

阶段结论：

- 对“更高墙”目标，quality ranker 有正向信号：地球从 6 层提升到 8 层，月球从 5 层提升到 6 层。
- 地球条件下质量网络同时提升了高度、可见层数、稳定石头数和结构分。
- 月球条件下高度略升、层数提升，但 RMSE 和结构分变差，说明质量分数还没有充分惩罚墙体形状偏移。
- 两个网络都没有达到严格成功，说明仅替代候选位姿排序还不够，必须继续增加石头几何编码、失败预测和修正策略。

## 5. 图像记录

质量网络的典型失败案例已渲染：

- `batch_runs/20260619_closed_loop_highwall_newrocks_seed94014_quality_holdout93013_top3_v1/captures_960x720`

地球高墙失败案例：

- `00_single_face_wall_high_v1_failure_statics_wall_earth_trial_00/wall_front_rgb.png`
- `00_single_face_wall_high_v1_failure_statics_wall_earth_trial_00/wall_front_depth.png`
- `00_single_face_wall_high_v1_failure_statics_wall_earth_trial_00/wall_top_rgb.png`
- `00_single_face_wall_high_v1_failure_statics_wall_earth_trial_00/wall_top_depth.png`

月球高墙失败案例：

- `01_single_face_wall_high_v1_failure_statics_wall_moon_trial_00/wall_front_rgb.png`
- `01_single_face_wall_high_v1_failure_statics_wall_moon_trial_00/wall_front_depth.png`
- `01_single_face_wall_high_v1_failure_statics_wall_moon_trial_00/wall_top_rgb.png`
- `01_single_face_wall_high_v1_failure_statics_wall_moon_trial_00/wall_top_depth.png`

## 6. 下一步执行计划

### 6.1 继续生成更多石头目录

目的：防止网络记住具体石头，而不是学习几何规律。

建议新增至少 5 到 8 个全候选高墙目录：

- `high_wall` profile：主训练目录
- `single_face_wall` profile：检查泛化
- `screening_stress` profile：增加难例

每个目录都应保留完整 candidate pose log，不能只保留神经网络 top-k 后的截断候选。

### 6.2 石头几何编码器

输入：

- 石头点云或 mesh 采样点
- 当前已有的几何特征表

输出：

- stone embedding
- role suitability：base / middle / cap / chock / tie / buttress
- 几何风险：过尖、过扁、支撑面不足、重心偏高、可互锁性弱

推荐模型：

- PointNet 作为第一版，因为数据量小、可解释、训练快。
- 后续再比较 PointNet++ 或 DGCNN。

验证方式：

- 整批石头 seed holdout。
- 看 embedding 是否能提升候选质量 ranker 的 top-3 和闭环高墙层数。

### 6.3 失败预测器

输入：

- 支撑图
- 候选姿态
- 石头几何 embedding
- 重力条件
- 层数和角色

输出：

- 是否大概率失败
- 是否会产生大扰动
- 是否目标误差过大
- 是否速度未收敛
- 是否低重力下慢漂移

标签来源：

- `target_error_xy_m`
- `placed_disturbance_xy_m`
- `velocity_inf_norm_after_place`
- `support_balance_error_m`
- `height_gain_m`
- `failure_cases.csv`

用途：

- 对 imitation / quality ranker 给出的 top-k 做 veto。
- 如果 top-k 都高风险，则触发换石头或生成新的候选姿态。

### 6.4 全局 planner

现阶段不做大端到端 planner，先做轻量级模块：

- 输入当前墙体状态和剩余石头集合。
- 输出下一槽位优先级、角色需求、候选石头池。
- 结合 dry stacking 先验：大而稳定石头放底层，长石/桥接石用于跨缝，楔形/小石用于锁固，顶部优先低重心盖石。

验证方式：

- 高墙目标先看可见层数、高度、RMSE、漂移和失败原因。
- 不急着扩展四面墙，先把单面高墙做稳。

## 7. 当前结论

多小网络路线是合理的。今天新增的 quality / risk ranker 不是最终答案，但已经证明“换监督目标”会改变闭环行为，并且在同一新石头 seed 上提升了高墙层数。下一阶段要把质量网络从单一 `candidate_score` 监督升级为多头风险预测，并引入石头几何编码器，才能真正减少无效尝试并稳定堆更高的墙。
