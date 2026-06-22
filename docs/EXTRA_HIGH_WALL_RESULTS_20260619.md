# 10 层单面高墙阶段实验记录

日期：2026-06-19

目标：在已有 8 层 `single_face_wall_high_v1` 基础上，继续尝试更高的单面干叠石墙。为此新增 10 层目标 `single_face_wall_extra_high_v1`，并使用扩展数据训练后的 quality / risk ranker 进行闭环堆叠。

## 1. 新增目标结构

代码位置：

- `moon_rock_stack/structured.py`

新增目标：

- `single_face_wall_extra_high_v1`

结构：

- 10 个 course
- 44 个目标槽位
- 单面 tapered wall：底层更宽，上层逐渐收窄
- 槽位数分布：
  - course 0：7
  - course 1：6
  - course 2：6
  - course 3：5
  - course 4：5
  - course 5：4
  - course 6：4
  - course 7：3
  - course 8：2
  - course 9：2

这比上一阶段的 8 层 `single_face_wall_high_v1` 更难，目标不是立刻严格成功，而是逼出更高墙的失败模式和训练数据。

## 2. 扩展训练数据

新增并入训练表的全候选数据：

- `20260619_newrocks_highwall_seed93014_fullcandidates_highonly_parallel_v1`
- `20260619_newrocks_screening_seed93015_fullcandidates_highonly_parallel_v1`

扩展学习集：

- `batch_runs/20260619_learning_dataset_multicatalog_presim_wall_plus93013_93014_93015_v1`

规模：

| 项目 | 数量 |
| --- | ---: |
| run dirs | 14 |
| run examples | 47 |
| placement examples | 874 |
| candidate pose examples | 20355 |
| assignment candidate examples | 504 |

张量目录：

- `batch_runs/20260619_local_support_maps_multicatalog_wall_plus93013_93014_93015_v1`

张量规模：

- row_count：20355
- shard_count：10
- map shape：`[8, 64, 64]`

## 3. 新版 quality / risk ranker

模型目录：

- `batch_runs/20260619_torch_support_map_cnn_quality_ranker_expanded_temp20_holdout93015_v1`

训练设置：

- target mode：`score`
- quality temperature：20
- dropout：0.30
- hidden：160
- batch size：96
- epochs：90
- input：pre-sim support map + pre-sim numeric geometry features
- explicitly excluded post-sim inputs：`candidate_score`, `support_overlap`, `support_contact_count`, `support_balance_error_m`, `bearing_pressure_proxy`

整批 holdout：

- `20260619_newrocks_screening_seed93015_fullcandidates_highonly_parallel_v1`

结果：

| 指标 | 数值 |
| --- | ---: |
| test top-1 | 0.315 |
| test top-3 | 0.722 |
| train top-1 | 0.379 |
| train top-3 | 0.771 |
| test top-1 quality regret | 17.418 |
| test top-3 quality regret | 4.269 |

分重力：

| 条件 | top-1 | top-3 | top-1 regret | top-3 regret |
| --- | ---: | ---: | ---: | ---: |
| 地球高墙 | 0.270 | 0.702 | 22.454 | 5.365 |
| 月球高墙 | 0.359 | 0.742 | 12.383 | 3.173 |

解释：

- 扩展数据后，stress profile holdout 的 top-3 达到 0.722，说明继续生成不同石头目录是有效的。
- 月球条件 top-3 更高，但闭环仍可能因低重力慢漂移和结构误差失败。

## 4. 10 层 extra-high 闭环实验

实验目录：

- `batch_runs/20260619_closed_loop_extra_highwall_seed94016_quality_expanded_top3_v1`

参数：

- rocks：260
- rock profile：`high_wall`
- clusters：14
- seed：94016
- target：`single_face_wall_extra_high_v1`
- strategy：`statics_wall`
- gravities：earth, moon
- candidates：5
- candidate top-k：3
- steps-per-rock：380
- hold-steps：950

结果：

| 重力 | 严格成功 | 可见层 | 稳定石头 | 失败数 | 高度 m | RMSE m | 最大目标误差 m | 结构分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 地球 | 0 | 7 | 13 | 8 | 0.430 | 0.465 | 1.845 | -1.577 |
| 月球 | 0 | 5 | 11 | 8 | 0.274 | 0.204 | 0.444 | -0.439 |

阶段判断：

- 地球条件下高度达到 `0.430 m`，这是目前高墙高度上的一个突破，但形状误差很大，说明它更像“局部堆高成功”，还不是可接受墙体。
- 月球条件下可见层只有 5，速度残差高，说明低重力下结构进入慢漂移/未充分收敛状态。
- 10 层目标对当前位姿 ranker 来说已经明显超出稳定能力，下一步必须加入 failure predictor 和修正策略。

## 5. 图像记录

图片目录：

- `batch_runs/20260619_closed_loop_extra_highwall_seed94016_quality_expanded_top3_v1/captures_960x720`

地球典型失败：

- `00_single_face_wall_extra_high_v1_failure_statics_wall_earth_trial_00/wall_front_rgb.png`
- `00_single_face_wall_extra_high_v1_failure_statics_wall_earth_trial_00/wall_front_depth.png`
- `00_single_face_wall_extra_high_v1_failure_statics_wall_earth_trial_00/wall_top_rgb.png`
- `00_single_face_wall_extra_high_v1_failure_statics_wall_earth_trial_00/wall_top_depth.png`

月球典型失败：

- `01_single_face_wall_extra_high_v1_failure_statics_wall_moon_trial_00/wall_front_rgb.png`
- `01_single_face_wall_extra_high_v1_failure_statics_wall_moon_trial_00/wall_front_depth.png`
- `01_single_face_wall_extra_high_v1_failure_statics_wall_moon_trial_00/wall_top_rgb.png`
- `01_single_face_wall_extra_high_v1_failure_statics_wall_moon_trial_00/wall_top_depth.png`

## 6. 下一步

下一步不应只继续增加 course 数，而应解决高墙失败机制：

1. 训练 failure predictor。
   - 输入仍然使用 pre-sim 支撑图、候选姿态、石头几何。
   - 标签来自 `target_error_xy_m`, `placed_disturbance_xy_m`, `velocity_inf_norm_after_place`, `failure_cases.csv`。
   - 用途是在 quality ranker top-k 之后 veto 高风险候选。

2. 训练 role-aware stone selector。
   - 区分底层、腰部、顶部对石头形状的要求。
   - 底层优先厚、大、支撑面广、低重心石头。
   - 中上层优先较轻、支撑面明确、扰动小的石头。

3. 增加 repair policy。
   - 如果某层出现大目标误差或侧向偏移，下一块石头不继续盲目堆高，而是优先修正支撑面。
   - 对月球条件尤其需要慢漂移检测和更长 hold/probe。

4. 继续生成全候选数据，但重点转向失败丰富的 extra-high 目标。
   - 需要保留全候选日志，不能只保存神经 top-k 截断数据。
   - 建议下一批生成 `single_face_wall_extra_high_v1` 的全候选数据，用于专门训练高墙 failure predictor。

结论：10 层高墙已经能在地球条件下局部堆到 `0.430 m`，但目前不是结构成功。下一阶段的科学问题已经很明确：不是简单“选更高分候选”，而是要学会识别高墙中后期的失稳风险，并主动修正支撑结构。
