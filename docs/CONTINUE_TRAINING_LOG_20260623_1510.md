# 2026-06-23 15:10 续跑记录：c12 数据飞轮与 4 层对照评估

记录时间：2026-06-23 15:10:40 +08:00  
仓库路径：`D:\MoonStack\experiments\moon_rock_stack`  
约束：只追加/修改日志、代码、模型、统计结果；不删除任何实验数据。

## 当前目的

本轮继续围绕 3-4 层单面墙数据飞轮展开，重点不是盲目冲更高层，而是验证：

1. c12 新收集数据是否能训练出更好的石头-槽位前筛网络。
2. c12 新的局部深度/支撑图是否能训练出更好的候选位姿排序网络。
3. 新模型在闭环 MuJoCo 评估中是否真的提高 4 层月面墙成功率。
4. 同 seed 下和 v19 基线对照，避免把随机性误判为方法提升。

## c12 飞轮阶段结果

飞轮主目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_autonomous_wall_flywheel_master_v2_c12_flywheel_3to4`

采集子任务：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_autonomous_wall_flywheel_master_v2_c12_flywheel_3to4_collect_exploit_00_seed206252708`

阶段状态：

| 阶段 | 状态 | 结果 |
| --- | --- | --- |
| collect | 完成 | 13:10:16 - 14:20:40，返回码 0 |
| dataset | 完成 | 14:20:40 - 14:20:47，返回码 0 |
| depth export | 完成 | 14:20:47 - 14:57:50，返回码 0 |
| StoneSlotNet | 完成 | 14:57:50 - 14:58:04，返回码 0 |
| SupportMap | 完成 | 14:58:04 - 14:59:14，返回码 0 |
| WallStateCritic | 完成 | 14:59:14 - 14:59:37，返回码 0 |
| closed-loop eval | 运行中 | PID 20828 |

## c12 数据集规模

数据集目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_autonomous_wall_flywheel_master_v2_c12_flywheel_3to4_learning_dataset`

统计：

- run_dir_count：9
- run_example_count：34
- placement_example_count：807
- candidate_pose_example_count：22167
- assignment_candidate_example_count：6498
- 月面 placement：351 条，其中 success 168、failure 65、skipped 118
- 地球 placement：456 条，其中 success 249、failure 69、skipped 138

角色分布：

- base：237 条，success 167，failure 12，skipped 58
- middle：401 条，success 198，failure 73，skipped 130
- cap：169 条，success 52，failure 49，skipped 68

解释：cap 层仍是主要困难区，失败率明显高于 base，这和 4 层墙上层漂移/无可行姿态问题一致。

## depth/support map 导出质量

导出目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_autonomous_wall_flywheel_master_v2_c12_flywheel_3to4_mujoco_depth_maps`

统计：

- source_row_count_after_filter：21360
- selected_group_count：768
- row_count：2639
- shard_count：3
- grid_size：64
- channels：10
- skipped_row_count：0

重要通道均值/方差：

- `render_top_depth_norm` mean 0.4706，std 0.0380
- `render_top_valid` mean 1.0，std 0.0
- `top_target_gaussian` mean 0.0095，std 0.0683
- `top_candidate_footprint` mean 0.0174，std 0.1308
- `front_candidate_silhouette` mean 0.0309，std 0.1730

解释：top depth 不再是单纯全黄无效图；真正有价值的是 top depth 与 target/candidate footprint 的组合。front depth 仍作为辅助，核心决策应优先依赖 top depth 和支撑 footprint。

## c12 StoneSlotNet

模型目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_autonomous_wall_flywheel_master_v2_c12_flywheel_3to4_stone_slot_net`

输入：

- 石头几何先验：体积、表面积、bbox、elongation、flatness、sphericity、roughness、angularity、spike_score、compactness、stability_score 等。
- 面结构先验：major_face_count、largest_face_area_ratio、top3_face_area_ratio、face_area_entropy、normal_concentration、support_face_count、opposing_face_pair_count 等。
- 槽位先验：target、role、course。
- 不使用 candidate_rank 或仿真后验成功率作为输入。

指标：

- row_count：6498
- positive_count：416
- test accuracy：0.748
- test precision：0.0897
- test recall：0.292
- test F1：0.137
- test_group top1_hit_rate：0.293
- test_group top3_hit_rate：0.415

解释：StoneSlotNet 仍是召回导向的石头-槽位前筛网络。它不适合单独决定最终堆叠，但可减少无意义石头候选。

## c12 SupportMap 默认模型

模型目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_autonomous_wall_flywheel_master_v2_c12_flywheel_3to4_pose_ranker_structure`

输入：

- MuJoCo 渲染的局部 top/front depth proxy。
- target/candidate footprint。
- 石头几何和候选位姿数值特征。
- 不使用 post-simulation features 作为输入。

指标：

- row_count：2639
- rankable_group_count：768
- test_groups：42
- test_top1_hit_rate：0.333
- test_top3_hit_rate：1.000
- 4 层月面 top1：0.308
- 4 层月面 top3：1.000
- 4 层月面 top1 quality regret：17.815
- 4 层月面 top3 quality regret：0.000

解释：top3 很强，说明它很适合作为候选压缩器；top1 仍不够，最终必须保留 top-k 与物理 probe。

## WallStateCritic

模型目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_autonomous_wall_flywheel_master_v2_c12_flywheel_3to4_wall_state_critic`

测试结果：

- test group top1_hit_rate：0.104
- test group top3_hit_rate：0.358
- target_x_error MAE：1.029 m
- target_y_error MAE：0.310 m
- placed_disturbance_xy MAE：0.233 m

解释：WallStateCritic 当前不适合强接入控制决策。它可以作为失败分析模型继续保留，但不应作为当前 4 层墙主策略。

## v20/v21/v22 消融

### v20：旧 v19 数据上的高层加权模型

目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260623_supportmap_v20_structureaware_highcourse_train`

和 v19 对比：

- v19 4 层月面 top1：0.269，top3：0.642
- v20 4 层月面 top1：0.202，top3：0.563

结论：v20 退化，不接入主流程。

### v21：c12 数据上的角色/层级加权消融

目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260623_c12_supportmap_v21_roleweighted_train`

和 c12 默认 SupportMap 对比：

- c12 默认 4 层月面 top1：0.308，top3：1.000
- v21 4 层月面 top1：0.269，top3：1.000

结论：top3 持平，top1 下降；不替换默认 c12 模型。

### v22：c12 数据 seed-check

目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260623_c12_supportmap_v22_seedcheck_train`

指标：

- test_groups：54
- test_top1_hit_rate：0.500
- test_top3_hit_rate：1.000
- 4 层月面 top1：0.545
- 4 层月面 top3：1.000

解释：v22 指标更好，但 test run 与 c12 默认不同，说明 c12 数据集的 SupportMap 指标对 split 很敏感。结论应依赖闭环 MuJoCo 评估，而不是只看离线 top-k。

## 正在运行的闭环评估

### c12 主飞轮 closed-loop eval

目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_autonomous_wall_flywheel_master_v2_c12_flywheel_3to4_closed_loop_eval`

进程：PID 20828  
目标：3 层 + 4 层月面墙，验证 c12 最新 StoneSlotNet + SupportMap + PoseRisk。  
当前进度：3 层墙 trial 0 已进入第 1 层 middle 附近。

### c12 最新模型 4 层专测

目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260623_c12_latest_models_4course_parallel_eval_seed206258901`

进程：PID 22524  
目标：只测 4 层月面墙，使用 c12 最新 StoneSlotNet + c12 默认 SupportMap + PoseRisk。  
当前进度：4 层墙 trial 0 base 层进行中。

### v19 同 seed 基线 4 层专测

目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260623_v19_baseline_4course_parallel_eval_seed206258901`

进程：PID 22804  
目标：同 seed 对照旧 v19 方案，判断 c12 新模型是否真正提升。  
当前进度：4 层墙 trial 0 base 层进行中。

## 当前判断

1. c12 数据飞轮已经完成一轮完整闭环：采集、数据集、depth/support map、StoneSlotNet、SupportMap、WallStateCritic、闭环 eval。
2. StoneSlotNet 原生崩溃修复有效，c12 训练顺利完成。
3. SupportMap 比 StoneSlotNet 更直接影响候选落点，当前 top3 指标更有价值。
4. WallStateCritic 当前不应作为强控制器，需要更多失败样本和更明确的风险标签。
5. 必须等待 4 层同 seed 对照评估完成，才能判断 c12 新模型是否真的优于 v19。

## 下一步

1. 等待三条 MuJoCo 评估完成。
2. 如果 c12 最新模型 4 层成功率高于 v19，同步更新后续 master 调度使用的候选模型路径。
3. 如果 c12 与 v19 持平或更差，继续保留 v19 作为主策略，同时把 c12 新数据纳入更大规模训练集。
4. 对成功或典型失败样本保存正视 RGB、俯视 depth、过程 GIF。
5. 下一轮日志必须统计：方案、数据集、seed、trials、success_rate、visible_courses、height、max_drift。

