# 2026-06-25 08:42 继续提升日志：4 层墙数据飞轮与 drift-guarded SupportMap

## 当前实验目的

当前目标不是盲目冲更高层，而是先把 3-4 层单面石墙的闭环成功率和失败数据做扎实，让神经网络逐步替代启发式搜索。重点指标包括：

- 3 层墙能否稳定成功；
- 4 层墙是否能保持 `visible_courses=4`；
- 4 层失败是否主要来自横向漂移、墙线变宽、末端速度或无可行位姿；
- 神经网络参与后是否减少候选位姿搜索成本，而不是只记住历史石头编号。

## 新增合并训练分支

昨晚新增并完成了一个合并训练分支：

- 脚本：`scripts/run_merged_c12_c16_supportmap_flywheel.ps1`
- 数据集：`batch_runs/20260624_merged_c12_c16_wall_dataset_20260624_153059`
- 深度/支撑图：`batch_runs/20260624_merged_c12_c16_mujoco_depth_maps_20260624_153146`
- 模型：`batch_runs/20260624_merged_c12_c16_supportmap_drift_guarded_train_20260624_190945`

训练数据来自 c12-c16 及早期 4 层墙正负样本，共 `14170` 行候选位姿，`2200` 个 rankable group。输入仍然是几何特征和观测支撑图，训练时使用 `--exclude-postsim-features`，也就是后验漂移/成功信息只做监督标签，不作为运行时输入。

模型指标：

- overall test top1/top3：`0.295 / 0.803`
- 4course moon holdout top1/top3：`0.289 / 0.729`
- 3course moon holdout top1/top3：`0.200 / 0.567`

解释：该模型离线 top1 不强，但目标是 drift-guarded，不是单纯拟合启发式 `candidate_score`。必须用闭环 MuJoCo 评测判断它是否减少 4 层漂移。

## c12-c19 闭环结果快照

下面是真实 MuJoCo closed-loop 结果，不是离线网络指标：

| cycle | target | success | shape | avg visible | max height | avg drift |
|---|---:|---:|---:|---:|---:|---:|
| c12 | 3course moon | 1/2 | 2/2 | 3.0 | 0.3000 | 0.0769 |
| c12 | 4course moon | 0/2 | 0/2 | 3.0 | 0.1789 | 0.2299 |
| c13 | 3course moon | 2/2 | 2/2 | 3.0 | 0.3392 | 0.0086 |
| c13 | 4course moon | 0/2 | 0/2 | 4.0 | 0.3621 | 0.3304 |
| c14 | 3course moon | 0/2 | 1/2 | 3.0 | 0.3002 | 0.1061 |
| c14 | 4course moon | 0/2 | 0/2 | 3.5 | 0.3401 | 0.1869 |
| c15 | 3course moon | 1/2 | 1/2 | 3.0 | 0.3173 | 0.1069 |
| c15 | 4course moon | 0/2 | 0/2 | 4.0 | 0.4179 | 0.1832 |
| c16 | 3course moon | 1/2 | 1/2 | 3.0 | 0.2953 | 0.1984 |
| c16 | 4course moon | 0/2 | 0/2 | 4.0 | 0.3130 | 0.5116 |
| c17 | 3course moon | 2/2 | 2/2 | 3.0 | 0.3162 | 0.0068 |
| c17 | 4course moon | 0/2 | 0/2 | 4.0 | 0.4372 | 0.2495 |
| c18 | 3course moon | 1/2 | 1/2 | 3.0 | 0.2851 | 0.2185 |
| c18 | 4course moon | 0/2 | 0/2 | 3.5 | 0.3427 | 0.2533 |
| c19 | 3course moon | 1/2 | 1/2 | 3.0 | 0.3244 | 0.1305 |

阶段判断：

- 3 层最好的是 c13/c17，均为 `2/2` 成功，其中 c17 漂移最低，`avg_drift=0.0068`。
- 4 层目前仍是 `0/2`，但 c13/c15/c16/c17 能保持 `visible_courses=4`，说明几何高度已经能搭出来，失败主要来自墙线质量和漂移。
- c17 是当前最值得作为 StoneSlot 基线的周期：StoneSlot test group top1/top3 为 `0.326 / 0.419`，比 c18/c19 更稳。

## line-lock 与漂移实验

已有 line-lock 试验：

- 输出：`batch_runs/20260624_linelock_c12stone_c16pose_4course_moon_seed206260404`
- 4course moon：`0/2` success，`0/2` shape，`avg_visible=4.0`，`max_height=0.3313`，`avg_drift=0.3441`

解释：line-lock 没有直接提高 success，但能把结构保持在可见 4 层，说明它适合继续作为 4 层探索策略。问题是漂移仍然大，特别是部分 trial 的 `velocity_inf_norm` 和 y 向展宽不可接受。

## 正在运行的新对比评测

为了让昨晚的新 drift-guarded SupportMap 真正进入闭环，启动了两个 4 层月面墙对照：

1. `batch_runs/20260625_merged_driftguard_c17stone_linelock_4course_moon_seed206270101`
   - strategy: `statics_wall_line_lock`
   - StoneSlot: `c17_flywheel_3to4_stone_slot_net`
   - SupportMap: `20260624_merged_c12_c16_supportmap_drift_guarded_train_20260624_190945`
   - PoseRisk: `20260622_poserisk_v18b_recent_3to4_train`
   - purpose: 测试 drift-guarded 支撑图网络是否能降低 line-lock 4 层漂移。

2. `batch_runs/20260625_merged_driftguard_c17stone_statics_4course_moon_seed206270102`
   - strategy: `statics_wall`
   - 其余网络相同
   - purpose: 作为非 line-lock 对照，判断改进来自网络还是来自墙线策略。

这两个任务在当前沙盒下通过 `cmd_start_b` 后台启动。之前 `async_process.py` 和 `Start-Process` 都出现了瞬时退出或 `Path/PATH` 环境键冲突；失败目录全部保留，没有删除。

## 当前瓶颈

4 层失败的主要模式仍然是：

- middle slot 后半段出现 `no_feasible_pose`；
- 结构能到 `visible_courses=4`，但横向漂移使 `shape_success=0`；
- 部分 trial 第 4 层有效高度增加，但墙面 y 向展宽过大，变成“堆”而不是墙；
- hard gate 会减少可行候选，soft risk 更容易保高度，但不一定保形状。

## 下一步决策规则

- 如果新 drift-guarded + line-lock 的 `avg_drift` 明显低于 c17 旧闭环的 `0.2495` 或出现 `success=1/2`，则把该 SupportMap 升级为主评测模型。
- 如果新模型保持 `visible_courses=4` 但漂移仍大，则继续收集其失败样本，训练下一版 PoseRisk/SupportMap，重点加权 course 2-3 的 middle/cap 位姿。
- 如果普通 `statics_wall` 明显好于 line-lock，则说明 line-lock 的几何先验过强，需要把墙线约束从硬启发式改成网络损失或软 penalty。
- 若两个都失败但 c19 旧调度出现 4 层高可见样本，则把 c19 的 candidate_pose_log 合并到下一轮训练集。

## 2026-06-25 09:05 追加记录

继续运行后观察到：

- c19 旧调度已经写出 4course trial0：`success=0`，`shape_success=0`，`visible_courses=4`，`stack_height=0.1779`，`max_horizontal_drift=0.2615`，`velocity_inf_norm=0.0128`。
- c19 4course trial1 继续推进到 middle 后半段，出现多次 `no_feasible_pose`，但已经放置到 14 块以上。它仍然是有价值的负样本来源。
- 新 drift-guarded + line-lock 评测进入 base 第 6-7 个 slot，速度较慢但稳定推进。
- 新 drift-guarded + ordinary statics_wall 评测进入 base 第 2 个 slot。

尝试过一个“直接合并 c17-c19 tensor，不重新渲染 MuJoCo depth”的方案：

- 新增脚本：`scripts/merge_support_map_tensor_dirs.py`
- 目的：按 `example_id` 去重合并 c17-c19 已渲染 tensor，减少重新渲染成本。
- 结果：逐样本重打包过慢，即使改为未压缩 `np.savez`，仍在前台超时。
- 失败输出目录保留：
  - `batch_runs/20260625_merged_c17_c19_mujoco_depth_maps_dedup`
  - `batch_runs/20260625_merged_c17_c19_mujoco_depth_maps_dedup_20260625_085829`

结论：当前不应继续走逐样本 tensor 重打包路线。更务实的路线是等 c19 与新 drift-guarded 对照闭环结束后，直接用 `candidate_pose_log.csv` 重新构建下一轮学习表，或修改训练器支持多个 tensor_dir streaming 读取，而不是生成一个巨大的重打包 tensor 目录。

## 2026-06-25 09:15 追加记录：multi-tensor 训练器

为了避开逐样本 tensor 重打包的低效问题，已修改 `scripts/train_torch_support_map_ranker.py`：

- 新增参数：`--extra-tensor-dir`
- 作用：训练时直接读取多个已有 tensor 目录，在内存中拼接，不重新渲染、不重打包、不改原始数据。
- smoke 输出：`batch_runs/20260625_supportmap_c17_c19_multi_tensor_smoke`
- 正式模型：`batch_runs/20260625_supportmap_c17_c19_multi_tensor_drift_guarded_train`

正式模型使用：

- primary tensor：`c17_flywheel_3to4_mujoco_depth_maps`
- extra tensor：`c18_flywheel_3to4_mujoco_depth_maps`
- extra tensor：`c19_flywheel_3to4_mujoco_depth_maps`
- rows：`7862`
- rankable groups：`2072`
- target mode：`drift_guarded`
- 输入仍为几何特征 + 支撑图，`exclude_postsim_features=true`

正式模型指标：

| split | top1 | top3 | top1 regret | top3 regret |
|---|---:|---:|---:|---:|
| overall test | 0.373 | 0.939 | 43.795 | 2.167 |
| 4course moon | 0.368 | 0.918 | 53.218 | 2.412 |
| 4course earth | 0.382 | 0.973 | 38.582 | 1.238 |
| 3course moon | 0.240 | 0.520 | 30.773 | 17.269 |

对比昨晚 c12-c16 merged drift-guarded 模型：

- c12-c16 merged overall top1/top3：`0.295 / 0.803`
- c17-c19 multi-tensor overall top1/top3：`0.373 / 0.939`
- c12-c16 merged 4course moon top1/top3：`0.289 / 0.729`
- c17-c19 multi-tensor 4course moon top1/top3：`0.368 / 0.918`

阶段结论：multi-tensor 支持是有效工程改进，避免了低效 tensor 重打包，并得到一个离线指标明显更好的 SupportMap。它应进入小规模闭环 probe。

已启动 probe：

- 输出：`batch_runs/20260625_c17_c19_multitensor_driftguard_linelock_4course_moon_seed206271201`
- strategy：`statics_wall_line_lock`
- trial：`1`
- StoneSlot：`c17_flywheel_3to4_stone_slot_net`
- SupportMap：`20260625_supportmap_c17_c19_multi_tensor_drift_guarded_train`
- PoseRisk：`v18b_recent_3to4_train`
- 当前状态：已完成 `model_loaded`，进入第 0 个 base slot。

现在并发任务较多，后续不再增加新的仿真并发，先等待 c19、merged driftguard line-lock/statics、c17-c19 probe 的闭环结果。
