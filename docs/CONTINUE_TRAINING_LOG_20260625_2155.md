# 2026-06-25 21:55 继续训练记录：c19-c21 闭环趋势与 c21 高层瓶颈

当前目标仍然是把 3-4 层单面墙的数据飞轮做稳，而不是盲目冲 10 层。今天晚间重点观察到：3 层已经能出现稳定结构，4 层仍未严格成功，但已经有高度、墙形和漂移都接近成功的样本。下一步应优先提高 4 层成功率和神经网络参与度，而不是单纯增加层数。

## 闭环结果快照

| run | target | trials | success | shape | avg height | max height | avg rmse | avg drift |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| c19 closed-loop | 3course moon | 2 | 1 | 1 | 0.320 | 0.324 | 0.051 | 0.131 |
| c19 closed-loop | 4course moon | 2 | 0 | 0 | 0.197 | 0.215 | 0.131 | 0.357 |
| c20 closed-loop | 3course moon | 2 | 2 | 2 | 0.288 | 0.290 | 0.025 | 0.002 |
| c20 closed-loop | 4course moon | 2 | 0 | 0 | 0.299 | 0.402 | 0.098 | 0.283 |
| c21 closed-loop | 3course moon | 2 | 1 | 1 | 0.342 | 0.357 | 0.025 | 0.012 |
| c21 closed-loop | 4course moon | 1 done + 1 running | 0 | 0 | 0.285 | 0.285 | 0.144 | 0.334 |
| c21 strict eval | 4course moon | 2 | 0 | 0 | 0.354 | 0.398 | 0.102 | 0.228 |

## 阶段性判断

- 3 层已经不是随机堆石头：c20 达到 `2/2`，c21 虽然 `1/2`，但失败 trial 的 `target_rmse=0.0318 m`、`drift=0.0161 m`、`velocity=0.0339` 都很低，说明它接近“形状判据边界失败”，不是散乱土堆。
- 4 层仍未严格成功，但已经出现多个科学上有价值的近成功：c20 trial1 高度 `0.402 m`，c21 strict trial1 高度 `0.398 m`、`target_rmse=0.0479 m`、`max_error=0.1196 m`、`drift=0.0131 m`、`wall_y_span=0.0376 m`、`outlier=0`。这说明墙形可以形成，主要失败来自跳过槽位和上层支撑不足。
- c21 closed-loop 4course trial1 运行到第 19 个 slot 时，已经进入 cap 层，但从 middle 后半段开始连续出现 `no_feasible_pose`。这条失败链很重要：上层失败不是简单塌落，而是中层缺可行姿态，导致 cap 层没有足够连续支撑。
- 当前更应该优化的是“石头-槽位可行性”和“候选姿态覆盖”，而不是继续硬冲更高层。具体来说，中层需要更宽的候选姿态、更强的 StoneSlot 筛选，以及对 course 2-3 的训练加权。

## 神经网络参与情况

- StoneSlotNet 输入仍只使用石头几何先验和目标槽位信息，不使用测试后验成功率。c21 StoneSlotNet test group top1/top3 为 `0.382 / 0.559`，说明它能参与选石，但还不足以单独解决中高层。
- c21 单轮 SupportMap 使用 RGB-D proxy + 石头几何 + 目标槽位，排除后验特征；但对新采样 run 的 holdout top1/top3 只有 `0.151 / 0.562`，4course moon top1/top3 为 `0.146 / 0.604`。这表明单轮 c21 数据过窄，容易泛化不足。
- c17-c19 multi-tensor drift-guarded SupportMap 离线指标明显更好：overall top1/top3 `0.373 / 0.939`，4course moon `0.368 / 0.918`。但它的一次 line-lock 闭环 probe 表现较差，说明离线 top-k 指标不能直接等价为闭环成功率。

## 今晚的合并训练尝试

尝试把 c17-c21 五轮 tensor 直接合并训练一个 course-weighted SupportMap：

- 目标模型目录：`batch_runs/20260625_supportmap_c17_c21_driftguard_courseweighted_train_foreground_2140`
- 输入：c17,c18,c19,c20,c21 的 MuJoCo depth/support maps。
- 训练目标：`drift_guarded`，并保留 `--exclude-postsim-features`。
- 加权：course 2 权重 `1.25`，course 3 权重 `1.65`，目的是提高 middle/cap 决策学习强度。
- 结果：前台训练 15 分钟未完成，被工具超时中断；未得到可用模型。空输出目录保留，作为调度失败/训练成本记录。

工程结论：

- 全量 c17-c21 RGB-D ranking 训练不能再用前台长命令跑，必须拆成较小数据窗口，或由稳定后台调度器管理。
- 下一轮更合理的训练方式是先用 c19-c21 或 c20-c21 做 25-40 epoch 的小模型，验证闭环收益；如果有效再扩大到 c17-c21。
- 不能只追求离线 top1/top3，需要把闭环指标写入模型选择：`success`、`shape_success`、`visible_courses`、`skipped_slot_count`、`no_feasible_pose` 次数、`wall_y_span`、`drift`、`velocity`。
- 针对 4 层瓶颈，下一轮策略应优先增加 course 1-2 middle 槽位的候选覆盖，而不是只增加 cap 层搜索。因为 cap 层失败经常是中层支撑缺口的后果。

## 下一步执行策略

1. 等 c21 4course trial1 完整落盘，不中断。
2. 若 trial1 形成高墙但失败，优先保存 RGB、top depth、wall front/top depth，并标注为“可行姿态不足型高层失败”。
3. 用 c20-c21 或 c19-c21 训练小规模 course-weighted SupportMap，控制训练时间，先验证闭环收益。
4. 下一轮 4 层实验增加 middle 槽位候选覆盖，保留 good priors，但让网络参与候选落点排序。
5. 如果 4 层成功率仍低，暂不进入 5 层，继续围绕 4 层收集正负样本和 near-success 样本。

## 2026-06-25 22:05 补充：c21 trial1 完整结果与图像记录

c21 closed-loop 4course trial1 已完整落盘。过程里一度放到 13 块并进入 cap 层，但 final hold 后稳定统计退化，最终指标如下：

- target: `single_face_wall_4course_v1`
- gravity: `moon`
- success / shape_success: `0 / 0`
- visible_courses: `4`
- stable_count / failure_count: `8 / 5`
- skipped_slot_count: `11`
- stack_height: `0.1577 m`
- target_rmse: `0.1442 m`
- target_max_error: `0.2840 m`
- max_horizontal_drift: `0.2906 m`
- wall_y_span: `0.2817 m`
- wall_outlier_count: `5`
- velocity_inf_norm: `0.0201`

解释：这是一个典型“中层缺槽导致上层支撑不足”的负样本。它不是高速崩塌型失败，最终速度很低；主要问题是 `no_feasible_pose` 多、跳过槽位多，导致 final hold 后墙体退化、墙面横向散开。

本轮自动抓图已经完成：

- capture root: `batch_runs/20260622_autonomous_wall_flywheel_master_v2_c21_flywheel_3to4_closed_loop_eval/captures_960x720`
- case 0: 3course success trial0
- case 1: 4course failure trial1，典型“高层缺槽/退化”负样本
- case 2: 4course failure trial0，典型“漂移/墙面散开”负样本
- case 3: 3course failure trial1，接近成功但形状判据失败
- 文件数量：`84` 张 PNG，`56` 个 depth NPY，包含 front、wall_front、right、back、left、top、wall_top 等视角，以及 RGB/depth/object_depth。

下一轮训练标签建议：

- 对 trial1 增加 `no_feasible_pose_count` 或由 `skipped_slot_count` 派生的负标签权重。
- 让 course 1-2 的 middle slot 获得更高训练权重，因为 cap 层失败是中层缺支撑的结果。
- 在候选姿态生成阶段增加 middle 槽位覆盖，而不是只加 cap 层搜索次数。
- 保留 `visible_courses=4` 作为 near-success 指标，但不能把它直接当成功；必须同时约束 `wall_y_span`、`outlier_count`、`skipped_slot_count` 和 `drift`。

## 2026-06-25 22:45 补充：低高度释放假设与第 4 层冲击失败

用户观察到一个关键失败机制：当前石头释放位置偏高，尤其到第 4 层时，石头下落动能会把已经堆好的中上层结构冲散。这个判断非常重要，因为它指出失败不一定来自选石或落点本身，也可能来自“执行动作”带来的额外动量。真实 dry stacking 中，放石头通常是尽量贴近支撑面、小高度释放或缓慢放置；如果仿真中直接从较高位置自由落下，会把策略问题和执行冲击混在一起。

本轮已经把这个假设转成可复现实验开关：

- 新增参数：`--low-release-search`
- 新增参数：`--release-search-step-m`，本轮使用 `0.004 m`
- 新增参数：`--release-extra-clearance-m`，本轮使用 `0.003 m`
- 实现位置：`moon_rock_stack/structured.py`
- 扫描逻辑：对 course > 0 的候选姿态进行 MuJoCo 接触扫描，从原始释放高度向下搜索，找到“即将接触但还没有穿透/碰撞”的最低释放高度，然后只保留很小的安全间隙。
- 记录字段：`release_original_z`、`release_z`、`release_drop_reduction_m`、`release_search_checks`、`release_contact_z`、`release_contact_clearance_m` 等。
- 为了避免 base 层浪费大量接触扫描成本，低高度释放暂时只用于 course > 0。base 层本来就是落在地面上，不是当前“高层冲散结构”假设的主要对象。

低高度释放 smoke 证明了这个机制不是微小误差：

- output: `batch_runs/20260625_low_release_smoke_2course_moon`
- 候选姿态数量：`198`
- 平均降低释放高度：`0.02259 m`
- 最大降低释放高度：`0.089 m`
- 超过 `1 cm` 的候选：`87`

随后做了一条 4 层月面 fast probe：

- output: `batch_runs/20260625_low_release_fastprobe_4course_moon_c21models`
- target: `single_face_wall_4course_v1`
- gravity: `moon`
- candidate_count: `6`
- neural modules: c21 StoneSlotNet + c21 SupportMap + PoseRisk v18b
- success / shape_success: `0 / 0`
- visible_courses: `3`
- stable_count / failure_count: `12 / 2`
- skipped_slot_count: `10`
- stack_height: `0.3663 m`
- target_rmse: `0.0786 m`
- max_horizontal_drift: `0.2373 m`
- wall_y_span: `0.2557 m`
- wall_outlier_count: `2`
- velocity_inf_norm: `0.0547`

和旧 c21 4 层 trial0 对比，低高度释放虽然还没有形成严格成功，但方向是正的：

| 指标 | 旧 c21 trial0 | low-release fast probe | 变化 |
|---|---:|---:|---:|
| stable_count | 12 | 12 | 持平 |
| failure_count | 4 | 2 | 降低 |
| stack_height_m | 0.2854 | 0.3663 | 提高 |
| target_rmse_xy_m | 0.1440 | 0.0786 | 降低 |
| max_horizontal_drift_m | 0.3337 | 0.2373 | 降低 |
| wall_y_span_m | 0.3975 | 0.2557 | 降低 |
| wall_outlier_count | 4 | 2 | 降低 |

release scan 的分层统计进一步支持“高层更需要低释放”：

| course | count | mean drop reduction | max drop reduction | >1cm | >3cm |
|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.00990 m | 0.033 m | 142 | 4 |
| 2 | 300 | 0.01416 m | 0.057 m | 142 | 43 |
| 3 | 250 | 0.02429 m | 0.073 m | 191 | 83 |

解释：越到高层，原始候选释放高度和真实接触高度之间的差越大；如果不做低释放，第 3-4 层更容易因为额外下落动能而把墙面冲散。这个机制应该纳入后续动作模型和神经网络输出：网络不只输出“放在哪里”，还应该输出或约束“以多低的安全高度释放/接近”。

本轮低释放典型失败样例已抓图：

- capture root: `batch_runs/20260625_low_release_fastprobe_4course_moon_c21models/captures_960x720_low_release_20260625`
- case: `00_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00`
- 包含：front、wall_front、right、back、left、top、wall_top 的 RGB/depth/object_depth。
- 这条样例适合作为“低释放改善但仍缺候选覆盖”的负样本：它比旧 trial 更高、更直、更少外点，但因为 candidate_count 和 top-k 较小，仍有 `10` 个 skipped slots，导致无法严格形成 4 层墙。
- 人工复查图像后，`wall_front_rgb.png` 能清楚看到左侧形成三层/局部四层趋势、右侧缺槽；`top_depth.png` 和 `wall_top_depth.png` 对墙体连通性最有价值；`wall_front_depth.png` 能看高度但背景动态范围偏大，汇报中应优先展示 wall_front RGB + top/wall_top depth。

下一步执行策略：

1. 将低高度释放作为 4 层及以上默认动作先验，尤其用于 course 2-3。
2. 做一条更公平的 full probe：恢复较高 candidate/top-k 设置，只改变 low-release 开关，与旧 c21 4 层 trial0 做单变量对照。
3. 如果 full probe 仍失败，优先统计失败类型：`no_feasible_pose`、高层冲击扰动、墙面横向散开、cap 层支撑不足。
4. 后续网络化方向要把低释放纳入输出或损失：候选落点网络负责位置，动作执行模块负责安全接近高度，避免把可行动作失败误判成石头/槽位不可行。

## 2026-06-26 00:15 补充：旧释放动作 c23 严格 4 层基线

并行调度器完成了一条旧释放动作的 c23 strict 4course 基线。注意这条没有启用 `--low-release-search`，因此只能作为“未修正释放高度”的对照，不应该和新的低释放动作混为同一类策略。

- output: `batch_runs/20260622_autonomous_wall_flywheel_master_v2_c23_strict_4course`
- trials: `2`
- low_release_search_requested: `0`
- success / shape_success: `0 / 0`
- mean_visible_courses: `3.0`
- mean_stable_count: `10.5`
- mean_failure_count: `3.0`
- mean_stack_height: `0.2355 m`
- mean_target_rmse: `0.1075 m`
- mean_max_drift: `0.2794 m`
- mean_velocity_inf_norm: `0.0281`

trial 级现象：

| trial | stable | failure | skipped | visible | height | rmse | drift | y_span |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9 | 3 | 12 | 3 | 0.1799 | 0.0958 | 0.1975 | 0.2092 |
| 1 | 12 | 3 | 9 | 3 | 0.2911 | 0.1193 | 0.3612 | 0.3739 |

解释：旧动作基线没有出现严格 4 层成形。它仍然表现为中高层 `no_feasible_pose` 和墙面横向扩散。和低释放 fast probe 相比，旧基线平均高度低、漂移大、外形更散；这进一步支持下一轮必须用低释放 full probe 做公平对照。

已启动新的公平对照：

- job: `batch_runs/async_jobs/20260625_225500_c21_low_release_fullprobe_4course_v2`
- output: `batch_runs/20260625_c21_low_release_fullprobe_4course_moon_seed206266789_v2`
- 差异变量：启用 `--low-release-search`
- 保持较高搜索强度：`candidate_count=10`、`candidate_pose_top_k=8`、`stone_fit_top_k=14`
- 目标：判断低释放是否能在高候选配置下减少第 4 层冲散和中高层退化。

调度系统也已接入低释放：

- `scripts/auto_wall_scale_scheduler.py` 新增默认开启的 `--low-release-search/--no-low-release-search`。
- 调度器 strict eval 现在会把 `--low-release-search --release-search-step-m 0.004 --release-extra-clearance-m 0.003` 传给结构化实验。
- 调度器 data flywheel 也会把同样参数传给 `scripts/run_wall_data_flywheel.py`。
- `scripts/run_wall_data_flywheel.py` 已支持并转发低释放参数到 collection 和 closed-loop eval。
- dry-run 验证目录：
  - `batch_runs/auto_wall_scale/20260625_low_release_scheduler_dryrun`
  - `batch_runs/20260625_low_release_flywheel_dryrun`
- 验证结果：strict、collection、closed-loop eval 命令都包含 `--low-release-search`，因此后续新样本不会继续默认使用高处释放动作。

旧 master 接管处理：

- 旧 v2 master: `batch_runs/async_jobs/20260622_231000_cmd_autonomous_wall_flywheel_master_v2`
- 该 master 是低释放改动前启动的长期进程，不会自动吸收新参数。
- 已停止旧 master 进程 `12556`，只停止调度循环；已生成的 c24 子任务和全部输出保留，不删除。
- 已在旧 master 目录写入停止说明：`STOPPED_BY_CODEX_20260626_LOW_RELEASE.txt`
- 新 low-release master: `batch_runs/async_jobs/20260626_093000_cmd_autonomous_wall_flywheel_lowrelease_master`
- 新 session: `batch_runs/auto_wall_scale/20260626_low_release_wall_master_v1`
- 新 master 第 0 轮识别到两个 active job 并等待：
  - `batch_runs/async_jobs/20260625_225500_c21_low_release_fullprobe_4course_v2`
  - `batch_runs/async_jobs/20260626_054025_20260622_autonomous_wall_flywheel_master_v2_c24_flywheel_3to4_data_train`
- 因此当前不会立刻堆新任务，等这两条完成后才会发下一轮低释放 strict/flywheel。

c24 旧释放动作飞轮已结束，作为对照保留：

- job: `batch_runs/async_jobs/20260626_054025_20260622_autonomous_wall_flywheel_master_v2_c24_flywheel_3to4_data_train`
- exit_code: `0`
- eval output: `batch_runs/20260622_autonomous_wall_flywheel_master_v2_c24_flywheel_3to4_closed_loop_eval`
- low_release_search_requested: `0`
- 3course moon: `2/2` success，说明 3 层任务已稳定。
- 4course moon: `0/2` success，mean visible courses `3.5`，mean stable_count `7.5`，mean failure_count `5.5`，mean skipped_slot_count `11.0`，mean height `0.1546 m`，mean RMSE `0.1884 m`，mean drift `0.4276 m`。

4course trial 级结果：

| trial | stable | failure | skipped | visible | height | rmse | drift | y_span | outliers | velocity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 6 | 10 | 3 | 0.1594 | 0.1971 | 0.3990 | 0.4018 | 6 | 3.3338 |
| 1 | 7 | 5 | 12 | 4 | 0.1497 | 0.1798 | 0.4561 | 0.6712 | 5 | 0.0226 |

解释：c24 的神经网络能把 3 层做稳定，但旧释放动作在 4 层仍然会产生严重退化。trial0 的 final velocity 很高，属于明显冲击/散落型失败；trial1 虽然 visible_courses 到 4，但 skipped slots 多、墙面横向散开严重。后续低释放数据应优先替代这类旧动作样本。

## 2026-06-26 10:10 补充：low-release full probe 4 层 near-success 结果

公平对照 full probe 已完成。它使用更高候选配置，只把动作执行改成低高度释放：

- job: `batch_runs/async_jobs/20260625_225500_c21_low_release_fullprobe_4course_v2`
- output: `batch_runs/20260625_c21_low_release_fullprobe_4course_moon_seed206266789_v2`
- target: `single_face_wall_4course_v1`
- gravity: `moon`
- candidate_count: `10`
- candidate_pose_top_k: `8`
- stone_fit_top_k: `14`
- low_release_search_requested: `1`
- release_search_step_m: `0.004`
- release_extra_clearance_m: `0.003`
- success / shape_success: `0 / 0`
- visible_courses: `4`
- stable_count / failure_count: `15 / 1`
- skipped_slot_count: `8`
- stack_height: `0.3285 m`
- target_rmse: `0.0549 m`
- target_max_error: `0.1344 m`
- max_horizontal_drift: `0.1337 m`
- wall_y_span: `0.1859 m`
- wall_outlier_count: `1`
- velocity_inf_norm: `2.1225`

和旧释放动作 c24 4course 均值相比：

| 指标 | c24 old-release mean | low-release full probe | 变化 |
|---|---:|---:|---:|
| success | 0/2 | 0/1 | 未严格成功 |
| visible_courses | 3.5 | 4.0 | 提高 |
| stable_count | 7.5 | 15 | 显著提高 |
| failure_count | 5.5 | 1 | 显著降低 |
| skipped_slot_count | 11.0 | 8 | 降低 |
| stack_height_m | 0.1546 | 0.3285 | 显著提高 |
| target_rmse_xy_m | 0.1884 | 0.0549 | 显著降低 |
| max_horizontal_drift_m | 0.4276 | 0.1337 | 显著降低 |
| wall_outlier_count | 5-6 | 1 | 显著降低 |

分层跳槽统计：

| course | placed | skipped | skip reason |
|---:|---:|---:|---|
| 0 | 7 | 0 | - |
| 1 | 5 | 1 | `no_feasible_pose` |
| 2 | 3 | 3 | `no_feasible_pose` |
| 3 | 1 | 4 | `no_feasible_pose` |

解释：低释放显著减少了冲击散落和整体漂移，墙体已经更像“墙”而不是“堆”。但它没有解决高层候选覆盖问题，尤其 course 2 后半段和 cap 层仍然大量 `no_feasible_pose`。因此下一步不应该再单纯降低释放高度，而应该把“低释放动作先验 + 高层连续支撑槽覆盖”一起作为训练目标。

release scan 统计：

| course | candidates | mean drop reduction | max drop reduction | >1cm | >3cm |
|---:|---:|---:|---:|---:|---:|
| 1 | 672 | 0.00777 m | 0.053 m | 226 | 32 |
| 2 | 672 | 0.01092 m | 0.045 m | 230 | 136 |
| 3 | 560 | 0.00862 m | 0.037 m | 216 | 4 |

图像记录：

- capture root: `batch_runs/20260625_c21_low_release_fullprobe_4course_moon_seed206266789_v2/captures_960x720_low_release_fullprobe_20260626`
- case: `00_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00`
- 推荐展示：
  - `wall_front_rgb.png`：能清楚显示这已经是单面墙，不是土堆。
  - `wall_top_depth.png`：能看出墙体连通和 cap 层缺口。
  - `top_depth.png`：能看出俯视高度分布。

下一步：

1. 新 low-release master 已接管，等待当前 active jobs 完成后自动发下一轮。
2. 后续训练应提高 course 2-3 的槽位覆盖权重，特别是对 `no_feasible_pose` 的 hard negative 建模。
3. 神经网络目标应从“单块候选评分”进一步转为“局部连续支撑窗口评分”：不仅判断某个石头能不能放，还要判断放完后是否给下一层留下可用支撑。
4. 对 cap 层增加候选生成宽度和局部重排策略；否则会出现当前这种“下部墙体已成形，但顶层没有连续可行位姿”的 near-success。
