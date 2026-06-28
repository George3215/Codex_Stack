# 2026-06-28 随机多面体石头堆叠记录

## 实验目的

本轮继续使用非平滑、多棱角、多面体石头进行月面单面墙堆叠。重点不是把石头堆成土堆，而是检查随机多面体石头在 3 层单面墙目标下的失败模式，并为后续神经网络替代启发式策略提供负样本和近成功样本。

本轮新增一个对照：在相同 `nasa_like_wall_v3` 随机多面体生成规则下，比较不开启与开启 `base_support_prior`、`base_continuity_prior` 后，墙线连续性、高度、漂移和失败类型的变化。

## 实验环境

- 仿真器：MuJoCo
- 重力：月面重力，`1.624 m/s^2`
- 目标结构：`single_face_wall_3course_v1`
- 策略：`statics_wall`
- 石头 profile：`nasa_like_wall_v3`
- 形状约束：多面体、棱角化、无尖刺、避免特别扁的片状石
- 网络参与：
  - `StoneSlotNet`：石头/槽位候选筛选
  - `CandidatePoseRankNet`：候选位姿排序
  - `PoseRiskNet`：候选位姿风险惩罚
- 执行策略：低释放高度搜索，避免高处自由落体动能把结构冲散

## 批次 A：随机多面体 3 层墙

输出目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_random_polyhedral_v3_3course_batch_fg_v1`

参数摘要：

- `rocks=120`
- `clusters=10`
- `trials=2`
- `candidates=5`
- `stone_fit_top_k=12`
- `candidate_pose_top_k=5`
- `pose_risk_weight=0.45`
- `commit_best_rejected=true`
- `base_support_prior=false`
- `base_continuity_prior=false`

结果：

| 指标 | 数值 |
| --- | ---: |
| trial 数 | 2 |
| 严格成功率 | 0.0 |
| 形状成功率 | 0.0 |
| 平均稳定石头数 | 12.0 / 15 |
| 平均失败石头数 | 3.0 |
| 平均可见层数 | 3.0 |
| 平均结构分数 | 0.7214 |
| 平均目标 RMSE | 0.1777 m |
| 平均最大目标误差 | 0.5478 m |
| 平均堆叠高度 | 0.1956 m |
| 平均最大水平漂移 | 0.0161 m |
| 平均速度无穷范数 | 0.4134 |

失败分布：

| 失败类型 | course | role | 数量 |
| --- | ---: | --- | ---: |
| `missed_target` | 0 | base | 3 |
| `missed_target` | 1 | middle | 2 |
| `unstable_structure` | 2 | cap | 1 |

分析：

- 这批不是严重坍塌，最大水平漂移较低，说明低释放和风险网络有一定稳定作用。
- 主要问题是墙形目标误差，尤其是第一层和第二层没有排成连续墙线。
- 俯视深度图显示墙线被分裂成多段，正视图显示局部达到 3 层但整体不像单面墙。

典型失败图：

- 正视 RGB：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_random_polyhedral_v3_3course_batch_fg_v1\captures_random_v3_batch_failures_v1\00_single_face_wall_3course_v1_failure_statics_wall_moon_trial_00\wall_front_rgb.png`
- 俯视 object depth：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_random_polyhedral_v3_3course_batch_fg_v1\captures_random_v3_batch_failures_v1\00_single_face_wall_3course_v1_failure_statics_wall_moon_trial_00\wall_top_object_depth.png`

## 批次 B：加入 base support 与 base continuity 先验

输出目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_random_polyhedral_v3_continuity_3course_batch_fg_v1`

参数摘要：

- `rocks=140`
- `clusters=12`
- `trials=1`
- `candidates=4`
- `stone_fit_top_k=14`
- `candidate_pose_top_k=4`
- `pose_risk_weight=0.45`
- `commit_best_rejected=true`
- `base_support_prior=true`
- `base_support_prior_weight=0.9`
- `base_continuity_prior=true`
- `base_continuity_prior_weight=1.1`

结果：

| 指标 | 数值 |
| --- | ---: |
| trial 数 | 1 |
| 严格成功率 | 0.0 |
| 形状成功率 | 0.0 |
| 稳定石头数 | 12 / 15 |
| 失败石头数 | 3 |
| 可见层数 | 3 |
| 结构分数 | 0.7891 |
| 目标 RMSE | 0.1531 m |
| 最大目标误差 | 0.4192 m |
| 堆叠高度 | 0.2802 m |
| 最大水平漂移 | 0.0704 m |
| 速度无穷范数 | 1.3977 |
| 墙体 x 跨度 | 0.9762 m |
| 墙体 y 跨度 | 0.4319 m |
| 外点数 | 2 |

失败分布：

| 失败类型 | course | role | 数量 |
| --- | ---: | --- | ---: |
| `missed_target` | 0 | base | 1 |
| `missed_target` | 1 | middle | 1 |
| `unstable_structure` | 2 | cap | 1 |

分析：

- 加入支撑/连续性先验后，高度从批次 A 的平均 `0.1956 m` 提升到 `0.2802 m`。
- 目标 RMSE 从 `0.1777 m` 降到 `0.1531 m`，最大目标误差也下降。
- 但最大水平漂移从 `0.0161 m` 升到 `0.0704 m`，速度无穷范数从 `0.4134` 升到 `1.3977`。
- 正视图显示结构更高，但出现前后厚度和散落石头，形态更像石堆而不是薄单面墙。
- 因此，单独强化 base support/continuity 会鼓励更高堆叠，但不足以保证墙体薄度、外点控制和高层稳定。

典型失败图：

- 正视 RGB：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_random_polyhedral_v3_continuity_3course_batch_fg_v1\captures_continuity_failure_v1\00_single_face_wall_3course_v1_failure_statics_wall_moon_trial_00\wall_front_rgb.png`
- 俯视 object depth：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_random_polyhedral_v3_continuity_3course_batch_fg_v1\captures_continuity_failure_v1\00_single_face_wall_3course_v1_failure_statics_wall_moon_trial_00\wall_top_object_depth.png`

## 结论

1. 低释放高度是必要的，但不是充分条件。本轮批次 A 的平均释放降低约 `0.0321 m`，漂移小；批次 B 的平均释放降低约 `0.0045 m`，漂移显著增大。后续需要同时记录释放高度、速度、漂移，不应只看是否达到层数。
2. 随机多面体石头能够形成 3 层可见结构，但严格单面墙成功仍为 0。失败主要不是高度不足，而是墙线连续性、墙厚控制、目标槽位误差和上层 cap 稳定性。
3. `base_support_prior` 与 `base_continuity_prior` 有阶段性价值：它们提升高度并降低部分目标误差，但会把策略推向更厚、更散的石堆形态。
4. 下一轮网络/损失函数不能只奖励高度或稳定数量，必须同时惩罚：
   - `target_rmse_xy_m`
   - `target_max_xy_error_m`
   - `wall_y_span_m`
   - `wall_outlier_count`
   - `max_horizontal_drift_m`
   - `velocity_inf_norm`
   - 高层 `cap` 的不稳定

## 下一步

- 保留 `nasa_like_wall_v3` 作为随机多面体数据源，继续扩充负样本和近成功样本。
- 对 3 层墙优先优化“薄墙形态”，不盲目追求高度。
- 在 candidate pose 网络中加入墙厚和外点惩罚，把“不是石堆”的判别标准变成训练标签或损失项。
- 对 base/middle/cap 分层训练风险模型：base 关注支撑宽度与连续性，middle 关注 load path 和横向误差，cap 关注局部稳定和扰动。
- 当 3 层随机多面体墙能稳定达到较低 RMSE、低墙厚、低漂移后，再重新推进 4 层墙。

## 附：本轮异步状态

- 旧 c11 飞轮任务仍在运行，使用 `high_wall` profile，不属于本轮随机多面体对照。
- 本轮尝试启动 `20260628_random_polyhedral_v3_continuity_3course_batch_v1` detached 异步任务，PID 很快退出，stdout/stderr 为空，没有产生实验输出目录。
- 该失败启动记录已保留在：
  `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260628_142057_20260628_random_polyhedral_v3_continuity_3course_batch_v1`
- 为保证有效数据，本轮改用前台小批次产出 `20260628_random_polyhedral_v3_continuity_3course_batch_fg_v1`。
- 随后改用 `.cmd` 后台方式启动第三个随机多面体对照批次：
  `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260628_cmd_random_polyhedral_v3_mixed_3course_v1`
- 第三个批次输出目录：
  `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_random_polyhedral_v3_mixed_3course_cmd_v1`
- 第三个批次目的：降低 `base_support_prior`/`base_continuity_prior` 权重，同时提高 `pose_risk_weight=0.65`，检查是否能保留高度收益但减少石堆化、外点和漂移。
- 启动检查：`started_at.txt` 已写入，输出目录已产生 `meshes`、`mjcf`、`features.csv`、`structured_progress.csv`，说明 `.cmd` 后台方式有效。
