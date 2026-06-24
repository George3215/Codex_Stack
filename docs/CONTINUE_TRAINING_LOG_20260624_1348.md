# 2026-06-24 续跑记录：c13-c16 飞轮结果与漂移门控对照

记录时间：2026-06-24 13:48:50 +08:00  
仓库路径：`D:\MoonStack\experiments\moon_rock_stack`  
约束：只追加/修改实验日志、模型、统计和代码；不删除任何数据。

## 当前判断

昨晚到今天上午，master 自动调度从 c13 跑到 c16。严格 4 层墙没有新增成功，c10 仍是唯一严格 4 层正例。现在主要瓶颈不是“完全堆不到第 4 层”，而是：

1. 第 4 层经常能形成 visible 4，但墙线和最大水平漂移超出严格成功阈值。
2. 3 层墙已经能阶段性达到较高成功率，例如 c13 飞轮 3 层为 2/2。
3. 4 层失败主要集中在上层/封顶层漂移、支撑连续性不足、局部候选位姿被离线 top-k 选中但后续长时间 hold 不稳定。

因此本轮不继续盲目增加层数，而是追加“漂移门控对照实验”，检验更长 probe、更严格 moon gate、更高 PoseRisk 权重能否把 visible 4 转成严格成功。

## c09-c16 严格 4 层窗口统计

| cycle | trials | success | rate | max_height_m | best_success_drift_m | mean_visible_courses | mean_drift_m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c09 | 2 | 0 | 0.000 | 0.2606 | - | 3.5 | 0.3955 |
| c10 | 2 | 1 | 0.500 | 0.4562 | 0.0122 | 4.0 | 0.1097 |
| c11 | 2 | 0 | 0.000 | 0.3934 | - | 4.0 | 0.2830 |
| c12 | 2 | 0 | 0.000 | 0.3436 | - | 4.0 | 0.3588 |
| c13 | 2 | 0 | 0.000 | 0.2828 | - | 3.0 | 0.3624 |
| c14 | 2 | 0 | 0.000 | 0.3529 | - | 3.5 | 0.1808 |
| c15 | 2 | 0 | 0.000 | 0.2855 | - | 3.0 | 0.1924 |
| c16 | 2 | 0 | 0.000 | 0.1772 | - | 2.5 | 0.2170 |

解释：

- c10 仍是当前最好严格成功样本：height 0.4562 m，drift 0.0122 m。
- c11/c12 能看到 4 层，但漂移过大。
- c14/c15 漂移比 c11/c12 降低，但层数/高度不足。
- c16 严格 4 层退化，说明仅靠继续同一策略随机采样，收益已经较低。

## c13-c15 飞轮闭环结果

| cycle | target | trials | success_rate | shape_success_rate | visible_courses | height_m | drift_m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c13 | 3course moon | 2 | 1.0 | 1.0 | 3.0 | 0.3298 | 0.0086 |
| c13 | 4course moon | 2 | 0.0 | 0.0 | 4.0 | 0.3208 | 0.3304 |
| c14 | 3course moon | 2 | 0.0 | 0.5 | 3.0 | 0.2507 | 0.1061 |
| c14 | 4course moon | 2 | 0.0 | 0.0 | 3.5 | 0.2700 | 0.1869 |
| c15 | 3course moon | 2 | 0.5 | 0.5 | 3.0 | 0.3036 | 0.1069 |
| c15 | 4course moon | 2 | 0.0 | 0.0 | 4.0 | 0.3129 | 0.1832 |

解释：

- c13 的 3 层结果很好，说明低层和中层策略不是完全失效。
- c13/c15 的 4 层 visible_courses=4.0，但 drift 仍大于严格阈值，直接指向“第 4 层漂移/整体墙线维护”问题。
- 下一步应该重点压 drift，而不是只增加候选数或继续堆更高。

## c12 与 v19 同 seed 对照结论

对照路径：

- c12 最新模型：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260623_c12_latest_models_4course_parallel_eval_seed206258901`
- v19 基线：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260623_v19_baseline_4course_parallel_eval_seed206258901`

结果：

| scheme | trials | success_rate | visible_courses | height_m | max_drift_m | velocity_inf_norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c12 latest | 2 | 0.0 | 3.5 | 0.2788 | 0.2557 | 0.0494 |
| v19 baseline | 2 | 0.0 | 3.5 | 0.2866 | 0.2850 | 1.1767 |

解释：

- 两者都没有成功。
- c12 latest 稍微降低 drift 和 residual velocity，但没有达到严格成功。
- 不能把 c12 模型提升为主策略，只能把其数据纳入更大训练集。

## c16 新模型离线指标

### StoneSlotNet

目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_autonomous_wall_flywheel_master_v2_c16_flywheel_3to4_stone_slot_net`

指标：

- test_group top1：0.286
- test_group top3：0.357
- test recall：0.473
- test precision：0.0956

解释：StoneSlotNet 仍是弱前筛，不应单独主导选石。

### SupportMap

目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_autonomous_wall_flywheel_master_v2_c16_flywheel_3to4_pose_ranker_structure`

指标：

- test_groups：100
- test top1：0.450
- test top3：1.000
- 4 层月面 top1：0.479
- 4 层月面 top3：1.000
- 4 层月面 top1 quality regret：14.732
- 4 层月面 top3 quality regret：0.000

解释：c16 SupportMap 是目前离线指标最好的候选落点网络，但是否能带来严格成功必须看闭环 MuJoCo。

### WallStateCritic

目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_autonomous_wall_flywheel_master_v2_c16_flywheel_3to4_wall_state_critic`

指标：

- test group top1：0.360
- test group top3：1.000
- placed_disturbance_xy_m test MAE：0.162

解释：c16 WallStateCritic 比 c12 好，但仍没有接入控制链路。当前先作为风险分析模型保留。

## 当前新增漂移对照实验

### 1. c12 StoneSlot + c14 SupportMap + 漂移硬门控

目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260624_driftgate_c12stone_c14pose_4course_moon_seed206260401`

进程：PID 22976  
设置：

- target：`single_face_wall_4course_v1`
- gravity：moon
- trials：2
- candidates：12
- hold_steps：1440
- candidate_probe_steps：120
- `--candidate-probe-hard-gate`
- `--moon-gate-strict`
- StoneSlot：c12
- SupportMap：c14
- PoseRisk weight：0.65

目的：验证硬拒绝 probe 漂移候选是否能降低 4 层最大 drift。

### 2. c12 StoneSlot + c14 SupportMap + 软风险提高

目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260624_softrisk_c12stone_c14pose_4course_moon_seed206260402`

进程：PID 9900  
设置：

- 与上面相同，但不启用 hard gate / moon strict。
- PoseRisk weight：0.65
- candidate_probe_steps：120
- hold_steps：1440

目的：验证“更长 probe + 更高风险权重”是否比硬门控更稳，避免 hard gate 造成无可行候选。

### 3. c12 StoneSlot + c16 SupportMap + 漂移硬门控

目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260624_driftgate_c12stone_c16pose_4course_moon_seed206260403`

进程：PID 21740  
设置：

- 与硬门控 c14 版本一致。
- SupportMap 换成 c16。

目的：验证 c16 更好的离线落点网络能否配合硬漂移筛选突破 4 层。

## 当前自动飞轮

master 主控仍在运行：

- PID：12556
- session：`20260622_autonomous_wall_flywheel_master_v2`

c16 飞轮当前闭环评估：

- PID：22032
- 目录：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_autonomous_wall_flywheel_master_v2_c16_flywheel_3to4_closed_loop_eval`
- 使用：c16 StoneSlot + c16 SupportMap + PoseRisk v18b

## 远端 1080Ti

旧远端调度器于 2026-06-24 00:33:34 跑完 96 个周期后退出，期间远端始终 SSH 超时。

已重启新调度器：

- session：`20260624_remote_1080ti_wall_worker_restart`
- PID：5304
- cycles：192
- poll_seconds：900
- 当前状态：cycle 0 仍显示 `desktop-m57fdie.tail83f520.ts.net` SSH 22 端口超时。

## 下一步判据

1. 等待三个漂移对照与 c16 默认闭环 eval 完成。
2. 比较：
   - success_rate
   - visible_courses
   - stack_height_m
   - max_horizontal_drift_m
   - target_rmse_xy_m
   - skipped/failure_count
3. 若 hard gate 降低 drift 但导致 visible_courses 降低，说明硬门控太强，需要转向 soft risk / differentiable risk ranking。
4. 若 soft risk 降低 drift 且维持 visible_courses=4，可作为下一轮 master 的候选参数。
5. 若 c16 SupportMap + hard gate 优于 c14，则后续漂移实验优先使用 c16 SupportMap。

