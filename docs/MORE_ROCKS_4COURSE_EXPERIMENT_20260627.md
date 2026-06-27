# 2026-06-27 更多石头池 4 层月面墙实验

## 目的

当前 4 层墙的主要失败不是完全无法形成 4 层，而是：

- `no_feasible_pose` 较多；
- `skipped_slot_count` 较高；
- 有些 trial 已经出现 `visible_courses = 4`，但槽位没有填完整；
- 有些结构稳定但不满足 strict wall success。

因此本轮增加石头池大小，验证“更多候选石头是否能减少 skipped slot / no feasible pose，并提高 4 层墙完整性”。

## 对照基线

使用当前 c06 闭环评估作为基线：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_low_release_wall_master_v1_c06_flywheel_3to4_closed_loop_eval
```

c06 已经完成的 3 层结果：

```text
3 层月面墙 strict success = 2/2
3 层月面墙 shape success = 2/2
```

c06 的 4 层部分在本记录生成时仍在运行。

## 本轮新增实验

输出目录：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260627_more_rocks_4course_moon_c06models_260rocks_top24_v1
```

异步任务目录：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260627_cmd_more_rocks_4course_moon_c06models_v1
```

## 参数变化

| 参数 | c06 基线 | 本轮更多石头 |
|---|---:|---:|
| target | 3/4 层 | 4 层 |
| gravity | moon | moon |
| rocks | 130 | 260 |
| clusters | 10 | 14 |
| stone_fit_top_k | 14 | 24 |
| candidate_pose_top_k | 8 | 8 |
| candidates per slot | 10 | 10 |
| candidate_probe_steps | 50 | 50 |
| pose_risk_weight | 0.45 | 0.45 |
| low_release_search | on | on |
| trials | 2 | 2 |

这轮尽量只扩大石头池和石头候选数量，位姿网络、PoseRisk 权重和低释放策略保持 c06 设置，便于判断“更多石头”本身是否改善 4 层墙。

## 使用模型

```text
StoneSlotNet:
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_low_release_wall_master_v1_c06_flywheel_3to4_stone_slot_net

SupportMapRanker:
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_low_release_wall_master_v1_c06_flywheel_3to4_pose_ranker_structure

PoseRiskNet:
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_low_release_wall_master_v1_c06_flywheel_3to4_pose_risk_net
```

## 需要观察的指标

重点不是只看 strict success，而是同时看：

- `success`
- `shape_success`
- `visible_courses`
- `skipped_slot_count`
- `stable_count / rock_count`
- `failure_count`
- `target_rmse_xy_m`
- `max_horizontal_drift_m`
- `no_feasible_pose` 在 `structured_progress.csv` 和 `placement_log.csv` 中的数量

如果 `skipped_slot_count` 明显下降，但 drift / failure 上升，说明更多石头增加了填充机会，但稳定性判别仍不足。

如果 `skipped_slot_count` 下降且 stable fraction 保持高，说明增加石头池是有效方向，后续应把更多石头池纳入数据飞轮。

如果 `skipped_slot_count` 不下降，说明瓶颈不是石头池数量，而是位姿生成、支撑约束或网络排序。
