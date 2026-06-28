# 20260628 回到多面体堆叠记录

日期: 2026-06-28  
用户指令: 注意力先不要在 NASA 相关石头上，回到之前的多面体堆叠。  
安全边界: 不删除、不覆盖历史数据；只停止非核心后台计算并保留已写出的部分结果。

## 1. 已执行的切换

停止了正在占用 CPU/GPU 的非多面体任务，包括:

```text
high_wall 4/5 层严格评估
c12 high_wall 3/4 flywheel
旧 wall4 network replacement fast/v2
旧自动 high_wall scheduler
```

这些任务的已有 `batch_runs` 输出、日志和部分数据均保留，没有删除。

## 2. 当前只关注训练用多面体

本轮明确不把注意力放在 NASA-like 石头上:

```text
nasa_like_wall
nasa_like_wall_v2
nasa_like_wall_v3
```

这些仍作为 held-out 测试/泛化诊断，不进入当前训练主线。

当前多面体主线:

```text
encyclopedic_poly_train
convex_poly_wall_train
convex_poly_diverse_train
```

重点是:

```text
1. 多面体 3/4 层单面墙采样。
2. 从多面体正负样本训练 StoneSlotNet / PoseRiskNet / SupportMapRanker / WallStateCritic。
3. 用多面体网络逐步缩小候选搜索空间。
4. 最终评估 single_face_wall_4course_v1 月面严格成功率。
```

## 3. 新启动任务

任务名:

```text
20260628_polyhedral_wall4_flywheel_v1
```

启动脚本:

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260628_cmd_polyhedral_wall4_flywheel_v1\run.cmd
```

流水线输出:

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_polyhedral_wall4_flywheel_v1
```

新采样输出:

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_polyhedral_wall4_flywheel_v1_collect_explore_00_seed20628620
```

## 4. 数据来源

本轮使用 `--no-default-prior-runs`，因此不会混入默认 high_wall 历史 run。

prior run 只包含多面体:

```text
20260628_4to5_encyclopedic_poly_curriculum_cmd_v1
20260628_convex_poly_wall_train_3course_cmd_v1
20260628_convex_poly_wall_train_smoke_v1
20260628_encyclopedic_poly_train_smoke_v1
```

新采样:

```text
rock_profile: encyclopedic_poly_train
targets: single_face_wall_3course_v1,single_face_wall_4course_v1
gravity: moon
collect_mode: explore
commit_best_rejected: true
```

## 5. 网络替代启发式设置

采样阶段:

```text
explore + commit-best
```

目的:

```text
先扩充多面体正负样本，特别是 no_feasible_pose、best rejected、上层漂移和 target error 样本。
```

训练阶段:

```text
StoneSlotNet: stone-slot 几何匹配
PoseRiskNet: 候选位姿风险过滤
SupportMapRanker: top-depth/support map 候选排序
WallStateCritic: 局部墙体状态评分
```

评估阶段:

```text
stone-fit-top-k: 12
candidate-pose-top-k: 3
stone-fit-ranker-max-course: -1
candidate-pose-ranker-max-course: -1
pose-risk-ranker-max-course: -1
pose-risk-weight: 0.55
eval_commit_best_rejected: false
```

解释:

- 启发式仍负责候选生成。
- 网络负责将石头候选和位姿候选压缩到小集合。
- `max-course=-1` 表示网络覆盖所有层，不只覆盖前 3 层。
- 严格评估不使用 `commit-best`，因此可统计真实成功率。

## 6. 当前早期进度

已确认进程命令行:

```text
rock-profile encyclopedic_poly_train
targets single_face_wall_3course_v1,single_face_wall_4course_v1
gravities moon
```

早期 MuJoCo 进度:

```text
single_face_wall_3course_v1 / moon / trial 0 已开始。
base 第 0、1、2 个槽位已完成放置。
structured_progress.csv、features.csv、cluster_summary.csv 已开始写入。
```

## 7. 后续判据

任务完成后必须汇报:

```text
3 层多面体 strict success_rate
4 层多面体 strict success_rate
shape_success_rate
visible_courses
stable_count
target_rmse_xy_m
target_max_xy_error_m
max_horizontal_drift_m
velocity_inf_norm
no_feasible_pose 按 course/role 分布
哪些 face_count / rectangularity / roundness_proxy / concavity_proxy 更适合 base/middle/cap
```

如果 4 层 strict success 暂时仍为 0，也要记录:

```text
1. 是否比之前更完整地达到 visible_courses=4。
2. no_feasible_pose 是否减少。
3. target RMSE 是否下降。
4. 网络 top-k 是否过窄导致错过可行候选。
```
