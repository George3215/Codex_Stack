# 20260628 4/5 层石墙堆叠任务记录

日期: 2026-06-28  
任务: 启动四层、五层单面墙堆叠实验。  
安全边界: 只创建新目录、日志和实验输出，不删除、不覆盖历史数据。

## 1. 本轮目标

1. 对 `single_face_wall_4course_v1` 和 `single_face_wall_5course_v1` 进行月面单面墙堆叠。
2. 将严格评估和 curriculum 数据采集分开:
   - 严格评估不使用 `--commit-best-rejected`，结果可用于成功率统计。
   - curriculum 使用 `--commit-best-rejected`，用于保存 best rejected、失败样本和负样本，不计入严格成功率。
3. 继续使用低释放高度搜索，减少高处自由落体带来的冲击动能。
4. 底层使用大支撑先验和连续性先验，避免小石头底层导致上层支撑面积不足。

## 2. 当前已有背景任务

运行本轮任务前，机器上仍有旧 c11 闭环评估在运行:

- 输出: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_low_release_wall_master_v1_c11_flywheel_3to4_closed_loop_eval`
- 当前观察: 已推进到 `single_face_wall_4course_v1` 月面 trial 1，产生额外 4 层数据。

因此本轮新任务均设置 `--workers 1`，避免同时开太多 MuJoCo worker。

## 3. 任务 A: high_wall 4/5 层严格评估

用途: 作为严格成功率统计。

输出:

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_4to5_highwall_c11_strict_eval_cmd_v1
```

启动脚本:

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260628_cmd_4to5_highwall_c11_strict_eval_v1\run.cmd
```

日志:

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260628_cmd_4to5_highwall_c11_strict_eval_v1\stdout.log
D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260628_cmd_4to5_highwall_c11_strict_eval_v1\stderr.log
```

核心参数:

```text
rock_profile: high_wall
targets: single_face_wall_4course_v1,single_face_wall_5course_v1
gravity: moon
trials: 2
rocks: 180
clusters: 12
candidates: 10
steps_per_rock: 300
hold_steps: 1200
candidate_probe_steps: 60
commit_best_rejected: false
low_release_search: true
release_search_step_m: 0.003
release_extra_clearance_m: 0.002
base_support_prior_weight: 1.4
base_continuity_prior_weight: 0.5
```

小网络:

```text
StoneSlotNet:
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_low_release_wall_master_v1_c11_flywheel_3to4_stone_slot_net

Candidate/support-map ranker:
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_low_release_wall_master_v1_c11_flywheel_3to4_pose_ranker_structure

PoseRiskNet:
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_low_release_wall_master_v1_c11_flywheel_3to4_pose_risk_net
```

早期进度:

```text
已创建 features.csv、cluster_summary.csv、4/5 层 target slots、structured_progress.csv。
已进入 single_face_wall_4course_v1 / moon / trial 0。
观察到前两个 base 槽位完成，说明任务已经真正进入 MuJoCo 试验。
```

## 4. 任务 B: encyclopedic_poly_train 4/5 层 curriculum

用途: 采集符合新几何先验的多面体石头 4/5 层正负样本。

输出:

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_4to5_encyclopedic_poly_curriculum_cmd_v1
```

启动脚本:

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260628_cmd_4to5_encyclopedic_poly_curriculum_v1\run.cmd
```

日志:

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260628_cmd_4to5_encyclopedic_poly_curriculum_v1\stdout.log
D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260628_cmd_4to5_encyclopedic_poly_curriculum_v1\stderr.log
```

核心参数:

```text
rock_profile: encyclopedic_poly_train
targets: single_face_wall_4course_v1,single_face_wall_5course_v1
gravity: moon
trials: 1
rocks: 160
clusters: 12
candidates: 6
steps_per_rock: 220
hold_steps: 900
candidate_probe_steps: 40
commit_best_rejected: true
low_release_search: true
release_search_step_m: 0.003
release_extra_clearance_m: 0.002
base_support_prior_weight: 1.4
base_continuity_prior_weight: 0.5
```

石头几何先验:

```text
face_count <= 80
包含块状/近矩形多面体
包含等轴/较圆化多面体
包含凸多面体和局部凹陷代理
不使用 NASA-like 训练石头
```

早期进度:

```text
已创建 features.csv、cluster_summary.csv、4/5 层 target slots、structured_progress.csv。
已进入 single_face_wall_4course_v1 / moon / trial 0。
该任务的结果用于训练数据和失败分析，不作为严格成功率。
```

## 5. 后续统计标准

任务完成后需要读取:

```text
results.csv
summary.json
failure_cases.csv
placement_log.csv
candidate_pose_log.csv
structured_progress.csv
```

必须统计:

```text
4 层 strict success_rate
4 层 shape_success_rate
5 层 strict success_rate
5 层 shape_success_rate
每个 target 的 stable_count、visible_courses、stack_height_m
RMSE、max error、max drift、velocity_inf_norm
no_feasible_pose 出现在哪些 course/role
哪些石头几何特征更常出现在 base/middle/cap 成功或近成功样本
```

特别注意:

- high_wall strict 结果和 encyclopedic curriculum 结果不能直接合并成功率。
- curriculum 的 `commit-best` 样本适合训练和失败分析，但不能当作严格成功。
- 5 层结果应先看 visible courses 和 shape_success，再看 strict success；完全成功很可能仍然稀少。
