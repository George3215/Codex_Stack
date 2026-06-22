# 2026-06-22 恢复实验记录 10:00-10:14

## 实验目的

先验证异步调度链路，再继续围绕 3/4 层单面墙做 hard-negative mining 和小网络替代启发式。当前不盲目冲 10 层，优先把 3 层成功率与 4 层阶段性结构做扎实。

## 异步问题定位

- 原 `scripts/launch_async_wall_scheduler.py` 能创建任务目录，但子任务 stdout/stderr 为空且快速退出。
- 前台 smoke 证明训练入口和 MuJoCo 飞轮入口本身正常，因此问题主要在 Windows/Codex 沙箱对子进程的回收方式。
- PowerShell `Start-Process` 被当前环境变量 `Path/PATH` 重复键拦住。
- 普通 Python `Popen` 子进程会被当前工具调用回收，`DETACHED_PROCESS` heartbeat 也未写出。
- `CREATE_BREAKAWAY_FROM_JOB` 被系统拒绝访问。
- `schtasks /Query` 在当前 `desktop-ml5vhfk\codexsandboxoffline` 用户下报 `ERROR: The system cannot find the path specified.`，Task Scheduler 后端不可用。
- 最终采用已授权的 `cmd.exe /c start "" /B <run.cmd>` 在沙箱外启动长期任务，实测 started_at 文件、Python 进程和 GPU 计算进程均出现。

## 前台 smoke 结果

### PoseRiskNet 1 epoch smoke

- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_resume_smoke_pose_risk_net_1000`
- 数据：34836 行，正样本 34647，说明原 slot/outcome 标签极度偏正。
- 测试：accuracy=0.430，top1_safe_rate=0.384，top3_safe_rate=0.973。
- 结论：链路正常，但标签过偏，必须继续做 hard negative，不能用“几乎全安全”的标签设计直接替代启发式。

### 3/4 层 moon 飞轮 smoke

- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_resume_smoke_3to4_moon`
- 3 层：strict success=0，stable=11/15，failure=4，RMSE=1.166 m，drift=0.291 m。
- 4 层：strict success=0，stable=11/24，failure=13，RMSE=1.002 m，drift=0.045 m。
- 主要失败：`missed_target`，其次是 `unstable_structure`。
- 结论：这轮不是成功案例，但适合补充 hard-negative 数据，尤其是“能稳定但没落在墙线/槽位上”的负样本。

## 新训练结果

Candidate-metric 标签版 PoseRiskNet：

- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_resume_poserisk_candidate_metric_v17_80ep_20260622_100955`
- 训练数据：`20260622_negative_mining_wall_dataset_v17_course3net_upperheuristic_605606`
- 输入策略：只使用放置前可获得的信息，即石头几何、目标槽位、重力、候选位姿、类别信息；候选位姿仿真后的误差、漂移、速度只作为标签。
- 测试：accuracy=0.739，precision=0.952，recall=0.751，F1=0.839。
- 分组排序：rankable_group_count=419，top1_safe_rate=0.632，top3_safe_rate=0.976。
- 结论：相比 1 epoch/旧标签 smoke 的 top1_safe_rate=0.384 有明显提升，可以作为下一轮候选 pose 风险排序器参与实验，但仍需要通过闭环墙体成功率验证。

## 已启动后台任务

1. PoseRiskNet candidate-metric 训练，已完成。
   - job：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_100955_cmd_pose_risk_metric_train_v17`
   - exit_code：0

2. 3/4 层 moon hard-negative 采样，正在运行。
   - job：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_100955_cmd_3to4_wall_hardnegative_collect`
   - session：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_resume_3to4_hardnegative_20260622_100955`
   - 设计：2 个 batch，mixed explore/exploit；explore 产边界失败样本，exploit 使用成熟 StoneSlotNet + support-map CNN + 旧 PoseRiskNet。
   - 当前观察：前两个 collect 子任务已经开始写 `structured_progress.csv`；exploit 已进入 middle 槽位。

## 下一步

- 等待 hard-negative 采样完成。
- 读取 `summary.json`、`failure_cases.csv`、`dataset_summary.json`。
- 用新 candidate-metric PoseRiskNet 参与一轮 3/4 层闭环 eval/collect。
- 对比旧 PoseRiskNet、explore baseline 与新 PoseRiskNet 在 3/4 层墙上的成功率、RMSE、drift、velocity 和 failure_reason。

## 10:22-10:27 第一批采样与图像记录

### explore_00 结果

- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_resume_3to4_hardnegative_20260622_100955_collect_explore_00_seed206222101`
- 3 层月面单面墙：strict success=1.0，shape_success=1.0，stable=14/15，failure=1，RMSE=0.074 m，max error=0.276 m，height=0.345 m，drift=0.0014 m。
- 4 层月面单面墙：strict success=0，stable=8/24，failure=16，RMSE=2.467 m，max error=11.086 m，height=0.201 m，drift=0.031 m。
- 经验：3 层已经可偶发严格成功，但同批 4 层会突然恶化，说明 4 层不是简单多放一层，而是会出现更强的目标偏移累积、支撑不连续和 cap/middle 选择耦合。

### exploit_00 结果

- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_resume_3to4_hardnegative_20260622_100955_collect_exploit_00_seed206231102`
- 3 层：strict success=0，stable=11/15，failure=4，RMSE=0.220 m，drift=0.0086 m。
- 4 层：strict success=0，stable=17/24，failure=7，RMSE=0.511 m，drift=0.030 m。
- 经验：成熟网络/启发式组合没有拿到 strict success，但 4 层 stable_count 明显好于 explore_00；explore_00 的 3 层成功说明随机探索仍能发现局部好解，后续应把这些正样本转成网络训练数据，而不是继续依赖随机。

### 图像与相机修正

- 成功/失败图像目录：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_resume_3to4_hardnegative_20260622_100955_collect_explore_00_seed206222101\capwf2`
- 3 层成功案例：
  - `capwf2\00_single_face_wall_3course_v1_success_statics_wall_moon_trial_00\wall_front_rgb.png`
  - `capwf2\00_single_face_wall_3course_v1_success_statics_wall_moon_trial_00\wall_top_depth.png`
- 4 层失败案例：
  - `capwf2\01_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00\wall_front_rgb.png`
  - `capwf2\01_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00\wall_top_depth.png`
- 路径长度问题：长 capture 目录名触发过 `FileNotFoundError`，实际原因是 Windows 路径过长。后续 capture 目录名应短，例如 `capwf2`。
- 相机问题：原 `wall_front` 使用 azimuth=0，更像沿墙轴方向看，不是墙面正视；已将 `scripts/capture_cases.py` 中的 `wall_front` 改为 azimuth=90。`wall_top_depth` 仍作为主要深度证据。

## 10:32-10:52 新 PoseRiskNet 闭环对比

### 新网络训练

- 网络：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_resume_poserisk_candidate_metric_hardnegative_100955_80ep`
- 数据：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_resume_3to4_hardnegative_20260622_100955_learning_dataset`
- 数据规模：candidate_pose=25471，placement=924。
- 测试：accuracy=0.735，precision=0.915，recall=0.774，F1=0.839。
- 分组排序：rankable_group_count=232，top1_safe_rate=0.651，top3_safe_rate=0.940。
- 解释：相比 v17 版 top1 略高，但 top3 更低，说明这个网络更激进；必须做闭环验证，不能只看离线指标。

### 闭环 1：top3+w0.35

- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_newposerisk_hardnegative_eval_3to4_moon_seed206232501`
- 3 层：strict=0，stable=11/11，skipped=4，failure=0，RMSE=0.028 m，drift=0.0055 m。
- 4 层：strict=0，stable=12/13，skipped=11，failure=1，visible_courses=3，RMSE=0.076 m，drift=0.229 m。
- 结论：网络过于保守，几何对齐很好，但跳过太多槽位；这是“低风险但不完整”的失败。

### 闭环 2：top5+w0.25

- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_newposerisk_hardnegative_eval_3to4_moon_top5_w025_seed206232777`
- 3 层：strict=0，stable=8/12，skipped=3，failure=4，RMSE=0.176 m，drift=0.379 m。
- 4 层：strict=0，stable=10/15，skipped=9，failure=5，visible_courses=4，RMSE=0.157 m，drift=0.399 m。
- 结论：简单降低风险权重会显著增加 post-hold drift，不是正确方向。

### 闭环 3：top5+w0.35

- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_newposerisk_hardnegative_eval_3to4_moon_top5_w035_seed206233003`
- 3 层：strict=1，shape=1，stable=13/13，skipped=2，failure=0，RMSE=0.033 m，drift=0.0016 m，height=0.321 m。
- 4 层：strict=0，stable=16/16，skipped=8，failure=0，visible_courses=4，RMSE=0.036 m，drift=0.0054 m，height=0.322 m。
- 图像：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_newposerisk_hardnegative_eval_3to4_moon_top5_w035_seed206233003\capwf2`
- 结论：这是当前最好的组合。它证明“扩大候选池 + 保持强风险约束”比“降低风险约束”更有效。剩余主要问题不是单石稳定性，而是 skipped slots 导致结构不完整；下一步应训练/设计 slot-completion 或 no-feasible fallback，而不是继续降低风险权重。

## 11:19 后台统计批次

已启动 4-trial 统计批次，用来验证 `top5+w0.35` 是否稳定，而不是单次偶然成功。

- job：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_110140_cmd_newposerisk_top5_w035_stats_3to4_moon`
- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_newposerisk_top5_w035_stats_3to4_moon_trials4_20260622_110140`
- 参数：moon，3/4 层单面墙，trials=4，rocks=96，workers=4，candidate_top_k=5，pose_risk_weight=0.35，StoneSlotNet top_k=12。
- 状态：started_at 已写入，`structured_progress.csv` 正在更新，4 个 Python worker 运行中。
- 目的：统计 3 层 strict success 是否能超过单次偶然；观察 4 层是否继续表现为“低漂移、低 RMSE、但 skipped slots 导致 strict=0”。

## 14:59 strictdrift 完整 4-trial 统计完成

### 数据集

- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top5_w035_stats_3to4_moon_trials4_20260622_140938`
- 模型：`20260622_resume_poserisk_strictdrift_hardnegative_100955_100ep`
- 参数：moon，3/4 层单面墙，rocks=96，clusters=10，trials=4，candidate top5，pose risk weight=0.35。

### 3 层结果

- strict success：1/4 = 25%
- shape success：2/4 = 50%
- mean stable count：12.25
- mean failure count：0.50
- mean RMSE：0.054 m
- mean max drift：0.138 m
- mean velocity inf norm：0.767
- 代表性成功：trial 3，stable=13/13，RMSE=0.012 m，drift=0.009 m，velocity=0.067。
- 边界失败：trial 2 已经 shape success，stable=13/13，RMSE=0.021 m，drift=0.035 m，但 velocity=0.766，说明几何形状可行并不等于动态稳定。

### 4 层结果

- strict success：0/4 = 0%
- shape success：0/4 = 0%
- mean stable count：10.75
- mean failure count：7.00
- mean RMSE：0.164 m
- mean max drift：0.367 m
- mean velocity inf norm：0.688
- 主要失败：`missed_target+post_hold_drift` 19 次，`unstable_structure` 6 次，`post_hold_drift` 3 次。

### 经验

- strictdrift 方向有阶段性价值：相对 candidate-metric 多 trial 的 3 层 strict 0/4，提升到 1/4 strict 和 2/4 shape。
- strictdrift 不是充分解：3 层仍不稳定，4 层仍为 0/4。
- 当前失败更像动态稳定问题，而不是单纯几何落点问题。需要把候选评估时间、速度残余、漂移风险或短时预筛 probe 纳入闭环。
- 下一步先做 small-batch long-settle：增加每块石头 settle steps 和最终 hold steps，看成功率是否上升。如果上升，说明当前候选评估时间过短；如果不上升，再提高风险权重或训练更严格速度标签。

## 16:24 新一轮异步提升任务

### 任务 A：strictdrift long-settle 小批量闭环验证

- job：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_162145_cmd_strictdrift_top5_w035_longsettle_3to4_moon`
- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top5_w035_longsettle_3to4_moon_trials2_20260622_162145`
- 目的：验证上一轮 3/4 层失败是否部分来自候选评估时间过短。
- 改动：上一轮 `steps_per_rock=260`、`hold_steps=1000`；本轮改为 `steps_per_rock=420`、`hold_steps=1800`。
- 其它参数：moon，3/4 层单面墙，rocks=96，clusters=10，trials=2，workers=4，candidate top5，pose risk weight=0.35。
- 预期判断：如果 3 层 strict/shape 明显提升，说明低重力慢漂移需要更长的评估窗口；如果没有提升，说明仅靠延长 settle 不够，需要更强速度标签或候选预筛。

### 任务 B：velocity-strict PoseRiskNet 训练

- job：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_162650_cmd_pose_risk_velocity_strict_train`
- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_resume_poserisk_velocity_strict_hardnegative_100955_90ep`
- 数据集：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_resume_3to4_hardnegative_20260622_100955_learning_dataset`
- 目的：训练更严格惩罚速度残余和漂移的候选位姿风险网络，为下一轮替代启发式搜索做准备。
- 标签阈值：`target_error_limit=0.08`，`target_y_error_limit=0.035`，`disturbance_limit=0.025`，`velocity_limit=0.08`。
- 输入约束：运行时仍只使用放置前可获得的石头几何、目标槽位、候选姿态、支撑/类别等信息；仿真后的 RMSE、漂移、速度只作为训练标签和评价指标。
- 资源安排：CPU MuJoCo 4 worker + GPU 小网络训练并行，启动时 RTX 2080 Ti 显存约 1.3 GB/11 GB。

### 任务 B 结果：velocity-strict 离线排序未优于 strictdrift

- exit code：0
- row count：25471
- risky positive count：23858
- test accuracy：0.783
- precision：0.943
- recall：0.816
- F1：0.875
- test group rankable groups：261
- top1 safe rate：0.517
- top3 safe rate：0.950

对比上一版 strictdrift：top1 safe rate 约 0.596，top3 safe rate 1.000。新模型的速度阈值更严格，但离线候选排序变差，因此暂不直接替换闭环模型。这个负结果说明：单纯把速度阈值收紧会扩大风险标签比例，但未必让网络更会排序；后续可能需要重采样、pairwise/ranking loss，或者把速度残余从二分类标签改成连续风险回归。

## 16:33 通用候选 probe 机制

### 修改目的

上一轮失败统计显示，很多失败不是候选完全放不上去，而是短 settle 后看似可行，最终 hold 时出现 `post_hold_drift` 或速度残余。因此新增一个可选机制：候选在正式提交前额外短 hold 一段时间，将慢漂移提前暴露出来。

### 代码改动

- 新增 CLI：`--candidate-probe-steps`
- 默认值：0，旧实验行为不变。
- 作用范围：`_place_for_target_slot`，因此可用于 `statics_wall`、`literature_wall`、柱子以及普通候选路径。
- 新增候选字段：
  - `candidate_probe_steps`
  - `candidate_probe_rock_drift_m`
  - `candidate_probe_placed_disturbance_m`
  - `candidate_probe_speed`
- 评分逻辑：开启 probe 后，对 probe 中石头自身慢漂移、扰动已放石头、速度未衰减的候选加惩罚。
- 可行性逻辑：对 strict single-face wall 的中高层候选，probe 漂移/扰动/速度超过阈值时拒绝。

### 代码 smoke

- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_candidate_probe_code_smoke_20260622_1633`
- 配置：16 rocks，2 层单面墙，moon，1 trial，`steps_per_rock=20`，`hold_steps=40`，`candidate_probe_steps=10`。
- 结果：程序正常结束，exit code=0；`PROTOCOL.md` 写入 `candidate_probe_steps: 10`；`results.csv` 写入 effective `candidate_probe_steps=24`；`candidate_pose_log.csv` 写入 probe 漂移/速度字段。
- 说明：这是代码验证，不作为科学成功率统计；步数故意极短，失败是预期结果。

## 16:55 strictdrift long-settle 完整结果

### 数据集

- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top5_w035_longsettle_3to4_moon_trials2_20260622_162145`
- job：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_162145_cmd_strictdrift_top5_w035_longsettle_3to4_moon`
- 参数：moon，3/4 层单面墙，rocks=96，clusters=10，trials=2，candidate top5，pose risk weight=0.35。
- 变化：`steps_per_rock=420`，`hold_steps=1800`；上一轮短窗口是 `steps_per_rock=260`，`hold_steps=1000`。

### 3 层

- strict success：1/2 = 50%
- shape success：1/2 = 50%
- mean stable count：11.50
- mean failure count：1.50
- mean RMSE：0.077 m
- mean max drift：0.155 m
- mean velocity inf norm：0.022
- 成功 trial：trial 1，stable=12/12，skipped=3，RMSE=0.031 m，drift=0.0007 m，velocity=0.030。
- 失败 trial：trial 0，stable=11/14，skipped=1，RMSE=0.124 m，drift=0.310 m，velocity=0.013。

### 4 层

- strict success：0/2 = 0%
- shape success：0/2 = 0%
- mean stable count：11.50
- mean failure count：6.50
- mean RMSE：0.203 m
- mean max drift：0.577 m
- mean velocity inf norm：0.051
- failure counts：`missed_target+post_hold_drift` 8 次，`post_hold_drift` 3 次，`unstable_structure` 2 次。

### 经验

- 更长 settle 能降低速度残余，并使 3 层出现更稳的成功样例。
- 更长 settle 也会暴露更多不可行候选，导致 4 层 skipped/no-feasible 增多。
- 4 层失败不是靠单纯加长 settle 能解决；需要 candidate probe 或网络排序在提交前区分“短期可放但长期会滑”的候选。
- 下一步启动 `candidate_probe_steps=80` 的小批次，用更低成本的短时 probe 代替盲目加长每个候选的 settle/hold。

## 17:09 candidate-probe 小批次启动

- job：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_164000_cmd_strictdrift_top5_w035_candidateprobe80_3to4_moon`
- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top5_w035_candidateprobe80_3to4_moon_trials2_20260622_164000`
- 参数：moon，3/4 层单面墙，rocks=96，clusters=10，trials=2，workers=4。
- 网络：StoneSlotNet + support-map CNN + strictdrift PoseRiskNet。
- 核心变化：`candidate_probe_steps=80`，`steps_per_rock=260`，`hold_steps=1000`。
- 实验目的：对比 long-settle。long-settle 是整体加长候选 settle 和最终 hold；candidate-probe 是在候选提交前加短 probe，用更局部、更低成本的方式暴露慢漂移。
- 启动状态：`started_at.txt` 已写入，`structured_progress.csv` 已开始更新，stderr 初始长度为 0。

### 17:24 probe hard-gate 修正

`candidate_probe_steps=80` 的 3 层早期结果显示，probe 作为硬拒绝会导致 skipped slots 增多。尤其 trial 0 几何和漂移都很好，但 skipped=4 导致 shape fail。因此把 probe 拆成两种模式：

- 默认：soft probe，只把 probe 漂移、扰动、速度加入候选评分，不硬拒绝。
- 显式 `--candidate-probe-hard-gate`：才启用硬拒绝。

代码 smoke：

- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_candidate_probe_soft_code_smoke_20260622_1724`
- 命令参数：`candidate_probe_steps=10`，未开启 hard gate。
- 结果：exit code=0；`results.csv` 写入 `candidate_probe_steps=24`、`candidate_probe_hard_gate=0`；`candidate_pose_log.csv` 写入 probe 漂移/速度字段。
- 下一步：等当前 hard-gate 风格的 `probe=80` 批次结束后，启动 soft-probe 小批次，优先尝试 `candidate_probe_steps=40`。

## 17:12 long-settle 代表案例图像

- 图像目录：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top5_w035_longsettle_3to4_moon_trials2_20260622_162145\capwf_probe_ready`
- 3 层成功案例：`00_single_face_wall_3course_v1_success_statics_wall_moon_trial_01`
  - 正视 RGB：`wall_front_rgb.png`
  - 俯视深度：`wall_top_depth.png`
- 3 层失败案例：`01_single_face_wall_3course_v1_failure_statics_wall_moon_trial_00`
- 4 层失败案例：`02_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00` 和 `03_single_face_wall_4course_v1_failure_statics_wall_moon_trial_01`
- 检查结果：`wall_front_rgb.png` 已是墙面正视方向，不再是边视；`wall_top_depth.png` 有有效深度结构，不是无信息纯色图。

## 17:35 candidate-probe80 hard-gate 完整结果

### 数据集

- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top5_w035_candidateprobe80_3to4_moon_trials2_20260622_164000`
- 参数：moon，3/4 层单面墙，rocks=96，clusters=10，trials=2，workers=4。
- 网络：StoneSlotNet + support-map CNN + strictdrift PoseRiskNet。
- 核心变化：`candidate_probe_steps=80`，本批次实际作为 hard-gate 使用；有效 probe steps 在 `results.csv` 中记录为 192。

### 3 层结果

- strict success：0/2 = 0%
- shape success：0/2 = 0%
- mean stable count：10.50
- mean failure count：1.50
- mean RMSE：0.084 m
- mean max drift：0.131 m
- mean velocity inf norm：1.112
- 关键观察：trial 0 几何和最终漂移都很好，stable=11/11、failure=0、RMSE=0.031 m、drift=0.0019 m，但 skipped=4，所以墙体缺块失败。

### 4 层结果

- strict success：0/2 = 0%
- shape success：0/2 = 0%
- mean stable count：13.50
- mean failure count：3.50
- mean RMSE：0.109 m
- mean max drift：0.174 m
- mean velocity inf norm：0.340
- 关键观察：trial 1 stable=17/17、failure=0、visible_courses=4、height=0.356 m、RMSE=0.044 m、drift=0.033 m，但 skipped=7。也就是说，hard-gate 让剩下的石头更稳，但牺牲了槽位完整性。

### 经验

- `probe=80 hard-gate` 没有提升成功率，3 层和 4 层都是 0/2。
- 它不是无意义失败：它证明“候选提交前 probe”确实能过滤风险，但不能直接作为硬拒绝规则，否则 no-feasible/skipped 会成为主要失败源。
- 下一步采用 `probe=40 soft`：probe 只进入候选评分，不直接拒绝。目标是在保留槽位完整性的同时，把慢漂移候选排到更低。
- 这个结果也说明当前系统不能只优化单石稳定性；墙体任务需要同时优化稳定性、槽位覆盖、course 连续性和支撑面嵌合。

## 17:41 candidate-probe40 soft 启动

- job：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_172500_cmd_strictdrift_top5_w035_candidateprobe40soft_3to4_moon`
- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top5_w035_candidateprobe40soft_3to4_moon_trials2_20260622_172500`
- 参数：moon，3/4 层单面墙，rocks=96，clusters=10，trials=2，workers=4。
- 网络：StoneSlotNet + support-map CNN + strictdrift PoseRiskNet。
- 核心变化：`candidate_probe_steps=40`，不开启 `--candidate-probe-hard-gate`，即 probe 只作为软评分惩罚。
- 设计目的：保留 candidate probe 暴露慢漂移候选的能力，但避免 `probe=80 hard-gate` 造成的过度 skipped slots。判断标准不是单石稳定，而是 strict/shape success、skipped slot、RMSE、drift 和 velocity 的综合变化。
- 启动检查：`started_at.txt` 已写入，输出目录已生成，`structured_progress.csv` 已开始写入；启动后 stderr 长度为 0，4 个 MuJoCo worker 和 1 个主 Python 进程运行中。

### 17:49 soft-probe 首次运行失败与修复

- 失败 job：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_172500_cmd_strictdrift_top5_w035_candidateprobe40soft_3to4_moon`
- 部分输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top5_w035_candidateprobe40soft_3to4_moon_trials2_20260622_172500`
- 结果：exit code=1，未写出 `results.csv`，不是物理失败，属于代码路径失败。
- 错误原因：`_literature_candidate_is_feasible()` 使用了 `probe_hard_gate`，但该变量没有从 `selected["candidate_probe_hard_gate"]` 读取，soft-probe 运行到 literature feasibility 分支时报 `NameError`。
- 修复：在 `moon_rock_stack/structured.py` 中补充 `probe_hard_gate = int(float(selected.get("candidate_probe_hard_gate", 0.0)))`。
- 验证：`py_compile` 通过；前台 smoke 输出 `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_candidate_probe_soft_fix_smoke_1748`，`results.csv` 写入 `candidate_probe_hard_gate=0`，`candidate_pose_log.csv` 写入 probe 漂移和速度字段。
- 重跑 job：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_174900_cmd_strictdrift_top5_w035_candidateprobe40soft_fix_3to4_moon`
- 重跑输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top5_w035_candidateprobe40soft_fix_3to4_moon_trials2_20260622_174900`

## 18:32 candidate-probe40 soft 完整结果

### 数据集

- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top5_w035_candidateprobe40soft_fix_3to4_moon_trials2_20260622_174900`
- job：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_174900_cmd_strictdrift_top5_w035_candidateprobe40soft_fix_3to4_moon`
- exit code：0
- 参数：moon，3/4 层单面墙，rocks=96，clusters=10，trials=2，workers=4，`candidate_probe_steps=40`，不开启 hard-gate。
- 记录：`results.csv` 中有效 `candidate_probe_steps=96`，`candidate_probe_hard_gate=0`。

### 3 层结果

- strict success：1/2 = 50%
- shape success：1/2 = 50%
- mean stable count：11.00
- mean failure count：1.50
- mean RMSE：0.064 m
- mean max drift：0.140 m
- mean velocity inf norm：0.258
- 成功 trial 1：rock_count=12，stable=12/12，failure=0，skipped=3，RMSE=0.017 m，drift=0.0028 m，velocity=0.022，height=0.287 m。
- 失败 trial 0：stable=10/13，failure=3，skipped=2，RMSE=0.112 m，drift=0.278 m，velocity=0.494。

### 4 层结果

- strict success：0/2 = 0%
- shape success：0/2 = 0%
- mean stable count：10.00
- mean failure count：5.00
- mean RMSE：0.157 m
- mean max drift：0.342 m
- mean velocity inf norm：1.757
- trial 0：stable=9/13，skipped=11，visible_courses=3，height=0.193 m。
- trial 1：stable=11/17，skipped=7，visible_courses=4，height=0.309 m。
- 主要失败：4 层 `missed_target+post_hold_drift` 5 次，4 层 `unstable_structure` 4 次，4 层 `post_hold_drift` 1 次。

### 图像记录

- 图像目录：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top5_w035_candidateprobe40soft_fix_3to4_moon_trials2_20260622_174900\capwf_soft_probe40`
- 3 层成功案例：`00_single_face_wall_3course_v1_success_statics_wall_moon_trial_01`
  - 正视 RGB：`wall_front_rgb.png`
  - 俯视深度：`wall_top_depth.png`
- 3 层失败案例：`01_single_face_wall_3course_v1_failure_statics_wall_moon_trial_00`
- 4 层失败案例：`02_single_face_wall_4course_v1_failure_statics_wall_moon_trial_01` 和 `03_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00`
- 检查结果：3 层成功案例的正视 RGB 和俯视深度均有效；4 层失败案例的正视 RGB 也能清楚显示墙体缺块和局部坍塌。

### 经验

- soft-probe 比 hard-gate 合理：`probe=80 hard-gate` 的 3 层是 0/2，`probe=40 soft` 恢复到 1/2，说明 probe 不应该直接变成拒绝规则。
- 3 层成功样例质量高，说明当前网络和启发式组合可以在低重力下找到有效落点。
- 4 层仍然失败，说明瓶颈已经不是“能不能放一块石头”，而是多层墙体中 support continuity、槽位覆盖、cap/middle 选择和 post-hold drift 的组合问题。
- 下一步不要继续粗暴增加 probe。更合理的方向是：只对中高层或高风险候选触发 probe；把 no-feasible/skipped 作为显式代价；训练连续风险回归或 pairwise ranking，让网络直接偏好“稳定且不跳槽”的候选。

## 18:36 下一组对照：candidate-probe40 soft + top8

- job：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_183600_cmd_strictdrift_top8_w035_candidateprobe40soft_3to4_moon`
- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top8_w035_candidateprobe40soft_3to4_moon_trials2_20260622_183600`
- 改动：在 `probe=40 soft` 的基础上，把 `candidate_pose_top_k` 从 5 扩到 8。
- 目的：判断 4 层失败是否主要来自候选池太窄。如果 top8 能减少 skipped/no-feasible 并改善 4 层 RMSE/drift，说明下一步应扩大候选池并让网络做更强排序；如果 top8 仍失败，则说明问题主要在支撑连续性、槽位代价和 cap/middle 结构目标函数。

## 19:12 candidate-probe40 soft + top8 完整结果

### 数据集

- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top8_w035_candidateprobe40soft_3to4_moon_trials2_20260622_183600`
- job：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_183600_cmd_strictdrift_top8_w035_candidateprobe40soft_3to4_moon`
- exit code：0
- 参数：moon，3/4 层单面墙，rocks=96，clusters=10，trials=2，workers=4，`candidate_probe_steps=40`，`candidate_pose_top_k=8`，不开启 hard-gate。

### 3 层结果

- strict success：1/2 = 50%
- shape success：2/2 = 100%
- mean stable count：12.50
- mean failure count：1.00
- mean RMSE：0.053 m
- mean max drift：0.115 m
- mean velocity inf norm：0.028
- trial 0：shape=1 但 strict=0，stable=11/13，skipped=2，RMSE=0.088 m，drift=0.227 m，velocity=0.028。
- trial 1：strict=1，shape=1，stable=14/14，skipped=1，RMSE=0.018 m，drift=0.0036 m，velocity=0.027。

### 4 层结果

- strict success：0/2 = 0%
- shape success：0/2 = 0%
- mean stable count：14.50
- mean failure count：6.00
- mean RMSE：0.163 m
- mean max drift：0.433 m
- mean velocity inf norm：0.157
- trial 0：rock_count=23，stable=16/23，failure=7，skipped=1，visible_courses=4，height=0.294 m。
- trial 1：rock_count=18，stable=13/18，failure=5，skipped=6，visible_courses=4，height=0.275 m。
- 主要失败：4 层 `missed_target+post_hold_drift` 8 次，`unstable_structure` 2 次，`post_hold_drift` 2 次。

### 图像记录

- 图像目录：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top8_w035_candidateprobe40soft_3to4_moon_trials2_20260622_183600\capwf_top8_probe40`
- 3 层成功案例：`00_single_face_wall_3course_v1_success_statics_wall_moon_trial_01`
- 3 层失败但 shape 成功案例：`01_single_face_wall_3course_v1_failure_statics_wall_moon_trial_00`
- 4 层关键失败案例：`02_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00`，该案例 rock_count=23、skipped=1，但因 drift/unstable 失败。
- 4 层另一失败案例：`03_single_face_wall_4course_v1_failure_statics_wall_moon_trial_01`
- 检查结果：3 层成功图像和 4 层关键失败图像均可用；4 层失败图显示墙体已经接近完整，但中高层存在漂移和局部失稳。

### 经验

- top8 是当前最有价值的方向之一：它没有提高 3 层 strict success，但把 3 层 shape 提到 2/2，并显著改善 4 层 skipped，trial 0 只 skipped=1。
- 4 层失败类型发生转移：从“候选不足导致缺槽”转为“结构更完整但 post-hold drift/unstable 失败”。这说明扩大候选池解决了槽位覆盖的一部分，但不能独立解决承载路径和高层漂移。
- 下一步需要把支撑连续性显式建模，而不是继续单纯扩大 top-k。可选方案：对 middle/cap 引入 course-level support continuity penalty；让网络输出连续风险分数；或训练 pairwise ranking，使模型直接学习“同一槽位下哪个候选更不漂移且不破坏整体墙线”。

## 19:15 启动 top8+probe40 4-trial 统计批次

- job：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_191500_cmd_strictdrift_top8_w035_candidateprobe40soft_stats_3to4_moon`
- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top8_w035_candidateprobe40soft_stats_3to4_moon_trials4_20260622_191500`
- 参数：与 2-trial top8+probe40 相同，仅把 `trials` 从 2 提到 4，seed 改为 206238040。
- 目的：验证 3 层 shape=2/2 和 4 层 skipped 改善是否可复现，避免被单批 seed 误导。

## 19:29 top8+probe40 4-trial 统计结果

### 数据集

- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top8_w035_candidateprobe40soft_stats_3to4_moon_trials4_20260622_191500`
- job：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_191500_cmd_strictdrift_top8_w035_candidateprobe40soft_stats_3to4_moon`
- exit code：0
- 参数：moon，3/4 层单面墙，rocks=96，clusters=10，trials=4，`candidate_pose_top_k=8`，`candidate_probe_steps=40`，`pose_risk_weight=0.35`，网络参与到 `course<=2`。

### 3 层结果

- strict success：4/4 = 100%
- shape success：4/4 = 100%
- mean stable count：13.25
- mean failure count：0.00
- mean RMSE：0.022 m
- mean max drift：0.0056 m
- mean velocity inf norm：0.051
- 经验：3 层已经从“偶然成功”进入“阶段性稳定成功”。这说明 `top8 + strictdrift PoseRiskNet + probe40 soft` 的组合在 3 层单面墙上已经足够强。

### 4 层结果

- strict success：0/4 = 0%
- shape success：0/4 = 0%
- mean stable count：12.25
- mean failure count：4.50
- mean RMSE：0.125 m
- mean max drift：0.281 m
- mean velocity inf norm：0.185
- visible courses mean：3.5
- 主要失败：`missed_target+post_hold_drift` 14 次，`unstable_structure` 2 次，`post_hold_drift` 2 次。
- 经验：4 层失败已经不是 3 层子问题，瓶颈集中在高层 cap/middle 的承载路径和 post-hold drift。

### 图像记录

- 图像目录：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_strictdrift_top8_w035_candidateprobe40soft_stats_3to4_moon_trials4_20260622_191500\c4`
- 3 层成功案例：`00_single_face_wall_3course_v1_success_statics_wall_moon_trial_00`
- 4 层失败案例：`01_single_face_wall_4course_v1_failure_statics_wall_moon_trial_01` 和 `02_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00`
- 说明：第一次 capture 使用较长目录名触发 Windows 路径过长错误，未删除该部分残留；使用短目录名 `c4` 后成功保存正视 RGB 和俯视深度图。

## 19:30 support-continuity 数据字段上线

### 代码改动

- 文件：`D:\MoonStack\experiments\moon_rock_stack\moon_rock_stack\structured.py`
- 改动：在 `_target_candidate_metrics()` 中新增 `_support_continuity_metrics()`，只写日志，不改变当前候选评分。
- 新增字段包括：
  - `same_course_placed_count`
  - `left_neighbor_present`
  - `right_neighbor_present`
  - `neighbor_gap_left_m`
  - `neighbor_gap_right_m`
  - `neighbor_gap_max_positive_m`
  - `course_height_std_after_m`
  - `course_y_std_after_m`
  - `course_y_abs_max_after_m`
  - `direct_support_count_course_below`
  - `support_load_path_count`
  - `support_span_x_m`
  - `support_span_cover_ratio`
  - `support_underhang_left_m`
  - `support_underhang_right_m`
  - `support_underhang_max_m`

### 验证

- smoke 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_support_continuity_field_smoke_1930`
- 结果：`candidate_pose_log.csv` 和 `placement_log.csv` 均写入新增字段；middle 层能看到 `direct_support_count_course_below=2/3`、`support_load_path_count=2/3`、`support_span_x_m` 等非空值。
- 目的：后续训练 support-continuity critic 或 pairwise ranking 时，可以把“支撑链是否连续、同层 gap 是否过大、高度是否断裂”作为网络输入/标签的一部分。

## 19:35 course=3 网络参与对照启动

- job：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_193500_cmd_top8_probe40_course3net_3to4_moon`
- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_top8_probe40_course3net_3to4_moon_trials2_20260622_193500`
- 参数：moon，3/4 层单面墙，trials=2，workers=2，`candidate_pose_top_k=8`，`candidate_probe_steps=40`。
- 核心变化：`stone_fit_ranker_max_course=3`、`candidate_pose_ranker_max_course=3`、`pose_risk_ranker_max_course=3`。
- 目的：让三个小网络第一次参与 4 层墙最顶层 `course=3` 的石头选择、姿态排序和风险评估，同时收集新增 support-continuity 字段。
- 启动检查：`started_at.txt` 已写入，输出目录已生成，2 个 worker 正常运行，stderr 初始为 0。
