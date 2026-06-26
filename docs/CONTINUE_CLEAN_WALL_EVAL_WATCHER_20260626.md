# 2026-06-26 清洗版墙体网络评估续跑记录

## 目的

当前清洗版 SupportMap / WallCritic 训练仍在运行，不能在 GPU 接近占用时强行启动新的闭环仿真。为避免空转，也避免重复启动同类任务，本轮新增了一个 watcher：

- 等待清洗版训练 manifest 写入 `finished_at`。
- 等待清洗版 `StoneSlotNet`、`SupportMapRanker`、`PoseRiskNet` 三类模型文件全部出现。
- 等待 GPU 显存低于阈值。
- 条件满足后自动启动 3/4 层月面单面墙闭环评估。

这个 watcher 不删除任何数据，也不停止任何已有训练。

## 新增脚本

```text
D:\MoonStack\experiments\moon_rock_stack\scripts\wait_for_clean_wall_eval.py
```

核心判定条件：

```text
batch_runs\20260626_clean_wall34_supportmap_train_v1\flywheel_manifest.json
batch_runs\20260626_clean_wall34_supportmap_train_v1_stone_slot_net\stone_fit_net.npz
batch_runs\20260626_clean_wall34_supportmap_train_v1_pose_ranker_structure\support_map_cnn_ranker.pt
batch_runs\20260626_clean_wall34_supportmap_train_v1_pose_risk_net\pose_risk_net.npz
```

只有这些文件存在，并且 manifest 中出现 `finished_at`，才会进入评估。

## 异步任务

异步任务目录：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260626_cmd_clean_wall34_eval_after_train_v1
```

启动脚本：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260626_cmd_clean_wall34_eval_after_train_v1\run.cmd
D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260626_cmd_clean_wall34_eval_after_train_v1\run.ps1
```

watcher 日志：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260626_cmd_clean_wall34_eval_after_train_v1\watch_eval.log
```

第一次检查记录：

```text
2026-06-26T22:51:56 watch_start train_session=20260626_clean_wall34_supportmap_train_v1 eval_session=20260626_clean_wall34_supportmap_eval_v1
2026-06-26T22:51:56 check attempt=1 missing=3 manifest_finished=False gpu_used_mib=3260 active_train=True
```

含义：

- `missing=3`：三个待训练模型文件还没全部产出。
- `manifest_finished=False`：清洗版训练还没完成。
- `gpu_used_mib=3260`：GPU 没有满载，但训练还在运行。
- `active_train=True`：清洗版训练进程仍然活跃。

## 后续自动评估配置

条件满足后将自动启动：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_clean_wall34_supportmap_eval_v1
```

评估任务：

```text
targets = single_face_wall_3course_v1,single_face_wall_4course_v1
gravity = moon
rocks = 128
clusters = 10
trials = 2
candidates = 10
workers = 1
candidate_pose_top_k = 1
stone_fit_top_k = 6
pose_risk_weight = 0.65
ranker_max_course = -1
low_release_search = on
base_support_prior = on
base_continuity_prior = on, weight 0.35
```

本轮评估目的不是盲目堆更高，而是验证清洗数据训练出的网络是否能在 3/4 层墙闭环中提升成功率，并把结果纳入下一份数据流/增长率报告。

## 当前训练状态

清洗版深度/支撑图导出目录：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_clean_wall34_supportmap_train_v1_mujoco_depth_maps
```

截至本记录时，已经产出 `support_maps_0001` 到 `support_maps_0005`，约每 16 分钟一个 shard。stderr 中存在 MuJoCo `Renderer.__del__` 和 GLFW context warning，但导出仍在继续产出文件，暂时不视作致命错误。

## 下一步

1. 等待清洗版训练完成。
2. watcher 自动启动闭环评估。
3. 评估完成后重新运行：

```powershell
C:\Users\all\miniconda3\envs\moon-rock-stack\python.exe -m scripts.generate_experiment_progress_report `
  --batch-root batch_runs `
  --output docs\progress_reports\20260626_clean_wall_eval_report_v1 `
  --target-contains single_face_wall `
  --dataset batch_runs\20260626_clean_policy_wall34_dataset_v1
```

4. 对比 `20260626_dataflow_growth_report_v4` 和新报告中的 3/4 层成功率、shape success、stable fraction、PoseRiskNet / SupportMapRanker 指标，判断是否进入新瓶颈。
