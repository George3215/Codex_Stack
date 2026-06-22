# 继续训练记录 2026-06-22 23:55

## 本轮执行目的

继续推进 3/4 层单面墙数据飞轮，同时保存当前已经出现的典型 4 层成功和失败样例。当前策略仍然是先提高 4 层稳定成功率，不因为单次成功直接进入 5 层。

## 当前后台任务

仍在运行的任务：

- `20260622_231000_cmd_autonomous_wall_flywheel_master_v2`
- `20260622_225000_cmd_supportmap_v19_train_eval4`
- `20260622_224800_cmd_poserisk_v18b_train_eval4`
- `20260622_221000_cmd_auto_flywheel_3to4`

已完成的任务：

- `20260622_221000_cmd_auto_strict_4course`

## 4 层 strict 结果

输出目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_cmd_auto_strict_4course_221000`

结果：

- trial 0：strict success `1`，shape success `1`
  - visible courses: `4`
  - stack height: `0.4095 m`
  - max horizontal drift: `0.0128 m`
  - stable count: `21`
- trial 1：strict success `0`，shape success `0`
  - visible courses: `4`
  - stack height: `0.3162 m`
  - max horizontal drift: `0.3268 m`
  - 主要失败模式：高层多次 `no_feasible_pose`，cap/course 3 阶段支撑不足并产生大漂移。

解释：

- trial 0 是当前很有价值的 4 层正样本：高度、漂移、形状都满足严格单面墙目标。
- trial 1 不是简单“没堆起来”，而是已经堆到 4 个可见层，但高层局部支撑链不稳定，适合作为 hard negative。

## 已保存图像与视频

成功/失败多视角截图：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_cmd_auto_strict_4course_221000\cap_strict_success_failure_20260622_2348`

包含：

- success trial 0 的 `wall_front_rgb/depth/object_depth`
- success trial 0 的 `wall_top_depth/object_depth`
- failure trial 1 的 `wall_front_rgb/depth/object_depth`
- failure trial 1 的 `wall_top_depth/object_depth`
- `capture_manifest.csv`

典型成功过程 GIF：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_cmd_auto_strict_4course_221000\process_video_strict_4course_20260622_2350\single_face_wall_4course_v1_statics_wall_moon_trial_00_process\process.gif`

GIF 信息：

- target: `single_face_wall_4course_v1`
- gravity: `moon`
- success: `1`
- rock count: `21`
- frame count: `115`
- view: `front`

## 网络和数据飞轮状态

PoseRiskNet v18b 评估：

- 当前已产出 1 个 4 层 moon trial。
- 当前结果为 failure，但 visible courses 仍为 `4`，drift 较大，适合作为负样本。

Support-map v19：

- 当前处于 `export_support_maps`。
- 已写出至少 8 个 shard：`support_maps_0001` 到 `support_maps_0008`。
- 说明导出进程不是空转，仍在持续生成训练图。

Flywheel 3/4：

- collect 和 dataset 已完成。
- 当前正在导出 `20260622_cmd_auto_flywheel_3to4_221000_mujoco_depth_maps`。
- 已开始写 support-map shard。

## 远端 1080 Ti

远端调度器仍在运行：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\remote_1080ti_scheduler\20260622_remote_1080ti_wall_worker\scheduler.log`

当前状态：

- 第 0 轮远端 SSH 超时。
- 调度器会每 15 分钟继续尝试。
- 远端恢复在线后会自动上传 worker 并启动轻量 3/4 层 moon wall 采样。

## 当前判断

4 层已经有明确成功样例，但稳定性还不够。当前 `1/2` strict 成功不能直接说明达到 60% 门槛。正确路线仍然是：

1. 让 support-map v19 和 flywheel 导出完成。
2. 让 master v2 在模型齐全后自动发下一轮 4 层改进任务。
3. 将 trial 0 作为正样本，将 trial 1 和 PoseRisk eval failure 作为 hard negative。
4. 等累计 4 层 strict success rate 达到 `60%` 后，再进入 5 层探测。

