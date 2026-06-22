# 2026-06-22 4-to-5 层自动墙体升阶实验记录

## 实验目的

当前阶段不盲目冲 10 层，而是把 4 层单面墙作为 gate：

- 如果 4 层严格成功率出现阶段性通过样本，则自动进入 5 层单面墙。
- 如果 4 层仍未通过，则继续做 4 层严格评估、3/4 层数据飞轮、负样本采集和小网络训练。
- 严格评估不使用 `commit-best-rejected`，避免把课程样本当成成功率。
- 数据采集允许使用 `commit-best-rejected`，目的是保存 hard negative、near miss 和可学习失败样本。

## 本轮新增方法

1. 新增目标 `single_face_wall_5course_v1`。
   - 用作 4 层之后的明确 5 层 gate。
   - 仍属于严格单面墙目标，要求可见 5 层、足够高度、较窄 y 厚度、足够 x 跨度和 wall aspect。

2. 新增调度脚本 `scripts/auto_wall_scale_scheduler.py`。
   - 读取观察中的实验输出。
   - 若 4 层 strict success 达到阈值，启动 5 层 probe。
   - 否则继续启动 4 层改进批次。
   - 支持 `cmd` 启动方式，避免 Python detached 进程在当前桌面沙箱中无日志退出。

3. 修正 `scripts/run_wall_data_flywheel.py`。
   - 让后续 collect/eval 都可以透传 `--candidate-probe-steps`。
   - 后续自动批次会把 candidate-probe 与严格评估保持一致。

## 当前已完成的观察结果

### `20260622_top8_probe40_course3net_3to4_moon_trials2_20260622_193500`

- 3 层墙：2 次试验，strict success `1/2`，shape success `1/2`。
- 4 层墙：2 次试验，strict success `0/2`，shape success `0/2`。
- 4 层平均可见层数 `2.5`，平均高度约 `0.216 m`，平均漂移约 `0.242 m`。
- 结论：当前 course=3 网络参与没有稳定提升 4 层，4 层仍被 post-hold drift、跳槽/skipped slot 和高层支撑连续性限制。

截图已保存：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_top8_probe40_course3net_3to4_moon_trials2_20260622_193500\cap`

## 当前正在运行的任务

### 4 层严格评估

- job: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_221000_cmd_auto_strict_4course\run.cmd`
- output: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_cmd_auto_strict_4course_221000`
- 设置：moon、`single_face_wall_4course_v1`、2 trials、120 rocks、candidate top-8、candidate probe 40、net active through course 3。
- 用途：给 4 层严格成功率增加独立样本。

### 3/4 层数据飞轮

- job: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_221000_cmd_auto_flywheel_3to4\run.cmd`
- output session: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_cmd_auto_flywheel_3to4_221000`
- 设置：3/4 层 wall、exploit collection、`commit-best-rejected`、然后重建数据集、导出深度/支撑图、训练 StoneSlotNet / support-map ranker / wall-state critic。
- 注意：该批次启动时 collect 命令尚未带 candidate-probe；脚本已修正，后续自动批次会带 probe。

### 自动 monitor

- job: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_222500_cmd_auto_wall_scale_monitor\run.cmd`
- monitor state: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\auto_wall_scale\20260622_auto_wall_scale_monitor_cmd\manifest.json`
- 行为：
  - 当前检测到两个 `cmd_auto_*` 任务仍在运行，所以不重复发任务。
  - 任务结束后每 15 分钟检查一次结果。
  - 如果 4 层 strict success 达到阈值，就启动 `single_face_wall_5course_v1`。
  - 否则继续发下一轮 4 层改进批次。

## 当前科学判断

- 3 层阶段仍可作为网络参与的训练/验证入口，但本次 `1/2` 说明网络参与需要继续用更多样本稳住泛化，而不能只看前一轮 `4/4` 的小样本结果。
- 4 层的核心问题不是“能不能放到目标附近”，而是上层支撑链、同层连续性和 hold 后漂移。新增的 support-continuity 字段会用于后续 hard-negative 学习。
- 下一步的价值不是直接随机冲 10 层，而是把 4 层成功率拉起来，并让小网络替代更多启发式候选筛选；当 4 层出现稳定 strict success 后，再把相同数据飞轮迁移到 5 层。

## 继续训练任务：2026-06-22 晚

### PoseRiskNet v18 / v18b

- v18 任务：
  - job: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_224500_cmd_poserisk_v18_train_eval4`
  - 状态：失败并保留日志。
  - 失败原因：脚本路径启动时缺少 `PYTHONPATH`，`build_learning_dataset.py` 找不到本地 `moon_rock_stack` 包。
- v18b retry：
  - job: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_224800_cmd_poserisk_v18b_train_eval4`
  - dataset: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_poserisk_v18b_recent_3to4_dataset`
  - model: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_poserisk_v18b_recent_3to4_train`
  - eval: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_poserisk_v18b_eval_4course_moon`
  - 数据量：11 个运行，48 个 run examples，936 个 placement examples，64984 个 candidate-pose examples，10384 个 assignment-candidate examples。
  - 训练结果：test accuracy `0.749`，test F1 `0.842`，candidate group top1 safe `0.605`，top3 safe `0.860`。
  - 解释：这是严格候选风险模型，不是直接预测结构成功；价值在于把高风险候选排到后面，减少 4 层 cap/middle 的无效搜索。

### Support-map CNN v19

- job: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_225000_cmd_supportmap_v19_train_eval4`
- tensor maps: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_supportmap_v19_recent_3to4_maps`
- model: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_supportmap_v19_recent_3to4_train`
- eval: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_supportmap_v19_poserisk_v18b_eval_4course_moon`
- 输入：候选石头几何、候选位姿、当前墙体局部 top/front depth/support map。
- 训练目的：把“观察当前石堆状态再选落点”的能力交给 CNN ranker，逐步减少启发式候选排序。
- 加权策略：middle/cap 和高层 course 权重更高，因为 4 层失败主要发生在中上层与 cap。

### Neural v19 monitor

- job: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260622_225500_cmd_neural_v19_wall_monitor`
- state: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\auto_wall_scale\20260622_neural_v19_wall_monitor\manifest.json`
- 行为：
  - 等待当前 4 层严格评估、3/4 数据飞轮、PoseRiskNet v18b、SupportMap v19 结束。
  - 后续默认使用 StoneSlotNet + SupportMap v19 + PoseRiskNet v18b。
  - 如果观察到 4 层 strict success，进入 `single_face_wall_5course_v1`；否则继续 4 层改进。
