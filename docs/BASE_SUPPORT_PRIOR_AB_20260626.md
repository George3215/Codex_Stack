# 2026-06-26 底层支撑先验 A/B 实验记录

## 实验目的

验证用户观察到的现象：当前第一层经常使用体积较小、支撑面积较小的石头。这可能不符合 dry stacking 的结构先验，因为底层承担更大的力流路径需求，如果底层 footprint 不足，上层石头会缺少连续支撑窗口，堆叠越高越容易横向漂移或局部失稳。

本轮实验的目标不是追求一次性 4 层成功，而是严谨验证：

1. 原策略是否系统性偏向小底层石头。
2. 增加底层支撑先验后，底层石头的几何统计是否显著改变。
3. 底层变大后，是否带来更高墙体或更低漂移。
4. 如果失败，新的失败模式是什么，下一步该怎么修正。

## 代码改动

新增参数：

- `--base-support-prior`
- `--base-support-prior-weight`

影响范围：

- `moon_rock_stack/structured.py`
  - 在 wall 类策略的 `course=0, role=base` 石头池排序中加入底层承重先验。
  - 日志记录 `base_support_prior_enabled`、`base_support_prior_weight`、`base_support_prior_score`。
- `moon_rock_stack/run_structured_experiment.py`
  - 增加 CLI 参数并写入 `results.csv`。
- `scripts/auto_wall_scale_scheduler.py`
  - 自动调度默认开启底层支撑先验。
- `scripts/run_wall_data_flywheel.py`
  - 数据采样和闭环评估可透传底层支撑先验。

编译验证通过：

```powershell
C:\Users\all\miniconda3\envs\moon-rock-stack\python.exe -m py_compile moon_rock_stack\structured.py moon_rock_stack\run_structured_experiment.py scripts\auto_wall_scale_scheduler.py scripts\run_wall_data_flywheel.py
```

## 底层先验定义

这个先验不是简单选择最大石头。它综合考虑：

- `footprint = 0.5 * (bbox_x + bbox_y)`
- `footprint_area = bbox_x * bbox_y`
- `volume`
- `support_face_area_ratio`
- `support_face_count`
- `opposing_face_area_ratio`
- `compactness`

同时惩罚：

- 尖刺过强的石头
- 过高瘦的石头
- 过长或过扁导致支撑不均的石头
- 更适合 cap/chock/tie 的石头出现在 base 层

重要设计原则：小石头没有被全局删除或禁用。它们仍然可以用于中上层找平、楔紧和收口。这个先验只改变第一层 base 的选石排序。

## 半程探针

输出目录：

```text
batch_runs/20260626_base_support_prior_lowrelease_4course_probe_v2
```

参数摘要：

- target: `single_face_wall_4course_v1`
- gravity: `moon`
- strategy: `statics_wall`
- low release: on
- old StoneSlotNet / SupportMap / PoseRisk: on
- `stone_fit_top_k=14`
- `candidate_pose_top_k=8`
- `base_support_prior=1`

状态：

- 前台运行 30 分钟后被工具超时截断。
- 没有生成最终 `results.csv`。
- `structured_progress.csv` 保留到 slot 14。
- 保留该目录，不删除，用作半程负样本和耗时证据。

已放置底层统计：

| metric | value |
|---|---:|
| base_count | 6 |
| base_mean_volume | 0.000922 |
| base_mean_area | 0.01895 |
| base_mean_footprint | 0.13716 |

对照旧 low-release fullprobe：

| run | base_mean_volume | base_mean_area |
|---|---:|---:|
| old low-release fullprobe | 0.000452 | 0.01023 |
| base prior half probe | 0.000922 | 0.01895 |

结论：底层支撑先验确实把底层石头从“小体积/小 footprint”推向更像地基的承重块。

## 同参数快速 A/B

为了得到完整结果，又跑了一个同参数快速 A/B。两组唯一核心差异是是否开启 `--base-support-prior`。

### 不开启底层先验

输出目录：

```text
batch_runs/20260626_no_base_support_prior_lowrelease_4course_fastprobe_v1
```

结果：

| metric | value |
|---|---:|
| success | 0 |
| shape_success | 0 |
| visible_courses | 3 |
| stable_count | 12 |
| failure_count | 2 |
| skipped_slot_count | 10 |
| stack_height_m | 0.2023 |
| max_horizontal_drift_m | 0.2490 |
| target_rmse_xy_m | 0.0932 |
| base_count | 7 |
| base_mean_volume | 0.000482 |
| base_mean_area | 0.01065 |
| base_mean_footprint | 0.1033 |

底层来源：

```text
course_block_clast, course_block_clast, equant_clast, compact_block_clast, equant_clast, interlock_block_clast, wall_block_clast
```

### 开启底层先验

输出目录：

```text
batch_runs/20260626_base_support_prior_lowrelease_4course_fastprobe_v1
```

结果：

| metric | value |
|---|---:|
| success | 0 |
| shape_success | 0 |
| visible_courses | 3 |
| stable_count | 10 |
| failure_count | 3 |
| skipped_slot_count | 11 |
| stack_height_m | 0.2834 |
| max_horizontal_drift_m | 0.2224 |
| target_rmse_xy_m | 0.0936 |
| base_count | 5 |
| base_mean_volume | 0.001137 |
| base_mean_area | 0.02250 |
| base_mean_footprint | 0.1483 |

底层来源：

```text
buttress_clast, buttress_clast, equant_clast, bearing_block_clast, equant_clast
```

## A/B 总结

| run | base prior | success | visible courses | stable | failure | skipped | height m | drift m | rmse m | base mean volume | base mean area | base mean footprint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `20260626_no_base_support_prior_lowrelease_4course_fastprobe_v1` | 0 | 0 | 3 | 12 | 2 | 10 | 0.2023 | 0.2490 | 0.0932 | 0.000482 | 0.01065 | 0.1033 |
| `20260626_base_support_prior_lowrelease_4course_fastprobe_v1` | 1 | 0 | 3 | 10 | 3 | 11 | 0.2834 | 0.2224 | 0.0936 | 0.001137 | 0.02250 | 0.1483 |

直接结论：

- 用户观察成立：原策略底层系统性偏小。
- 底层支撑先验有效改变选石分布：base 平均面积约翻倍。
- 开启先验后，墙体高度从 0.2023 m 提升到 0.2834 m。
- 开启先验后，最大水平漂移从 0.2490 m 降到 0.2224 m。
- 但严格 4 层成功率没有提升，两组都是 0/1。
- 开启先验后跳槽略增加，稳定石头数下降，说明“大底层”带来新的几何可达性问题。

## 失败分布

| run | course 0 skipped | course 1 skipped | course 2 skipped | course 3 skipped |
|---|---:|---:|---:|---:|
| no base prior | 0 | 1 | 4 | 5 |
| base prior | 2 | 0 | 4 | 5 |

解释：

- 不开启先验时，第一层能填满，但底层太小，整体高度不足，上层支撑窗口不强。
- 开启先验时，底层变大，墙更高、漂移略低，但第一层自身出现 2 个 `no_feasible_pose`。
- 新问题不是“底层要不要大”，而是“底层应当足够大，同时不能堵住相邻槽位和下一层支撑窗口”。

## 图片记录

base prior 失败案例：

```text
batch_runs/20260626_base_support_prior_lowrelease_4course_fastprobe_v1/captures_960x720_base_support_prior_ab/00_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00
```

no base prior 失败案例：

```text
batch_runs/20260626_no_base_support_prior_lowrelease_4course_fastprobe_v1/captures_960x720_no_base_prior_ab/00_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00
```

建议展示图片：

- `wall_front_rgb.png`
  - 正视图，能看出开启底层先验后底部更宽、更高，但仍未形成连续 4 层墙。
- `wall_top_depth.png`
  - 俯视深度图，这次有效，不是全黄图。
  - 能看到墙线方向断裂，以及中部大块对后续路径的挤压。
- `top_depth.png`
  - 用于展示整体高度分布。

## 下一步策略

1. 保留 `base_support_prior`，但不要继续盲目增大底层石头。
2. 新增 `base continuity / slot reachability` 指标：
   - 底层石头被选中后，相邻 base 槽必须仍有可行放置空间。
   - course 1 必须有连续支撑窗口。
   - 尽量形成错缝路径，而不是单个大石头孤立占位。
3. 将神经网络目标从“某石头适不适合某槽”推进到“放下这个石头后，局部未来 1-2 层是否仍有连续可行支撑窗口”。
4. 负样本标注拆成两类：
   - `base_too_small_unstable`
   - `base_too_large_blocks_future_slots`
5. 后续自动调度默认开启底层支撑先验，并单独统计 base volume/area/footprint 与 3-4 层成功率、跳槽率、漂移的相关性。
