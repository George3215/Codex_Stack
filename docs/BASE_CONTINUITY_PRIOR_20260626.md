# 2026-06-26 Base Continuity Prior 实验记录

## 实验目的

上一轮 `base_support_prior` 验证了用户观察：第一层过小会限制上层支撑面积。它能显著增大底层石头并提高墙体高度，但也带来新问题：底层大石头会堵住相邻槽位，导致 base 或 course 1/2 可达性下降。

本轮实验增加 `base_continuity_prior`，目标是把底层规则从“底层要大”推进到“底层要足够大，同时保留连续支撑窗口和相邻槽位可达性”。

## 新增方法

新增参数：

```text
--base-continuity-prior
--base-continuity-prior-weight
```

设计：

- `base_support_prior_score`
  - 负责避免底层石头过小。
  - 奖励较大的 footprint、volume、支撑面比例、对置面比例和 compactness。
- `base_continuity_prior_score`
  - 负责避免底层石头过大。
  - 惩罚超过槽位间距窗口的 footprint、投影面积、长边、短边和高度比。
  - 边缘槽位允许略大，内部槽位更严格。

调度默认值：

- `auto_wall_scale_scheduler.py` 默认开启 `base_continuity_prior`。
- 调度默认 `base_continuity_prior_weight=0.35`。
- 核心实验脚本默认关闭，需要显式传参，便于 A/B。

## 受控实验

所有实验保持同一套基础条件：

- target: `single_face_wall_4course_v1`
- strategy: `statics_wall`
- gravity: `moon`
- rock profile: `high_wall`
- rocks: `130`
- seed: `206266789`
- low release: on
- StoneSlotNet / SupportMap / PoseRisk: 使用旧 c24 3to4 模型
- `stone_fit_top_k=8`
- `candidate_pose_top_k=5`
- `pose_risk_weight=0.45`

## 四组结果对比

| run | base support | base continuity | continuity weight | visible | stable | failure | skipped | height m | drift m | rmse m | speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `20260626_no_base_support_prior_lowrelease_4course_fastprobe_v1` | 0 | 0 | - | 3 | 12 | 2 | 10 | 0.2023 | 0.2490 | 0.0932 | 0.0165 |
| `20260626_base_support_prior_lowrelease_4course_fastprobe_v1` | 1 | 0 | - | 3 | 10 | 3 | 11 | 0.2834 | 0.2224 | 0.0936 | 0.0336 |
| `20260626_base_support_continuity_lowrelease_4course_fastprobe_v1` | 1 | 1 | 1.00 | 4 | 11 | 7 | 6 | 0.1814 | 0.3229 | 0.1732 | 0.4442 |
| `20260626_base_support_continuity035_lowrelease_4course_fastprobe_v1` | 1 | 1 | 0.35 | 3 | 14 | 2 | 8 | 0.3342 | 0.1765 | 0.0760 | 0.0236 |

## 关键发现

1. `base_continuity_prior_weight=1.0` 太强。
   - 它把跳槽数从 11 降到 6，并且让 cap 层实际放上了 2 块。
   - 但结构变得动态不稳定，failure=7，speed=0.4442，height 反而只有 0.1814 m。
   - 解释：它过度牺牲了承重块规模和稳定性，虽然增加了可达性，但墙体不能稳定保持。

2. `base_continuity_prior_weight=0.35` 是当前更好的平衡点。
   - stable=14，是四组最高。
   - height=0.3342 m，是四组最高。
   - drift=0.1765 m，是四组最低。
   - rmse=0.0760 m，也是最好。
   - speed=0.0236，最终状态很安静。

3. 0.35 版本仍然没有 4 层 strict success。
   - visible_courses=3。
   - course 0 只跳 1 个槽。
   - course 1 完整放置。
   - course 2 跳 2 个槽。
   - course 3 cap 全部跳槽。

4. cap 层失败不是“没有支撑”。
   - cap best rejected 的 support count 通常是 4-7。
   - support overlap 约 0.77-0.85。
   - 很多 target error 并不大。
   - 主要失败原因是放置 cap 时扰动已有结构，`velocity_inf_norm_after_place` 约 3，`placed_disturbance_xy_m` 可到 6-14 cm。

## 更长 settle/probe 对照

为了验证 cap 是否只是月面低重力下 settle 时间不够，又跑了更长 settle/probe：

```text
batch_runs/20260626_base_support_continuity035_longsettle_4course_probe_v1
```

参数差异：

- `steps_per_rock`: 240 -> 420
- effective steps under Moon: 576 -> 1008
- `hold_steps`: 900 -> 1500
- effective hold under Moon: 2160 -> 3600
- `candidate_probe_steps`: 30 -> 60
- effective probe under Moon: 72 -> 144

结果：

| run | visible | stable | failure | skipped | height m | drift m | rmse m | speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.35 fastprobe | 3 | 14 | 2 | 8 | 0.3342 | 0.1765 | 0.0760 | 0.0236 |
| 0.35 longsettle | 3 | 12 | 3 | 9 | 0.3240 | 0.2369 | 0.1058 | 0.0571 |

结论：单纯增加 settle/probe 没有解决 cap 问题，反而略差。cap 失败不是简单的“等得不够久”，更像是 cap 候选与已有局部结构的低扰动兼容性不足。

## 图片记录

当前最有价值的 near-success 样本：

```text
batch_runs/20260626_base_support_continuity035_lowrelease_4course_fastprobe_v1/captures_960x720_base_continuity035_ab/00_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00
```

建议展示：

- `wall_front_rgb.png`
  - 正视图能看到连续墙体和高局部结构，不是土堆。
- `wall_top_depth.png`
  - 顶视深度有效，能看到墙线连续性和顶部断裂位置。

## 下一步

1. 自动调度采用 `base_support_prior=1`、`base_continuity_prior=1`、`base_continuity_prior_weight=0.35`。
2. cap 层单独建模，不再靠底层规则间接解决：
   - 建立 `cap_low_disturbance_negative` 标签。
   - 训练或微调 cap 专用 pose/risk ranker。
   - 输入应包括 top-depth / local support map / 已放石头局部稳定性。
   - 输出不只是落点分数，还要预测 `placed_disturbance_xy_m` 和 `velocity_inf_norm_after_place`。
3. 继续收集 3/4 层 hard negative：
   - `base_too_small_unstable`
   - `base_too_large_blocks_future_slots`
   - `cap_has_support_but_disturbs_structure`
4. 当前不建议继续盲目堆 5 层。4 层 cap 稳定闭合前，5 层成功没有科学意义。
