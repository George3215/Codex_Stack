# 2026-06-28 多面体石头外观修正与堆叠实验记录

## 目的

用户指出上一版 NASA-like 石头集合外观和原始石头集合差别很大。这个判断是正确的：上一版主要按几何特征分布筛选，但 mesh 本身仍偏低面数规则多面体，视觉上不像真实月面/陨石类不规则块状石头。

本轮目标：

- 保留“有棱角、多面体、非平滑、无尖刺、不过扁”的石头先验。
- 不把 NASA 原始 OBJ 放进训练集，继续保持 test-only 隔离。
- 改进生成器，使 NASA-like 石头更像不规则块状多面体。
- 继续做 3 层和 4 层单面墙 dry stacking，记录成功和失败。

## 生成器版本对比

| 版本 | 生成方式 | 外观 | 可堆叠性 | 结论 |
|---|---|---|---|---|
| `nasa_like_wall` | 20 面左右低面数多面体 | 太规则，集合外观和 NASA 原始石头差异大 | 3 层 smoke 可达到 shape success，但不够可信 | 不能作为最终 NASA-like 外观 |
| `nasa_like_wall_v2` | 80 面左右不规则多面体 | 更接近不规则块状石头 | 支撑面不足，3 层中等步数失败 | 外观改善，但 dry stacking 性能下降 |
| `nasa_like_wall_v3` | 80 面不规则多面体 + 宽支撑面 | 保留不规则外观，同时加入 dry stacking 主面先验 | 3 层候选数 8 达到 strict success | 当前最合理 |

核心代码：

- `D:\MoonStack\experiments\moon_rock_stack\moon_rock_stack\fractal_rocks.py`
- 新增 profile：`nasa_like_wall_v2`
- 新增 profile：`nasa_like_wall_v3`
- 新增生成风格：`nasa_irregular`
- 新增生成风格：`nasa_stackable_irregular`

`nasa_stackable_irregular` 的关键变化：

- 使用 42 个顶点、80 个三角面，而不是 12 个顶点、20 个三角面。
- 添加方向微扰和 lobe/chip 扰动，形成更自然的不规则多面体。
- 显式生成若干宽支撑面，避免变成很多小面片拼成的“看起来像石头但没法堆”的形状。
- 继续使用 `spike_score`、`flatness`、`short_to_mid` 做几何筛选，防止尖刺和薄片进入候选池。

## v2 筛选与堆叠

筛选目录：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_nasa_like_wall_v2_screen_v1`

筛选结果：

- generated rocks：260
- accepted：252
- rejected：8
- accepted rate：96.9%

v2 说明：

- v2 的外观比 `nasa_like_wall` 更不规则。
- 但 v2 没有显式生成宽支撑面，许多石头虽然不是尖刺也不是薄片，但有效接触主面不足。

v2 中等步数 3 层实验：

- 目录：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_nasa_like_wall_v2_3course_mid_v1`
- target：`single_face_wall_3course_v1`
- gravity：moon
- candidates：4
- steps per rock：180
- hold steps：650
- strict success：0
- shape success：0
- stable count：9
- failure count：6
- visible courses：3
- stack height：0.2007 m
- target RMSE：0.2588 m
- max drift：0.0274 m

结论：

- v2 没有大幅漂移，说明低释放和 settling 仍有效。
- 失败主要来自可用支撑不足和目标误差，而不是高速散架。
- 单纯追求外观更像 NASA 原始石头是不够的，必须把 dry stacking 的宽主面和支撑面先验合进去。

## v3 筛选

筛选目录：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_nasa_like_wall_v3_screen_v1`

筛选结果：

- generated rocks：260
- accepted：243
- rejected：17
- accepted rate：93.5%
- average `support_face_area_ratio`：0.2424
- max `support_face_area_ratio`：0.6980
- average `spike_score`：0.1421
- max `spike_score`：0.1600
- average `flatness`：1.3655
- max `flatness`：1.6129

4 层墙贪心结构分配：

```json
{
  "bearing_block_clast": 3,
  "buttress_clast": 2,
  "cap_block_clast": 2,
  "compact_block_clast": 9,
  "course_block_clast": 6,
  "subangular_block": 1,
  "wall_block_clast": 1
}
```

解释：

- v3 的筛选通过率略低于 v2，但支撑面质量明显更好。
- `support_face_area_ratio` 的均值达到 0.242，说明许多石头具有可用于堆叠的主支撑面。
- `spike_score` 和 `flatness` 有一些样本接近阈值，因此几何筛选仍然必要。

## v3 3 层堆叠结果

成功实验目录：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_nasa_like_wall_v3_3course_c8_v1`

参数：

- target：`single_face_wall_3course_v1`
- gravity：moon，`1.624 m/s^2`
- candidates：8
- steps per rock：180
- hold steps：650
- candidate probe steps：24
- StoneSlotNet：c08 最新模型
- PoseRanker：c08 最新 support map 模型
- PoseRiskNet：c08 最新模型

结果：

| 指标 | 数值 |
|---|---:|
| strict success | 1 |
| shape success | 1 |
| stable count | 15 |
| failure count | 0 |
| visible courses | 3 |
| target RMSE | 0.0299 m |
| target max error | 0.0908 m |
| stack height | 0.2642 m |
| max drift | 0.0160 m |
| velocity inf norm | 0.2140 |

图像目录：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_nasa_like_wall_v3_3course_c8_v1\captures_v3_success_v1`

关键图：

- `wall_front_rgb.png`
- `wall_front_depth.png`
- `wall_front_object_depth.png`
- `wall_top_depth.png`
- `wall_top_object_depth.png`

结论：

- v3 在 3 层墙上已经能形成严格成功样本。
- 与 v2 对比，主要提升来自“宽支撑面 + 候选数 8”，不是简单增加随机尝试。
- 这个结果支持后续把 `support_face_area_ratio`、`support_plane_quality`、top-depth/support map 一起作为网络输入和损失设计依据。

## v3 4 层堆叠结果

实验目录：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_nasa_like_wall_v3_4course_c8_v1`

结果：

| 指标 | 数值 |
|---|---:|
| strict success | 0 |
| shape success | 0 |
| stable count | 13 |
| failure count | 11 |
| visible courses | 4 |
| target RMSE | 0.5623 m |
| target max error | 2.6594 m |
| stack height | 0.1898 m |
| max drift | 0.2162 m |
| velocity inf norm | 0.0869 |

失败原因统计：

| role / reason | count |
|---|---:|
| middle / unstable_structure | 5 |
| cap / missed_target+post_hold_drift | 2 |
| middle / missed_target | 2 |
| base / missed_target | 1 |
| middle / post_hold_drift | 1 |

按层统计：

| course / role | count |
|---|---:|
| 0 / base | 1 |
| 1 / middle | 5 |
| 2 / middle | 3 |
| 3 / cap | 2 |

图像目录：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_nasa_like_wall_v3_4course_c8_v1\captures_v3_4course_failure_v1`

结论：

- 4 层可见，但已经偏向低矮散开，不能算墙。
- 关键失败不是最后 cap 单点问题，而是 base 和 middle 早期误差累积。
- `base missed_target` 导致后续 middle/cap 的支撑几何被污染；4 层需要更严格的 base gate。
- 后续不能直接冲 5 层，应该先提高 v3 4 层的 base/middle 成功率。

## 和 c08 原始高墙评估的关系

c08 最新闭环评估已经完成：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_low_release_wall_master_v1_c08_flywheel_3to4_closed_loop_eval`

3 层：

- 2 个 trial 中 1 个 strict success。
- 最好样本：14 stable、0 failure、RMSE 0.0209 m、max drift 0.0058 m。

4 层：

- 2 个 trial 都失败。
- 两个 trial 都能显示 4 个 visible courses，但 shape/strict 没过。
- 失败仍集中在上层可行位姿、漂移和 early-course error accumulation。

说明：

- `nasa_like_wall_v3` 的 3 层成功和 c08 原始 profile 的 3 层成功是一致方向。
- 4 层仍是共同瓶颈，不是单一生成器的问题。

## 下一步

短期优先：

1. 对 v3 4 层失败样本做数据清洗，重点抽出：
   - `base/missed_target`
   - `middle/unstable_structure`
   - `cap/missed_target+post_hold_drift`
2. 提高 base gate：
   - base 层 candidate 必须满足更严格的 target error 和支撑面投影。
   - base 层宁可跳过不合适石头，也不要让一个大 missed target 污染后续层。
3. 训练/微调 PoseRanker：
   - 输入继续包含 top-depth/support map。
   - 增加 `support_face_area_ratio`、`support_plane_quality` 对高层稳定性的权重。
   - 对 v3 的负样本重采样，避免网络只记住旧 20 面石头。
4. 在 v3 上继续跑 3 层多 trial，把 strict success 率统计出来，而不是只看单个成功样本。

中期目标：

- v3 3 层 strict success 率达到 60%-80% 后，再系统推 4 层。
- 4 层先解决 base/middle 误差累积，再考虑 5 层。

