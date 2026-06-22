# 2026-06-21 神经网络负样本飞轮实验记录

## 实验目的

本轮实验的核心不是继续盲目堆高，而是检验“更多 4 层墙正负样本 + 小网络重训”能否替代一部分启发式搜索，并提高月面 4 层单面墙的稳定率。

具体问题：

- 新增失败样本能否训练出更好的候选位姿网络。
- support-map CNN 是否能替代旧 v4 候选位姿 ranker。
- PoseRiskNet 是否能从新增失败中学到“看起来对齐但会滑/散”的候选。
- 更长单石 settling 是否能解决当前 4 层墙速度和漂移问题。

## 数据和模型输入输出

### Support-map CNN

输入：

- 当前槽位周围的局部支撑图，64 x 64，共 8 个通道：已有支撑高度、占据、支撑计数、目标高斯、候选 footprint、候选高度、重力比例、层数比例。
- 候选位姿和石头几何特征：目标位置、候选位姿、bbox、体积、粗糙度、棱角度、主面数量、支撑面质量等。
- 不输入 candidate 后验仿真结果；训练时用后验质量作为标签。

输出：

- 同一槽位、同一候选石头下各候选位姿的排序分数。
- 在闭环中作为 `--candidate-pose-ranker-dir`，用于裁剪候选位姿搜索空间。

### PoseRiskNet

输入：

- 重力、层数、目标位置、候选位姿四元数、石头几何特征、类别特征。
- 不输入“历史成功率”或任何测试后统计成功率。

输出：

- 候选位姿的预仿真风险概率。
- 在闭环中通过 `--pose-risk-weight` 对高风险候选加惩罚。

### WallStateCritic

输入：

- front/top 局部支撑张量 + 石头几何和候选数值特征。

输出：

- 预测放置后的目标误差、扰动、速度、height gain 和失败概率。

状态：

- 已训练并评估，但还未接入闭环；当前 top-1 能力不足，只适合做分析和候选池诊断。

## 实验顺序

1. 从 v10 数据集导出局部支撑图：

   `batch_runs/20260621_negative_mining_support_maps_v10_weight_sweep`

2. 训练 WallStateCritic：

   `batch_runs/20260621_wall_state_critic_v10_closedloop_holdout`

3. 训练 support-map CNN v10 structure-aware：

   `batch_runs/20260621_support_map_cnn_v10_structure_holdout`

4. 用 v10 structure-aware 替换旧 v4 ranker，做 4 层月面墙闭环：

   `batch_runs/20260621_4course_moon_supportv10_structure_stonefit_poseriskv5_w035_medium_v1`

5. 放宽 support-map 候选保留到 top-3：

   `batch_runs/20260621_4course_moon_supportv10_structure_top3_stonefit_poseriskv5_w035_medium_v1`

6. 训练 support-map CNN v10 risk-adjusted：

   `batch_runs/20260621_support_map_cnn_v10_riskadjusted_holdout`

7. 用 v10 risk-adjusted 做 top-2 和 top-3 闭环：

   `batch_runs/20260621_4course_moon_supportv10_riskadj_top2_stonefit_poseriskv5_w035_medium_v1`

   `batch_runs/20260621_4course_moon_supportv10_riskadj_top3_stonefit_poseriskv5_w035_medium_v1`

8. 回到旧 v4 最佳组合，测试更长单石 settling：

   `batch_runs/20260621_4course_moon_supportv4_stonefit_poseriskv5_w035_steps240_hold900_v1`

9. 合并 v10 + 新失败实验，生成 v11 数据集：

   `batch_runs/20260621_negative_mining_wall_dataset_v11_neural_failures`

10. 用 v11 重训 PoseRisk：

    `batch_runs/20260621_pose_risk_net_candidate_metric_negative_mining_v8_neural_failures`

    `batch_runs/20260621_pose_risk_net_candidate_metric_negative_mining_v8b_relaxed_neural_failures`

## 关键统计

### 数据集

v11 数据集：

- run dirs: 23
- run examples: 26
- placement examples: 659
- candidate pose examples: 22404
- assignment candidate examples: 7944
- Moon placement: 635，其中 success 352、failure 216、skipped 67
- Earth placement: 24，其中 success 15、failure 9

按角色：

| role | examples | success | failure | skipped |
|---|---:|---:|---:|---:|
| base | 181 | 177 | 2 | 2 |
| middle | 351 | 139 | 172 | 40 |
| cap | 127 | 51 | 51 | 25 |

解释：

- base 基本不是瓶颈。
- middle 和 cap 是主要失败来源。
- 这支持下一步把网络目标集中到“中层互锁”和“封顶稳定”。

### 离线模型指标

| 模型 | 数据 | 测试 top-1 | 测试 top-3 | 备注 |
|---|---|---:|---:|---|
| WallStateCritic v10 | v10 support maps | 0.482 | 1.000 | top-1 不够，暂不接入闭环 |
| Support-map CNN v10 structure-aware | v10 support maps | 0.598 | 1.000 | 离线提升，但闭环不稳 |
| Support-map CNN v10 risk-adjusted | v10 support maps | 0.615 | 1.000 | 离线最好，但 top-2 闭环很差 |
| PoseRisk v8 strict | v11 | 0.726 | 1.000 | 低于 v5，风险标签过密 |
| PoseRisk v8b relaxed | v11 | 0.705 | 1.000 | 放宽阈值仍未改善 |
| 当前闭环首选 PoseRisk v5 | v6 | 0.772 | 0.994 | 仍保留为默认 |

### 4 层月面墙闭环对比

| 实验 | stable/failure | RMSE m | max error m | height m | drift m | velocity | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| 旧 v4 + PoseRisk v5 w0.35 | 16/8 | 0.2165 | 0.9013 | 0.2613 | 0.0454 | 0.3844 | 当前最佳稳定性 |
| v10 structure top-2 | 15/9 | 0.3486 | 1.3887 | 0.2960 | 0.1475 | 0.0337 | 更高但更散 |
| v10 structure top-3 | 13/11 | 0.1575 | 0.3500 | 0.1949 | 0.2145 | 0.0147 | 几何误差低但稳定差 |
| v10 risk-adjusted top-2 | 8/16 | 0.4986 | 2.0857 | 0.2007 | 0.2281 | 0.0126 | 明显失败 |
| v10 risk-adjusted top-3 | 13/11 | 0.1575 | 0.3500 | 0.1949 | 0.2145 | 0.0147 | top-3 基本等价于不裁剪 |
| 旧 v4 + steps 240 | 14/10 | 0.7659 | 3.6610 | 0.2444 | 0.0713 | 0.0350 | 单纯增加 settling 无效 |

## 典型失败图像

典型失败案例：

`batch_runs/20260621_4course_moon_supportv10_riskadj_top2_stonefit_poseriskv5_w035_medium_v1/cap_v10risk2`

图像目录：

`batch_runs/20260621_4course_moon_supportv10_riskadj_top2_stonefit_poseriskv5_w035_medium_v1/cap_v10risk2/00_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00`

包含：

- `front_rgb.png`
- `wall_front_depth.png`
- `wall_front_object_depth.png`
- `top_rgb.png`
- `wall_top_depth.png`
- `wall_top_object_depth.png`
- right/back/left 多角度 RGB 和 depth/object_depth

该失败是典型的“离线候选排序看起来好，但 top-2 裁剪后互锁不足，墙体散开”的负样本。

## 当前结论

1. 仅扩大数据集并重训 support-map CNN，没有自动提高闭环成功率。

   离线 top-1/top-3 变好，但闭环稳定性下降，说明当前标签主要学习了局部几何对齐，还没有充分表达“后续层会不会依赖这个支撑”和“互锁是否可靠”。

2. top-3 指标很高不等于可闭环。

   在 `candidates=3` 的设置下，top-3 近似没有裁剪能力。它说明候选池里有好姿态，但不能证明网络能替代搜索。

3. PoseRisk v8/v8b 暂不替代 v5。

   v11 新负样本让 risky 标签比例过高，模型倾向于保守判危险，候选组 top-1 safe 反而下降。

4. 更长单石 settling 不是主要瓶颈。

   steps 从 180 增加到 240 后，最终速度降低，但墙线误差明显恶化，说明问题不是简单“没等够”，而是选择和互锁策略不够好。

5. 当前闭环默认仍应使用：

   - support-map CNN v4 highwall foreground
   - PoseRisk v5
   - stone-fit v2
   - pose-risk weight 0.35

## 下一步建议

1. 不再直接把 v10 support-map CNN 替换进闭环。

   它适合作为候选池分析器，但不适合作为 top-2 强裁剪器。

2. 训练“层间互锁/后续可支撑” critic。

   新网络目标应从单个候选的局部对齐，改成预测该候选是否会让下一层 middle/cap 更容易成功。输入仍使用已知几何先验和当前墙体观测，不输入测试后成功率。

3. 对 middle/cap 做角色加权采样。

   v11 统计显示 middle/cap 是主要失败源，训练和数据采集应提高这两类槽位权重。

4. 闭环策略暂时保守：

   前 2-3 层可以继续使用神经网络辅助缩小候选池；高层仍保留启发式和 MuJoCo 短仿真验证。完全退出启发式还不严谨。

5. 下一批实验应增加候选数量，例如 `candidates=5` 或 `candidates=6`。

   当前 `candidates=3` 下，top-3 指标无法体现网络有效性，也限制了学习信号。

## 架构图与数据流向图

为了后续 README、PPT 和论文草稿中更清楚地说明实验逻辑，已补充一组论文风格 SVG 架构图：

图注和使用说明：

`docs/STACKING_ARCHITECTURE_FIGURES_20260621.md`

图片文件：

- `docs/figures/fig1_data_flywheel.svg`：月面干叠石墙数据飞轮总览。
- `docs/figures/fig2_local_placement_analysis.svg`：单个槽位的局部落点分析、support map、候选 footprint 和目标槽关系。
- `docs/figures/fig3_modular_network_policy.svg`：StoneSlotNet、SupportMap CNN、PoseRiskNet、WallStateCritic 的模块化策略。
- `docs/figures/fig4_failure_negative_loop.svg`：失败案例如何进入负样本数据集并回流到下一轮训练。

这组图强调两点：

1. 网络输入只包含放置前可获得的信息：石头几何、当前墙体观测、目标槽位、重力和候选位姿。
2. 仿真后的目标误差、漂移、速度、失败标签只用于监督学习和评估，不能作为运行时输入。

## 继续实验：candidates=5 与 interlock-aware 标签

新增详细记录：

`docs/CANDIDATES5_INTERLOCK_RESULTS_20260621.md`

关键结论：

- `candidates=5 + support-map v4 top-3 all courses` 明显失败，stable/failure 为 `10/14`，RMSE `1.1058 m`，drift `1.9560 m`。
- 不使用 support-map 裁剪、保留 PoseRisk v5 的候选全集更好，stable/failure 为 `13/11`，RMSE `0.4505 m`，drift `0.5384 m`。
- 新增 `interlock_aware` 标签并训练 `20260621_support_map_cnn_v12_interlock_holdout_candidates5`，离线 test top-1/top-3 为 `0.473/0.936`，闭环 stable/failure 为 `10/14`，未进入默认策略。
- 课程门控更合理：`course <= 2` 使用 support-map v4 top-3，cap 层回退完整候选，得到 stable/failure `13/11`、RMSE `0.1680 m`、stack height `0.2989 m`。
- PoseRisk 权重从 `0.35` 提到 `0.50` 能降低 drift 到 `0.1176 m`，但 RMSE 变差到 `0.2726 m`，说明仍需单独训练 drift/future-support critic。
