# 2026-06-22 前三层网络 + 高层启发式堆叠实验记录

## 实验目的

本轮任务不是继续盲目冲高，而是围绕一个可复用的数据飞轮做验证：

- 前 3 层使用已有相对成熟的小网络缩小候选空间。
- 第 4-5 层保留启发式规则、几何先验和 MuJoCo 短仿真验证，继续收集正负样本。
- 新训练的小网络必须只使用放置前可获得的信息，不能把“这个石头历史成功率”或测试后统计量作为运行时输入。
- 重点观察 4-5 层单面墙中，哪些失败来自几何发散、哪些失败来自月面低重力下的残余速度或横向漂移。

## 当前闭环策略

默认闭环仍然使用上一轮最稳的成熟组合：

- StoneSlotNet：`batch_runs/20260621_wall_flywheel_3course_focus_v2_stone_slot_net_20260621_014556`
- support-map CNN v4 highwall foreground：`batch_runs/20260621_support_map_cnn_negative_mining_v4_highwall_foreground`
- PoseRisk v5：`batch_runs/20260621_pose_risk_net_candidate_metric_negative_mining_v5_after_mlpfix`

具体执行方式：

1. `course <= 2`：StoneSlotNet 先按槽位选石头，support-map CNN 只保留 top-3 候选位姿，PoseRiskNet 对高风险位姿加惩罚。
2. `course >= 3`：不再强制用网络裁剪，回到完整候选池、干叠几何先验和 MuJoCo 短仿真验证。
3. 所有失败都保留到 `candidate_pose_log.csv`、`placement_log.csv` 和 learning dataset 中，作为后续负样本。

这样做的原因：当前新网络的 top-1 能力还不够，直接让网络裁剪高层候选会把可行位姿剪掉；但前三层使用成熟网络可以降低早期搜索成本，并生成更稳定的中高层训练分布。

## 新增训练目标

本轮在 `scripts/train_torch_support_map_ranker.py` 中新增了 `drift_guarded` 监督目标。它只作为训练标签使用，运行时输入仍然不含仿真后信息。

运行时输入包括：

- 当前局部 support map：支承高度、占据、目标高斯、候选 footprint、重力比例、层数比例。
- 候选位姿：位置、姿态四元数、候选编号。
- 石头几何先验：体积、包围盒、长扁度、粗糙度、角度、主面数量、支承面质量等。

监督标签惩罚：

- 高层和 cap 层的 y 方向偏移。
- 放置后扰动。
- 残余速度。
- 支承重叠不足。
- 支承接触数量不足。
- 支承平衡误差过大。
- 高度增益不足。

这个目标的设计目的，是让网络学习“月面低重力下不容易横向漂移、也不会牺牲墙线形状”的候选，而不是单纯模仿当前启发式搜索。

## 数据集版本

### v15

路径：`batch_runs/20260621_negative_mining_wall_dataset_v15_course3net_upperheuristic_seed602`

- run dirs：30
- run examples：35
- placement examples：889
- candidate pose examples：28980
- assignment candidate examples：9784
- Moon placement：865，其中 success 488、failure 310、skipped 67

按角色：

| role | examples | success | failure | skipped |
|---|---:|---:|---:|---:|
| base | 244 | 236 | 6 | 2 |
| middle | 473 | 192 | 241 | 40 |
| cap | 172 | 75 | 72 | 25 |

### v16

路径：`batch_runs/20260622_negative_mining_wall_dataset_v16_course3net_upperheuristic_603604`

新增 seed603/604 的 4 层和 5 层样本。

- run dirs：34
- run examples：39
- placement examples：999
- candidate pose examples：32132
- assignment candidate examples：10664
- Moon placement：975，其中 success 555、failure 353、skipped 67

### v17

路径：`batch_runs/20260622_negative_mining_wall_dataset_v17_course3net_upperheuristic_605606`

新增 seed605 和 seed606 的 5 层失败样本。

- run dirs：36
- run examples：41
- placement examples：1061
- candidate pose examples：34836
- assignment candidate examples：11360
- Moon placement：1037，其中 success 592、failure 378、skipped 67

按角色：

| role | examples | success | failure | skipped |
|---|---:|---:|---:|---:|
| base | 286 | 277 | 7 | 2 |
| middle | 573 | 234 | 299 | 40 |
| cap | 202 | 96 | 81 | 25 |

结论：base 仍然不是瓶颈，middle 和 cap 是主要失败来源。后续网络训练应继续对 middle/cap 加权，而不是平均看所有层。

## 本轮闭环实验结果

| run | 目标 | 配置 | stable/failure | RMSE m | height m | drift m | velocity | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `20260621_course3net_upperheuristic_4to5_moon_candidates5_seed601_v1` | 4 层 | w0.35, candidates=5 | 15/9 | 0.2264 | 0.2616 | 0.0451 | 0.0270 | 动力学较稳，但几何误差偏大 |
| `20260621_course3net_upperheuristic_4to5_moon_candidates5_seed602_v1` | 4 层 | w0.35, candidates=5 | 20/4 | 0.1158 | 0.3611 | 0.4136 | 0.1446 | 目前最有结构感的 4 层阶段性案例，但严格成功被 drift 卡住 |
| `20260622_course3net_upperheuristic_4to5_moon_candidates5_seed603_w035_v1` | 4 层 | w0.35, candidates=5 | 12/12 | 1.4404 | 0.1750 | 2.8095 | 1.3978 | 严重几何和动力学发散，作为 hard negative |
| `20260622_course3net_upperheuristic_4to5_moon_candidates5_seed604_w050_v1` | 4 层 | w0.50, candidates=5 | 17/7 | 0.2209 | 0.2589 | 0.0954 | 0.0108 | w0.50 明显压低 drift/velocity，但墙线仍不够好 |
| `20260622_course3net_upperheuristic_5course_moon_candidates5_seed603_w035_v1` | 5 层 | w0.35, candidates=5 | 19/12 | 0.2385 | 0.3623 | 0.0473 | 1.6987 | 高度不错、漂移低，但残余速度很大 |
| `20260622_course3net_upperheuristic_5course_moon_candidates5_seed604_w050_v1` | 5 层 | w0.50, candidates=5 | 19/12 | 0.4307 | 0.2628 | 0.1842 | 0.0201 | 速度低，但几何发散 |
| `20260622_course3net_upperheuristic_5course_moon_candidates5_seed605_w050_v1` | 5 层 | w0.50, candidates=5 | 19/12 | 0.7923 | 0.2211 | 0.0126 | 0.0221 | 几乎是“稳定地堆歪”，不是有效墙 |
| `20260622_course3net_upperheuristic_5course_moon_candidates8_seed606_w025_longsettle_v1` | 5 层 | w0.25, candidates=8, long settle | 18/13 | 0.9562 | 0.2764 | 0.0031 | 0.0337 | 长 settling 和更多候选不能自动解决墙线形状 |

本轮没有 strict success=1 的完整墙；但是有阶段性结构案例和大量高价值负样本。当前严格失败主要不是“放不上去”，而是墙线形状、局部堆散、漂移和残余速度之间存在冲突。

## 新训练模型结果

### v15 drift_guarded support-map CNN

路径：`batch_runs/20260621_support_map_cnn_v15_drift_guarded_holdout`

- row count：28980
- rankable groups：9784
- test top-1：0.440
- test top-3：0.924
- 4 层 moon top-1/top-3：0.471/0.946
- 5 层 moon top-1/top-3：0.381/0.881

结论：可作为候选分析模型，但不应替换成熟 v4 ranker。

### v16 drift_guarded support-map CNN

路径：`batch_runs/20260622_support_map_cnn_v16_drift_guarded_holdout`

- row count：32132
- rankable groups：10664
- test top-1：0.396
- test top-3：0.911
- 4 层 moon top-1/top-3：0.423/0.935
- 5 层 moon top-1/top-3：0.360/0.878

结论：hard negatives 加入后，单纯加重 drift/velocity 标签没有提升泛化，说明目标函数还没有表达“墙线形状 + 可支承下一层 + 低漂移”的联合结构。

### v16 WallStateCritic

路径：`batch_runs/20260622_wall_state_critic_v16_course3net_upperheuristic_603604`

- test rows：6304
- failure accuracy：0.668
- failure F1：0.798
- group top-1：0.362
- group top-3：0.877

结论：它能做离线诊断，但 top-1 太低，暂时不能闭环裁剪候选。

### v16 PoseRiskNet

路径：`batch_runs/20260622_pose_risk_net_candidate_metric_v16_course3net_upperheuristic_603604`

- rows：32132
- risky positives：29229
- test accuracy：0.790
- test F1：0.882
- top1 safe：0.442
- top3 safe：0.891

结论：不如旧 PoseRisk v5 稳，不能替换默认风险网络。风险标签仍然过于偏正类，模型偏保守。

## 图像记录

本轮 capture 可以正常生成 RGB 和 depth 图。

### seed604 4 层典型失败

路径：`batch_runs/20260622_course3net_upperheuristic_4to5_moon_candidates5_seed604_w050_v1/captures_960x720_20260622/00_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00`

包含：

- `wall_front_rgb.png`
- `wall_front_depth.png`
- `wall_front_object_depth.png`
- `wall_top_rgb.png`
- `wall_top_depth.png`
- `wall_top_object_depth.png`
- front/back/left/right/top 多角度 RGB 和 depth。

该案例数值上 drift/velocity 较好，但图像上仍明显偏局部堆散，说明单看动力学稳定不够。

### seed602 4 层阶段性结构案例

路径：`batch_runs/20260621_course3net_upperheuristic_4to5_moon_candidates5_seed602_v1/captures_960x720_20260622/01_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00`

该案例 stable/failure 为 20/4，RMSE 0.1158 m，高度 0.3611 m，图像上能看到更明显的竖向结构，但 drift 0.4136 m，仍未达到 strict success。

## 当前经验总结

1. 前三层网络化是可行的，但高层不能过早让新网络强裁剪。
   当前成熟网络用于前三层可以稳定生成 4-5 层训练数据；新训练的 v15/v16 ranker、critic、PoseRisk 都还不足以替代启发式。

2. `pose_risk_weight=0.50` 倾向于降低 drift/velocity，但可能牺牲墙线形状。
   seed604 和 seed605 都出现“低漂移、低速度、但 RMSE 高”的情况。

3. `pose_risk_weight=0.35` 有时能产生更高结构，但速度风险更大。
   seed603 的 5 层高度达到 0.3623 m、drift 0.0473 m，但 velocity 1.6987，说明月面下的残余运动仍是关键失败模式。

4. 更长 settling 和更多候选不是充分条件。
   seed606 使用 candidates=8、long settle 后 drift 和速度非常低，但 RMSE 0.9562 m，说明它把错误形状稳定下来了。

5. middle/cap 是主要学习对象。
   v17 中 middle failure 299、cap failure 81，而 base failure 只有 7。后续数据采样和损失函数应继续对 middle/cap 重点加权。

6. 下一代网络目标应从“单候选好不好”升级为“放完后是否给下一层形成可支承墙线”。
   也就是说，loss 需要同时看当前 slot 的误差、墙线一致性、未来层支承能力和动力学稳定，而不是单独看 drift 或速度。

## 下一步建议

1. 不立即把 v16 `drift_guarded`、v16 WallStateCritic 或 v16 PoseRiskNet 接入闭环强裁剪。
2. 继续用成熟 v4/v5/v2 组合采集 4-5 层数据，重点复验 w0.42-w0.50 的 4 层墙，以及 w0.35 的 5 层速度失败。
3. 新建一个 future-support / wall-line critic：输入当前墙体观测 + 候选石头几何 + 目标槽位，输出候选是否会提高下一层 middle/cap 的可支承性。
4. 训练时把 v17 中“低 drift 但高 RMSE”的样本作为特殊 hard negative，避免网络学成“稳定地堆歪”。
5. capture 对典型失败和阶段性结构案例继续保留 RGB、top depth、wall-front depth；top depth 当前有价值，其他侧向 depth 只作为辅助。
