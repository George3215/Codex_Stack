# 2026-06-30 深层 MLP 数据飞轮继续实验记录

## 当前目的

当前首要目标不是盲目冲 4 层或 5 层，而是先把 3 层月面单面墙的数据飞轮跑顺：

1. 用已有启发式和物理先验制造 3 层单面墙正/负样本。
2. 训练更深的过渡网络，让网络逐步替代一部分启发式搜索。
3. 在严格 3 层评估成功率达到约 60% 之后，再推进 4 层和 5 层。

## 已完成的数据集

数据集路径：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_positive_mining_3course_moon_v1_learning_dataset`

数据规模：

- run examples: 62
- placement examples: 1236
- candidate pose examples: 44700
- assignment candidate examples: 13599
- moon placement examples: 675
- earth placement examples: 561

按角色统计：

- base: 406 examples, 309 success, 31 failure, 66 skipped
- middle: 548 examples, 306 success, 95 failure, 147 skipped
- cap: 282 examples, 131 success, 62 failure, 89 skipped

初步结论：顶层 `cap` 的成功比例明显低于底层 `base`，说明越往上越受支撑面积、局部倾斜、误差累计和释放扰动影响。

## 已完成的深层 MLP

## 失败机制细标注与旧数据复用

### 新增脚本

新增脚本：

`D:\MoonStack\experiments\moon_rock_stack\scripts\enrich_failure_mechanisms.py`

目的：把旧学习数据集合并去重，并追加更细的失败机制弱标签。这个脚本只新增输出目录，不修改旧数据。

机制标签：

- `mechanism_bottom_support_insufficient`: 底层或下层支撑不足
- `mechanism_upper_contact_too_few`: 上层接触点或直接支撑数量不足
- `mechanism_release_disturbance_excessive`: 释放或 probe 后扰动/速度过大
- `mechanism_geometry_mismatch`: 石头几何与槽位或姿态不匹配
- `mechanism_neighbor_gap_too_large`: 邻接缝隙或侧向连续性不足
- `mechanism_target_miss`: 落点偏离目标
- `mechanism_no_feasible_pose`: 未找到可行姿态
- `mechanism_post_hold_drift`: hold 后发生漂移
- `mechanism_low_or_fallen`: 低矮或掉落

这些标签是后验弱标签，只能作为监督信号、诊断统计或辅助 loss 使用，不能作为动作网络推理输入。未来端到端模型的输入应限制为石头几何、目标槽位、候选位姿、重力和真实观测图；机制标签作为 auxiliary target。

### 已完成的近期墙体增强数据集

输出：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_enriched_recent_wall_failure_mechanisms_v1`

输入数据集：13 个近期墙体/low-release/多面体数据集，包含 2026-06-29/30 数据、2026-06-28 多面体墙、2026-06-26 到 2026-06-28 low-release 飞轮数据。

去重后规模：

- run_examples: 114
- placement_examples: 2192
- candidate_pose_examples: 118570
- assignment_candidate_examples: 25831

placement 级标签：

- success: 1375
- negative: 817

placement 级主失败机制分布：

- success_or_safe: 1375
- no_feasible_pose: 378
- upper_contact_too_few: 240
- release_disturbance_excessive: 89
- target_miss: 55
- bottom_support_insufficient: 44
- neighbor_gap_too_large: 11

按层级看，当前最需要解决的是：

- middle 层 `no_feasible_pose`: 183
- middle 层 `upper_contact_too_few`: 136
- cap 层 `no_feasible_pose`: 123
- cap 层 `upper_contact_too_few`: 104

解释：目前失败瓶颈不是单一“随机搜索不够”，而是上层可行姿态、上层接触点、下层支撑连续性共同不足。

### 用增强数据训练的新中间网络

StoneSlotNet：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_enriched_recent_wall_stone_slot_net_v1`

- row_count: 25831
- input_dim: 113
- hidden_layers: 768, 384, 192
- activation: SiLU
- parameter_count: 456961
- test accuracy: 0.8199
- test precision: 0.0717
- test recall: 0.1525
- test F1: 0.0976
- group top1 hit rate: 0.1737
- group top3 hit rate: 0.3174

结论：StoneSlotNet 继续显示为弱项。旧数据扩大后，跨 run 泛化仍差，说明仅靠石头几何 + 槽位角色去预测“选哪块石头”还不够，后续必须加入当前墙面观测，例如 top depth/support map/local height field，再做 pairwise ranking 或 point-cloud/transformer 融合。

PoseRiskNet：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_enriched_recent_wall_pose_risk_net_v1`

- row_count: 118570
- input_dim: 135
- hidden_layers: 768, 384, 192
- activation: SiLU
- parameter_count: 473857
- test accuracy: 0.7734
- test precision: 0.8488
- test recall: 0.8531
- test F1: 0.8509
- group top1 safe rate: 0.6273
- group top3 safe rate: 0.8892

结论：PoseRiskNet 对候选姿态风险过滤是有价值的，可以继续保留在飞轮中。后续应该让它输出机制辅助头，例如支撑不足、接触点不足、释放扰动、目标偏移等，从“安全/不安全”升级成“为什么不安全”。

### 后台全历史合并任务

全历史合并任务已启动：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260630_cmd_all_history_failure_mechanism_merge_v1`

预期输出前缀：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_all_history_enriched_failure_mechanisms_v1`

这个任务会合并全部非 NASA 学习数据集，排除 NASA held-out 测试集。它较慢，作为后台任务运行，完成后再决定是否用全历史增强数据重训网络。

### StoneSlotNet

路径：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_positive_mining_3course_moon_v1_deep_mlp_train_stone_slot_net_deep`

网络：

- 输入维度：98
- 输出维度：1
- hidden layers: 512, 256, 128
- activation: SiLU
- dropout: 0.12
- 参数量：215041

验证指标：

- test accuracy: 0.7966
- test precision: 0.1210
- test recall: 0.2209
- test F1: 0.1564
- group top1 hit rate: 0.2292
- group top3 hit rate: 0.3750

解释：StoneSlotNet 现在还只是过渡网络。它能参与排序，但还不能完全替代启发式。主要问题是正样本稀少、跨 run 泛化弱、石头-槽位匹配本身比姿态风险过滤更难。

### PoseRiskNet

路径：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_positive_mining_3course_moon_v1_deep_mlp_train_pose_risk_net_deep`

网络：

- 输入维度：122
- 输出维度：1
- hidden layers: 640, 320, 160
- activation: SiLU
- dropout: 0.14
- 参数量：335361

验证指标：

- test accuracy: 0.7037
- test precision: 0.8172
- test recall: 0.7914
- test F1: 0.8041
- group top1 safe rate: 0.6879
- group top3 safe rate: 1.0000

解释：PoseRiskNet 比 StoneSlotNet 更可用，说明候选姿态风险过滤已经能学习到稳定信号。当前更大的瓶颈是“选择哪块石头放到哪个槽位”，不是单纯“某个 pose 是否安全”。

## 正在运行的任务

### 3 层成功样本收割支线

作业：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260630_cmd_positive_mining_3course_success_harvest_v1`

session：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_positive_mining_3course_success_harvest_v1`

目的：当前严格成功样本数量不足，因此新增一条偏向正样本收割的支线。它和 v2 的区别是：v2 保留 `commit-best-rejected` 以扩大负样本和失败分布；本支线关闭 `commit-best-rejected`，避免把被判定为高风险的候选强行放进墙里，从而提高完整 3 层墙正样本概率。

参数：

- rocks: 420
- clusters: 12
- candidates: 28
- trials per batch: 2
- batches: 2
- parallel jobs: 1
- stone-fit-top-k: 44
- candidate-pose-top-k: 5
- pose-risk-weight: 0.40
- base-support-prior-weight: 2.10
- base-continuity-prior-weight: 0.80
- release-search-step: 0.0025 m
- release-extra-clearance: 0.0015 m
- commit-best-rejected: false

预期：单 trial 速度会比 v2 慢，但更有机会拿到完整严格成功样本；若成功率明显提升，后续将把这条策略作为 3 层正样本挖掘的主策略。

### controller support-map 阶段

控制器：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260630_cmd_data_flywheel_controller_v1`

当前正在导出 support-map tensor：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_data_flywheel_controller_v1_support_maps`

当前已经写出：

- `support_maps_0001.csv`
- `support_maps_0001.npz`

这个阶段后续会训练 support-map CNN ranker，然后再做严格 3 层评估，判断是否推进 4 层。

### 深层 MLP exploit 追加采集

正式后台作业：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260630_cmd_positive_mining_3course_deepmlp_exploit_v2`

session：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_positive_mining_3course_deepmlp_exploit_v2`

策略：

- 目标：`single_face_wall_3course_v1`
- 重力：moon
- 采集模式：exploit
- batches: 3
- trials per batch: 3
- parallel jobs: 2
- rocks: 320
- candidates: 18
- stone ranker: deep StoneSlotNet
- pose risk ranker: deep PoseRiskNet
- pose ranker: previous support-map ranker
- 保留：low-release search, base-support prior, base-continuity prior, commit-best-rejected

启动时已经生成两个采集 worker：

- `20260630_positive_mining_3course_deepmlp_exploit_v2_collect_exploit_00_seed206302900`
- `20260630_positive_mining_3course_deepmlp_exploit_v2_collect_exploit_01_seed206303037`

早期进度：

- 两个 worker 都已经完成模型加载。
- 两个 worker 都已经完成第 0 个 base slot。
- 当前正在继续第 0 个 trial 的第 1 个 base slot。

## 资源状态

读数时间：2026-06-30 20:54 +08:00

- RTX 2080 Ti 显存占用约 10.7 GB / 11.3 GB。
- GPU 利用率约 24%。
- 主要 CPU 进程包括 support-map/depth-map 导出和两个新 MuJoCo 采集 worker。

因此当前不再继续加本机并行任务，避免显存顶满导致训练或导出崩溃。

## 当前判断

1. 3 层月面墙已经进入“深层 MLP 参与采集”的阶段。
2. PoseRiskNet 已经比较有用，可以稳定参与候选 pose 排序/风险过滤。
3. StoneSlotNet 还需要更多正样本，尤其是 middle 和 cap 的成功样本。
4. 下一轮重点不是继续扩大网络，而是制造更多高质量成功样本，再训练 StoneSlotNet。
5. 严格 4 层推进应该等待 controller 的 3 层严格评估结果；如果不到 60%，继续做 3 层数据飞轮更有科学价值。

## 不删除原则

本阶段所有任务均为追加写入：

- 未删除任何旧实验。
- v1 半启动 session 被保留为失败/中断记录。
- 正式继续任务使用 v2 独立 session。
