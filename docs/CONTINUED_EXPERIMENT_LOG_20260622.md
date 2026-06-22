# 2026-06-22 继续实验记录

## 本轮恢复目的

用户完成阶段性汇报 PPT 后要求继续实验。本轮不再盲目冲 10 层，而是继续围绕 3-4 层墙的可复现成功率、负样本积累和网络替代启发式展开。

当前原则：

- 不删除任何已有数据，只追加新 batch_runs、async_jobs 和日志。
- 本端 RTX 2080 Ti 作为主训练端，先做小网络训练和 3-4 层数据飞轮。
- 暂时不让新 v15/v16 网络直接接管高层闭环；成熟组合仍作为默认策略。
- 运行时输入只使用放置前可获得的信息：石头几何、当前 support map/depth、目标槽位、候选位姿、重力和层级。
- 仿真后指标，如历史成功率、drift、velocity、RMSE，只作为监督标签和实验评估。

## 当前默认策略

- 前 3 层：StoneSlotNet + support-map CNN + PoseRiskNet 缩小候选空间。
- 第 4 层：保留更多启发式和 MuJoCo 验证，避免不成熟网络误删好候选。
- 主要目标：提高 3-4 层成功率，继续收集 middle/cap 的 hard negatives。

## 本轮计划启动任务

1. 本地异步训练/采样调度。
   - 目标：3 层和 4 层单面墙。
   - 重力：以 moon 为主，保留后续 earth 对照。
   - 输出：`batch_runs/async_jobs/` 和新的 `batch_runs/<session>...`。

2. 采样结果进入 learning dataset。
   - 重点统计：placement success/failure、candidate pose、role、course、drift、velocity、wall RMSE。

3. 若出现典型成功或失败案例。
   - 继续用 RGB + top depth 为主的相机记录。
   - 不再依赖侧向 depth 做主要判断。

## 本轮关注指标

- 3 层 wall strict 或 placement 成功率是否继续提高。
- 4 层是否出现更多 seed602 类型的阶段性结构，而不是 seed604/605 类型的稳定石堆。
- middle/cap 的失败是否减少。
- 网络 top-1 是否提升，而不是只看 top-3。
- 墙线 RMSE、drift、velocity 是否能同时改善。
