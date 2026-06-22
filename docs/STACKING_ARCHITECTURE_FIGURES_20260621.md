# 月面干叠石墙实验架构图 2026-06-21

这组图用于后续 README、PPT 或论文草稿，目标是把“石头生成、局部落点分析、网络参与决策、失败样本回流”的数据流向讲清楚。图中刻意区分了两类信息：

- 放置前可获得的输入：石头几何、目标槽位、当前墙体 RGB-D/depth/support map、重力、候选位姿。
- 仿真后才知道的标签：目标误差、漂移、速度、扰动、是否失败、最终结构指标。

后者只能用于训练标签和评估，不能作为运行时网络输入。

## Figure 1：数据飞轮总览

文件：

`docs/figures/fig1_data_flywheel.svg`

用途：

- 说明从文献/力学先验到石头生成、MuJoCo 实验、数据集、网络训练、闭环策略的完整循环。
- 强调失败样本回流，不删除负样本。
- 说明当前路线不是让网络盲目端到端控制，而是用小网络逐步减少启发式搜索。

推荐图注：

> 图 1. 月面干叠石墙的数据飞轮。系统首先根据 dry stacking 与材料力学先验生成多凸多面体石头库，随后在 Earth/Moon 重力下进行 MuJoCo 结构化堆叠实验。实验产生 placement、candidate pose、RGB-D 和失败日志，进入数据集构建与小网络训练。失败案例作为负样本回流，用于下一轮策略改进。

## Figure 2：局部落点分析

文件：

`docs/figures/fig2_local_placement_analysis.svg`

用途：

- 解释“当前墙体观测 + 候选石头 + 目标 slot”如何形成候选位姿。
- 展示 top-view support map、target slot、candidate footprint 的关系。
- 明确 support-map CNN / PoseRiskNet 用于候选排序和风险惩罚，而 MuJoCo 短仿真仍负责最终物理验证。

推荐图注：

> 图 2. 单个槽位的局部落点分析。系统从当前墙体 front/top depth 或局部支撑图中提取支撑区域，与候选石头几何和目标槽位组合生成多个候选释放位姿。网络对候选进行排序和风险估计，随后少量候选进入 MuJoCo 短时 settle 验证，最终结果写入候选日志和失败日志。

## Figure 3：模块化小网络策略

文件：

`docs/figures/fig3_modular_network_policy.svg`

用途：

- 说明当前不是单个大网络，而是 StoneSlotNet、SupportMap CNN、PoseRiskNet、WallStateCritic 等小网络组合。
- 解释每个网络的输入和输出。
- 强调当前网络只能缩小候选池，高层仍需要启发式与物理验证。

推荐图注：

> 图 3. 模块化小网络闭环策略。StoneSlotNet 负责槽位-石头适配性，SupportMap CNN 负责局部支撑感知的候选位姿排序，PoseRiskNet 提供预仿真风险惩罚，WallStateCritic 是下一阶段用于预测后续层可支撑性的模块。当前系统仍保留干砌先验和 MuJoCo 物理验证，以避免网络单点决策造成结构失稳。

## Figure 4：失败样本回流

文件：

`docs/figures/fig4_failure_negative_loop.svg`

用途：

- 解释为什么失败不是坏结果，而是负样本。
- 展示典型失败 `v10 risk-adjusted top-2` 如何进入 v11 数据集。
- 说明如果某个网络路线闭环退化，就记录并暂时不作为默认策略。

推荐图注：

> 图 4. 失败案例负样本回流。以 v10 risk-adjusted top-2 闭环失败为例，系统保留最终状态、候选位姿日志、失败原因、RGB-D 图像和结构指标。这些数据进入 v11 数据集用于重训和诊断。如果新网络离线指标提升但闭环退化，则不替换默认策略，而是把失败机制作为下一轮互锁 critic 的监督来源。

## 当前可放进报告的核心结论

1. 当前最稳闭环仍是旧 v4 support-map CNN + PoseRisk v5 + stone-fit v2。
2. v10 support-map CNN 离线 top-1/top-3 指标更好，但闭环退化，说明单点候选排序还没有学会互锁与后续可支撑性。
3. v11 数据集显示失败主要集中在 middle/cap，而不是 base。
4. 下一步应训练“层间互锁/后续可支撑 critic”，而不是继续单纯提高候选位姿 top-1。

## 图片引用

Markdown 中可直接引用：

```markdown
![Figure 1](docs/figures/fig1_data_flywheel.svg)
![Figure 2](docs/figures/fig2_local_placement_analysis.svg)
![Figure 3](docs/figures/fig3_modular_network_policy.svg)
![Figure 4](docs/figures/fig4_failure_negative_loop.svg)
```

如果后续做 PPT，可以把 SVG 直接拖入 PowerPoint，或者用浏览器/矢量软件导出为 PNG。
