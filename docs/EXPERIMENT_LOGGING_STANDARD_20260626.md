# 石头堆叠实验日志与 README 标准

## 目的

后续日志不能只写“跑了什么”和“有没有成功”。每一轮 README 必须让人一眼看出：

- 数据从哪里来，经过哪些过滤，流向哪些网络。
- 系统获取了什么信息，例如石头几何、候选位姿、深度图、支撑图、墙体状态。
- 使用了哪些判别标准，例如低释放高度、底层支撑、连续性、PoseRisk、SupportMap。
- 哪些判别标准看起来有效，哪些没有效果或证据不足。
- 每个任务用了多少数据、训练多少次、仿真尝试多少次、成功率是否提升。
- 当前是否进入瓶颈，瓶颈更像是数据瓶颈、模型瓶颈、策略瓶颈还是物理释放/稳定瓶颈。

## 每轮 README 必须包含的结构

### 1. 实验目的

必须写清楚本轮实验到底验证什么。推荐格式：

```text
本轮目的：
1. 验证某个判别标准是否能提高 3/4 层单面墙成功率。
2. 验证某个网络是否能替代一部分启发式搜索。
3. 收集某类正负样本，用于下一轮训练。
```

不要只写“继续训练”或“继续堆叠”。如果只是增加数据，也要写清楚增加什么数据、服务哪个网络。

### 2. 数据流动

必须用表格或图描述数据流：

| 阶段 | 输入 | 处理 | 输出 | 用途 |
|---|---|---|---|---|
| 数据采集 | MuJoCo results / placement / candidate pose | 过滤异常与无关任务 | clean dataset | 训练小网络 |
| 石头筛选 | 石头几何 + slot | StoneSlotNet / 几何先验 | top-k stone | 降低石头搜索空间 |
| 位姿筛选 | 候选位姿 + 石头几何 | PoseRiskNet | top-k low-risk pose | 降低扰动风险 |
| 局部结构判断 | 深度图 / 支撑图 / wall state | SupportMapRanker / WallCritic | wall-aware score | 判断是否破坏墙体 |
| 物理验证 | top-k 方案 | MuJoCo settling | success / failure / metrics | 生成下一轮标签 |

如果某一阶段没有运行，也要标注“未运行”或“本轮跳过”，避免后续误读。

### 3. 获取了什么信息

每轮必须列出实际使用的信息，不要只写“训练神经网络”。

推荐字段：

| 信息类型 | 示例 | 来源 | 是否用于输入 | 是否只作为标签 |
|---|---|---|---|---|
| 石头几何 | 体积、bbox、粗糙度、棱角、主面数、支撑面比例 | rock generator / mesh analysis | 是 | 否 |
| 候选位姿 | x/y/z、四元数、候选编号 | pose search | 是 | 否 |
| 墙体状态 | top depth、support map、局部高度图 | MuJoCo render/export | 是 | 否 |
| 后验物理指标 | target error、drift、velocity、disturbance | MuJoCo validation | 否 | 是 |
| 成功标签 | strict success、shape success、placement success | results / placement log | 否 | 是 |

特别注意：后验成功率、测试后统计的石头成功率不能作为真实部署时的输入，只能作为标签或分析指标。

### 4. 判别标准与启发式

每轮必须列出启用的判别标准：

| 判别标准 | 本轮是否启用 | 输入信息 | 预期作用 | 统计观察 | 结论 |
|---|---|---|---|---|---|
| low release search | 是/否 | 碰撞前释放高度 | 降低冲击动能 | 成功率/漂移/速度变化 | 有效/无效/不确定 |
| base support prior | 是/否 | 底层 footprint / 支撑面积 | 提高底层可承载面积 | base/middle/cap 成功变化 | 有效/无效/不确定 |
| base continuity prior | 是/否 | 底层连续性 | 减少墙线断裂 | RMSE / y span / drift | 有效/无效/不确定 |
| PoseRiskNet | 是/否 | 几何 + 候选位姿 | 剔除高扰动位姿 | top1/top3 safe rate | 有效/无效/不确定 |
| SupportMapRanker | 是/否 | 局部墙体状态 + 候选 | 学习结构支撑关系 | top1/top3 hit / 闭环成功率 | 有效/无效/不确定 |
| StoneSlotNet | 是/否 | 石头几何 + slot | 粗筛石头 | top1/top3 hit | 有效/无效/不确定 |

“统计观察”可以先用启用/未启用对照，但 README 里要注明这不是严格因果证明。严格因果需要后续做同 seed、同数据、同目标的 A/B。

### 5. 数据量、训练量、尝试量

每轮必须写：

| 项目 | 数量 |
|---|---:|
| 新增 run |  |
| 新增 placement examples |  |
| 新增 candidate pose examples |  |
| 新增 assignment candidate examples |  |
| 训练样本 |  |
| 测试样本 |  |
| 训练 epochs |  |
| MuJoCo 尝试次数 |  |
| strict success |  |
| shape success |  |

如果是异步任务，必须写清楚“已完成”和“仍在后台运行”的部分，不要把未完成任务当成结果。

### 6. 效果增长率

后续 README 必须报告增长率，而不是只报当前成功率。

推荐公式：

```text
strict_success_rate = strict_success_count / trials
shape_success_rate = shape_success_count / trials
stable_fraction = stable_count_sum / rock_count_sum
growth_vs_previous = current_rate - previous_comparable_rate
growth_vs_best = current_rate - best_previous_rate
```

推荐表格：

| 任务 | 上一可比 run | 当前 run | trials | success rate 变化 | shape rate 变化 | stable fraction 变化 | 判断 |
|---|---|---|---:|---:|---:|---:|---|
| 3 层 moon | A | B |  |  |  |  | 提升/退化/持平 |
| 4 层 moon | A | B |  |  |  |  | 提升/退化/持平 |

### 7. 图表要求

每个阶段性 README 至少包含以下图表中的 2 个：

- 成功率随时间变化折线图。
- 不同任务 attempts / success / shape success 柱状图。
- 不同判别标准启用前后指标对比柱状图。
- 模型 top1/top3 指标对比柱状图。
- 数据量增长图，例如 placement / candidate pose / assignment candidate。

图表统一放在：

```text
docs/progress_reports/<report_name>/figures
```

README 中必须用相对路径或绝对路径引用图片，便于打开检查。

### 8. 瓶颈判断

推荐使用以下规则判断是否进入瓶颈：

| 现象 | 可能瓶颈 |
|---|---|
| strict success 长期不涨，但 stable fraction 和 RMSE 改善 | 成功标准过严或动态稳定瓶颈 |
| shape success 涨，strict success 不涨 | 速度、漂移、settling 或释放高度瓶颈 |
| top3 网络指标高，top1 很低 | 网络排序分辨率不足，不能直接 top1 部署 |
| StoneSlotNet top1/top3 都低 | 只看石头几何不够，缺墙体局部观测 |
| base 成功率高，middle/cap 成功率低 | 上层支撑和误差传播瓶颈 |
| 4 层 moon 反复失败，3 层 moon 高成功 | 课程迁移瓶颈，需要更多 4 层 hard negative |
| 启用某先验后成功率提升但 drift/velocity 变差 | 先验只改善形状，不改善动态稳定 |

如果最近 3 个可比 run 的 strict success 增长都小于 5 个百分点，并且 shape/stable 指标也没有提升，应在 README 中明确写“疑似进入瓶颈”，并提出改变数据或模型输入，而不是继续堆更多随机实验。

## 自动报告脚本

后续优先使用：

```powershell
C:\Users\all\miniconda3\envs\moon-rock-stack\python.exe -m scripts.generate_experiment_progress_report `
  --batch-root batch_runs `
  --output docs\progress_reports\<report_name> `
  --target-contains single_face_wall
```

该脚本会生成：

```text
README.md
success_by_run.csv
task_growth.csv
criterion_effectiveness.csv
dataset_flow.csv
model_metrics.csv
figures/
```

报告生成后，再根据具体实验补充人工解释。自动统计负责给出事实，人工日志负责解释为什么这样设计、下一步怎么改。
