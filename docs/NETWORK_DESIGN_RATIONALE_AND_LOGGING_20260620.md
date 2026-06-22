# 网络设计目的与实验日志规范 2026-06-20

## 为什么需要重新说明

当前已经有若干小网络参与堆叠实验，但如果只说“训练了 CNN / MLP，然后成功率是多少”，是不够严谨的。必须说明：

1. 网络到底在替代哪一部分启发式。
2. 输入是不是任务真实可观测的信息。
3. 输出是不是实际机器人或自动堆叠系统能使用的动作。
4. 训练标签是不是只来自仿真后验，是否存在过拟合风险。
5. 该网络的科学意义是什么，不只是拟合已有数据。

用户指出的问题是正确的：如果网络只是看石头外观或几个几何统计量，然后拟合已有候选结果，本质上可能只是数据拟合；这种模型只能作为过渡工具，不能作为最终方案。

## 当前网络的真实定位

### 当前 pose CNN ranker 是什么

当前 `support_map_cnn_ranker` 不是最终端到端策略。它是一个过渡型候选排序器，用于回答一个较窄的问题：

> 在已有启发式生成出的若干候选释放位姿中，能否用网络减少需要真实 MuJoCo settle 的候选数量？

因此它替代的是“每块石头试 8 个候选姿态”的一部分搜索成本，而不是完整替代干叠规划。

### 当前输入

当前输入包含两类信息：

1. 石头自身信息：
   - bbox、体积、质量、compactness、flatness、angularity、spike score。
   - 主面/支承面等预仿真几何先验。
   - 候选位姿 `pose_x/y/z/qw/qx/qy/qz`。

2. 当前堆叠区域代理信息：
   - 局部 support map。
   - 包括当前支承高度、支承占据、目标高斯、候选 footprint、重力比例、层数比例。

注意：这个 support map 是从仿真日志重建的结构化高度图，不是直接来自相机深度图。因此它已经有“石堆区域状态”，但还不是最终想要的 RGB-D 感知输入。

### 当前输出

当前输出是候选姿态 ranking score。它不直接输出完整 SE(3) 放置动作，而是在候选集合里排序。

这有两个原因：

1. 数据规模还不够直接训练连续动作策略。
2. 候选排序更容易验证是否能减少搜索成本。

### 当前模型的意义

当前模型有意义，但意义有限：

- 有意义：验证“石头几何 + 当前支承区域状态”可以学习候选姿态排序，减少候选仿真数量。
- 有限：它仍依赖启发式候选生成，不能单独完成真实任务。
- 有风险：如果训练/测试 catalog 太接近，模型可能记住数据分布，而不是学到稳定堆叠规律。

因此它应该被记录为 `Phase 1: candidate-ranker baseline`，而不是最终 stacking policy。

## 后续真正需要的网络

后续目标应该是“感知-几何联合”的小网络或模块组合，而不是只看石头外观。

### 输入设计

网络输入应该至少包含四类：

1. 待放石头几何
   - 石头点云或 mesh sample。
   - 石头法向、主面、支承面。
   - 可选：多视角深度图或 PointNet/PointNet++ embedding。

2. 当前石堆/墙体观测
   - RGB-D 相机图。
   - 深度图转 height map 或 point cloud。
   - wall_front 深度图：看墙面偏移、倾斜、空洞、外凸。
   - wall_top 深度图：看 footprint、支承区域、墙厚、前后漂移。

3. 目标结构信息
   - 当前要放的 slot / course / role。
   - 目标墙体 mask 或目标 height field。
   - 已放置石头与目标槽位的误差。

4. 物理条件
   - 重力：Earth / Moon。
   - 摩擦参数、密度、接触设置。
   - 当前释放模式：低高度释放、未来 servo placement。

### 输出设计

短期不需要一步到位端到端输出完整动作，可以分阶段：

1. 小输出版本：候选排名
   - 输入：石头几何 + 当前石堆 depth/height map + 少量候选姿态。
   - 输出：每个候选姿态的稳定概率或质量分数。
   - 意义：替代大部分候选物理搜索。

2. 中等输出版本：位姿增量
   - 输入：当前候选位姿和观测。
   - 输出：`dx, dy, dz, dyaw, droll, dpitch` 和 confidence。
   - 意义：从启发式候选升级为网络修正候选。

3. 最终版本：直接动作/策略
   - 输入：待放石头点云、墙体 RGB-D/点云、目标结构。
   - 输出：释放位姿、接触策略、是否放弃该石头。
   - 意义：真正接近机器人任务。

## 推荐网络结构路线

### Phase 1：当前小网络基线

目的：

- 减少候选姿态仿真。
- 验证数据闭环和标签构造。
- 找出失败模式。

模型：

- local support map CNN + geometry MLP。
- 输出候选 ranking score。

局限：

- support map 不是相机深度图。
- 候选仍由启发式生成。
- 闭环成功率不等于离线 top-k 指标。

### Phase 2：RGB-D / Depth conditioned ranker

目的：

- 让网络真的利用观测信息，而不是只拟合石头外观。
- 将 wall_front_depth、wall_top_depth 纳入训练输入。

模型建议：

- 深度图 encoder：小型 ResNet / U-Net encoder。
- 石头几何 encoder：PointNet 或几何 MLP。
- 融合：concat + MLP / cross-attention。
- 输出：候选位姿质量分数。

输入示例：

- `stone_points`: `[N, 3]`
- `stone_normals`: `[N, 3]`
- `wall_front_depth`: `[1, H, W]`
- `wall_top_depth`: `[1, H, W]`
- `target_mask`: `[1, H, W]`
- `candidate_pose`: `[7]`
- `gravity`: `[1]`

输出示例：

- `quality_score`
- `predicted_target_error`
- `predicted_post_hold_drift`
- `failure_probability`

### Phase 3：动作修正网络

目的：

- 不只在候选里排序，而是修正候选。

输出：

- `dx, dy, dz, dyaw, droll, dpitch`
- `release_clearance`
- `reject_candidate`

这一步需要更多成功/失败 paired 数据。

## 训练与评估必须避免的问题

1. 不把后验成功率作为输入。
   - 不输入“这个石头历史成功率”。
   - 不输入“该候选仿真后的 target error”。
   - 不输入“该候选仿真后的 support contact count”。

2. 后验量只能作为标签。
   - 可以监督 predicted drift。
   - 可以监督 predicted target error。
   - 可以监督 failure probability。

3. 必须做跨 catalog / 跨 seed 测试。
   - 同一批石头内随机切分不够。
   - 需要 held-out rock catalog。
   - 需要 Earth/Moon 分开统计。

4. 闭环成功率优先于离线 hit rate。
   - 离线 top-3 高不代表结构成功。
   - 四层实验已经证明这一点。

## 标准实验顺序

后续每个实验记录必须说明如下顺序：

1. 实验目的
   - 要验证什么假设。
   - 当前实验在长期目标中的位置。

2. 目标结构
   - 例如 `wall_segment_v1`、`single_face_wall_4course_v1`。
   - 层数、槽位数、角色分布。

3. 石头生成与筛选
   - profile、数量、seed、几何先验。
   - 是否使用目标条件筛选。

4. 放置方式
   - 当前是低高度静止释放。
   - clearance、settle steps、hold steps。

5. 网络设计
   - 输入是什么。
   - 输出是什么。
   - 替代哪一段启发式。
   - 为什么这样设计。
   - 局限是什么。

6. 数据来源
   - 使用哪些 run 构建学习集。
   - candidate-pose 样本数。
   - 是否跨 catalog 测试。

7. 训练指标
   - top-1/top-3 hit。
   - regret。
   - 是否剔除 post-simulation features。

8. 闭环实验配置
   - target、strategy、gravity、trials。
   - candidates、top-k、workers。
   - steps per rock、hold steps。

9. 闭环结果
   - success rate。
   - shape success。
   - stable count。
   - failure count。
   - RMSE、max error、height、drift、velocity。

10. 失败模式
   - 按 gravity / role / failure reason 统计。
   - 特别记录 cap、middle、base 的失败差异。

11. 图像与深度图
   - 必须保存 RGB 与 depth。
   - 至少包含 `wall_front_rgb.png`、`wall_front_depth.png`、`wall_top_rgb.png`、`wall_top_depth.png`。
   - 必须写出 `capture_manifest.csv` 路径。

12. 决策
   - 这个实验说明什么。
   - 下一步继续、修改还是停止该方向。

## 图片记录规范

后续有成功或典型失败时，必须保存：

- `wall_front_rgb.png`
- `wall_front_depth.png`
- `wall_front_depth.npy`
- `wall_top_rgb.png`
- `wall_top_depth.png`
- `wall_top_depth.npy`
- 其他视角：front/right/back/left/top
- `capture_manifest.csv`

日志中必须写：

- 截图目录。
- manifest 路径。
- 成功案例数量。
- 失败案例数量。
- 是否包含深度图。

## 当前工作修正

从现在开始，不再把“训练 ranker”简单表述成“网络学会堆叠”。更准确表述是：

- 当前网络：候选姿态排序器。
- 当前目的：减少候选物理搜索，并产生失败/成功数据。
- 下一阶段网络：感知-几何联合 ranker，输入石头几何和石堆 RGB-D/深度图。
- 更后阶段：位姿修正或直接动作输出。

这能避免把一个过渡模型误解成最终策略，也能让每个实验有明确科学目的。

## 已完成的第一步感知化尝试

为了开始向“石头几何 + 石堆观测”过渡，新增了双视角深度代理导出器：

- 脚本：`scripts/export_depth_observation_maps.py`
- 数据：`batch_runs/20260620_depth_observation_maps_4course_curriculum_v2`
- 模型：`batch_runs/20260620_torch_depth_proxy_cnn_4course_curriculum_score_v1`

### 输入形式

该导出器输出 13 个通道：

- `top_height_before_m`
- `top_support_occupancy`
- `top_support_count_clipped`
- `top_target_gaussian`
- `top_candidate_footprint`
- `top_candidate_height_m`
- `front_support_silhouette`
- `front_support_depth_proxy`
- `front_target_gaussian`
- `front_candidate_silhouette`
- `front_candidate_depth_proxy`
- `gravity_ratio`
- `course_ratio`

其中 top view 近似俯视深度/高度图，front view 近似墙面正视深度图。注意这仍然是从日志和几何重建出的 proxy，不是真正相机 RGB-D 原始图。

### 训练结果

| model | rows | channels | top-1 | top-3 | top-1 regret | top-3 regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4course support-map CNN | 12,484 | 8 | 0.436 | 0.989 | 9.931 | 0.033 |
| 4course depth-proxy CNN | 12,484 | 13 | 0.420 | 0.985 | 10.571 | 0.066 |

结论：

- depth-proxy 模型可以训练并在线推理，但离线指标没有超过旧 support-map 模型。
- 这说明“简单拼接代理深度通道”还不够。
- 下一步应该导出真正的相机深度图或点云，而不是继续只用几何 proxy。

### 闭环烟测

路径：`batch_runs/20260620_highcourse_4course_depth_proxy_top3_seed97004_smoke_v1`

配置：

- target：`single_face_wall_4course_v1`
- trials：Earth 1、Moon 1
- ranker：`batch_runs/20260620_torch_depth_proxy_cnn_4course_curriculum_score_v1`
- top-k：3

结果：

| gravity | success | shape | stable | failure | height m | RMSE m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 0/1 | 0/1 | 16/24 | 2 | 0.415 | 0.105 |
| Moon | 0/1 | 0/1 | 14/24 | 4 | 0.193 | 0.137 |

该烟测的意义不是成功率，而是确认新通道可以进入在线 ranker 闭环。失败图和深度图已保存：

- `batch_runs/20260620_highcourse_4course_depth_proxy_top3_seed97004_smoke_v1/captures_960x720`
- `capture_manifest.csv`
- 2 个失败案例，28 张 PNG，14 个 depth NPY。

## 已完成的第二步：MuJoCo 渲染深度数据

这一轮的目的不是马上追求更高墙，而是验证一个更接近真实感知输入的数据链路：

- 候选石头自身几何仍然作为已知先验输入。
- 当前墙体/石堆状态不再只用几何统计，而是导出正视图和俯视图深度观测。
- 网络输出仍然是候选 pose 的 ranking score，用来替代一部分候选 pose 搜索。

新增脚本：

- `scripts/export_mujoco_depth_observation_maps.py`

该脚本会根据已有 placement/candidate 日志重建每个候选放置前的支撑状态，在 MuJoCo 中隐藏无关自由石头，只渲染已经成功放置的支撑石头和当前候选石头，导出 10 通道张量：

- `render_front_depth_norm`
- `render_front_valid`
- `render_top_depth_norm`
- `render_top_valid`
- `top_target_gaussian`
- `top_candidate_footprint`
- `front_target_gaussian`
- `front_candidate_silhouette`
- `gravity_ratio`
- `course_ratio`

为了避免候选排序训练被随机抽行破坏，导出脚本新增了按候选组采样：

- group key: `run_name/target_name/strategy/gravity/trial/slot_id/candidate_rock_index`
- 参数：`--max-groups`
- 默认对 candidate 数据使用 `candidate-groups` 模式。

这点很重要。候选 pose ranker 的训练样本必须保留同一个 slot 下的多个候选，否则网络无法学习排序关系。

### 真实渲染数据集

小样本 smoke：

- `batch_runs/20260620_mujoco_depth_observation_maps_4course_group_smoke_v1`
- 8 个候选组，24 条样本。
- 渲染跳过：0。
- 目的：验证导出和训练接口。

正式小数据集：

- `batch_runs/20260620_mujoco_depth_observation_maps_4course_groups256_v1`
- 来源：`batch_runs/20260620_learning_dataset_4course_curriculum_v1`
- target: `single_face_wall_4course_v1`
- 候选组：256
- 样本：809
- shard: 3
- 渲染跳过：0
- 地球样本：518
- 月球样本：291
- run 数：6

### 真实渲染深度 CNN

模型目录：

- `batch_runs/20260620_torch_mujoco_depth_cnn_4course_groups256_score_v1`

训练设置：

- PyTorch + CUDA AMP
- GPU: NVIDIA GeForce RTX 2080 Ti
- target mode: `score`
- post-simulation features 已从输入中剔除。
- split: `--split-by-run`
- rows: 809
- rankable groups: 256
- train groups: 249
- test groups: 7

离线指标：

| metric | value |
| --- | ---: |
| test top-1 hit | 0.429 |
| test top-3 hit | 0.857 |
| train top-1 hit | 0.442 |
| train top-3 hit | 0.980 |
| test top-1 quality regret | 1.609 |
| test top-3 quality regret | 0.566 |

解释：

- top-3 在 held-out run 上有一定可用性，但 test 只有 7 个组，所以不能视为泛化结论。
- train top-3 很高而 top-1 一般，说明模型更适合作为 top-k 缩小搜索，而不是直接独立输出最终 pose。
- 这仍然是候选 pose ranker，不是完整端到端 stacking policy。

### 在线闭环兼容

当前闭环模拟还没有逐候选在线 MuJoCo 相机渲染，因此在 `moon_rock_stack/structured.py` 中增加了一个保守兼容层：

- `render_front_depth_norm` 在线用 `front_support_depth_proxy` 和候选前视深度 proxy 填充。
- `render_front_valid` 在线用前视轮廓填充。
- `render_top_depth_norm` 在线用高度 proxy 填充。
- `render_top_valid` 在线设为全 1。

这使得用真实渲染数据训练的模型可以先进入闭环，但必须记录清楚：

- 训练数据是 MuJoCo 渲染深度。
- 在线推理目前是 proxy 深度。
- 这不是最终 RGB-D 在线策略。

下一步如果要严谨，应把在线 ranker 输入升级为真正的当前 MuJoCo 相机渲染，或者直接从当前仿真状态导出点云/深度图。

### 闭环结果

闭环目录：

- `batch_runs/20260620_highcourse_4course_mujoco_depth_proxyonline_top3_seed97005_v1`

配置：

- target: `single_face_wall_4course_v1`
- strategy: `statics_wall`
- trials: Earth 2, Moon 2
- rocks: 140
- candidates: 8
- pose ranker: `batch_runs/20260620_torch_mujoco_depth_cnn_4course_groups256_score_v1`
- top-k: 3

结果：

| gravity | trials | success | shape | mean stable | mean failure | mean RMSE m | mean height m | mean drift m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 2 | 0/2 | 0/2 | 11.0 | 5.0 | 0.151 | 0.374 | 0.232 |
| Moon | 2 | 0/2 | 0/2 | 13.0 | 3.5 | 0.113 | 0.314 | 0.256 |

典型单次结果：

- Moon trial 0: stable 15/16, RMSE 0.064 m, height 0.398 m, visible courses 4, but shape success = 0。
- Earth trial 1: stable 9/13, height 0.423 m, shape success = 0。

失败模式：

- 不是完全散乱石堆。
- 也不是合格石墙。
- 主要问题是墙体横向覆盖不足，正视图看起来更像窄柱或局部高塔。
- `missed_target` 和 `post_hold_drift` 仍是主要失败原因。

截图和深度图：

- `batch_runs/20260620_highcourse_4course_mujoco_depth_proxyonline_top3_seed97005_v1/captures_960x720`
- `capture_manifest.csv`
- 4 个典型失败案例。
- 包含 `wall_front_rgb.png`、`wall_front_depth.png`、`wall_top_rgb.png`、`wall_top_depth.png` 以及对应 NPY。

### 本轮结论

这轮不是成功提升，而是有价值的负结果：

1. 仅用候选质量分数训练 ranker 会偏向局部稳定和高度，容易把墙堆成柱。
2. 网络标签里必须加强结构性目标，例如横向覆盖、错缝、墙厚、outlier 和 course balance。
3. top-k ranker 仍可以减少启发式搜索，但不能独立保证墙体结构。
4. 下一轮应把 label 从单一 `candidate_score` 改成 structure-aware score：
   - 稳定性
   - 目标误差
   - 横向 span 增益
   - wall aspect
   - y 厚度约束
   - course/slot 覆盖
   - post-hold drift 风险

这更符合 dry stacking 的目标：不是堆高一堆石头，而是形成有明确结构功能的墙。

## 标签改进实验：risk 与 structure-aware

在 MuJoCo 渲染深度 score 模型之后，又训练了两个标签改进版本，目的是验证“只改候选 pose 标签”是否足以修正墙体形状。

### risk-adjusted

模型：

- `batch_runs/20260620_torch_mujoco_depth_cnn_4course_groups256_risk_v1`

标签：

- `candidate_score`
- 减去 target error
- 减去 placed disturbance
- 减去 velocity risk
- 减去 radial distance risk
- 惩罚负 height gain

闭环：

- `batch_runs/20260620_highcourse_4course_mujoco_depth_risk_proxyonline_top3_seed97006_v1`

结果：

| gravity | success | shape | stable | height m | drift m | y span m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 0/2 | 0/2 | 15.0 | 0.235 | 0.160 | 0.511 |
| Moon | 0/2 | 0/2 | 13.0 | 0.248 | 0.306 | 0.445 |

解释：

- Earth 稳定数提高，但墙厚没有改善。
- Moon 没有稳定收益。
- 说明风险标签仍然主要处理单次放置风险，不能控制全局墙形。

### structure-aware

代码：

- `scripts/train_torch_support_map_ranker.py`
- 新增 `--target-mode structure_aware`

标签：

- 在 `risk_adjusted` 基础上进一步惩罚 `target_y_error_m`。
- 额外惩罚 `target_x_error_m`。
- 对 middle/cap 层的高度不足和 y 偏差加重惩罚。

模型：

- `batch_runs/20260620_torch_mujoco_depth_cnn_4course_groups256_structure_v1`

闭环：

- `batch_runs/20260620_highcourse_4course_mujoco_depth_structure_proxyonline_top3_seed97007_v1`

结果：

| gravity | success | shape | stable | height m | drift m | y span m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 0/2 | 0/2 | 15.0 | 0.248 | 0.096 | 0.493 |
| Moon | 0/2 | 0/2 | 10.5 | 0.174 | 0.263 | 0.625 |

解释：

- Earth 漂移下降，说明结构标签有局部效果。
- 但墙厚仍严重超标。
- Moon 变差，说明简单加重 y 惩罚会牺牲低重力可放置性。

### 设计结论

现在可以明确：

1. candidate pose ranker 可以替代一部分启发式 pose 搜索。
2. 但只做 pose ranker 不足以把 4 层单面墙成功率拉起来。
3. 失败不只在“某个候选姿态不好”，还在：
   - 哪块石头被分配到哪个 slot。
   - 当前墙体横向覆盖是否均匀。
   - 当前墙厚是否已经超标。
   - 上层是否有足够支撑面。
   - 是否应该换石头，而不是强行从当前石头中选 pose。

下一步网络应拆为更合理的小模块：

- `StoneSlotNet`: 输入石头几何和目标 slot/role/course，输出石头是否适合该位置。
- `PoseRankNet`: 输入候选 pose、石头几何、局部 wall_front/wall_top depth，输出 pose score。
- `WallStateCritic`: 输入当前墙体 depth/height map 和目标结构 mask，输出未来 wall_y_span、wall_x_span、outlier risk、shape_success probability。
- `RejectRepairHead`: 决定继续放、换石头、调整 pose，还是修复局部结构。

这比继续堆更高更有意义。当前 4 层墙已经足够暴露学习问题，应先把 2-4 层的结构成功率和形状一致性做上去，再继续 5-6 层或 10 层。
