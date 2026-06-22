# 四层单面墙课程实验记录 2026-06-20

## 目的

在三层短墙 `wall_segment_v1` 的 pose CNN top-3 已经达到月球 83.3% 成功率后，开始向更高结构推进。本轮先选择 `single_face_wall_4course_v1`，它是 4 层、24 个槽位的长单面墙，比 9 槽三层短墙明显更难，但还没有直接进入 8/10 层高墙的强偶然区。

本实验的直接目的不是证明网络已经学会“堆墙”，而是验证：

1. 低层训练得到的候选姿态排序器能否迁移到更高的四层墙。
2. 四层墙失败主要来自姿态候选不足，还是来自上层石头选择和支承结构。
3. 四层墙是否值得作为后续感知-几何联合网络的课程数据。

## 放置假设

沿用当前低高度静止释放方法：

- 先把石头瞬移到候选释放位姿。
- 释放位姿高度为 `support_top + 0.5 * bbox_z + drop_clearance`。
- 对 `statics_wall`，底层 clearance 为 0.004 m，上层 clearance 为 0.014 m。
- 石头初始速度清零，然后 MuJoCo 自由下落、接触和 settle。
- 全部石头放完后执行整堆 hold，检查 drift、掉落和目标误差。

该方法先保留，用于筛选石头和规则；未来升级方向是 servo placement，而不是直接在本轮加入机械臂轨迹。

## 网络设计说明

本轮使用的 pose CNN ranker 是一个过渡模型。它的定位是：

- 输入：石头几何、候选释放位姿、局部支承高度图、目标槽位和重力。
- 输出：候选位姿 ranking score。
- 替代对象：把每块石头的 8 个候选位姿减少到 top-3 进行 MuJoCo settle。
- 不替代：石头池选择、完整操作轨迹、连续动作规划、相机感知。

这不是最终希望使用的网络。最终应该是“待放石头几何 + 当前石堆 RGB-D/深度图 + 目标结构”联合输入，然后输出候选位姿质量、位姿修正或直接释放动作。

本轮仍然使用 support map，而不是直接用相机深度图。它有价值，因为它验证了当前支承区域信息对候选排序有用；但它也有局限，因为它不是原始观测，不能代表真实机器人感知输入。

后续升级方向已经记录在：

- `docs/NETWORK_DESIGN_RATIONALE_AND_LOGGING_20260620.md`

## 实验 1：低层 pose CNN top-3 迁移到四层墙

路径：`batch_runs/20260620_highcourse_4course_presim_pose_top3_seed97001_v1`

配置：

- target：`single_face_wall_4course_v1`
- slots：24 个，4 层
- strategy：`statics_wall`
- rocks：140
- profile：`high_wall`
- candidates：8
- pose ranker：`batch_runs/20260620_torch_support_map_cnn_lowwall_presim_score_v1`
- pose top-k：3
- steps per rock：420
- hold steps：1800
- workers：4

运行说明：

- 原计划 8 个地球 trial + 8 个月球 trial。
- 运行 30 分钟超时，但已完整写出 8 个地球 trial 和 4 个月球 trial。
- 没有删除 partial run，结果可继续用于训练。

实验顺序：

1. 用 `high_wall` profile 生成 140 块多面体石头。
2. 提取石头几何和主面/支承面先验。
3. 构建 `single_face_wall_4course_v1` 的 24 个目标槽位。
4. 每个槽位由 `statics_wall` 生成候选石头池。
5. 每块石头生成 8 个候选释放位姿。
6. pose CNN ranker 先对 8 个候选位姿排序，只保留 top-3 做 MuJoCo settle。
7. 对 top-3 候选分别低高度释放、自由 settle，选评分最好者提交。
8. 全墙放完后执行 hold，统计 stable count、目标误差和失败原因。
9. 对成功和典型失败保存 RGB/depth 图。

结果：

| gravity | trials | success | shape | mean stable | mean failure | RMSE m | height m | drift m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 8 | 0/8 = 0.0% | 0.0% | 12.75/24 | 3.75 | 0.144 | 0.236 | 0.159 |
| Moon | 4 | 1/4 = 25.0% | 25.0% | 12.25/24 | 3.25 | 0.132 | 0.322 | 0.072 |

失败模式：

- Earth middle `missed_target`：11 次。
- Earth middle `missed_target+post_hold_drift`：6 次。
- Earth cap `missed_target+post_hold_drift`：4 次。
- Earth cap `missed_target`：4 次。
- Moon middle `missed_target`：3 次。
- Moon cap `missed_target`：3 次。

解释：

- 4 个可见层通常能形成，但严格成功很低。
- 失败集中在 middle/cap，说明上层槽位的支承几何和目标命中还不够可靠。
- 地球重力下 post-hold drift 更严重。

图像：

- 截图目录：`batch_runs/20260620_highcourse_4course_presim_pose_top3_seed97001_v1/captures_960x720`
- manifest：`batch_runs/20260620_highcourse_4course_presim_pose_top3_seed97001_v1/captures_960x720/capture_manifest.csv`
- 包含 1 个成功案例、2 个典型失败案例。
- 成功案例：`00_single_face_wall_4course_v1_success_statics_wall_moon_trial_00`
- 包含 wall_front 和 wall_top 的 RGB/depth 图。
- 每个案例包含：
  - `wall_front_rgb.png`
  - `wall_front_depth.png`
  - `wall_front_depth.npy`
  - `wall_top_rgb.png`
  - `wall_top_depth.png`
  - `wall_top_depth.npy`

## 实验 2：高墙 expanded quality ranker 对照

路径：`batch_runs/20260620_highcourse_4course_expanded_quality_top3_seed97002_v1`

配置：

- target：`single_face_wall_4course_v1`
- pose ranker：`batch_runs/20260619_torch_support_map_cnn_quality_ranker_expanded_temp20_holdout93015_v1`
- pose top-k：3
- 其余配置基本同实验 1。

运行说明：

- 原计划 4 个地球 trial + 4 个月球 trial。
- 运行 20 分钟超时，但已完整写出 4 个地球 trial 和 3 个月球 trial。

结果：

| gravity | trials | success | shape | mean stable | mean failure | RMSE m | height m | drift m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 4 | 0/4 = 0.0% | 0.0% | 15.50/24 | 3.25 | 0.128 | 0.319 | 0.088 |
| Moon | 3 | 0/3 = 0.0% | 0.0% | 9.00/24 | 3.33 | 0.163 | 0.230 | 0.181 |

解释：

- expanded quality ranker 的高墙 holdout top-3 指标更好，但迁移到当前 4 层闭环没有带来严格成功。
- 它能在 Earth 上提高平均 stable count 和 height，但 shape success 仍为 0。
- 说明 ranker 离线 top-k 指标不能直接等价为闭环结构成功率。

## 实验 3：四层课程数据专用 ranker

路径：`batch_runs/20260620_highcourse_4course_curriculum_ranker_top3_seed97003_v1`

训练数据：

- 学习表：`batch_runs/20260620_learning_dataset_4course_curriculum_v1`
- support-map 张量：`batch_runs/20260620_local_support_maps_4course_curriculum_v1`
- ranker：`batch_runs/20260620_torch_support_map_cnn_4course_curriculum_score_v1`

训练数据来源：

- 早期 `single_face_wall_4course_v1` dense candidate minibatch。
- 早期 4course baseline/neural 闭环数据。
- 本轮 `seed97001` 和 `seed97002` 四层 partial run。

离线训练指标：

| metric | value |
| --- | ---: |
| candidate rows | 12,484 |
| rankable groups | 3,992 |
| test top-1 | 0.436 |
| test top-3 | 0.989 |
| Earth top-3 | 0.982 |
| Moon top-3 | 1.000 |

注意：这是随机 group split，同一批 run 内部存在相关性，所以不能把 0.989 top-3 解释为真实泛化成功率。

闭环配置：

- target：`single_face_wall_4course_v1`
- trials：Earth 2、Moon 2
- candidates：8
- pose top-k：3
- rocks：140
- profile：`high_wall`

闭环结果：

| gravity | trials | success | shape | mean stable | mean failure | RMSE m | height m | drift m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 2 | 0/2 = 0.0% | 0.0% | 12.50/24 | 3.50 | 0.139 | 0.291 | 0.157 |
| Moon | 2 | 0/2 = 0.0% | 1/2 = 50.0% | 15.00/24 | 1.50 | 2.094 | 0.372 | 0.029 |

解释：

- 离线 top-3 很高，但闭环没有严格成功，说明该模型很可能在当前数据划分下过拟合 run 内部分布。
- Moon 有一个 shape success trial，但另一个 trial 出现极大 RMSE，说明少量 outlier 会破坏长墙指标。
- 该结果支持用户的担心：只拟合候选日志是不够的，需要引入真正的当前墙体观测，例如正视/俯视深度图。

## 实验 4：双视角深度代理 ranker

该实验是对用户提出问题的直接响应：网络不应该只看石头几何或拟合候选日志，还应该结合当前石堆/墙体观测。

### 实验目的

验证一个更接近感知输入的设计是否可行：

- 顶视图：近似俯视深度/高度图，用于看支承 footprint、墙厚和候选 footprint。
- 正视图：近似墙面深度/轮廓图，用于看墙面高度、外凸、上层位置和目标槽位。
- 石头自身：仍使用几何特征和主面/支承面先验。
- 输出：候选释放位姿 ranking score。

这个实验仍不是最终 RGB-D 模型，因为 depth map 是从日志和几何重建的 proxy，不是真实相机原始图。

### 数据与模型

- 导出脚本：`scripts/export_depth_observation_maps.py`
- 张量目录：`batch_runs/20260620_depth_observation_maps_4course_curriculum_v2`
- 模型目录：`batch_runs/20260620_torch_depth_proxy_cnn_4course_curriculum_score_v1`

输入通道：

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

离线指标：

| model | rows | channels | top-1 | top-3 | top-1 regret | top-3 regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4course support-map CNN | 12,484 | 8 | 0.436 | 0.989 | 9.931 | 0.033 |
| 4course depth-proxy CNN | 12,484 | 13 | 0.420 | 0.985 | 10.571 | 0.066 |

解释：

- depth-proxy 输入可以训练，且已排除 post-simulation features。
- 但离线指标没有超过旧 support-map 模型。
- 因此“简单拼接代理深度通道”还不能证明更强，需要进一步使用真实渲染深度图/点云和跨 catalog 测试。

### 闭环烟测

路径：`batch_runs/20260620_highcourse_4course_depth_proxy_top3_seed97004_smoke_v1`

配置：

- target：`single_face_wall_4course_v1`
- ranker：`batch_runs/20260620_torch_depth_proxy_cnn_4course_curriculum_score_v1`
- trials：Earth 1、Moon 1
- pose top-k：3

结果：

| gravity | success | shape | stable | failure | height m | RMSE m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 0/1 | 0/1 | 16/24 | 2 | 0.415 | 0.105 |
| Moon | 0/1 | 0/1 | 14/24 | 4 | 0.193 | 0.137 |

失败模式：

- Moon middle `missed_target`：4 次。
- Earth middle `missed_target`：1 次。
- Earth cap `missed_target`：1 次。

图像：

- 截图目录：`batch_runs/20260620_highcourse_4course_depth_proxy_top3_seed97004_smoke_v1/captures_960x720`
- manifest：`batch_runs/20260620_highcourse_4course_depth_proxy_top3_seed97004_smoke_v1/captures_960x720/capture_manifest.csv`
- 包含 2 个失败案例。
- 包含 RGB、depth PNG 和 depth NPY。
- 关键视角包含 `wall_front_rgb/depth` 与 `wall_top_rgb/depth`。

### 实验结论

该实验说明双视角观测输入管线已经跑通，但当前 proxy 观测没有带来明确性能提升。下一步应从 proxy 升级到真实渲染深度图或点云：

1. 对每个候选放置前状态渲染 `wall_front_depth` 和 `wall_top_depth`。
2. 对待放石头导出 point cloud / mesh embedding。
3. 用 depth encoder + stone geometry encoder 融合。
4. 输出候选质量、失败概率和位姿修正量。

## 当前结论

1. 四层墙已经进入新的难度区间。
   - 24 槽长墙使 middle/cap 误差不断积累。
   - 低层 ranker 可以得到一个月球成功案例，但成功率不高。

2. 继续单纯增加候选或换旧 ranker 不够。
   - 失败不是只发生在候选姿态排序，而是出现在上层石头选择、支承重心和保持期漂移。

3. 下一步应该用四层墙本身的数据训练专门模型。
   - 将本轮 partial run 转成 learning dataset。
   - 导出 `single_face_wall_4course_v1` support-map tensors。
   - 训练 4course-specific pose ranker。
   - 只在训练完成后继续尝试更多 4 层和高墙 run。

4. 4 层墙的数据有科学价值。
   - 包含严格成功、shape 失败、middle/cap missed target、post-hold drift。
   - 可以作为从三层到高墙的课程学习数据。

## 设计反思

用户提出的关键问题是：如果网络只是根据石头外观或简单几何统计选择位姿，本质上可能只是拟合已有数据，科学意义有限。

这个判断是正确的。本轮 pose CNN 的科学意义不是“已经学会石墙堆叠”，而是：

- 证明候选姿态排序可以作为一个可控的小任务。
- 证明必须把当前石堆/墙体状态纳入输入，否则单看石头几何不够。
- 发现四层墙失败集中在 middle/cap，说明下一步应设计针对上层支承状态的视觉/深度输入。

因此下一步不应该继续只扩大同类 ranker，而应该升级为：

1. 输入当前墙体 `wall_front_depth` 和 `wall_top_depth`。
2. 输入待放石头点云或 mesh embedding。
3. 输入目标槽位/目标结构 mask。
4. 输出候选位姿质量、位姿修正或失败概率。

这才是更接近真实 dry stacking 的网络设计。

## 实验 5：MuJoCo 渲染深度 ranker 与在线 proxy 闭环

### 实验目的

用户指出只根据石头外观或几何统计选择位姿，本质上可能只是拟合已有候选日志，科学意义有限。本实验把输入推进到“候选石头几何 + 当前墙体观测”：

- 用 MuJoCo 渲染候选放置前的正视深度图和俯视深度图。
- 继续输入石头的几何先验，例如主面、支撑面、bbox、粗糙度、棱角度等。
- 网络仍输出候选 pose ranking score，用来替代一部分启发式候选搜索。

注意：训练数据是真 MuJoCo 渲染深度；当前在线闭环还没有逐候选实时渲染相机，因此运行时用几何 proxy 填充同名 render 通道。这一步是过渡实验，不是最终 RGB-D policy。

### 数据导出

脚本：

- `scripts/export_mujoco_depth_observation_maps.py`

导出设置：

```powershell
conda run -n moon-rock-stack python -m scripts.export_mujoco_depth_observation_maps `
  --dataset batch_runs\20260620_learning_dataset_4course_curriculum_v1 `
  --output batch_runs\20260620_mujoco_depth_observation_maps_4course_groups256_v1 `
  --source candidate `
  --grid-size 64 `
  --window-m 0.9 `
  --front-height-m 0.55 `
  --shard-size 384 `
  --target-contains single_face_wall_4course_v1 `
  --max-groups 256 `
  --sample-seed 629
```

导出结果：

| item | value |
| --- | ---: |
| candidate groups | 256 |
| rows | 809 |
| shards | 3 |
| skipped render rows | 0 |
| Earth rows | 518 |
| Moon rows | 291 |
| source runs | 6 |

输入通道：

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

### 模型训练

模型：

- `batch_runs/20260620_torch_mujoco_depth_cnn_4course_groups256_score_v1`

训练命令：

```powershell
conda run -n moon-rock-stack python -m scripts.train_torch_support_map_ranker `
  --tensor-dir batch_runs\20260620_mujoco_depth_observation_maps_4course_groups256_v1 `
  --output batch_runs\20260620_torch_mujoco_depth_cnn_4course_groups256_score_v1 `
  --epochs 60 `
  --batch-size 64 `
  --hidden 128 `
  --dropout 0.25 `
  --lr 0.001 `
  --weight-decay 0.0003 `
  --test-fraction 0.2 `
  --split-by-run `
  --seed 630 `
  --amp `
  --target-mode score `
  --quality-temperature 20 `
  --exclude-postsim-features
```

离线指标：

| metric | value |
| --- | ---: |
| rows | 809 |
| rankable groups | 256 |
| train groups | 249 |
| test groups | 7 |
| split | by run |
| test top-1 | 0.429 |
| test top-3 | 0.857 |
| train top-1 | 0.442 |
| train top-3 | 0.980 |
| test top-1 regret | 1.609 |
| test top-3 regret | 0.566 |

解释：

- 该模型在 held-out run 上 top-3 有一定效果，但测试组只有 7 个，不能夸大。
- top-3 明显优于 top-1，说明它适合缩小候选搜索，而不是单独输出最终 pose。
- 输入中已经剔除 post-simulation features，避免把仿真后验作为网络输入。

### 闭环实验

路径：

- `batch_runs/20260620_highcourse_4course_mujoco_depth_proxyonline_top3_seed97005_v1`

配置：

```powershell
conda run -n moon-rock-stack python -m moon_rock_stack.run_structured_experiment `
  --rocks 140 `
  --rock-profile high_wall `
  --clusters 10 `
  --trials 2 `
  --targets single_face_wall_4course_v1 `
  --strategies statics_wall `
  --gravities earth,moon `
  --candidates 8 `
  --steps-per-rock 420 `
  --hold-steps 1800 `
  --workers 2 `
  --seed 97005 `
  --candidate-pose-ranker-dir batch_runs\20260620_torch_mujoco_depth_cnn_4course_groups256_score_v1 `
  --candidate-pose-top-k 3 `
  --output batch_runs\20260620_highcourse_4course_mujoco_depth_proxyonline_top3_seed97005_v1
```

闭环结果：

| gravity | trials | success | shape | stable | failure | RMSE m | height m | drift m | visible courses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 2 | 0/2 | 0/2 | 11.0 | 5.0 | 0.151 | 0.374 | 0.232 | 4.0 |
| Moon | 2 | 0/2 | 0/2 | 13.0 | 3.5 | 0.113 | 0.314 | 0.256 | 4.0 |

单次典型结果：

| case | stable | failure | RMSE m | height m | x span m | y span m | aspect | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Moon trial 0 | 15/16 | 1 | 0.064 | 0.398 | 0.624 | 0.233 | 2.68 | 稳定数高，但 shape=0 |
| Earth trial 1 | 9/13 | 4 | 0.143 | 0.423 | 0.719 | 0.355 | 2.02 | 高度可以，但厚度和 outlier 不合格 |

失败统计：

- `missed_target`: 8
- `missed_target+post_hold_drift`: 7
- `unstable_structure`: 2

### 图像记录

截图目录：

- `batch_runs/20260620_highcourse_4course_mujoco_depth_proxyonline_top3_seed97005_v1/captures_960x720`

manifest：

- `batch_runs/20260620_highcourse_4course_mujoco_depth_proxyonline_top3_seed97005_v1/captures_960x720/capture_manifest.csv`

已保存：

- 4 个典型失败案例。
- 每个案例包含 front/right/back/left/top RGB 与 depth。
- 包含墙体专用 `wall_front_rgb.png`、`wall_front_depth.png`、`wall_top_rgb.png`、`wall_top_depth.png`。
- 深度图同时保存 PNG 和 NPY。

目检结论：

- 最好的 Moon trial 不是松散石堆，但也不是合格墙。
- 它更像窄柱或局部高塔，墙体横向覆盖不足。
- 俯视深度图显示石头集中在窄区域，而不是沿墙长方向均匀展开。

### 实验结论

这是一个负结果，但很有价值：

1. 真实渲染深度输入链路已经跑通。
2. 当前模型能减少候选搜索，但不能保证墙体结构。
3. 单一 `candidate_score` 标签会偏向局部稳定和高度，容易把墙堆成柱状。
4. 后续不能只刷 top-k 或继续堆高；应先把 4-6 层墙的结构成功率提高。
5. 下一轮标签和评分应加入 structure-aware 项：
   - 横向 span 增益。
   - 墙厚 y span 上限。
   - wall aspect。
   - course 覆盖。
   - 错缝/bonding。
   - outlier 惩罚。
   - post-hold drift 风险。

下一步决策：

- 继续使用小网络路线，但把 pose ranker 从“候选质量 ranker”升级为“结构感知候选 ranker”。
- 用 5-6 层最高单面墙的成功/失败数据收集更多结构标签。
- 在线输入继续保留 wall_front/wall_top depth，并尽快把在线 proxy 替换为真实在线渲染或点云。

## 实验 6：risk 与 structure-aware 标签对比

实验 5 说明单一 `candidate_score` 会偏向局部稳定和高度，但不能保证墙体结构。随后做了两个更保守的标签版本：

- `risk_adjusted`: 在 `candidate_score` 基础上惩罚 target error、扰动、速度、离目标半径和负高度增益。
- `structure_aware`: 在 `risk_adjusted` 基础上进一步加重 `target_y_error_m`、`target_x_error_m`、上层高度不足、cap/middle y 偏差。

### risk-adjusted 模型

模型：

- `batch_runs/20260620_torch_mujoco_depth_cnn_4course_groups256_risk_v1`

离线指标：

| metric | value |
| --- | ---: |
| test groups | 142 |
| test top-1 | 0.380 |
| test top-3 | 1.000 |
| train top-1 | 0.298 |
| train top-3 | 0.965 |

闭环：

- `batch_runs/20260620_highcourse_4course_mujoco_depth_risk_proxyonline_top3_seed97006_v1`

| gravity | trials | success | shape | stable | height m | drift m | y span m | aspect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 2 | 0/2 | 0/2 | 15.0 | 0.235 | 0.160 | 0.511 | 1.88 |
| Moon | 2 | 0/2 | 0/2 | 13.0 | 0.248 | 0.306 | 0.445 | 1.76 |

结论：

- Earth 稳定数从 score 版的 11.0 提高到 15.0。
- 但是高度降低，墙厚 `y span` 仍很大。
- Moon 没有改善，速度/漂移风险更高。
- 风险惩罚只能抑制一部分不稳定候选，不能解决墙体形状。

图像：

- `batch_runs/20260620_highcourse_4course_mujoco_depth_risk_proxyonline_top3_seed97006_v1/captures_960x720`
- 4 个失败案例，包含 `wall_front` 和 `wall_top` RGB/depth。

### structure-aware 模型

代码改动：

- `scripts/train_torch_support_map_ranker.py`
- 新增 `--target-mode structure_aware`
- 新增 `structure_aware_quality_targets`

模型：

- `batch_runs/20260620_torch_mujoco_depth_cnn_4course_groups256_structure_v1`

离线指标：

| metric | value |
| --- | ---: |
| test groups | 92 |
| test top-1 | 0.413 |
| test top-3 | 1.000 |
| train top-1 | 0.427 |
| train top-3 | 0.963 |

闭环：

- `batch_runs/20260620_highcourse_4course_mujoco_depth_structure_proxyonline_top3_seed97007_v1`

| gravity | trials | success | shape | stable | height m | drift m | y span m | aspect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 2 | 0/2 | 0/2 | 15.0 | 0.248 | 0.096 | 0.493 | 1.61 |
| Moon | 2 | 0/2 | 0/2 | 10.5 | 0.174 | 0.263 | 0.625 | 1.33 |

结论：

- Earth 漂移从 score 版 0.232、risk 版 0.160 进一步降到 0.096。
- 但 wall y span 仍约 0.49 m，远高于单面墙要求。
- Moon 稳定数和高度明显下降，说明当前 y-error 惩罚会牺牲低重力下的可放置性。
- top-3 离线指标仍然好，闭环结构仍失败，再次证明离线 top-k 不是最终目标。

图像：

- `batch_runs/20260620_highcourse_4course_mujoco_depth_structure_proxyonline_top3_seed97007_v1/captures_960x720`
- 4 个失败案例，包含 `wall_front` 和 `wall_top` RGB/depth。

### 三版对比

| model | gravity | success | shape | stable | height m | drift m | y span m | aspect |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| score | Earth | 0/2 | 0/2 | 11.0 | 0.374 | 0.232 | 0.428 | 1.87 |
| score | Moon | 0/2 | 0/2 | 13.0 | 0.314 | 0.256 | 0.378 | 2.03 |
| risk | Earth | 0/2 | 0/2 | 15.0 | 0.235 | 0.160 | 0.511 | 1.88 |
| risk | Moon | 0/2 | 0/2 | 13.0 | 0.248 | 0.306 | 0.445 | 1.76 |
| structure-aware | Earth | 0/2 | 0/2 | 15.0 | 0.248 | 0.096 | 0.493 | 1.61 |
| structure-aware | Moon | 0/2 | 0/2 | 10.5 | 0.174 | 0.263 | 0.625 | 1.33 |

最终判断：

- `risk_adjusted` 和 `structure_aware` 都不能单独把 4 层墙推到成功。
- 只改 candidate pose ranker 的标签不够，因为很多失败来自石头选择、slot assignment、course coverage 和全局墙厚控制。
- 下一阶段应从“单候选 pose 排序”升级到“局部墙状态 + 待放石头 + 目标结构”的多模块策略：
  - stone-slot selector: 先选适合该 slot/role/course 的石头。
  - pose ranker: 在少量候选 pose 中排序。
  - wall-state critic: 根据当前 wall_front/wall_top depth 预测放置后 wall span、y thickness、outlier 风险。
  - repair/reject head: 对高风险候选直接拒绝或要求换石头。

这也是当前实验的科学价值：不是证明网络已经会堆墙，而是定位了“仅替代 pose 搜索”这条路线的瓶颈。
