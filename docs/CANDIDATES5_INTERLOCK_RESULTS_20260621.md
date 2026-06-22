# 2026-06-21 candidates=5 与互锁标签实验记录

## 实验目的

上一轮结论是：`candidates=3` 时 top-3 基本等价于不裁剪，无法检验网络是否真的替代了候选搜索。因此本轮把 4 层月面单面墙提升到 `candidates=5`，并做三个问题的对照：

1. 增加候选数后，旧 support-map v4 的 top-3 是否还能稳定缩小搜索空间。
2. 不使用 support-map 裁剪、保留 PoseRisk 的候选全集是否更好。
3. 新增 `interlock_aware` 标签是否能让 support-map CNN 学到支撑互锁和后续可支撑性。

## 新增代码

修改文件：

`scripts/train_torch_support_map_ranker.py`

新增 target mode：

`interlock_aware`

该模式不改变网络架构和运行时接口，只改变训练标签。标签基于已有 candidate 后验结果和支撑几何构造，不作为运行时输入。新增惩罚项包括：

- support overlap 不足；
- support contact count 不足；
- support balance error 过大；
- middle/cap 横向墙线误差；
- height gain 不足；
- cap 和高 course 更高权重。

训练时仍使用：

`--exclude-postsim-features`

因此网络输入仍只有放置前可获得的信息：local support map、候选位姿、石头几何、目标 slot 和重力。

## 新增数据

v12 数据集：

`batch_runs/20260621_negative_mining_wall_dataset_v12_candidates5_interlock`

统计：

- run dirs: 25
- run examples: 28
- placement examples: 707
- candidate pose examples: 23940
- assignment candidate examples: 8328
- Moon placement: 683，其中 success 375、failure 241、skipped 67
- Earth placement: 24，其中 success 15、failure 9

按角色：

| role | examples | success | failure | skipped |
|---|---:|---:|---:|---:|
| base | 195 | 189 | 4 | 2 |
| middle | 375 | 146 | 189 | 40 |
| cap | 137 | 55 | 57 | 25 |

结论：瓶颈仍然集中在 middle 和 cap。base 几乎不是问题。

v12 support maps：

`batch_runs/20260621_negative_mining_support_maps_v12_candidates5_interlock`

## 闭环实验结果

所有实验均为：

- target: `single_face_wall_4course_v1`
- gravity: Moon
- strategy: `statics_wall`
- rocks: 150
- rock profile: `high_wall`
- seed: `206213401`
- stone-fit: `20260621_wall_flywheel_3course_focus_v2_stone_slot_net_20260621_014556`
- PoseRisk base: `20260621_pose_risk_net_candidate_metric_negative_mining_v5_after_mlpfix`

| 实验 | candidate pose ranker | top-k / gate | PoseRisk w | stable/failure | RMSE m | max error m | height m | drift m | velocity | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `20260621_4course_moon_supportv4_poseriskv5_w035_candidates5_top3_v1` | support-map v4 | top-3 all courses | 0.35 | 10/14 | 1.1058 | 5.3375 | 0.1820 | 1.9560 | 0.0169 | 增加候选后，全层 top-3 裁剪明显失败 |
| `20260621_4course_moon_fullposes_poseriskv5_w035_candidates5_v1` | none | all poses | 0.35 | 13/11 | 0.4505 | 2.0752 | 0.2209 | 0.5384 | 0.0158 | 候选全集比网络裁剪好，但仍不稳 |
| `20260621_4course_moon_interlockv12_top3_poseriskv5_w035_candidates5_v1` | interlock v12 | top-3 all courses | 0.35 | 10/14 | 0.7590 | 3.2071 | 0.1839 | 0.6347 | 0.0123 | 新互锁标签闭环未成功 |
| `20260621_4course_moon_supportv4_lowcourse_top3_capfull_poseriskv5_w035_candidates5_v1` | support-map v4 | course <= 2 top-3, cap full | 0.35 | 13/11 | 0.1680 | 0.4809 | 0.2989 | 0.2466 | 0.0184 | 课程门控显著改善墙线和高度 |
| `20260621_4course_moon_supportv4_lowcourse_top3_capfull_poseriskv5_w050_candidates5_v1` | support-map v4 | course <= 2 top-3, cap full | 0.50 | 13/11 | 0.2726 | 1.0863 | 0.2878 | 0.1176 | 0.0107 | 漂移和速度降低，但墙线误差变差 |

## interlock-aware 离线模型

模型目录：

`batch_runs/20260621_support_map_cnn_v12_interlock_holdout_candidates5`

训练设置：

- target mode: `interlock_aware`
- rows: 23940
- rankable groups: 8328
- test runs 包含两个 candidates=5 新 run 和两个旧强基线 run
- test top-1 hit: 0.473
- test top-3 hit: 0.936
- test mean top-1 regret: 4.099
- test mean top-3 regret: 0.392

结论：

- 这个标签方向有科学意义，但当前公式还不够好。
- top-3 没有达到 1.0，说明它会删掉一部分真实较优候选。
- 闭环结果验证了离线判断：不能作为默认策略。

## 科学结论

### 1. candidates=5 暴露了旧网络裁剪的问题

`candidates=5 + support-map v4 top-3 all courses` 比候选全集差很多：

- top-3 all courses: stable 10/24，RMSE 1.1058，drift 1.9560
- full poses: stable 13/24，RMSE 0.4505，drift 0.5384

这说明旧 support-map ranker 在高候选数和高层 cap 中会把关键候选裁掉。

### 2. 高层 cap 不应过早被网络裁剪

`course <= 2 top-3, cap full` 明显更像墙：

- RMSE 0.1680
- max error 0.4809
- height 0.2989

这支持“低层用网络，高层保留物理验证”的课程策略。当前网络还没有学会 cap 的互锁和后续稳定性。

### 3. PoseRisk 权重存在 tradeoff

在课程门控下：

- w0.35：墙线更好，RMSE 0.1680，但 drift 0.2466。
- w0.50：drift 降到 0.1176，velocity 降到 0.0107，但 RMSE 变差到 0.2726。

因此不能只靠调大风险权重解决问题。需要单独学习横向漂移和互锁风险。

### 4. interlock-aware 标签初版失败，但失败有价值

`interlock_aware` 直接把支撑 overlap/contact/balance 加入标签后，离线 top-3 降低，闭环也没有提升。这说明简单手工加权不够，需要更结构化的 future-support 标签，例如：

- 当前候选被放置后，下一层 middle/cap 是否还有足够可行候选；
- 后续槽位的 support map 是否变好；
- 该候选是否造成墙厚扩散或 outlier；
- 是否为下一层提供连续支撑面，而不是只在当前 slot 稳定。

## 图像记录

尝试对 `course <= 2 top-3, cap full, w0.35` 进行 RGB-D 捕获时，MuJoCo 渲染上下文失败：

- 默认 GLFW：`gladLoadGL error`
- `MUJOCO_GL=egl`：当前 Windows MuJoCo 安装不接受该值
- `MUJOCO_GL=osmesa`：当前 Windows MuJoCo 安装不接受该值

因此本轮图像暂未补拍。已保留完整 state 和日志，后续图形上下文恢复后可以从以下目录补拍：

`batch_runs/20260621_4course_moon_supportv4_lowcourse_top3_capfull_poseriskv5_w035_candidates5_v1`

已有 state：

`batch_runs/20260621_4course_moon_supportv4_lowcourse_top3_capfull_poseriskv5_w035_candidates5_v1/states/single_face_wall_4course_v1_statics_wall_moon_trial_00.npz`

## 下一步

1. 把课程门控作为 candidates=5 的当前最佳方向，而不是全层 top-k。
2. 训练显式 drift critic：输入 local support map + candidate pose + rock geometry，输出横向漂移/面外出墙风险。
3. 训练 future-support critic：标签不只看当前候选是否好，而看它是否让下一层候选池更好。
4. 继续把 candidates=5 的失败纳入下一版数据集，但暂不把 `interlock_aware` 作为默认 ranker。
