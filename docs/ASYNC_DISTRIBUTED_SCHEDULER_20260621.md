# 2026-06-21 异步分布式调度记录

## 目标

当前阶段不再盲目追求偶然的 10 层成功，而是先把 3-4 层墙的数据飞轮跑稳，再把经验迁移到 5 层墙。核心目标是：

- 提高 3 层墙和 4 层墙的候选石头/候选位姿选择效率；
- 收集 4 层和 5 层墙的成功、失败、跳槽、漂移、扰动数据；
- 训练多个小网络逐步替代启发式搜索，但保留 dry stacking 的几何/力学先验；
- 明确区分 curriculum 数据和 strict success 评估，commit-best 只用于采集失败/边界经验，不计入严格成功率。

## 资源分工

| 机器 | GPU | 角色 | 当前策略 |
|---|---:|---|---|
| 本机 | RTX 2080 Ti 11 GB | 主训练端 | 训练 PoseRiskNet、StoneSlotNet、姿态/墙状态小网络；运行主 4/5 层闭环评估 |
| 远端 `desktop-m57fdie` | GTX 1080 Ti 11 GB | 轻量辅助端 | 轻量 PoseRisk 训练、低成本 3/4 层采样、辅助数据收集 |

调度约束：

- 严禁删除已有实验结果；
- 所有新任务输出到新的时间戳目录；
- 本机 MuJoCo 并发保持较低，避免和已有 4 层任务互相拖慢；
- 本机 GPU 训练可以与 CPU 仿真并行；
- 远端只跑轻量任务，不再承担主训练。

## 本机已启动任务

调度器输出：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_scheduler\20260621_local_primary_4to5_scheduler_v1`

### 1. 本机 PoseRiskNet 主训练

- 目的：用本机 2080Ti 重新训练候选位姿风险网络，减少 4/5 层墙的无效候选位姿搜索。
- PID：`2368`，已完成。
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_local_primary_4to5_scheduler_v1_local_pose_risk_net`
- 训练数据：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_wall_flywheel_3course_pose_risk_augmented_learning_dataset`
- 结果摘要：
  - candidate pose rows：`26472`
  - train/test split：按 run 切分
  - test accuracy：`0.9271`
  - test F1：`0.9621`
  - rankable groups：`52`
  - top1 safe rate：`0.2115`
  - top3 safe rate：`0.6154`

解释：这个网络能有效识别高风险候选，但 Top-1 仍不够强，所以当前阶段仍应保留 Top-3 小候选搜索，而不是让网络完全单点决策。

### 2. 本机 4/5 层墙数据飞轮

- 目的：采集 4/5 层墙 curriculum 数据，重建学习数据集，训练小网络，并做 strict 闭环评估。
- 父进程 PID：`1172`
- 当前采样子进程 PID：`13468`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_local_primary_4to5_scheduler_v1_local_4to5_flywheel`
  - collect 输出：
    - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_local_primary_4to5_scheduler_v1_local_4to5_flywheel_collect_exploit_00_seed206212501`
- 目标：
  - `single_face_wall_4course_v1`
  - `single_face_wall_v1`，即 5 层单面墙
- 重力：
  - `moon`
- 采集模式：
  - `--collect-commit-best-rejected` 开启，只用于 curriculum 数据；
  - strict eval 不开启 commit-best。
- 网络输入：
  - StoneSlotNet：槽位几何角色、石头几何先验、类别特征；
  - 姿态网络/风险网络：候选位姿、目标槽位、石头几何、重力、类别；
  - 后验仿真统计只作为标签，不作为输入。

### 3. 旧 4 层 curriculum 任务

- PID：`8108`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_async_local_4course_moon_commitbest_n1`
- 目的：补一条 4 层 Moon commit-best curriculum 数据，主要用于分析失败槽位、可行候选缺口和石头类别选择问题。
- 当前结果：
  - strict success：`0/1`
  - visible courses：`4`
  - stable/failure：`14/10`
  - skipped slots：`0`
  - height：`0.267 m`
  - RMSE：`1.661 m`
  - max drift：`0.004 m`
  - failure reason：`missed_target=5`，`unstable_structure=5`
  - commit-best placements：`11/24`

解释：这条任务把 4 层问题从“跳过槽位/无候选”推进到了“候选落点偏离墙线/局部结构不稳”。因此下一轮网络训练要重点学习中层和 cap 的位姿风险，而不是只追求填满槽位。

### 4. 3/4 层 curriculum 学习数据集

- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_3to4course_curriculum_learning_dataset_v1`
- 来源 runs：
  - `20260621_wall_flywheel_3course_pose_risk_w035_eval`
  - `20260621_wall_flywheel_3course_pose_risk_w035_eval_n4_increment`
  - `20260621_gate_3course_moon_v1_n3`
  - `20260621_curriculum_4course_moon_from_3course_v1`
  - `20260621_curriculum_4course_moon_commitbest_v3`
  - `20260621_async_local_4course_moon_commitbest_n1`
- 样本量：
  - run examples：`19`
  - placement examples：`339`
  - candidate pose examples：`15255`
  - assignment candidate examples：`5085`
- 按角色统计：
  - base：`119` success / `1` failure / `0` skipped
  - middle：`69` success / `25` failure / `43` skipped
  - cap：`43` success / `13` failure / `26` skipped

解释：base 已经相对稳定，当前主要瓶颈在 middle 和 cap。后续模型/损失应该增加中高层候选位姿、支撑面质量、面外偏移惩罚的权重。

### 5. 新增本机增强训练任务

PoseRiskNet 3/4 curriculum：

- PID：`16192`
- job dir：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260621_150150_local_pose_risk_3to4_curriculum_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_pose_risk_net_3to4_curriculum_v1`
- 目的：把 4 层 missed target / unstable structure 的失败经验加入候选位姿风险模型。
- 结果：
  - test accuracy：`0.9340`
  - test F1：`0.9658`
  - top1 safe rate：`0.3571`
  - top3 safe rate：`1.0000`
- 使用判断：可用于 4 层 strict evaluation，但仍保留 Top-3 候选搜索，不做单点贪心。

StoneSlotNet 3/4 curriculum：

- PID：`19400`
- job dir：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260621_150150_local_stone_slot_3to4_curriculum_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_stone_slot_net_3to4_curriculum_v1`
- 目的：把 4 层中高层槽位的石头选择经验加入石头-槽位匹配模型。
- 结果：
  - test accuracy：`0.7803`
  - test F1：`0.1247`
  - top1 hit rate：`0.0833`
  - top3 hit rate：`0.2917`
- 使用判断：当前版本不能作为主石头选择器，只适合作为失败分析或弱辅助特征；下一轮 strict eval 暂时继续使用旧 StoneSlotNet。

### 6. 新增 4 层 strict 评估任务

- 任务：
  - `local_4course_strict_eval_oldstone_newposerisk_v1`
- PID：
  - `15108`
- job dir：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260621_150323_local_4course_strict_eval_oldstone_newposerisk_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_4course_moon_strict_eval_oldstone_newposerisk_v1`
- 配置：
  - target：`single_face_wall_4course_v1`
  - gravity：`moon`
  - trials：`2`
  - commit-best：关闭
  - stone selector：旧 StoneSlotNet
  - pose risk：`20260621_pose_risk_net_3to4_curriculum_v1`
- 目的：验证 4 层 strict success 是否因为新的 3/4 层 PoseRiskNet 而提升。

## 远端已启动任务

远端脚本：

- 本地备份：
  - `D:\MoonStack\deploy\desktop_m57fdie\start_light_remote_20260621.ps1`
- 远端位置：
  - `D:\ResearchManagerData\incoming\start_light_remote_20260621.ps1`

远端 job：

- job dir：
  - `D:\ResearchManagerData\runs\async_jobs\20260621_145153_remote_light_train_sample_v1`
- PID：
  - `14904`
- stdout：
  - `D:\ResearchManagerData\runs\async_jobs\20260621_145153_remote_light_train_sample_v1\stdout.log`
- stderr：
  - `D:\ResearchManagerData\runs\async_jobs\20260621_145153_remote_light_train_sample_v1\stderr.log`
- exit code：
  - `D:\ResearchManagerData\runs\async_jobs\20260621_145153_remote_light_train_sample_v1\exit_code.txt`

远端任务内容：

- 轻量 PoseRiskNet 训练，`12` epochs，hidden `96`；
- 低成本 3/4 层 Moon 采样；
- 输出：
  - `D:\ResearchManagerData\runs\20260621_145153_remote_light_pose_risk_v1`
  - `D:\ResearchManagerData\runs\20260621_145153_remote_light_sampling_4to5_v1`

## 当前实验顺序

1. 保留旧 4 层 curriculum 任务继续跑，补充 commit-best 边界数据。
2. 本机完成 PoseRiskNet 主训练，得到新的候选风险筛选器。
3. 本机 4/5 层 flywheel 先采集 curriculum 数据，再重建学习数据集，再训练 StoneSlot/姿态/墙状态小网络。
4. 本机 flywheel 最后做 strict evaluation，统计 4 层和 5 层是否真正稳定成墙。
5. 远端轻量任务补充低成本采样，用于后续合并数据集。

## 后续要统计的经验

从 `placement_log.csv`、`candidate_pose_log.csv`、`failure_cases.csv`、`results.csv` 中抽取：

- 哪些石头类别适合 base/middle/cap；
- 哪些几何特征能预测好支撑面：major face count、support face ratio、opposing face pair、flatness、angularity；
- 哪些槽位最容易出现 `no_feasible_pose`；
- commit-best 成功/失败的分布，用于判断是石头选择问题还是姿态选择问题；
- 4 层 strict 失败是否主要来自跳槽、面外偏移、残余速度，还是局部扰动；
- 5 层失败是否是底层承载不足、中层锁固不足，还是高层候选石头缺少适合的接触面。

## 继续实验记录：更好网络与更高堆叠

### Candidate Pose Groupwise Ranker

- 任务：
  - `local_candidate_pose_group_ranker_3to4_v1`
- PID：
  - `17264`
- job dir：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260621_150749_local_candidate_pose_group_ranker_3to4_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_candidate_pose_group_ranker_3to4_v1`
- 代码修正：
  - `scripts/train_candidate_pose_group_ranker.py` 现在会同时导出 `candidate_pose_rank_net.npz` 和 `candidate_pose_rank_net_schema.json`，可被闭环仿真直接加载。
- 目的：
  - 在同一 run/target/slot/stone 的候选姿态组内学习排序，补充 PoseRiskNet 的二分类风险判断。
  - 如果 Top-3 指标合格，下一轮 4/5 层会使用“groupwise pose ranker + PoseRiskNet”的组合。

### 8 层高单面墙 curriculum

- 任务：
  - `local_high_wall_8course_curriculum_oldstone_newrisk_v1`
- PID：
  - `14896`
- job dir：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260621_150906_local_high_wall_8course_curriculum_oldstone_newrisk_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_high_wall_8course_curriculum_oldstone_newrisk_v1`
- 配置：
  - target：`single_face_wall_high_v1`
  - gravity：`moon`
  - trials：`1`
  - commit-best：开启
  - stone selector：旧 StoneSlotNet
  - candidate pose ranker：旧 pose ranker
  - pose risk：`20260621_pose_risk_net_3to4_curriculum_v1`
- 目的：
  - 不把它作为 strict success 统计，而是收集 8 层墙的高层失败边界、槽位缺口、面外偏移和可堆高经验。

### Support-Map / Top-Structure CNN Ranker

support-map 数据：

- 任务：
  - `local_support_maps_3to4_curriculum_v1`
- PID：
  - `19712`，已完成
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_support_maps_3to4_curriculum_v1`
- 样本：
  - candidate rows：`15255`
  - shards：`8`
  - channels：`height_before_m`，`support_occupancy`，`support_count_clipped`，`target_gaussian`，`candidate_footprint`，`candidate_height_m`，`gravity_ratio`，`course_ratio`
- 设计目的：
  - 避免使用无效的侧视深度图；
  - 使用局部支撑高度图、候选 footprint、目标位置和层数/重力信息，近似 top-depth/结构观测；
  - 让网络输入包含石堆区域观测，而不是只看单块石头几何。

CNN 训练：

- 任务：
  - `local_support_map_cnn_3to4_structure_v1`
- PID：
  - `17912`
- job dir：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260621_151305_local_support_map_cnn_3to4_structure_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_support_map_cnn_3to4_structure_v1`
- 配置：
  - target mode：`structure_aware`
  - `--exclude-postsim-features`：开启
  - CUDA AMP：开启
- 当前状态：
  - 2080Ti 训练中，显存约 `2.5 GB`，GPU 利用率约 `30-40%`。
- 结果：
  - test top1：`0.4484`
  - test top3：`1.0000`
  - 4 层 Moon top1：`0.4472`
  - 4 层 Moon top3：`1.0000`
- 使用判断：
  - 可作为 Top-3 candidate-pose ranker 进入闭环；
  - 不能做 Top-1 单点贪心。

### 5 层 strict 对照：Support-Map CNN + PoseRiskNet

- 任务：
  - `local_5course_strict_eval_supportcnn_newrisk_v1`
- PID：
  - `10124`
- job dir：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260621_151831_local_5course_strict_eval_supportcnn_newrisk_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_5course_moon_strict_eval_supportcnn_newrisk_v1`
- 配置：
  - target：`single_face_wall_v1`
  - gravity：`moon`
  - trials：`1`
  - commit-best：关闭
  - stone selector：旧 StoneSlotNet
  - candidate pose ranker：`20260621_support_map_cnn_3to4_structure_v1`
  - pose risk：`20260621_pose_risk_net_3to4_curriculum_v1`
- 目的：
  - 用局部支撑观测网络替换旧姿态 ranker，验证 5 层 strict 成墙是否改善。

### Support-Map CNN Selected-Mode 对照

- 任务：
  - `local_support_map_cnn_3to4_selected_v1`
- PID：
  - `8832`
- job dir：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260621_151831_local_support_map_cnn_3to4_selected_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_support_map_cnn_3to4_selected_v1`
- 目的：
  - 和 `structure_aware` 版本对比，判断“模仿当前搜索选择”是否比“结构质量监督”更适合闭环。

## 查看命令

本机异步任务：

```powershell
conda run -n moon-rock-stack python scripts\async_process.py list --job-root batch_runs\async_jobs
```

本机调度 manifest：

```powershell
Get-Content D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_scheduler\20260621_local_primary_4to5_scheduler_v1\manifest.json
```

远端轻量任务 stdout：

```powershell
ssh all@desktop-m57fdie.tail83f520.ts.net "cmd /c type D:\ResearchManagerData\runs\async_jobs\20260621_145153_remote_light_train_sample_v1\stdout.log"
```

## 继续实验记录：PointNet / Support-Map 对照

### Support-Map CNN selected-mode 结果

- 任务：
  - `local_support_map_cnn_3to4_selected_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_support_map_cnn_3to4_selected_v1`
- 结果：
  - test top1：`0.4463`
  - test top3：`1.0000`
  - 4 层 Moon holdout top1：`0.4463`
  - 4 层 Moon holdout top3：`1.0000`
- 判断：
  - selected-mode 是模仿旧启发式搜索的行为克隆。
  - structure-aware 版本 test top1 为 `0.4484`，top3 同为 `1.0000`，且监督目标包含目标误差、面外偏移和漂移惩罚，因此闭环仍优先使用 `structure_aware` 版本。
  - 当前网络 Top-1 不足以单点贪心；Top-3 稳定，所以闭环应继续采用“网络缩小候选集 + 物理/风险先验做最后排序”的策略。

### PointNet + Support-Map 消融任务

- 新增脚本：
  - `D:\MoonStack\experiments\moon_rock_stack\scripts\run_pointnet_supportmap_ablation.py`
- 任务：
  - `local_pointnet_supportmap_ablation_3to4_v1`
- PID：
  - `11556`
- job dir：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260621_152432_local_pointnet_supportmap_ablation_3to4_v1`
- 计划输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_pointnet_supportmap_3to4_ablation_v1`
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_pointnet_supportmap_3to4_ablation_v1_pointclouds`
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_pointnet_supportmap_3to4_ablation_v1_pointnet_encoder`
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_pointnet_supportmap_3to4_ablation_v1_support_map_pointnet_structure_aware`
- 数据：
  - dataset：`20260621_3to4course_curriculum_learning_dataset_v1`
  - support maps：`20260621_support_maps_3to4_curriculum_v1`
  - 自动读取 dataset 中的 6 个 run 目录，导出石头点云。
- 配置：
  - point samples：`768`
  - PointNet embedding dim：`128`
  - PointNet epochs：`70`
  - hybrid ranker epochs：`80`
  - target mode：`structure_aware`
  - post-simulation 输入泄漏：关闭，`--exclude-postsim-features`
- 目的：
  - 验证“点云形状 embedding + 局部支撑图”是否比纯手工几何特征 + 支撑图更好。
  - 这是消融，不是默认闭环策略。只有当同数据集 holdout top-k 指标和后续 MuJoCo strict-wall 成功率都提升时，才把它接入闭环。

结果：
- pointcloud：
  - run count：`6`
  - rock count：`1020`
  - points per rock：`768`
- PointNet：
  - input dim：`6`，包含 xyz + normal
  - embedding dim：`128`
  - test source-kind accuracy：`0.500`
  - test cluster-family accuracy：`1.000`，但 cluster-family 当前只有 `multi_facet_clast`，不能说明形状判别能力强。
- Hybrid Support-Map + PointNet：
  - 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_pointnet_supportmap_3to4_ablation_v1_support_map_pointnet_structure_aware`
  - test top1：`0.4256`
  - test top3：`1.0000`
  - 4 层 Moon holdout top1：`0.4315`
  - 4 层 Moon holdout top3：`1.0000`
- 对照结论：
  - 纯 Support-Map structure-aware CNN 的 test top1 为 `0.4484`，4 层 Moon top1 为 `0.4472`。
  - 当前 PointNet embedding 拼接后 top1 下降，因此不接入闭环主策略。
  - 后续如果继续做点云网络，应改成 affordance-specific 监督，例如预测 base/middle/cap 适配性、接触面质量、面外偏移风险，而不是先分类 source-kind 再直接拼 embedding。

### 新增策略：`statics_wall_line_lock`

- 修改文件：
  - `D:\MoonStack\experiments\moon_rock_stack\moon_rock_stack\structured.py`
  - `D:\MoonStack\experiments\moon_rock_stack\moon_rock_stack\run_structured_experiment.py`
- 编译检查：
  - `conda run -n moon-rock-stack python -m compileall moon_rock_stack\structured.py moon_rock_stack\run_structured_experiment.py`
  - 结果：通过。
- 设计目的：
  - 上一轮 4 层墙失败集中在 `missed_target`、`unstable_structure` 和墙线 outlier。
  - 新策略不改变石头生成，不改变 MuJoCo 物理，只改变在线候选位姿选择的约束。
  - 更小的 y jitter、更少 y 方向跟随支撑中心、更强 target/y-error/balance hard penalty。
- smoke run：
  - 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_line_lock_smoke_v1`
  - 作用：只验证新策略入口和 MuJoCo 路径不崩溃；由于步数极短，结果不作为科学统计。
- 正式对照任务：
  - 任务：`local_4course_line_lock_supportcnn_newrisk_v1`
  - PID：`21320`
  - job dir：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260621_153123_local_4course_line_lock_supportcnn_newrisk_v1`
  - 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_4course_moon_line_lock_supportcnn_newrisk_v1`
  - target：`single_face_wall_4course_v1`
  - strategy：`statics_wall_line_lock`
  - gravity：`moon`
  - trials：`1`
  - candidate pose ranker：`20260621_support_map_cnn_3to4_structure_v1`
  - pose risk：`20260621_pose_risk_net_3to4_curriculum_v1`
  - 目的：和正在运行的 `statics_wall` 4 层/5 层任务对照，判断更强墙线约束是否降低 missed-target/outlier。

### Wall-State Critic 3/4 Curriculum

- 任务：`local_wall_state_critic_3to4_structure_v1`
- PID：`19488`
- job dir：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260621_153258_local_wall_state_critic_3to4_structure_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_wall_state_critic_3to4_structure_v1`
- 输入：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_support_maps_3to4_curriculum_v1`
- 配置：
  - epochs：`90`
  - batch size：`160`
  - hidden：`192`
  - AMP：开启
  - post-simulation 输入泄漏：关闭，`--exclude-postsim-features`
- 目的：
  - 从支撑图和候选几何预测 target error、y error、扰动、速度和失败概率。
  - 暂时不直接替代闭环策略，先作为失败解释器；后续如果指标可靠，再接入候选过滤或 PoseRiskNet 的蒸馏监督。
- 结果：
  - 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_wall_state_critic_3to4_structure_v1`
  - test group top1：`0.3838`
  - test group top3：`1.0000`
  - target x error MAE：`0.1732` m
  - target y error MAE：`0.0833` m
  - disturbance MAE：`0.1939` m
- 判断：
  - Top-3 可用，但 top1 低于 support-map ranker 的 `0.4484`。
  - 暂时不作为主排序器；更适合做失败解释、风险辅助特征、或后续蒸馏到 PoseRiskNet。

### 当前 3/4 层经验统计

- 新增脚本：
  - `D:\MoonStack\experiments\moon_rock_stack\scripts\summarize_learning_dataset_experience.py`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_3to4course_curriculum_experience_summary_v1`
- 输入：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_3to4course_curriculum_learning_dataset_v1`
- 样本：
  - placement rows：`339`
- role 统计：
  - `base`：样本 `120`，成功 `119`，失败 `1`，跳过 `0`，committed 成功率 `0.992`
  - `middle`：样本 `137`，成功 `69`，失败 `25`，跳过 `43`，committed 成功率 `0.734`，跳过率 `0.314`
  - `cap`：样本 `82`，成功 `43`，失败 `13`，跳过 `26`，committed 成功率 `0.768`，跳过率 `0.317`
- 当前经验：
  - base 不是主要瓶颈；中高层的跳过率和失败率才是 4/5 层提升的关键。
  - base：`interlock_block_clast`、`tie_bridge_clast`、`bearing_block_clast`、`buttress_clast` 当前 committed 成功率都为 `1.000`。
  - cap：`tie_bridge_clast` committed 成功率 `0.909`，`bearing_block_clast` 为 `0.900`，比 `cap_block_clast` 和 `interlock_block_clast` 更稳。
  - middle：`course_block_clast` 样本少但 committed 成功率 `1.000`；`interlock_block_clast` committed 成功率 `0.875` 但跳过率 `0.600`，说明它一旦有可行位姿很好用，但可行位姿不容易找到。
- 几何信号：
  - cap 成功样本的 opposing face pair count、major face count、elongation 更高，flatness 和 support plane quality 反而略低。
  - middle 成功样本的 flatness、roughness、bbox_y 更低，elongation 和 major face count 略高。
- 对后续策略的直接影响：
  - StoneSlotNet 的中高层标签应提高 role-specific 权重，不能让 base 的高成功率主导训练。
  - 中高层候选位姿网络应更重视墙线 y error、support balance 和可行位姿存在性。
  - 点云网络下一步应做 role/affordance head：预测某块石头适合 base/middle/cap 的概率，而不是只学 source-kind。

### Role-Balanced StoneSlotNet 对照

- 修改文件：
  - `D:\MoonStack\experiments\moon_rock_stack\scripts\train_torch_stone_slot_net.py`
- 新增参数：
  - `--role-balance`：按 role 频次做样本权重均衡。
  - `--role-weight role=value`：对指定 role 做额外权重，例如 `middle=1.45`、`cap=1.45`。
- 默认行为：
  - 不传上述参数时，旧训练行为不变。
- 任务：
  - `local_stone_slot_3to4_rolebalanced_v1`
- PID：
  - `18928`
- job dir：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260621_153828_local_stone_slot_3to4_rolebalanced_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_stone_slot_net_3to4_rolebalanced_v1`
- 配置：
  - dataset：`20260621_3to4course_curriculum_learning_dataset_v1`
  - hidden：`160`
  - epochs：`180`
  - role balance：开启
  - role weights：`middle=1.45`，`cap=1.45`
- 目的：
  - 旧 3/4 StoneSlotNet 的 top1/top3 很弱，可能被 base 的高成功率和中高层稀疏正样本影响。
  - 新对照将 middle/cap 的 assignment 候选显式加权；只有 group top-k 明显提升后才接入闭环。
- 结果：
  - 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_stone_slot_net_3to4_rolebalanced_v1`
  - test accuracy：`0.7917`
  - test F1：`0.1414`
  - test group top1：`0.1000`
  - test group top3：`0.3000`
- 对照：
  - 旧 3/4 StoneSlotNet top1：`0.0833`
  - 旧 3/4 StoneSlotNet top3：`0.2917`
- 判断：
  - role-balance 只有轻微提升，仍不能作为主石头选择器。
  - 主要问题不是简单类别不平衡，而是 assignment 标签太稀疏、候选可行性受当前墙状态强影响；后续 StoneSlot 应加入支撑区域/slot 上下文，或者训练 role-affordance head 而不是只看单块石头和 slot。

### 4 层严格评估的第一条经验

- 任务：
  - `local_4course_strict_eval_oldstone_newposerisk_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_4course_moon_strict_eval_oldstone_newposerisk_v1`
- 当前已落盘 trial：
  - trial `0`
- 配置：
  - target：`single_face_wall_4course_v1`
  - strategy：`statics_wall`
  - gravity：`moon`
  - candidate pose ranker：旧 3 层结构 ranker
  - PoseRiskNet：`20260621_pose_risk_net_3to4_curriculum_v1`
  - commit-best：关闭
- 结果：
  - success：`0`
  - shape_success：`0`
  - visible_courses：`4`
  - stable_count / failure_count：`9 / 3`
  - skipped_slot_count：`12`
  - target_rmse_xy_m：`0.1744`
  - target_max_xy_error_m：`0.4765`
  - max_horizontal_drift_m：`0.4243`
  - wall_y_span_m：`0.4390`
  - wall_outlier_count：`3`
- 失败集中位置：
  - course `2` middle：`missed_target+post_hold_drift`
  - course `2` middle：`unstable_structure`
  - course `3` cap：`missed_target+post_hold_drift`
- 初步判断：
  - 这不是随机石堆，墙体已出现 4 个可见层，但墙线控制仍不够，y 向漂移和保持阶段漂移是主因。
  - 新 PoseRiskNet 对高层更保守，导致跳过 `12` 个槽位；这降低了盲目放置，但也使墙体高度和完整度不足。
  - 底层仍不是主要瓶颈；中层/顶层的候选位姿排序、可行位姿存在性和保持后稳定性是当前重点。

### Role-Weighted Support-Map CNN Ranker

- 修改文件：
  - `D:\MoonStack\experiments\moon_rock_stack\scripts\train_torch_support_map_ranker.py`
- 新增参数：
  - `--group-role-weight role=value`
  - `--group-course-weight course=value`
- 默认行为：
  - 不传新参数时，原始训练行为不变。
- 设计目的：
  - 当前 support-map CNN 的整体 top1 最高，但 4 层失败集中在 middle/cap。
  - 新训练不改变输入和运行时接口，只在 groupwise loss 中提高 middle、cap 和高 course 槽位权重，让网络更关注真正决定墙体高度和墙线的候选位姿。
  - 输入仍是可观测几何先验和局部支撑/目标图，不使用 post-simulation 特征作为输入。
- 任务：
  - `local_support_map_cnn_3to4_roleweighted_structure_v1`
- PID：
  - `21408`
- job dir：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260621_154535_local_support_map_cnn_3to4_roleweighted_structure_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_support_map_cnn_3to4_roleweighted_structure_v1`
- 配置：
  - target mode：`structure_aware`
  - post-simulation 输入泄漏：关闭，`--exclude-postsim-features`
  - role weights：`middle=1.25`，`cap=1.50`
  - course weights：`2=1.15`，`3=1.25`
- 通过标准：
  - 必须对照 `20260621_support_map_cnn_3to4_structure_v1`。
  - 如果整体 top1/top3 持平但 4 层 Moon 组 top1 或高层组 regret 下降，可以进入闭环 4 层/5 层评估。
  - 如果整体 top1 明显下降且没有高层收益，则保留为负结果，不接入闭环。
- 首轮结果：
  - 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_support_map_cnn_3to4_roleweighted_structure_v1`
  - test top1：`0.3942`
  - test top3：`1.0000`
  - 备注：holdout run 与原始模型不同，不能直接作为闭环依据；整体指标偏低。
- 同 holdout 对照：
  - 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_support_map_cnn_3to4_roleweighted_structure_sameholdout_v1`
  - test runs：`20260621_curriculum_4course_moon_from_3course_v1`，`20260621_gate_3course_moon_v1_n3`
  - overall test top1：`0.4450`
  - overall test top3：`1.0000`
  - 4 层 Moon top1：`0.4519`
  - 4 层 Moon top1 quality regret：`6.2981`
- 对照判断：
  - 原始 support-map CNN overall top1：`0.4484`
  - 原始 support-map CNN 4 层 Moon top1：`0.4472`
  - 原始 support-map CNN 4 层 Moon top1 quality regret：`6.6496`
  - 角色加权模型整体略低，但 4 层 Moon 子集略好且 regret 更低。
  - 暂不替代主模型；后续可作为 4 层专用闭环对照，尤其配合 `statics_wall_line_lock` 验证墙线漂移是否下降。

### PointNet Role-Affordance 3/4 Curriculum

- 新增脚本：
  - `D:\MoonStack\experiments\moon_rock_stack\scripts\train_torch_pointnet_role_affordance.py`
- 任务：
  - `local_pointnet_role_affordance_3to4_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_pointnet_role_affordance_3to4_v1`
- 输入：
  - 点云：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_pointnet_supportmap_3to4_ablation_v1_pointclouds`
  - 监督：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_3to4course_curriculum_learning_dataset_v1\placement_examples.csv`
- 设计目的：
  - 只用石头点云几何，预测该石头适合 `base`、`middle`、`cap` 的概率。
  - 这是离线石头选择候选，不观察当前墙体状态，因此不能单独替代闭环策略。
  - 不把历史成功率作为输入；历史 placement outcome 只作为训练标签。
- 结果：
  - rock_count：`1020`
  - labeled_rock_count：`166`
  - test observed accuracy：`0.7368`
  - test observed F1：`0.8485`
  - test observed MAE：`0.3213`
  - base：count `18`，positive_rate `1.000`，F1 `1.000`，MAE `0.0211`
  - middle：count `23`，positive_rate `0.609`，F1 `0.7568`，MAE `0.4714`
  - cap：count `16`，positive_rate `0.625`，F1 `0.7692`，MAE `0.4434`
- 判断：
  - base 指标虚高，因为 base 样本几乎全是成功，不能说明网络真的理解底层几何。
  - middle/cap 的 recall 高，但 MAE 很大，说明它倾向于把很多中高层石头都判为可用，区分失败样本能力弱。
  - 该网络暂时不接入闭环；下一版石头选择器必须加入 slot/wall local context，例如 support map、目标槽位高度、邻接支撑区域和当前墙线误差。

### 4/5 层 Flywheel Collect 结果和典型失败截图

- 任务：
  - `local_4to5_wall_flywheel`
- collect 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_local_primary_4to5_scheduler_v1_local_4to5_flywheel_collect_exploit_00_seed206212501`
- 配置：
  - gravity：`moon`
  - strategy：`statics_wall`
  - commit-best：开启
  - candidate pose ranker：旧 3 层结构 ranker
  - PoseRiskNet：旧 3 层 PoseRiskNet
  - targets：`single_face_wall_4course_v1`，`single_face_wall_v1`
- 4 层结果：
  - success / shape_success：`0 / 0`
  - visible_courses：`4`
  - stable_count / failure_count：`16 / 8`
  - skipped_slot_count：`0`
  - stack_height_m：`0.2878`
  - target_rmse_xy_m：`0.3300`
  - target_max_xy_error_m：`1.3343`
  - max_horizontal_drift_m：`0.1575`
  - wall_y_span_m：`0.5498`
  - wall_outlier_count：`6`
- 5 层结果：
  - success / shape_success：`0 / 0`
  - visible_courses：`5`
  - stable_count / failure_count：`23 / 8`
  - skipped_slot_count：`0`
  - stack_height_m：`0.3287`
  - target_rmse_xy_m：`0.3423`
  - target_max_xy_error_m：`1.3837`
  - max_horizontal_drift_m：`0.1289`
  - wall_y_span_m：`0.9472`
  - wall_outlier_count：`4`
- 分层误差：
  - 4 层 course 0 base：mean error `0.0079` m
  - 4 层 course 1 middle：mean error `0.0187` m
  - 4 层 course 2 middle：mean error `0.3648` m，mean y error `0.0982` m
  - 4 层 course 3 cap：mean error `0.0824` m
  - 5 层 course 2 middle：mean error `0.2432` m，mean disturbance `0.2001` m
  - 5 层 course 4 cap：mean error `0.1455` m，mean disturbance `0.1045` m
- 典型失败截图：
  - 输出目录：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_local_primary_4to5_scheduler_v1_local_4to5_flywheel_collect_exploit_00_seed206212501\cap4to5`
  - manifest：`capture_manifest.csv`
  - 4 层案例：`00_single_face_wall_4course_v1_failure_statics_wall_moon_trial_00`
  - 5 层案例：`01_single_face_wall_v1_failure_statics_wall_moon_trial_00`
  - 包含：RGB、depth、object_depth，多视角；其中 `front_*` 和 `top_*` 别名可直接用于汇报。
- 深度图抽查：
  - 4 层 top_object_depth 有效像素：`126532`，std：`0.0727`
  - 5 层 top_object_depth 有效像素：`163391`，std：`0.1091`
  - 结论：顶视 object depth 不是全黄空图，能体现墙线漂移和离群石块。
- 判断：
  - 这批结果证明当前策略可以形成 4/5 个可见层，但不是稳定墙体。
  - 主要失败不是底层承载，而是中层横向放置误差、y 向墙线漂移、以及高层扰动。
  - 下一步闭环优先级：`statics_wall_line_lock` + 4 层专用 role-weighted support-map ranker；如果降低 y span/outlier，再扩展到 5 层。

### Flywheel Learning Dataset 更新

- 数据集输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_local_primary_4to5_scheduler_v1_local_4to5_flywheel_learning_dataset`
- 生成时间：
  - `2026-06-21T16:02:27`
- 规模：
  - run_example_count：`34`
  - placement_example_count：`823`
  - candidate_pose_example_count：`20274`
  - assignment_candidate_example_count：`6777`
- 重力分布：
  - earth：examples `456`，success `249`，failure `69`，skipped `138`
  - moon：examples `367`，success `182`，failure `67`，skipped `118`
- role 分布：
  - base：examples `238`，success `168`，failure `12`，skipped `58`
  - middle：examples `415`，success `207`，failure `78`，skipped `130`
  - cap：examples `170`，success `56`，failure `46`，skipped `68`
- 数据意义：
  - 相比早前 3/4 curriculum，加入了 4/5 层高墙失败样本和更多地球/月球对照。
  - cap 的 failure+skipped 占比高，是后续 PoseRisk/support-map/stone-slot 改进重点。
  - 正在等待 `20260621_local_primary_4to5_scheduler_v1_local_4to5_flywheel_mujoco_depth_maps` 导出完成；完成后再进入训练阶段。

### PoseRiskNet 4/5 Flywheel Dataset

- 任务：
  - `local_pose_risk_4to5_flywheel_dataset_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_pose_risk_net_4to5_flywheel_dataset_v1`
- 输入：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_local_primary_4to5_scheduler_v1_local_4to5_flywheel_learning_dataset`
- 配置：
  - epochs：`150`
  - hidden：`224`
  - target error limit：`0.16` m
  - target y error limit：`0.075` m
  - disturbance limit：`0.08` m
  - velocity limit：`0.22`
- 数据：
  - candidate rows：`20274`
  - train rows：`18687`
  - test rows：`1587`
- 指标：
  - test accuracy：`0.9357`
  - test F1：`0.9668`
  - test group top1 safe：`0.3929`
  - test group top3 safe：`1.0000`
  - 4 层 Earth top1 safe：`0.4615`
  - 4 层 Moon top1 safe：`0.3333`
- 判断：
  - 和之前 PoseRiskNet 一样，top3 可靠、top1 不可靠。
  - 后续闭环中只作为候选风险惩罚或 Top-3 rerank 的辅助项，不单独贪心选 top1。
  - 月球 4 层仍是困难分组，下一轮需要和 `statics_wall_line_lock`、support-map ranker 一起验证。

### 8-Course High Wall 失败案例

- 任务：
  - `local_high_wall_8course_curriculum_oldstone_newrisk_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_high_wall_8course_curriculum_oldstone_newrisk_v1`
- 配置：
  - target：`single_face_wall_high_v1`
  - strategy：`statics_wall`
  - gravity：`moon`
  - commit-best：开启
  - candidate pose ranker：旧 3 层结构 ranker
  - PoseRiskNet：`20260621_pose_risk_net_3to4_curriculum_v1`
- 结果：
  - success / shape_success：`0 / 0`
  - visible_courses：`8`
  - stable_count / failure_count：`21 / 10`
  - stack_height_m：`0.2313`
  - target_rmse_xy_m：`0.3050`
  - target_max_xy_error_m：`1.2882`
  - wall_y_span_m：`1.7947`
  - wall_outlier_count：`8`
  - structure_score：`-1.2905`
- 失败原因：
  - course 2-6 多数为 `missed_target`
  - course 7 cap 出现 `missed_target+post_hold_drift`
  - 说明可以产生 8 个可见层，但不是有效高墙；横向/纵向墙线已严重散开。
- 截图：
  - 输出目录：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_high_wall_8course_curriculum_oldstone_newrisk_v1\cap8h`
  - 案例：`00_single_face_wall_high_v1_failure_statics_wall_moon_trial_00`
  - top_object_depth 有效像素：`201490`，std：`0.0759`
  - front_object_depth 有效像素：`209652`，std：`1.1020`
- 判断：
  - 暂时不继续追 8-10 层高度；该结果是失败数据，不是阶段性成功。
  - 下一轮应回到 4/5 层，用 line-lock 和新 support-map/PoseRisk 数据先降低 y span、outlier 和 course-2 missed_target。

### Support-Map 4/5 Flywheel Tensor Export

- 任务：
  - `local_support_maps_4to5_flywheel_dataset_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_support_maps_4to5_flywheel_dataset_v1`
- 输入：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_local_primary_4to5_scheduler_v1_local_4to5_flywheel_learning_dataset`
- 规模：
  - row_count：`19467`
  - shard_count：`10`
  - grid_size：`64`
  - channels：`height_before_m`、`support_occupancy`、`support_count_clipped`、`target_gaussian`、`candidate_footprint`、`candidate_height_m`、`gravity_ratio`、`course_ratio`
- 后续训练：
  - 任务：`local_support_map_cnn_4to5_flywheel_structure_v1`
  - PID：`21172`
  - 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_support_map_cnn_4to5_flywheel_structure_v1`
  - 目的：训练 4/5 数据集版 support-map CNN，和 `20260621_support_map_cnn_3to4_structure_v1` 及 role-weighted 4 层模型对照。
- 基线训练结果：
  - overall test top1：`0.4336`
  - overall test top3：`1.0000`
  - 4 层 Earth top1：`0.4219`
  - 4 层 Moon top1：`0.4453`
  - 4 层 Moon top1 quality regret：`12.2924`
- 判断：
  - 该模型的 top3 仍可靠，但 top1 不优于 3/4 support-map CNN，也不优于 4 层 role-weighted 模型。
  - 暂不作为默认闭环 ranker。
- 泛化对照：
  - 任务：`local_support_map_cnn_4to5_holdout_newcollect_v1`
  - PID：`19400`
  - 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_support_map_cnn_4to5_holdout_newcollect_v1`
  - test run：`20260621_local_primary_4to5_scheduler_v1_local_4to5_flywheel_collect_exploit_00_seed206212501`
  - 目的：把今天新增 4/5 collect 整个 hold out，检查旧数据训练的 support-map CNN 是否能泛化到今天的 4/5 失败分布。

### Line-Lock 4 层闭环负结果

- 任务：
  - `local_4course_line_lock_supportcnn_newrisk_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_4course_moon_line_lock_supportcnn_newrisk_v1`
- 配置：
  - target：`single_face_wall_4course_v1`
  - strategy：`statics_wall_line_lock`
  - gravity：`moon`
  - candidate pose ranker：`20260621_support_map_cnn_3to4_structure_v1`
  - PoseRiskNet：`20260621_pose_risk_net_3to4_curriculum_v1`
- 结果：
  - success / shape_success：`0 / 0`
  - visible_courses：`4`
  - stable_count / failure_count：`10 / 6`
  - skipped_slot_count：`8`
  - target_rmse_xy_m：`1.4439`
  - target_max_xy_error_m：`5.7487`
  - stack_height_m：`0.1357`
  - max_horizontal_drift_m：`0.2902`
  - wall_y_span_m：`5.6163`
  - wall_outlier_count：`6`
- 失败机制：
  - failure_cases 中 slot `1` base 出现最终 y 方向 `5.3089` m 的严重漂移。
  - placement_log 显示初始落点并非全部离谱，说明问题是后续/最终保持阶段传播失稳，而不是单纯瞬时放置偏差。
  - middle course 2 mean y error 达到 `0.2780` m。
- 截图：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_4course_moon_line_lock_supportcnn_newrisk_v1\capll`
- 判断：
  - 当前 `statics_wall_line_lock` 不能作为 5 层推进策略。
  - 下一步若继续 line-lock，必须先修复底层传播失稳：base 层需要更强的 post-settle/hold 稳定筛选，不能只看候选落点 y error。
  - 在修复前，闭环主线仍用 `statics_wall` + support-map/PoseRisk top3 组合，而不是 line-lock。

### 4 层 Strict Eval：新 PoseRisk 过保守

- 任务：
  - `local_4course_strict_eval_oldstone_newposerisk_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_4course_moon_strict_eval_oldstone_newposerisk_v1`
- 配置：
  - target：`single_face_wall_4course_v1`
  - strategy：`statics_wall`
  - gravity：`moon`
  - trials：`2`
  - commit-best：关闭
  - candidate pose ranker：旧 3 层结构 ranker
  - PoseRiskNet：`20260621_pose_risk_net_3to4_curriculum_v1`
- 平均结果：
  - success / shape_success：`0 / 0`
  - visible_courses：`4.0`
  - stable_count：`8.5`
  - failure_count：`3.5`
  - skipped_slot_count：`12.0`
  - target_rmse_xy_m：`0.1443`
  - target_max_xy_error_m：`0.3561`
  - structure_score：`0.4705`
  - stack_height_m：`0.1308`
  - max_horizontal_drift_m：`0.3269`
  - wall_y_span_m：`0.4099`
  - wall_outlier_count：`3.5`
- 判断：
  - 相比 commit-best 硬放满策略，strict + PoseRisk 的墙线误差更小，但跳过太多槽位，墙体高度和完整性不足。
  - 主要失败仍集中在 course 1/2 middle 和 cap 的 post_hold_drift/missed_target。
  - 下一轮不应进一步增大 PoseRisk 权重；更合理的是 Top-3 保留候选，再由支撑图/结构目标选择，同时对 skipped slot 做更宽松但有 hold-stability 的 fallback。

### 下一轮 4 层闭环对照：Commit-Best + Role-Weighted Ranker + 4/5 PoseRisk

- 任务：
  - `local_4course_commitbest_roleweighted_newrisk_v1`
- PID：
  - `17736`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_4course_moon_commitbest_roleweighted_newrisk_v1`
- 配置：
  - target：`single_face_wall_4course_v1`
  - strategy：`statics_wall`
  - gravity：`moon`
  - commit-best：开启
  - candidate pose ranker：`20260621_support_map_cnn_3to4_roleweighted_structure_sameholdout_v1`
  - pose risk：`20260621_pose_risk_net_4to5_flywheel_dataset_v1`
  - pose risk weight：`0.25`
  - candidate top-k：`3`
- 设计目的：
  - line-lock 负结果说明单纯收紧墙线会造成传播失稳。
  - strict 负结果说明 PoseRisk 权重/跳过过保守会导致缺块、低墙。
  - 新对照保留 commit-best 的完整放置能力，同时用 4 层 role-weighted support-map 和 4/5 PoseRisk 降低明显漂移候选。
- 判定标准：
  - 必须至少优于 4/5 collect 的 4 层：`wall_y_span_m < 0.55`，`wall_outlier_count < 6`，`stable_count > 16` 或 failure_count 明显下降。
  - 如果 visible_courses 仍为 4 且 target_rmse 明显下降，可进入 5 层试验；否则继续训练/修策略，不推进高度。

### 4/5 Support-Map 泛化 Holdout 结果

- 任务：
  - `local_support_map_cnn_4to5_holdout_newcollect_v1`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_support_map_cnn_4to5_holdout_newcollect_v1`
- 训练/测试划分：
  - 训练 run：2026-06-20 的 4 层旧数据。
  - 测试 run：`20260621_local_primary_4to5_scheduler_v1_local_4to5_flywheel_collect_exploit_00_seed206212501`
  - 目的：检查旧数据训练出的 support-map CNN 是否泛化到今天新增的 4/5 层失败分布。
- 指标：
  - row_count：`19467`
  - rankable_group_count：`6489`
  - train_groups / test_groups：`5664 / 825`
  - overall top1 / top3：`0.4279 / 1.0000`
  - overall mean top1 regret：`10.0303`
  - `single_face_wall_4course_v1|moon` top1 / top3：`0.4222 / 1.0000`
  - `single_face_wall_v1|moon` top1 / top3：`0.4323 / 1.0000`
- 判断：
  - top3 仍然可靠，top1 不足以直接替代启发式。
  - 该结果支持“网络负责缩小候选集合，启发式/物理仿真负责最终选择”的路线。
  - 这也说明当前网络还没有完全学会高层墙体的稳定约束，因此暂时不应让网络控制全部 4/5 层。

### Hybrid Course-Gate 策略实现

- 代码改动：
  - `moon_rock_stack/run_structured_experiment.py`
  - `moon_rock_stack/structured.py`
- 新参数：
  - `--candidate-pose-ranker-max-course`
  - `--pose-risk-ranker-max-course`
  - `--stone-fit-ranker-max-course`
  - 默认值都是 `-1`，表示旧行为不变，即网络在所有 course 都可用。
- 设计目的：
  - 前 3 层使用神经网络：stone-fit 选石、support-map pose ranker 缩小候选位姿、PoseRisk 做风险惩罚。
  - 第 4 层及以上暂时回退到启发式和 MuJoCo 候选仿真，不让训练不足的网络把高层结构带偏。
  - 这样既能利用 2/3/4 层已有数据，又能继续收集 4/5 层高层失败样本，形成数据飞轮。
- smoke 验证：
  - 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_hybrid_course_gate_smoke_v1`
  - 设置：`single_face_wall_2course_v1`，`max_course=0`。
  - candidate_pose_log 分组结果：course 0 的 `ranker_top_k=1`，course 1 的 `ranker_top_k=0`。
  - 结论：course gate 已真正改变每层的网络启用状态。

### Hybrid 4/5 层异步实验

- 4 层任务：
  - job：`local_4course_hybrid_neural3_heuristic_high_v1`
  - PID：`20388`
  - 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_4course_moon_hybrid_neural3_heuristic_high_v1`
  - target：`single_face_wall_4course_v1`
- 5 层任务：
  - job：`local_5course_hybrid_neural3_heuristic_high_v1`
  - PID：`7104`
  - 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_5course_moon_hybrid_neural3_heuristic_high_v1`
  - target：`single_face_wall_v1`
- 公共配置：
  - gravity：`moon`
  - strategy：`statics_wall`
  - rock_profile：`high_wall`
  - candidates：`8`
  - steps_per_rock / hold_steps：`400 / 1750`
  - commit-best：开启
  - stone-fit：`20260621_wall_flywheel_3course_focus_v2_stone_slot_net_20260621_014556`
  - candidate pose ranker：`20260621_support_map_cnn_3to4_roleweighted_structure_sameholdout_v1`
  - PoseRiskNet：`20260621_pose_risk_net_4to5_flywheel_dataset_v1`
  - 三个网络均设置 `max_course=2`，即 course 0/1/2 用网络，course 3 及以上用启发式和物理候选仿真。
- 判定标准：
  - 4 层优先看 `stable_count`、`failure_count`、`target_rmse_xy_m`、`wall_y_span_m`、`wall_outlier_count`。
  - 5 层优先看 `visible_courses` 是否从 strict 对照的 4 层提升到 5 层，同时不能出现明显 wall_y_span 爆炸。
  - 如果 hybrid 4 层优于全神经/strict 对照，则下一轮用它采集更多 4 层数据，再训练更强 support-map/PoseRisk。

### Course-Gate 数据飞轮调度器

- 代码改动：
  - `scripts/run_wall_data_flywheel.py`
- 新增参数：
  - `--candidate-pose-ranker-max-course`
  - `--pose-risk-ranker-max-course`
  - `--stone-fit-ranker-max-course`
- dry-run：
  - 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_hybrid_course_gate_scheduler_dryrun_v1`
  - 验证：`collect_jobs.json` 中 exploit collect 命令已经包含三个 `*-max-course 2` 参数。
- 意义：
  - 后续不需要手工拼接 hybrid 命令，可以直接让飞轮完成 collect -> dataset -> depth/support map export -> modular training -> eval。
  - 这使“前 3 层神经网络，后面启发式采样”的策略可以持续产数据，而不是只做单次验证。

### 5 层 Strict 对照截图

- 来源实验：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_5course_moon_strict_eval_supportcnn_newrisk_v1`
- 截图输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_5course_moon_strict_eval_supportcnn_newrisk_v1\cap_strict5`
- 案例：
  - `00_single_face_wall_v1_failure_statics_wall_moon_trial_00`
- 结果摘要：
  - stable_count / failure_count：`19 / 1`
  - visible_courses：`4`
  - stack_height_m：`0.2738`
  - target_rmse_xy_m：`0.0769`
  - target_max_xy_error_m：`0.2463`
- 图像内容：
  - RGB：`front_rgb.png`，`wall_front_rgb.png`，`top_rgb.png`，`wall_top_rgb.png` 等。
  - 深度：`front_depth.png`，`wall_front_depth.png`，`top_depth.png`，`wall_top_depth.png`。
  - object depth：`front_object_depth.png`，`wall_front_object_depth.png`，`top_object_depth.png`，`wall_top_object_depth.png`。
- 用途：
  - 作为 5 层 hybrid 的视觉对照：该 strict 案例多数石头能稳定，但最终没有形成 5 层可见墙体。

### 4/5 Role-Weighted Support-Map CNN 训练

- 任务：
  - `local_support_map_cnn_4to5_roleweighted_holdout_v1`
- PID：
  - `16192`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_support_map_cnn_4to5_roleweighted_holdout_newcollect_v1`
- 输入：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_support_maps_4to5_flywheel_dataset_v1`
- 配置：
  - epochs：`100`
  - hidden：`224`
  - target mode：`structure_aware`
  - held-out test run：`20260621_local_primary_4to5_scheduler_v1_local_4to5_flywheel_collect_exploit_00_seed206212501`
  - role weights：`middle=1.25`，`cap=1.50`
  - course weights：`2=1.15`，`3=1.25`，`4=1.35`
- 目的：
  - 与未加权 4/5 holdout 模型对比，看是否能改善 middle/cap 和高 course 的候选排序。
  - 如果 top1 或 regret 明显改善，可替换后续 hybrid 低层网络；如果没有改善，说明当前网络结构/输入还不足，需要更多观测或重新设计损失。
- 结果：
  - overall top1 / top3：`0.4182 / 1.0000`
  - overall mean top1 regret：`9.6300`
  - train top1：`0.6746`
  - `single_face_wall_4course_v1|moon` top1 / top3：`0.4306 / 1.0000`
  - `single_face_wall_4course_v1|moon` mean top1 regret：`11.4595`
  - `single_face_wall_v1|moon` top1 / top3：`0.4086 / 1.0000`
  - `single_face_wall_v1|moon` mean top1 regret：`8.2137`
- 判断：
  - 相比未加权 holdout：overall top1 从 `0.4279` 降到 `0.4182`，但 overall regret 从 `10.0303` 降到 `9.6300`。
  - 4 层 moon 分组有改善：top1 从 `0.4222` 到 `0.4306`，regret 从 `12.5881` 到 `11.4595`。
  - 5 层 moon 分组变差：top1 从 `0.4323` 到 `0.4086`，regret 从 `8.0500` 到 `8.2137`。
  - 因此它可以作为 4 层修正候选模型，但暂时不替换 5 层 hybrid 默认模型。

### 4/5 Regularized Support-Map CNN 训练

- 任务：
  - `local_support_map_cnn_4to5_regularized_holdout_v1`
- PID：
  - `21072`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_support_map_cnn_4to5_regularized_holdout_v1`
- 输入：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_support_maps_4to5_flywheel_dataset_v1`
- 设计原因：
  - 上一版 role/course-weighted 模型训练 top1 到 `0.6746`，测试 top1 只有 `0.4182`，存在明显过拟合。
  - 新模型降低 hidden，从 `224` 到 `192`；提高 dropout 到 `0.28`；提高 weight decay 到 `0.0008`；不再对 course 4 加权。
- 配置：
  - epochs：`90`
  - hidden：`192`
  - dropout：`0.28`
  - lr：`0.00055`
  - weight_decay：`0.0008`
  - role weights：`middle=1.15`，`cap=1.35`
  - course weights：`2=1.10`，`3=1.15`
- 判定：
  - 如果 overall top1 高于 `0.4279` 且 regret 不高于 `10.0303`，可替代未加权模型。
  - 如果 4 层指标改善但 5 层不降太多，可只用于 4 层 hybrid。
- 结果：
  - overall top1 / top3：`0.4158 / 1.0000`
  - overall mean top1 regret：`10.1424`
  - train top1：`0.4514`
  - `single_face_wall_4course_v1|moon` top1 / top3：`0.4000 / 1.0000`
  - `single_face_wall_4course_v1|moon` mean top1 regret：`13.6459`
  - `single_face_wall_v1|moon` top1 / top3：`0.4280 / 1.0000`
  - `single_face_wall_v1|moon` mean top1 regret：`7.4301`
- 判断：
  - 正则化确实降低了过拟合，train top1 从强加权模型的 `0.6746` 降到 `0.4514`。
  - 但整体 top1 和 regret 均未超过未加权模型，不应作为通用默认模型。
  - 4 层指标明显变差，不用于 4 层 hybrid。
  - 5 层 regret 优于未加权和强加权，可作为 5 层候选备选；需要等当前 5 层 hybrid 结果后再决定是否开新对照。

### 在线 Support-Map Ranker 推理优化

- 代码改动：
  - `moon_rock_stack/structured.py`
- 内容：
  - 在线 support-map CNN ranker 由固定 CPU 推理改成优先 CUDA，若 CUDA 不可用则回退 CPU。
  - checkpoint 仍先用 CPU `map_location` 读取，再把模型搬到目标设备。
- 影响范围：
  - 只影响后续新启动的 `run_structured_experiment` 任务。
  - 当前已经在跑的 PID `17736`、`20388`、`7104` 不会被中途改变。
- 目的：
  - course-gated hybrid 中底部 3 层会频繁调用 support-map CNN，未来新任务使用 GPU 推理可减轻 CPU 压力。

### 轻量 Hybrid Course-Gate Flywheel

- 任务：
  - `local_hybrid_course_gate_flywheel_v1`
- PID：
  - `6412`
- session：
  - `20260621_hybrid_course_gate_flywheel_v1`
- session 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_hybrid_course_gate_flywheel_v1`
- 目的：
  - 使用已经实现的 course gate 直接跑自动闭环：collect -> dataset -> depth/support map export -> modular training -> eval。
  - 让数据飞轮继续产 4/5 层 hybrid 数据，而不是只等待单次手工实验。
- collect/eval 配置：
  - targets：`single_face_wall_4course_v1,single_face_wall_v1`
  - gravity：`moon`
  - collect mode：`exploit`
  - rocks / eval-rocks：`150 / 150`
  - candidates：`6`
  - steps_per_rock / hold_steps：`320 / 1400`
  - candidate pose top-k：`2`
  - stone-fit top-k：`12`
  - 三个网络均 `max_course=2`
  - collect/eval 均开启 commit-best curriculum。
- 初始模型：
  - stone-fit：`20260621_wall_flywheel_3course_focus_v2_stone_slot_net_20260621_014556`
  - support-map ranker：`20260621_support_map_cnn_3to4_roleweighted_structure_sameholdout_v1`
  - PoseRiskNet：`20260621_pose_risk_net_4to5_flywheel_dataset_v1`
- 注意：
  - 该任务使用新代码，后续在线 support-map ranker 会优先走 CUDA。
  - 旧的 PID `17736`、`20388`、`7104` 已经启动，仍按启动时的代码继续跑，不中断。
- 当前状态更新：
  - 异步注册表显示父 PID `6412` 已退出，`collect_exploit_00.stdout.log` 和 `collect_exploit_00.stderr.log` 仍为空。
  - `flywheel_manifest.json` 已写入 collect 阶段和完整命令，但没有进入后续 dataset/training 阶段。
  - 暂不删除该目录；后续若需要重启，应优先用直接 `run_structured_experiment` 异步命令或等当前 3 条 MuJoCo 对照完成后再启动，避免包装层状态不清。

### Negative-Mining 数据集

- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_negative_mining_wall_dataset_v1`
- 来源 run：
  - `20260621_local_primary_4to5_scheduler_v1_local_4to5_flywheel_collect_exploit_00_seed206212501`
  - `20260621_4course_moon_line_lock_supportcnn_newrisk_v1`
  - `20260621_4course_moon_strict_eval_oldstone_newposerisk_v1`
  - `20260621_high_wall_8course_curriculum_oldstone_newrisk_v1`
  - `20260621_5course_moon_strict_eval_supportcnn_newrisk_v1`
- 目的：
  - 不把失败案例丢掉，而是把 `failure_cases.csv`、`placement_log.csv`、`candidate_pose_log.csv` 合成训练表。
  - 重点学习 cap/middle、高 course、y 向漂移、missed_target、post_hold_drift 等失败模式。
- 规模：
  - run_example_count：`7`
  - placement_example_count：`189`
  - candidate_pose_example_count：`9279`
  - assignment_candidate_example_count：`3093`
- placement 分布：
  - moon：success `106`，failure `40`，skipped `43`
  - base：success `46`，failure `1`，skipped `1`
  - middle：success `53`，failure `31`，skipped `25`
  - cap：success `7`，failure `8`，skipped `17`
- 判断：
  - 负样本主要集中在 middle/cap，符合当前 4/5 层墙失败机理。
  - base 层不应被过度惩罚；后续模型需要对 role/course 加权或至少分层观察。

### Strict Negative-Mining PoseRiskNet

- 任务：
  - `local_pose_risk_negative_mining_strict_v1`
- PID：
  - `10964`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_pose_risk_net_negative_mining_strict_v1`
- 输入：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_negative_mining_wall_dataset_v1`
- 配置：
  - epochs：`180`
  - hidden：`224`
  - dropout：`0.22`
  - lr：`0.00065`
  - weight_decay：`0.00035`
  - split：`split-by-run`
  - target_error_limit：`0.12`
  - target_y_error_limit：`0.055`
  - disturbance_limit：`0.055`
  - velocity_limit：`0.18`
- 设计目的：
  - 当前 PoseRisk top3 可用但 top1 不可靠；这版不是让它单独贪心选 pose，而是作为更严格的风险惩罚项。
  - 训练标签来自后验失败、目标误差、扰动和速度，但输入仍只使用几何、目标、重力和候选位姿等预先可观测信息。
  - 如果 top3 safe 仍保持高水平且 top1 safe 提升，可进入下一轮 4/5 hybrid 采集。
- 结果：
  - row_count：`9279`
  - risky positive_count：`9214`
  - test run：`20260621_high_wall_8course_curriculum_oldstone_newrisk_v1`
  - test accuracy / F1：`0.9648 / 0.9821`
  - test top1 safe / top3 safe：`0.3000 / 1.0000`
- 判断：
  - 阈值过严，几乎所有候选都被标为 risky。
  - 该模型可作为强负样本过滤器或风险惩罚上限，不适合单独替代当前 PoseRisk。
  - 已启动宽松版 negative-mining PoseRisk 作为对照。

### Negative-Mining Support-Map Tensor Export

- 任务：
  - `local_negative_mining_support_maps_v1`
- PID：
  - `7920`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_negative_mining_support_maps_v1`
- 输入：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_negative_mining_wall_dataset_v1`
- 配置：
  - source：`candidate`
  - grid_size：`64`
  - window_m：`0.9`
  - shard_size：`2000`
  - dtype：`float16`
  - target filter：`single_face_wall`
  - strategy filter：`statics_wall`
- 目的：
  - 为下一版 risk-adjusted / structure-aware support-map CNN 准备输入。
  - 让失败候选也进入局部支撑图学习，而不只是进入 tabular PoseRisk。
- 结果：
  - row_count：`9279`
  - shard_count：`5`
  - channels：`height_before_m`、`support_occupancy`、`support_count_clipped`、`target_gaussian`、`candidate_footprint`、`candidate_height_m`、`gravity_ratio`、`course_ratio`

### Moderate Negative-Mining PoseRiskNet

- 任务：
  - `local_pose_risk_negative_mining_moderate_v1`
- PID：
  - `18176`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_pose_risk_net_negative_mining_moderate_v1`
- 输入：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_negative_mining_wall_dataset_v1`
- 配置：
  - epochs：`160`
  - hidden：`224`
  - dropout：`0.20`
  - target_error_limit：`0.18`
  - target_y_error_limit：`0.085`
  - disturbance_limit：`0.09`
  - velocity_limit：`0.25`
- 目的：
  - 对照 strict PoseRisk，避免把几乎所有候选都判成 risky。
  - 如果 top3 safe 保持 1.0 且 top1 safe 高于 strict 版，可优先进入下一轮 4/5 hybrid。
- 结果：
  - risky positive_count：`9211 / 9279`
  - test run：`20260621_5course_moon_strict_eval_supportcnn_newrisk_v1`
  - test accuracy / F1：`0.9528 / 0.9758`
  - test top1 safe / top3 safe：`0.4375 / 1.0000`
- 判断：
  - top1 safe 比 strict 版 `0.3000` 明显更好，但 risky 标签仍接近全正类。
  - 后续需要弱化 velocity 标签，否则候选短 settle 阶段的速度会把大量几何可用候选误判为 risky。

### PoseRisk 标签阈值复查

- 数据源：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_negative_mining_wall_dataset_v1\candidate_pose_examples.csv`
- 观察：
  - 一个 base 候选 `target_error_xy_m=0.0151`、`target_y_error_m=0.0148`，但 `velocity_inf_norm_after_place=18.0571`。
  - 说明候选短 settle 阶段的速度字段会把很多低误差候选标成 risky。
- 若忽略 velocity，仅用 `target_y_error <= 0.10` 且 `disturbance <= 0.12`：
  - target_error <= `0.12`：safe `2628 / 9279`
  - target_error <= `0.22`：safe `2909 / 9279`
  - target_error <= `0.40`：safe `3214 / 9279`
- 若保持 `target_error <= 0.22`、`target_y_error <= 0.10`、`disturbance <= 0.12`，改变 velocity：
  - velocity <= `0.25`：safe `1290 / 9279`
  - velocity <= `5`：safe `2347 / 9279`
  - velocity <= `50`：safe `2909 / 9279`
- 结论：
  - 当前 negative-mining PoseRisk 不应强依赖 velocity。
  - 后续闭环中 velocity 更适合作为二级惩罚或后验统计，而不是主要二分类标签。

### No-Velocity-Bias Negative-Mining PoseRiskNet

- 任务：
  - `local_pose_risk_negative_mining_no_velocity_bias_v1`
- PID：
  - `17556`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_pose_risk_net_negative_mining_no_velocity_bias_v1`
- 输入：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_negative_mining_wall_dataset_v1`
- 配置：
  - epochs：`180`
  - hidden：`224`
  - target_error_limit：`0.22`
  - target_y_error_limit：`0.10`
  - disturbance_limit：`0.12`
  - velocity_limit：`50.0`
- 目的：
  - 训练一个不会把短 settle 高速度候选全部判死的 PoseRisk。
  - 如果 top3 safe 仍高、top1 safe 提升，并且 positive_count 不再接近全正类，可作为下一轮闭环优先 PoseRisk。
- 结果：
  - risky positive_count：`9183 / 9279`
  - test run：`20260621_4course_moon_strict_eval_oldstone_newposerisk_v1`
  - test accuracy / F1：`0.9032 / 0.9490`
  - test top1 safe / top3 safe：`0.4118 / 1.0000`
- 判断：
  - 弱化 velocity 后仍然接近全正类，原因是脚本把 `label_committed_success=0` 和 `failure_reason` 当成风险标签。
  - 这会把未被当前启发式选中的候选也错误标成 risky，不适合候选级风险训练。

### Candidate-Metric PoseRisk 标签修正

- 代码改动：
  - `scripts/train_torch_pose_risk_net.py`
- 新参数：
  - `--candidate-metric-labels`
- 设计：
  - 开启后只用每个 candidate pose 自身的 `target_error_xy_m`、`target_y_error_m`、`placed_disturbance_xy_m`、`velocity_inf_norm_after_place` 判定 risky。
  - 忽略 `label_committed_success` 和 `failure_reason`，因为它们是启发式选择/slot 级结果，不是候选本身的几何风险。
- 编译：
  - `python -m py_compile scripts\train_torch_pose_risk_net.py` 通过。

### Candidate-Metric Negative-Mining PoseRiskNet

- 任务：
  - `local_pose_risk_candidate_metric_negative_mining_v1`
- PID：
  - `21392`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_pose_risk_net_candidate_metric_negative_mining_v1`
- 输入：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_negative_mining_wall_dataset_v1`
- 配置：
  - epochs：`180`
  - hidden：`224`
  - target_error_limit：`0.22`
  - target_y_error_limit：`0.10`
  - disturbance_limit：`0.12`
  - velocity_limit：`50.0`
  - labels：`candidate-metric-labels`
- 目的：
  - 得到真正候选级的 PoseRisk，用于替代“把失败 slot 里全部候选都当负样本”的旧标签方式。
  - 如果 positive_count 接近统计预期的 `6370 / 9279`，且 top3 safe 保持高水平，可进入下一轮闭环。
- 结果：
  - risky positive_count：`6370 / 9279`
  - test run：`20260621_5course_moon_strict_eval_supportcnn_newrisk_v1`
  - test accuracy / F1：`0.7061 / 0.7848`
  - test top1 safe / top3 safe：`0.6762 / 1.0000`
- 判断：
  - 这是目前最合理的 PoseRisk 版本：负样本比例不再退化为近全正类，且 top3 safe 仍保持 1.0。
  - top1 safe 明显优于 strict/moderate/no-velocity-bias 版本。
  - 下一轮闭环优先使用该模型，但仍只作为风险惩罚和 top-k 辅助，不让它单独贪心选 pose。

### 4 层 Commit-Best 负案例

- 实验：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_4course_moon_commitbest_roleweighted_newrisk_v1`
- 配置：
  - target：`single_face_wall_4course_v1`
  - gravity：`moon`
  - strategy：`statics_wall`
  - commit-best：开启
  - candidate pose ranker：`20260621_support_map_cnn_3to4_roleweighted_structure_sameholdout_v1`
  - PoseRiskNet：`20260621_pose_risk_net_4to5_flywheel_dataset_v1`
- 结果：
  - success / shape_success：`0 / 0`
  - visible_courses：`4`
  - stable_count / failure_count：`13 / 11`
  - target_rmse_xy_m：`0.3818`
  - target_max_xy_error_m：`1.4325`
  - stack_height_m：`0.3182`
  - wall_y_span_m：`1.3999`
  - wall_outlier_count：`10`
  - structure_score：`-2.0306`
- 失败机理：
  - 该实验能放满并形成 4 个可见 course，但墙线严重散开。
  - 主要失败集中在 middle/cap，包含 `missed_target` 和 `unstable_structure`。
  - 这说明仅靠 commit-best 放满不等于结构成功，需要更强的墙线/支撑面风险模型。
- 截图：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_4course_moon_commitbest_roleweighted_newrisk_v1\cap_commitbest4_neg`

### Negative-Mining Risk-Adjusted Support-Map CNN

- 任务：
  - `local_support_map_cnn_negative_mining_riskadjusted_v1`
- PID：
  - `8460`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_support_map_cnn_negative_mining_riskadjusted_v1`
- 输入：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_negative_mining_support_maps_v1`
- 配置：
  - target_mode：`risk_adjusted`
  - held-out test run：`20260621_high_wall_8course_curriculum_oldstone_newrisk_v1`
  - epochs：`90`
  - hidden：`192`
  - dropout：`0.22`
  - role weights：`middle=1.20`，`cap=1.45`
  - course weights：`2=1.15`，`3=1.25`，`4=1.35`，`5=1.45`
- 目的：
  - 把负样本中的 target error、扰动、速度风险直接融入 support-map 排序目标。
  - 该模型如果在 high-wall heldout 上 top3/regret 好于旧模型，可作为后续 5 层候选 ranker。
- 结果：
  - row_count：`9279`
  - rankable_group_count：`3093`
  - test run：`20260621_high_wall_8course_curriculum_oldstone_newrisk_v1`
  - test top1 / top3：`0.4409 / 1.0000`
  - test mean top1 regret：`6.8046`
  - train top1 / top3：`0.4761 / 1.0000`
- 判断：
  - 该模型在 high-wall 负样本 heldout 上 top3 仍可靠，regret 明显低于早先 4/5 holdout 模型。
  - 可以作为下一轮 4/5 candidate-pose ranker 的候选，但仍应保留启发式最终筛选。

### Candidate-Metric Risk 4 层闭环

- 任务：
  - `local_4course_candidate_metric_risk_v1`
- PID：
  - `19780`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_4course_moon_candidate_metric_pose_risk_v1`
- 配置：
  - target：`single_face_wall_4course_v1`
  - gravity：`moon`
  - rocks：`150`
  - candidates：`5`
  - steps_per_rock / hold_steps：`300 / 1300`
  - stone-fit：`20260621_wall_flywheel_3course_focus_v2_stone_slot_net_20260621_014556`
  - candidate-pose ranker：`20260621_support_map_cnn_negative_mining_riskadjusted_v1`
  - PoseRiskNet：`20260621_pose_risk_net_candidate_metric_negative_mining_v1`
  - pose risk weight：`0.20`
  - 三个网络均 `max_course=2`
  - commit-best：开启
- 目的：
  - 快速检验 candidate-metric PoseRisk + negative-mining support-map 是否能改善 4 层墙线。
  - 本轮优先收集新模型下的正/负样本；若稳定性有明显突破，再补截图并汇报。

### Negative-Mining 数据集 v2

- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_negative_mining_wall_dataset_v2`
- 相比 v1 新增：
  - `20260621_4course_moon_commitbest_roleweighted_newrisk_v1`
- 规模：
  - run_example_count：`8`
  - placement_example_count：`213`
  - candidate_pose_example_count：`10575`
  - assignment_candidate_example_count：`3525`
- placement 分布：
  - success：`119`
  - failure：`51`
  - skipped：`43`
  - middle failure/skipped 仍是主问题：middle success `57`，failure `39`，skipped `25`
  - cap 也仍弱：cap success `9`，failure `11`，skipped `17`
- 意义：
  - 把 4 层 commit-best 的“放满但墙线散开”负样本并入训练。
  - 后续模型不只学习跳过/缺块，也学习 wall_y_span/outlier 类结构失败。

### Candidate-Metric Negative-Mining PoseRiskNet v2

- 任务：
  - `local_pose_risk_candidate_metric_negative_mining_v2`
- PID：
  - `9440`
- 输出：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_pose_risk_net_candidate_metric_negative_mining_v2`
- 输入：
  - `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_negative_mining_wall_dataset_v2`
- 配置：
  - labels：`candidate-metric-labels`
  - epochs：`180`
  - target_error_limit：`0.22`
  - target_y_error_limit：`0.10`
  - disturbance_limit：`0.12`
  - velocity_limit：`50.0`
- 目的：
  - 用包含 commit-best 失败的新数据重训 candidate-level PoseRisk。
  - 如果 v2 top1/top3 不低于 v1，后续闭环优先用 v2。
- 结果：
  - row_count：`10575`
  - risky positive_count：`7233`
  - test runs：`20260621_4course_moon_commitbest_roleweighted_newrisk_v1`，`20260621_local_primary_4to5_scheduler_v1_local_4to5_flywheel_collect_exploit_00_seed206212501`
  - test accuracy / F1：`0.7049 / 0.7872`
  - test top1 safe / top3 safe：`0.7044 / 1.0000`
- 判断：
  - v2 比 v1 的 top1 safe `0.6762` 更好，top3 safe 仍保持 `1.0000`。
  - 当前正在跑的 `local_4course_candidate_metric_risk_v1` 用的是 v1；下一轮闭环优先改用 v2。
## 2026-06-21 18:10-19:20 本地数据飞轮继续记录

### 新增高层近失败样本

- 新完成实验：
  - `batch_runs\20260621_5course_moon_hybrid_neural3_heuristic_high_v1`
- 结果：
  - gravity: `moon`
  - success / shape_success: `0 / 0`
  - visible_courses: `5`
  - rock_count: `31`
  - stable_count / failure_count: `20 / 11`
  - target_rmse_xy_m: `0.1652`
  - target_max_xy_error_m: `0.3437`
  - structure_score: `0.5653`
  - stack_height_m: `0.3182`
  - max_horizontal_drift_m: `0.0081`
  - wall_y_span_m: `0.6616`
  - wall_outlier_count: `4`
- 判断：
  - 这不是严格成功，但属于高层近失败样本：5 层可见、漂移低、形状误差仍偏大。
  - 该 run 已并入后续 v5 数据集，用于让风险模型学习高层墙体的离群/线宽问题。

### Negative-Mining 数据集 v4/v5

- `batch_runs\20260621_negative_mining_wall_dataset_v4`
  - run_example_count: `11`
  - placement_example_count: `285`
  - candidate_pose_example_count: `13657`
  - assignment_candidate_example_count: `4535`
  - placement success / failure / skipped: `158 / 73 / 54`
  - middle success / failure / skipped: `71 / 55 / 31`
  - cap success / failure / skipped: `14 / 17 / 21`
- `batch_runs\20260621_negative_mining_wall_dataset_v5`
  - 相比 v4 新增 `20260621_5course_moon_hybrid_neural3_heuristic_high_v1`
  - run_example_count: `13`
  - placement_example_count: `347`
  - candidate_pose_example_count: `16836`
  - assignment_candidate_example_count: `5448`
  - placement success / failure / skipped: `193 / 87 / 67`
  - middle success / failure / skipped: `88 / 67 / 40`
  - cap success / failure / skipped: `18 / 19 / 25`
- 数据意义：
  - middle/cap 仍是主要失败层位，说明“底层稳定”已经不是核心瓶颈，真正瓶颈是上层选择、墙线约束和局部支撑面。
  - v5 开始包含 5 层近失败样本，可以训练模型识别高层墙体的离群风险。

### PoseRisk 训练脚本稳定性修正

- 修改文件：
  - `scripts/train_torch_pose_risk_net.py`
- 问题：
  - v4 数据集训练时，训练本体 1 epoch 约 1.5 秒完成，但评估阶段卡在 NumPy/BLAS 的大矩阵 `predict()`。
  - 该卡点与 MuJoCo 并行任务和 MKL/BLAS 线程竞争有关；不是标签、数据读取或 torch 训练问题。
- 修正：
  - `predict()` 改为优先使用 torch CPU 小批量推理。
  - NumPy fallback 改为 1024 样本分块，并用 `np.einsum(..., optimize=False)` 避免走同一条 BLAS 大矩阵路径。
- 验证：
  - `python -m py_compile scripts\train_torch_pose_risk_net.py` 通过。
  - `20260621_pose_risk_net_candidate_metric_negative_mining_v3_smoke_afterpatch` 1 epoch smoke 成功。

### PoseRisk v3/v4 结果

- `batch_runs\20260621_pose_risk_net_candidate_metric_negative_mining_v3_controlled_afterpatch`
  - dataset: `20260621_negative_mining_wall_dataset_v4`
  - row_count: `13657`
  - risky positive_count: `9054`
  - test accuracy / F1: `0.6691 / 0.7251`
  - test top1 safe / top3 safe: `0.7596 / 0.9981`
  - 判断：top1 safe 明显提高，适合做 4 层候选风险排序，但 held-out 分布较难，accuracy 不高。
- `batch_runs\20260621_pose_risk_net_candidate_metric_negative_mining_v4_highwall`
  - dataset: `20260621_negative_mining_wall_dataset_v5`
  - row_count: `16836`
  - risky positive_count: `11270`
  - test accuracy / F1: `0.7643 / 0.8320`
  - test top1 safe / top3 safe: `0.7454 / 1.0000`
  - 判断：v4 top1 safe 略低于 v3，但包含 5 层高墙近失败，泛化目标更符合下一阶段，因此闭环优先使用 v4。

### Support-Map CNN v4_highwall

- 输出：
  - `batch_runs\20260621_support_map_cnn_negative_mining_v4_highwall_foreground`
- 输入：
  - `batch_runs\20260621_negative_mining_support_maps_v4_highwall`
- 配置：
  - epochs: `90`
  - device: `cuda`
  - target_mode: `risk_adjusted`
  - hidden: `192`
  - dropout: `0.22`
  - role weights: `middle=1.20`, `cap=1.45`
  - course weights: `2=1.15`, `3=1.25`, `4=1.35`, `5=1.45`
- 结果：
  - row_count: `16836`
  - rankable_group_count: `5448`
  - train top1 / top3: `0.5334 / 0.9975`
  - test top1 / top3: `0.4483 / 0.9022`
  - 4-course moon top1 / top3: `0.5149 / 0.9440`
  - 5-course moon top1 / top3: `0.4085 / 0.8772`
  - test mean top1 regret: `7.0592`
  - test mean top3 regret: `0.5263`
- 判断：
  - 4 层 top1 有提升，但 5 层仍弱。
  - 该模型可作为候选筛选研究结果，但目前不适合直接放入长闭环内环：实测闭环推理/数据准备开销过大。

### 闭环验证尝试与未完成 partial

- `batch_runs\20260621_4course_moon_supportv4_poseriskv4_eval`
  - 配置：support-map v4 + PoseRisk v4，`candidates=5`, `steps=320`, `hold=1400`
  - 状态：超过 20 分钟未完成，被工具超时中止。
  - 结果：无 `results.csv`，只生成初始 meshes/mjcf/features/protocol。
- `batch_runs\20260621_4course_moon_supportv4_poseriskv4_quick_eval`
  - 配置：support-map v4 + PoseRisk v4，`candidates=3`, `steps=180`, `hold=900`
  - 状态：超过 15 分钟未完成，被工具超时中止。
  - 结果：无 `results.csv`，不可作为训练样本。
- `batch_runs\20260621_4course_moon_poseriskv4_only_quick_eval`
  - 配置：PoseRisk v4 only，`candidates=3`, `steps=150`, `hold=700`
  - 状态：约 10 分钟后非零退出，无 `results.csv`。
  - 结果：不可作为训练样本。
- `batch_runs\20260621_4course_moon_poseriskv4_smoke_eval`
  - 配置：PoseRisk v4 smoke，`candidates=1`, `steps=50`, `hold=120`
  - 状态：约 23 秒非零退出，无 `results.csv`。
  - 结果：不可作为训练样本。
- `batch_runs\20260621_4course_moon_no_poserisk_smoke_eval`
  - 配置：无 PoseRisk，同 smoke 规模
  - 状态：超过 5 分钟未完成，被工具超时中止，无 `results.csv`。
- 结论：
  - 当前瓶颈已经转移到结构仿真内环和运行调度，不应继续盲目拉长单次 4 层闭环。
  - 没有 `results.csv` 的 partial 目录只保留为失败调度记录，不并入 learning dataset。

### 下一步调度策略

- 先优化/检查 `run_structured_experiment` 的闭环耗时：
  - 给 placement 内环增加阶段性进度日志，至少记录 slot、candidate、当前耗时。
  - 缓存 support-map CNN 和 PoseRisk runtime，避免每个 slot/candidate 重复做模型加载或昂贵特征构造。
  - 增加超时保护：单个 slot 超时后记录 skipped/failure，而不是整轮实验无结果。
- 数据飞轮继续方向：
  - 使用已完成的 v5 数据和 PoseRisk v4 作为当前最强小网络结果。
  - 暂不把 support-map v4 放入长闭环，先做离线排序和短小 smoke 修复。
  - 下一轮优先恢复可以稳定产出 `results.csv` 的 3-4 层闭环，再继续采集正/负样本。

## 2026-06-21 19:20-20:00 运行内环修复与 4 层近成功

### 进度日志修复

- 修改文件：
  - `moon_rock_stack\structured.py`
  - `moon_rock_stack\run_structured_experiment.py`
- 新增输出：
  - `structured_progress.csv`
- 字段：
  - `time_s`
  - `elapsed_s`
  - `event`
  - `target_name`
  - `strategy`
  - `gravity`
  - `trial`
  - `slot_index`
  - `slot_id`
  - `course`
  - `role`
  - `placed_count`
  - `rock_index`
  - `candidate_count`
  - `stone_pool_size`
  - `message`
- 事件：
  - `trial_start`
  - `model_loaded`
  - `slot_start`
  - `slot_done`
  - `final_hold_start`
  - `trial_done`
- 目的：
  - 避免长时间实验被中止后只留下 meshes/mjcf/features，而不知道卡在哪个阶段。
  - 以后没有 `results.csv` 的 partial 目录也能判断是模型加载、stone pool、slot 模拟还是 final hold 问题。
- 编译：
  - `python -m py_compile moon_rock_stack\structured.py moon_rock_stack\run_structured_experiment.py` 通过。

### 在线 MLP 推理修复

- 修改文件：
  - `moon_rock_stack\structured.py`
- 问题：
  - `stone_fit_ranker` 接入后，最小 3 层 smoke 卡在 `slot_start` 的第 0 个 base slot，尚未进入 MuJoCo 落石。
  - `structured_progress.csv` 证明瓶颈在 `_literature_stone_pool()` 调用 `_mlp_binary_prob()` 给全部候选石头打分。
  - 这是和 PoseRisk 训练评估类似的 NumPy/BLAS 小矩阵卡顿问题。
- 修复：
  - `_mlp_binary_prob()` 从 `x @ w` 改为 `np.einsum(..., optimize=False)`。
  - 这条路径服务 stone-fit、PoseRisk 和旧式 MLP candidate ranker。
- 验证：
  - 修复前：`20260621_3course_stonefit_progress_smoke_eval` 180 秒超时，progress 停在 `slot_start`。
  - 修复后：`20260621_3course_stonefit_progress_smoke_after_mlpfix` 1.7 秒完成，并产出 `results.csv`。

### MLP 修复后 4 层对照

- `batch_runs\20260621_4course_moon_stonefit_poseriskv4_quick_after_mlpfix`
  - 配置：stone-fit + PoseRisk v4，`candidates=2`, `steps=80`, `hold=240`
  - 用时：约 30 秒
  - success / shape_success: `0 / 0`
  - visible_courses: `4`
  - stable_count / failure_count: `8 / 16`
  - target_rmse_xy_m: `0.9722`
  - target_max_xy_error_m: `4.3233`
  - max_horizontal_drift_m: `0.8679`
  - velocity_inf_norm: `8.8246`
  - 判断：快速 smoke 可产出数据，但物理 settle 不充分，主要作为运行链路验证。
- `batch_runs\20260621_4course_moon_stonefit_poseriskv4_medium_after_mlpfix`
  - 配置：stone-fit + PoseRisk v4，`candidates=3`, `steps=180`, `hold=900`
  - success / shape_success: `0 / 0`
  - visible_courses: `4`
  - stable_count / failure_count: `15 / 9`
  - target_rmse_xy_m: `1.4929`
  - target_max_xy_error_m: `7.2785`
  - max_horizontal_drift_m: `4.3278`
  - velocity_inf_norm: `0.2370`
  - 判断：稳定数量尚可，但墙线/目标误差非常差，是“稳定但散线”的负样本。
- `batch_runs\20260621_4course_moon_supportv4_stonefit_poseriskv4_medium_after_mlpfix`
  - 配置：support-map v4 + stone-fit + PoseRisk v4，`candidates=3`, `steps=180`, `hold=900`
  - success / shape_success: `0 / 0`
  - visible_courses: `4`
  - stable_count / failure_count: `10 / 14`
  - target_rmse_xy_m: `0.7682`
  - target_max_xy_error_m: `2.5897`
  - max_horizontal_drift_m: `0.9575`
  - velocity_inf_norm: `0.0072`
  - 判断：support-map 明显改善墙形状和速度，但局部稳定性下降。

### 数据集 v6/v7/v8 与 PoseRisk v5/v6

- `batch_runs\20260621_negative_mining_wall_dataset_v6_after_mlpfix`
  - 新增 MLP 修复后的 v4 medium 两个对照。
  - run_example_count: `15`
  - placement_example_count: `395`
  - candidate_pose_example_count: `17796`
  - placement success / failure / skipped: `218 / 110 / 67`
  - middle success / failure / skipped: `96 / 83 / 40`
  - cap success / failure / skipped: `21 / 26 / 25`
- `batch_runs\20260621_pose_risk_net_candidate_metric_negative_mining_v5_after_mlpfix`
  - dataset: `20260621_negative_mining_wall_dataset_v6_after_mlpfix`
  - row_count: `17796`
  - risky positive_count: `11602`
  - test accuracy / F1: `0.7343 / 0.7719`
  - test top1 safe / top3 safe: `0.7724 / 0.9936`
  - 判断：top1 safe 比 v4/v3 更好，闭环优先使用 v5。
- `batch_runs\20260621_negative_mining_wall_dataset_v7_poseriskv5_nearmiss`
  - 新增 PoseRisk v5 `w=0.20` 近失败。
  - run_example_count: `16`
  - placement_example_count: `419`
  - candidate_pose_example_count: `18180`
  - placement success / failure / skipped: `230 / 122 / 67`
- `batch_runs\20260621_pose_risk_net_candidate_metric_negative_mining_v6_poseriskv5_nearmiss`
  - dataset: `20260621_negative_mining_wall_dataset_v7_poseriskv5_nearmiss`
  - test accuracy / F1: `0.7354 / 0.8001`
  - test top1 safe / top3 safe: `0.7043 / 0.9936`
  - 判断：held-out 更难，top1 safe 下降；当前闭环仍保留 v5。
- `batch_runs\20260621_negative_mining_wall_dataset_v8_w035_nearmiss`
  - 新增 PoseRisk v5 `w=0.35` 近失败。
  - 作为下一轮训练材料保留。

### PoseRisk 权重实验

- `batch_runs\20260621_4course_moon_supportv4_stonefit_poseriskv5_medium_after_mlpfix`
  - pose_risk_weight: `0.20`
  - success / shape_success: `0 / 0`
  - visible_courses: `4`
  - stable_count / failure_count: `12 / 12`
  - target_rmse_xy_m: `0.2842`
  - target_max_xy_error_m: `0.9500`
  - max_horizontal_drift_m: `0.1024`
  - wall_y_span_m: `0.5523`
  - wall_outlier_count: `12`
  - velocity_inf_norm: `0.0339`
  - failure reasons: `missed_target=8`, `unstable_structure=4`
  - 截图目录：`batch_runs\20260621_4course_moon_supportv4_stonefit_poseriskv5_medium_after_mlpfix\cap_supportv4_poseriskv5_nearmiss`
- `batch_runs\20260621_4course_moon_supportv4_stonefit_poseriskv5_w035_medium_after_mlpfix`
  - pose_risk_weight: `0.35`
  - success / shape_success: `0 / 0`
  - visible_courses: `4`
  - stable_count / failure_count: `16 / 8`
  - target_rmse_xy_m: `0.2165`
  - target_max_xy_error_m: `0.9013`
  - structure_score: `-0.0803`
  - stack_height_m: `0.2613`
  - max_horizontal_drift_m: `0.0454`
  - velocity_inf_norm: `0.3844`
  - failure reasons: `missed_target=5`, `unstable_structure=2`, `low_or_fallen=1`
  - 截图目录：`batch_runs\20260621_4course_moon_supportv4_stonefit_poseriskv5_w035_medium_after_mlpfix\cap_w035`
- 判断：
  - `w=0.35` 显著优于 `w=0.20`，稳定数从 `12/24` 提高到 `16/24`，RMSE 从 `0.284` 降到 `0.217`，漂移从 `0.102` 降到 `0.045`。
  - 主要代价是 final velocity 偏高，说明需要更长 hold 或更强 settle 检查，而不是继续盲目提高 pose risk 权重。
  - 目前最有希望的组合是 support-map v4 + stone-fit + PoseRisk v5 + `pose_risk_weight=0.35`。

### Hold 与重力对照

- `batch_runs\20260621_4course_moon_supportv4_stonefit_poseriskv5_w035_hold1500_after_mlpfix`
  - gravity: `moon`
  - seed: `206213298`
  - hold_steps: `1500`
  - success / shape_success: `0 / 0`
  - stable_count / failure_count: `15 / 9`
  - target_rmse_xy_m: `0.8597`
  - target_max_xy_error_m: `3.9042`
  - max_horizontal_drift_m: `0.1306`
  - velocity_inf_norm: `0.0273`
  - 判断：速度降低，但形状退化；该 run 与 w=0.35 主结果 seed 不同，只作为补充。
- `batch_runs\20260621_4course_moon_supportv4_stonefit_poseriskv5_w035_hold1500_seed297_after_mlpfix`
  - gravity: `moon`
  - seed: `206213297`
  - hold_steps: `1500`
  - success / shape_success: `0 / 0`
  - stable_count / failure_count: `15 / 9`
  - target_rmse_xy_m: `0.2178`
  - target_max_xy_error_m: `0.9013`
  - max_horizontal_drift_m: `0.1346`
  - velocity_inf_norm: `0.0110`
  - 与同 seed hold=900 对比：
    - hold=900: stable `16`, RMSE `0.2165`, drift `0.0454`, velocity `0.3844`
    - hold=1500: stable `15`, RMSE `0.2178`, drift `0.1346`, velocity `0.0110`
  - 判断：长 hold 能压速度，但结构会继续水平移动，说明缺的是几何互锁/局部支撑，而不是单纯 settle 时间。
- `batch_runs\20260621_4course_earth_supportv4_stonefit_poseriskv5_w035_medium_after_mlpfix`
  - gravity: `earth`
  - seed: `206213297`
  - hold_steps: `900`
  - success / shape_success: `0 / 0`
  - stable_count / failure_count: `15 / 9`
  - target_rmse_xy_m: `0.1373`
  - target_max_xy_error_m: `0.3146`
  - max_horizontal_drift_m: `0.2034`
  - velocity_inf_norm: `0.1265`
  - 判断：地球重力下目标位置更紧，RMSE 明显优于月面，但水平漂移更大，仍未严格成功。
- 数据集：
  - `batch_runs\20260621_negative_mining_wall_dataset_v9_gravity_hold_compare`
  - 新增 hold 和 Earth/Moon 对照，用于后续分析重力对目标误差、漂移、速度的影响。

### 当前科学结论

- 当前最强 4 层组合：
  - support-map v4
  - stone-fit ranker
  - PoseRisk v5
  - `pose_risk_weight=0.35`
  - `candidates=3`, `steps_per_rock=180`, `hold_steps=900`
- 阶段性改进：
  - 相比 `w=0.20`，`w=0.35` 将 stable_count 从 `12/24` 提到 `16/24`。
  - RMSE 从 `0.284` 降到 `0.217`，漂移从 `0.102` 降到 `0.045`。
  - failure reasons 从 `missed_target=8, unstable=4` 改善到 `missed_target=5, unstable=2, low_or_fallen=1`。
- 仍未解决：
  - 严格 shape_success 仍为 0。
  - cap/middle 层仍是主要失败来源。
  - 长 hold 暴露出结构会继续滑移，说明下一步需要让网络/启发式显式学习“互锁与抗横向漂移”，而不是只优化落点误差。

### PoseRisk 权重继续扫描

- `batch_runs\20260621_4course_moon_supportv4_stonefit_poseriskv5_w050_medium_after_mlpfix`
  - pose_risk_weight: `0.50`
  - success / shape_success: `0 / 0`
  - stable_count / failure_count: `14 / 10`
  - target_rmse_xy_m: `0.1461`
  - target_max_xy_error_m: `0.2824`
  - max_horizontal_drift_m: `0.1387`
  - velocity_inf_norm: `0.0156`
  - 判断：几何误差最好，已经接近 4 层墙阈值，但稳定数和漂移不如 `w=0.35`。
- `batch_runs\20260621_4course_moon_supportv4_stonefit_poseriskv5_w042_medium_after_mlpfix`
  - pose_risk_weight: `0.42`
  - success / shape_success: `0 / 0`
  - stable_count / failure_count: `14 / 10`
  - target_rmse_xy_m: `0.2270`
  - target_max_xy_error_m: `0.9013`
  - max_horizontal_drift_m: `0.1207`
  - velocity_inf_norm: `0.0260`
  - 判断：没有成为 0.35 和 0.50 之间的折中，整体不如 0.35。
- 当前权重结论：
  - `w=0.35`：稳定性最好，`stable=16/24`，漂移最低。
  - `w=0.50`：目标几何最好，RMSE 最低，但结构互锁不足。
  - 后续需要多目标选择：不是单一权重继续扫，而是引入“稳定/互锁 critic”或把 final drift、support balance 作为二级筛选。
- 数据集：
  - `batch_runs\20260621_negative_mining_wall_dataset_v10_weight_sweep`
  - 包含 `w=0.35`, `w=0.42`, `w=0.50`, hold 和地球/月球重力对照。

### PoseRisk v7 权重扫描数据训练

- `batch_runs\20260621_pose_risk_net_candidate_metric_negative_mining_v7_weight_sweep`
  - dataset: `20260621_negative_mining_wall_dataset_v10_weight_sweep`
  - row_count: `20100`
  - risky positive_count: `12366`
  - test accuracy / F1: `0.7036 / 0.7681`
  - test top1 safe / top3 safe: `0.7164 / 1.0000`
- 判断：
  - v7 的 test top1 safe 低于 v5 的 `0.7724`。
  - 当前闭环不替换 v5；v7 只保留为纳入权重扫描后数据的训练记录。
  - 后续更合理的方向是训练二级 stable/drift critic，而不是继续让单一 PoseRisk 模型同时承担目标误差、速度和互锁判断。
