# 并行资源调度实验记录

日期：2026-06-19

目的：在不让桌面系统失去响应的前提下，提高 CPU、GPU、显存和内存利用率，并同时推进高墙数据采集与小网络消融训练。

## 1. 资源策略

硬件条件：

- 内存：约 64 GB
- GPU：NVIDIA GeForce RTX 2080 Ti，显存 11 GB
- CPU 逻辑处理器：8

执行策略：

- 不追求把资源打到 100% 卡死，而是高利用并保留系统余量。
- CPU 侧并行运行 2 个 MuJoCo 全候选高墙数据采集任务。
- GPU 侧并行运行 3 个 PyTorch ranker 消融训练任务。
- 所有任务写入独立目录和日志，不删除任何旧数据。

调度脚本：

- `scripts/launch_parallel_modular_jobs_20260619.py`

队列 manifest：

- `batch_runs/parallel_jobs_20260619_modular_highwall_foreground_v1/jobs_manifest.json`

状态日志：

- `batch_runs/parallel_jobs_20260619_modular_highwall_foreground_v1/status.jsonl`

## 2. 实际资源占用

前台监督器从 15:57:44 运行到 16:31:19，5 个任务全部成功。

GPU 高负载阶段：

- 显存占用约 10.26 GB / 11.26 GB
- 空闲显存约 0.73 到 0.76 GB
- GPU utilization 约 70% 到 86%

后半段：

- 3 个 PyTorch 训练任务完成后，GPU 释放到约 0.33 GB 使用。
- 2 个 MuJoCo 数据任务继续占用 CPU，直到全部结束。

最终状态：

- running：0
- done：5
- failed：0

## 3. 并行任务清单

### 3.1 CPU / MuJoCo 全候选数据采集

任务 1：

- 名称：`cpu_fullcandidate_highwall_seed93014`
- 输出：`batch_runs/20260619_newrocks_highwall_seed93014_fullcandidates_highonly_parallel_v1`
- rock profile：`high_wall`
- seed：93014
- rocks：220
- target：`single_face_wall_high_v1`
- gravities：earth, moon
- candidates：5
- workers：2

任务 2：

- 名称：`cpu_fullcandidate_screening_seed93015`
- 输出：`batch_runs/20260619_newrocks_screening_seed93015_fullcandidates_highonly_parallel_v1`
- rock profile：`screening_stress`
- seed：93015
- rocks：220
- target：`single_face_wall_high_v1`
- gravities：earth, moon
- candidates：5
- workers：2

### 3.2 GPU / PyTorch 小网络消融训练

任务 3：

- 名称：`gpu_quality_temp20_dropout30`
- 输出：`batch_runs/20260619_torch_support_map_cnn_quality_ranker_temp20_dropout30_holdout93013_v1`
- target mode：`score`
- quality temperature：20
- dropout：0.30
- hidden：160
- holdout：`20260619_newrocks_highwall_seed93013_fullcandidates_highonly_fg_v1`

任务 4：

- 名称：`gpu_quality_temp55_dropout20`
- 输出：`batch_runs/20260619_torch_support_map_cnn_quality_ranker_temp55_dropout20_holdout93013_v1`
- target mode：`score`
- quality temperature：55
- dropout：0.20
- hidden：160
- holdout：`20260619_newrocks_highwall_seed93013_fullcandidates_highonly_fg_v1`

任务 5：

- 名称：`gpu_selected_dropout30_hidden160`
- 输出：`batch_runs/20260619_torch_support_map_cnn_selected_ranker_dropout30_hidden160_holdout93013_v1`
- target mode：`selected`
- dropout：0.30
- hidden：160
- holdout：`20260619_newrocks_highwall_seed93013_fullcandidates_highonly_fg_v1`

## 4. 新数据采集结果

### 4.1 high_wall / seed 93014

输出目录：

- `batch_runs/20260619_newrocks_highwall_seed93014_fullcandidates_highonly_parallel_v1`

新增候选位姿：

- `candidate_pose_log.csv`：2480 行

结果：

| 重力 | 严格成功 | 可见层 | 稳定石头 | 失败数 | 高度 m | RMSE m | 结构分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 地球 | 0 | 7 | 18 | 7 | 0.291 | 0.174 | 0.841 |
| 月球 | 0 | 8 | 11 | 7 | 0.267 | 0.208 | 0.248 |

### 4.2 screening_stress / seed 93015

输出目录：

- `batch_runs/20260619_newrocks_screening_seed93015_fullcandidates_highonly_parallel_v1`

新增候选位姿：

- `candidate_pose_log.csv`：2480 行

结果：

| 重力 | 严格成功 | 可见层 | 稳定石头 | 失败数 | 高度 m | RMSE m | 结构分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 地球 | 0 | 7 | 14 | 7 | 0.236 | 0.211 | 0.418 |
| 月球 | 0 | 8 | 15 | 8 | 0.298 | 0.201 | 0.079 |

解释：

- 两个新目录都能达到 7 到 8 个可见层，但严格成功仍为 0。
- 月球条件下可见层数较高，但结构分偏低，说明高墙容易形成“看起来高但形状偏移/稳定性不足”的状态。
- 这 4960 条全候选样本适合加入下一版训练集，尤其适合训练失败预测器和 quality ranker。

## 5. 小网络消融结果

共同训练集：

- `batch_runs/20260619_local_support_maps_multicatalog_wall_plus93013_v1`
- row_count：15395
- rankable groups：3269
- holdout：`20260619_newrocks_highwall_seed93013_fullcandidates_highonly_fg_v1`

### 5.1 quality temp20 / dropout30

模型：

- `batch_runs/20260619_torch_support_map_cnn_quality_ranker_temp20_dropout30_holdout93013_v1`

| 指标 | 数值 |
| --- | ---: |
| test top-1 | 0.302 |
| test top-3 | 0.710 |
| train top-1 | 0.433 |
| train top-3 | 0.823 |
| test top-1 quality regret | 18.045 |
| test top-3 quality regret | 4.700 |

分重力：

| 条件 | top-1 | top-3 | top-1 regret | top-3 regret |
| --- | ---: | ---: | ---: | ---: |
| 地球高墙 | 0.286 | 0.750 | 21.528 | 5.394 |
| 月球高墙 | 0.319 | 0.669 | 14.563 | 4.006 |

### 5.2 quality temp55 / dropout20

模型：

- `batch_runs/20260619_torch_support_map_cnn_quality_ranker_temp55_dropout20_holdout93013_v1`

| 指标 | 数值 |
| --- | ---: |
| test top-1 | 0.304 |
| test top-3 | 0.702 |
| train top-1 | 0.413 |
| train top-3 | 0.807 |
| test top-1 quality regret | 19.291 |
| test top-3 quality regret | 5.016 |

### 5.3 selected dropout30 / hidden160

模型：

- `batch_runs/20260619_torch_support_map_cnn_selected_ranker_dropout30_hidden160_holdout93013_v1`

| 指标 | 数值 |
| --- | ---: |
| test top-1 | 0.315 |
| test top-3 | 0.698 |
| train top-1 | 0.362 |
| train top-3 | 0.785 |

## 6. 阶段判断

这轮并行实验说明：

- GPU 可以安全地同时跑 3 个小模型训练，显存基本吃满但没有 OOM。
- CPU 可以同时跑 2 个 MuJoCo 全候选高墙任务，任务全部正常结束。
- `quality_temperature=20, dropout=0.30` 是当前 quality ranker 消融里较好的版本，93013 holdout top-3 达到 0.710。
- 新增的 93014 和 93015 全候选目录很有价值，因为它们都是未截断候选数据，适合并入下一轮训练。

下一步：

1. 用旧数据 + 93013 + 93014 + 93015 重建更大的 learning dataset。
2. 重新导出 support maps。
3. 训练新的 quality ranker 和 failure predictor，并用一个完全新 seed 做闭环验证。
4. 如果 GPU 空闲，可以继续用 2 到 3 个并行训练任务做温度、dropout、hidden size 消融；CPU 继续生成不同 rock profile 的全候选高墙数据。
