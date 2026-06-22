# 2026-06-19 新石头泛化与高墙闭环实验

## 为什么做这轮实验

上一轮闭环实验说明，pre-sim support-map CNN 可以把候选位姿从 4-6 个缩减到 top-2/top-3，并在 4 层墙上改善 RMSE、漂移和墙面纵深。

但存在一个关键风险：

> 网络可能记住了少数 catalog 里的石头，而不是学到了石头几何、局部支撑和目标 slot 之间的关系。

因此本轮目标是生成更多不同 seed/profile 的石头，并用完整 run/catalog 留出测试来检验泛化。

## 新生成的石头数据

### 新 catalog A：high_wall profile

路径：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_newrocks_highwall_seed93011_fullcandidates_v1`

配置：

- rocks: 180
- profile: `high_wall`
- seed: 93011
- targets:
  - `single_face_wall_4course_v1`
  - `single_face_wall_high_v1`
- gravity:
  - Earth
  - Moon
- candidates: 4
- ranker: none
- 目的：完整候选采集，不做 neural top-k 截断。

结果摘要：

| Target | Gravity | Success | Visible courses | Height m | RMSE m | Max drift m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4-course wall | Earth | 0 | 4 | 0.344 | 0.167 | 0.218 |
| 4-course wall | Moon | 0 | 4 | 0.303 | 0.171 | 0.324 |
| high wall | Earth | 0 | 8 | 0.278 | 0.181 | 0.012 |
| high wall | Moon | 0 | 8 | 0.208 | 0.203 | 0.289 |

候选位姿日志：

- `candidate_pose_log.csv`: 2,640 行

### 新 catalog B：single_face_wall profile

路径：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_newrocks_singleface_seed93012_fullcandidates_v1`

配置：

- rocks: 180
- profile: `single_face_wall`
- seed: 93012
- targets:
  - `single_face_wall_4course_v1`
  - `single_face_wall_high_v1`
- gravity:
  - Earth
  - Moon
- candidates: 4
- ranker: none

结果摘要：

| Target | Gravity | Success | Visible courses | Height m | RMSE m | Max drift m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4-course wall | Earth | 0 | 4 | 0.240 | 0.254 | 0.261 |
| 4-course wall | Moon | 0 | 4 | 0.258 | 0.193 | 0.262 |
| high wall | Earth | 0 | 7 | 0.258 | 0.191 | 0.491 |
| high wall | Moon | 0 | 7 | 0.238 | 0.192 | 0.075 |

候选位姿日志：

- `candidate_pose_log.csv`: 2,640 行

## 多 catalog 数据集

新建学习数据集：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_learning_dataset_multicatalog_presim_wall_v1`

构成：

- 11 个 run；
- 688 条 committed placement；
- 12,915 条 candidate pose；
- 新生成两批石头贡献 5,280 条 candidate pose；
- wall targets:
  - `single_face_wall_2course_v1`: 5,742 条；
  - `single_face_wall_4course_v1`: 3,577 条；
  - `single_face_wall_high_v1`: 3,596 条。

导出的 support-map 张量：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_local_support_maps_multicatalog_wall_v1`

张量规模：

| Item | Value |
| --- | ---: |
| Candidate examples | 12,915 |
| Shards | 7 |
| Map shape | `[8,64,64]` |
| dtype | float16 |

## 防记忆训练：按 run/catalog 留出

### 随机 run 留出版本

路径：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_torch_support_map_cnn_ranker_multicatalog_presim_splitrun_v1`

这个版本按完整 run 随机留出测试集，但随机抽到的是旧 run：

Test runs:

- `20260618_candidate_pose_single_face_wall4c_moon_c4_minibatch`
- `20260618_high_single_face_wall_v1_earth_fast`

结果：

| Metric | Value |
| --- | ---: |
| Train groups | 2,594 |
| Test groups | 179 |
| Test top-1 | 0.497 |
| Test top-3 | 0.983 |

解释：

- 按 run 留出仍能保持很高 top-3。
- 但这个版本没有专门把新生成的石头留出，因此还不能说明新石头泛化。

### 新石头显式 holdout 版本

路径：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_torch_support_map_cnn_ranker_multicatalog_presim_newrocks_holdout_v1`

显式测试 run：

- `20260619_newrocks_highwall_seed93011_fullcandidates_v1`
- `20260619_newrocks_singleface_seed93012_fullcandidates_v1`

训练集不包含这两批石头。

结果：

| Metric | Value |
| --- | ---: |
| Train groups | 1,453 |
| Test groups | 1,320 |
| Test top-1 | 0.351 |
| Test top-3 | 0.808 |
| Train top-1 | 0.374 |
| Train top-3 | 0.792 |

按任务拆分：

| Target | Gravity | Groups | Top-1 | Top-3 |
| --- | --- | ---: | ---: | ---: |
| 4-course wall | Earth | 288 | 0.396 | 0.847 |
| 4-course wall | Moon | 288 | 0.399 | 0.826 |
| high wall | Earth | 372 | 0.298 | 0.774 |
| high wall | Moon | 372 | 0.331 | 0.796 |

解释：

- 当测试集是完全没见过的新石头 catalog 时，top-3 从随机/旧 run 留出的 0.98 降到 0.81。
- 这说明之前确实存在一定 catalog 依赖。
- 但 top-3 仍然高于随机选择，说明模型学到了一部分可迁移的支撑/几何规律。
- high wall 比 4-course wall 更难泛化，尤其 Earth high wall top-3 只有 0.774。

## 全新 seed 高墙闭环验证

为了进一步检查泛化能力，用完全新的 seed=94013 生成高墙石头，并使用新石头 holdout 版本 ranker 做闭环：

路径：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_closed_loop_highwall_newrocks_seed94013_neural_top3_v1`

配置：

- rocks: 160
- profile: `high_wall`
- seed: 94013
- target: `single_face_wall_high_v1`
- strategy: `statics_wall`
- candidates: 4
- neural top-k: 3
- ranker: `20260619_torch_support_map_cnn_ranker_multicatalog_presim_newrocks_holdout_v1`

结果：

| Gravity | Success | Visible courses | Stable count | Failure count | Height m | RMSE m | Max drift m | Wall y span m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 0 | 5 | 10 | 4 | 0.211 | 0.219 | 0.548 | 0.647 |
| Moon | 0 | 8 | 9 | 8 | 0.215 | 0.200 | 0.339 | 0.605 |

解释：

- 在全新 seed 的没见过石头上，Moon 达到 8 个 visible courses。
- Earth 只达到 5 个 visible courses，且横向漂移很大。
- 两者 strict success 仍为 0。
- 这说明当前网络已经具备一定“堆高可见层数”的泛化能力，但还没有学会高墙稳定性。

## 图片输出

全新 seed high wall neural top3：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_closed_loop_highwall_newrocks_seed94013_neural_top3_v1\captures_960x720`

包含：

- wall-front RGB/depth；
- wall-top RGB/depth；
- front/left/right/back/top RGB/depth。

典型案例：

- Moon high wall failure:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_closed_loop_highwall_newrocks_seed94013_neural_top3_v1\captures_960x720\00_single_face_wall_high_v1_failure_statics_wall_moon_trial_00`

- Earth high wall failure:

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_closed_loop_highwall_newrocks_seed94013_neural_top3_v1\captures_960x720\01_single_face_wall_high_v1_failure_statics_wall_earth_trial_00`

## 当前科学判断

### 1. 网络没有只是在记石头，但泛化还不够强

证据：

- 新石头显式 holdout 测试集 top-3 = 0.808；
- 训练集 top-3 = 0.792；
- 说明模型并不是简单记住训练 catalog；
- 但 top-3 相比旧 run 留出的 0.983 明显下降，说明 catalog 分布变化仍然很影响模型。

### 2. 生成更多石头是必要的

新增两批石头后，模型对新 catalog 的评估更真实。接下来应继续生成更多：

- 不同 seed；
- 不同 profile；
- 不同 rock count；
- 更多 high_wall profile；
- 更多 wall_statics profile；
- 少量 screening_stress profile，用于增加困难负样本。

### 3. 当前网络能帮助“堆高可见层”，但不能保证高墙稳定

证据：

- 新 seed=94013 的 Moon high wall 达到 8 visible courses；
- 但 failure_count=8，max drift=0.339 m，strict success=0。

所以目标应从 imitation ranker 转向 success-aware/risk-aware ranker。

## 下一步建议

1. 继续生成更多新石头 catalog，至少扩展到 8-12 个不同 seed/profile。
2. 训练时强制按 catalog split，不能再只用随机 group split。
3. 给每个候选/placement 加 success-aware 标签：
   - final visible courses；
   - final stack height；
   - final wall y span；
   - max drift；
   - failure_count；
   - target RMSE；
   - gravity-specific failure。
4. 在 high wall 中加入 role-aware assignment plan，减少石头选择随机性，让 ranker 专注位姿。
5. 对高墙引入 repair policy：当上层失败或漂移过大时，自动选择 chock/lock stone 修复，而不是继续向上堆。
6. PointNet 继续做 affordance head，而不是 frozen embedding 拼接。
