# 2026-06-19 闭环神经位姿排序实验记录

## 实验目的

本轮实验回答两个问题：

1. 神经网络能不能替代一部分启发式候选位姿搜索？
2. 神经网络闭环选择候选位姿后，能不能帮助堆更高的单面墙？

这里的“闭环”指：网络不再只做离线 imitation top-k 评估，而是在 MuJoCo 实验中真正参与候选筛选。每个石头/slot 先生成若干候选位姿，PyTorch ranker 只允许 top-k 候选进入 MuJoCo 仿真，然后仍由物理仿真结果在 top-k 内做最终选择。

## 关键修正：训练 pre-simulation ranker

之前的 support-map CNN 使用了 `candidate_score`、`support_overlap`、`support_contact_count`、`support_balance_error_m`、`bearing_pressure_proxy` 等字段。这些字段只有在候选位姿已经经过 MuJoCo 仿真后才知道，不能用于真实任务中的仿真前筛选。

因此本轮重新训练了一个 pre-simulation 模型，只使用放置前可获得的信息：

- 当前局部支撑/高度图；
- 候选位姿；
- 石头几何特征；
- 目标 slot 的 course/target；
- 重力条件。

模型路径：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_torch_support_map_cnn_ranker_wall_presim_v1`

离线 imitation 指标：

| 指标 | 数值 |
| --- | ---: |
| 候选样本 | 11,418 |
| 可排序候选组 | 2,793 |
| test top-1 recovery | 0.394 |
| test top-3 recovery | 0.902 |
| train top-1 recovery | 0.649 |
| train top-3 recovery | 0.958 |

解释：

- top-1 明显下降，这是合理的，因为模型不能再看仿真后指标。
- top-3 仍有 0.902，说明它适合做“候选缩减器”：从 4-6 个候选里保留前 2-3 个，再交给 MuJoCo 物理验证。

## 代码改动

新增/修改：

- `scripts/train_torch_support_map_ranker.py`
  - 增加 `--exclude-postsim-features`。
  - 支持训练不含仿真后泄漏字段的 pre-sim ranker。
- `moon_rock_stack/run_structured_experiment.py`
  - `--candidate-pose-ranker-dir` 现在可自动识别 PyTorch `support_map_cnn_ranker.pt`。
- `moon_rock_stack/structured.py`
  - 在线渲染局部 support map。
  - 在线构造 pre-sim 数值特征。
  - PyTorch CNN 在候选仿真前给出 ranker score。
  - 已放置石头缓存 `last_center/last_top/last_quat`，用于下一步局部支撑图重建。

## Smoke 验证

路径：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_neural_ranker_smoke_presim_top2`

配置：

- target: `single_face_wall_2course_v1`
- strategy: `statics_wall`
- gravity: Earth
- candidates: 4
- neural top-k: 2

结果：

- 实验跑通；
- `candidate_pose_log.csv` 正常写出 `ranker_prob`、`ranker_rank`、`ranker_top_k`；
- 每组候选只仿真 top-2，说明闭环筛选有效。

## 4 层单面墙闭环实验

使用旧的 targeted wall-statistics catalog：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260618_targeted_rock_catalog_wall_statics_v1`

反推并确认 catalog seed 为 `79`，参数为：

- rocks: 480
- profile: `wall_statics`
- clusters: 12
- target: `single_face_wall_4course_v1`
- strategy: `wall_bonded`
- assignment plan: `assignment_plan_single_face_wall_4course_v1.csv`
- role screening: `role_screening.csv`

### Full6 baseline partial

路径：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_closed_loop_wall4_baseline_full6_v1`

配置：

- candidates: 6
- fallback: 5
- ranker: none
- full search: 仿真所有候选

结果：

| Gravity | Completed | Success | Visible courses | Height m | RMSE m | Max drift m | Wall y span m | Structure score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | yes | 0 | 4 | 0.289 | 0.125 | 0.158 | 0.403 | 1.173 |
| Moon | no, timeout | - | - | - | - | - | - | - |

说明：

- Earth 完成，可见 4 层，但严格失败。
- Earth+Moon 总命令 20 分钟超时，Moon 没写出结果。
- 这说明 full candidate baseline 成本过高。

### Neural top3

路径：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_closed_loop_wall4_neural_presim_top3_v1`

配置：

- candidates: 6
- fallback: 5
- ranker: pre-sim support-map CNN
- top-k: 3

结果：

| Gravity | Success | Visible courses | Height m | RMSE m | Max drift m | Wall y span m | Structure score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 0 | 4 | 0.225 | 0.113 | 0.090 | 0.348 | 1.631 |
| Moon | 0 | 4 | 0.349 | 0.077 | 0.001 | 0.089 | 2.372 |

解释：

- neural top-3 完整跑完 Earth 和 Moon。
- 两者都达到 4 个可见 course，但严格成功仍为 0。
- 相比 full6 baseline Earth，neural top-3 的 RMSE、漂移和 wall y span 更好，但高度较低。
- Moon 下结构很窄、漂移极小，但仍未通过严格成功判定。

### Full4 baseline partial

路径：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_closed_loop_wall4_baseline_full4_fallback3_v1`

配置：

- candidates: 4
- fallback: 3
- ranker: none

结果：

| Gravity | Completed | Success | Visible courses | Height m | RMSE m | Max drift m | Wall y span m | Structure score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | yes | 0 | 4 | 0.206 | 0.199 | 0.208 | 0.717 | -0.266 |
| Moon | no, timeout | - | - | - | - | - | - | - |

说明：

- 即使 candidates=4/fallback=3，baseline Earth+Moon 仍然 20 分钟超时。
- Earth 结果比 neural top-2 明显差，尤其 RMSE、漂移、wall y span。

### Neural top2

路径：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_closed_loop_wall4_neural_presim_top2_fallback3_v1`

配置：

- candidates: 4
- fallback: 3
- ranker: pre-sim support-map CNN
- top-k: 2

结果：

| Gravity | Success | Visible courses | Height m | RMSE m | Max drift m | Wall y span m | Structure score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 0 | 4 | 0.295 | 0.080 | 0.006 | 0.138 | 2.412 |
| Moon | 0 | 4 | 0.238 | 0.063 | 0.000 | 0.085 | 2.471 |

与 full4 baseline Earth 对比：

| Metric | Full4 baseline Earth | Neural top2 Earth |
| --- | ---: | ---: |
| Candidate top-k | all 4 | top 2 |
| Success | 0 | 0 |
| Visible courses | 4 | 4 |
| Stable count | 9 | 11 |
| Failure count | 7 | 0 |
| Height m | 0.206 | 0.295 |
| RMSE m | 0.199 | 0.080 |
| Max drift m | 0.208 | 0.006 |
| Wall y span m | 0.717 | 0.138 |
| Wall aspect | 1.207 | 5.139 |
| Structure score | -0.266 | 2.412 |

阶段结论：

- 在 Earth 4 层墙上，neural top-2 用一半候选仿真量，得到更好的墙形、漂移、稳定计数和结构分。
- 仍未达到严格 success，因为当前判定对全 slot 完成、稳定数、墙高、形状误差都很严格。
- Moon baseline 目前成本过高，没有完整 baseline 结果；但 neural top-2 能完整跑完并保持 4 层可见。

## High wall 神经闭环实验

路径：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_closed_loop_highwall_neural_presim_top2_v1`

配置：

- target: `single_face_wall_high_v1`
- strategy: `statics_wall`
- rocks: 120
- candidates: 4
- neural top-k: 2
- assignment plan: none

结果：

| Gravity | Success | Visible courses | Stable count | Failure count | Height m | RMSE m | Max drift m | Wall y span m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth | 0 | 7 | 15 | 9 | 0.263 | 0.209 | 0.470 | 0.538 |
| Moon | 0 | 4 | 11 | 5 | 0.143 | 0.139 | 0.139 | 0.467 |

解释：

- Earth high wall 达到 7 个可见 course，但不是严格成功，主要问题是漂移和较大 wall y span。
- Moon high wall 只保持 4 个可见 course，高度和稳定数都低于 Earth。
- 这说明神经 ranker 能把高墙“推到更高层可见”，但还不能保证高层结构稳定。

## 候选仿真日志规模

| Run | Candidate pose rows |
| --- | ---: |
| full6 baseline partial | 372 |
| neural top3 | 435 |
| full4 baseline partial | 168 |
| neural top2 | 248 |
| highwall neural top2 | 744 |

注意：

- 这些行数不是理论候选总数，因为 assignment gate、fallback、slot skip 会改变实际尝试数。
- baseline partial 只完成 Earth，所以不能直接和 neural Earth+Moon 总行数比较。
- 但运行时间已经说明，baseline full search 在 Moon 上成本明显过高；neural top-k 能完成同一批任务。

## 图片输出

4 层墙 neural top2：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_closed_loop_wall4_neural_presim_top2_fallback3_v1\captures_960x720`

High wall neural top2：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260619_closed_loop_highwall_neural_presim_top2_v1\captures_960x720`

每个 case 包含：

- front RGB/depth；
- wall-front RGB/depth；
- left/right/back RGB/depth；
- top RGB/depth；
- wall-top depth。

## 当前回答

### 1. 能不能替代启发式搜索？

可以替代一部分，但不能完全替代。

当前最可靠的用途是：

> 用神经网络做候选位姿缩减，从 4-6 个候选里选 top-2/top-3，再交给 MuJoCo 物理验证。

本轮最直接证据是 4 层墙 Earth：

- baseline full4 仿真全部候选，RMSE 0.199 m、max drift 0.208 m、wall y span 0.717 m；
- neural top2 只仿真一半候选，RMSE 0.080 m、max drift 0.006 m、wall y span 0.138 m。

所以它已经有实际用途：减少候选仿真成本，并改善部分墙形指标。

但它还不能完全替代 dry-stacking 启发式，因为：

- 仍依赖启发式生成候选位姿；
- 仍需要 MuJoCo 在 top-k 内做物理验证；
- 严格 success 还没有达到；
- Moon baseline 还没有完整跑完，统计样本太少。

### 2. 能不能堆更高的单面墙？

能把 high wall 推到更高可见层数，但还不能稳定成功。

本轮 high wall Earth：

- visible courses = 7；
- stack height = 0.263 m；
- 但 failure_count = 9，max drift = 0.470 m，严格 success = 0。

这说明网络确实能把结构堆到高层可见，但高层漂移和结构约束还没有解决。

## 下一步

1. 继续用 pre-sim ranker，不再使用带仿真后字段的模型做闭环筛选。
2. 对 4 层墙跑更多 trial，优先 `neural top2/top3`，baseline 只保留低成本对照。
3. 给 high wall 增加 assignment plan 或 role-aware catalog，避免高层随机构件选择过强。
4. 训练 success-aware ranker：标签不再只是 imitation，而是结合最终稳定、墙高、漂移、RMSE。
5. 增加 repair policy：当某层 slot 失败时，不是跳过，而是选择 chock/lock stone 做修复。
6. PointNet 不直接拼接 frozen embedding；下一步训练 affordance head，预测 base/middle/cap 适配性、接触面质量和漂移风险。
