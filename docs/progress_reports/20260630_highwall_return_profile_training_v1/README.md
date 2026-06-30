# 2026-06-30 high_wall 多面体石头堆叠与神经网络训练记录

## 实验目的

本轮目标是回到 `20260621_wall_flywheel_3course_stoneslot_v2_eval` 成功案例对应的 `high_wall` 多面体石头生成方法，而不是 NASA-like 或其它新石头风格。重点任务是：

1. 继续月面单面墙 3/4 层堆叠实验。
2. 用新采集数据训练小网络，让启发式搜索逐步退到 Top-K 候选保留和安全先验的位置。
3. 记录失败负样本，尤其是中层和盖层的 `missed_target + post_hold_drift`。
4. 自动安排下一轮 4 层 focused eval，避免人工等待主飞轮结束。

## 数据流

当前主飞轮会按以下顺序运行：

```text
high_wall 石头生成
  -> mixed explore/exploit 3/4 层月面采样
  -> build_learning_dataset
  -> MuJoCo support-map / depth-map 导出
  -> StoneSlotNet / PoseRiskNet / support-map pose ranker / WallStateCritic 训练
  -> 新模型闭环评估
  -> 典型案例截图
```

主会话：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260629_highwall_return_profile_v2`

学习数据集：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260629_highwall_return_profile_v2_learning_dataset`

数据规模：

| 数据表 | 样本数 | 用途 |
|---|---:|---|
| `placement_examples.csv` | 1362 | 已提交放置样本，用于放置成功/失败、角色、层级统计 |
| `candidate_pose_examples.csv` | 45822 | 候选位姿级样本，用于 PoseRiskNet 和候选位姿排序 |
| `assignment_candidate_examples.csv` | 12933 | 石头-槽位候选样本，用于 StoneSlotNet |
| `run_examples.csv` | 68 | run 级摘要 |

按重力的 placement 样本：

| 重力 | examples | success | failure | skipped |
|---|---:|---:|---:|---:|
| earth | 714 | 410 | 93 | 211 |
| moon | 648 | 388 | 93 | 167 |

按角色的 placement 样本：

| 角色 | examples | success | failure | skipped |
|---|---:|---:|---:|---:|
| base | 446 | 351 | 23 | 72 |
| middle | 606 | 335 | 88 | 183 |
| cap | 310 | 112 | 75 | 123 |

结论：当前主要难点不是底层，而是 middle/cap。cap 的成功样本稀疏且失败/跳过比例高，所以后续网络和损失权重必须对中高层更敏感。

## 主飞轮训练结果

### StoneSlotNet

输出：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260629_highwall_return_profile_v2_stone_slot_net`

输入：

- 石头几何特征：体积、表面积、面数、bbox、elongation、flatness、roughness、angularity、spike、rectangularity、roundness proxy、concavity proxy、support face 等。
- 槽位/角色：course、target_x、target_y、role、target_name。
- 类别：rock_source_kind、rock_cluster_label。

输出：

- `stone_fit_prob`，表示某块石头适合某个目标槽位的概率。

测试指标：

| 指标 | 数值 |
|---|---:|
| accuracy | 0.687 |
| precision | 0.116 |
| recall | 0.500 |
| F1 | 0.188 |
| group top1 | 0.244 |
| group top3 | 0.474 |

判断：StoneSlotNet 还不能单独替代启发式选石；它适合做 Top-K 缩小候选池。纯石头几何 + slot 角色不够，后续需要加入当前墙体局部观测。

### role-balanced StoneSlotNet

补充训练：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_highwall_tabular_train_v1_stone_slot_net_rolebalanced`

改动：

- `--role-balance`
- `middle=1.2`
- `cap=1.8`

测试指标：

| 指标 | 主飞轮 StoneSlotNet | role-balanced StoneSlotNet |
|---|---:|---:|
| group top1 | 0.244 | 0.286 |
| group top3 | 0.474 | 0.508 |
| F1 | 0.188 | 0.197 |

判断：role balancing 有小幅提升，说明中高层样本权重是有用规则；但提升幅度不足，不能说明 StoneSlotNet 已经学会结构性选石。

### PoseRiskNet

主飞轮输出：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260629_highwall_return_profile_v2_pose_risk_net`

补充 CUDA 输出：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_highwall_tabular_train_v2_pose_risk_net_candidate_metric_cuda`

输入：

- 重力、course、target、candidate pose、石头几何特征、role、source_kind、cluster_label。
- 不输入后验成功率，不输入测试后统计得到的成功概率。

标签：

- 使用候选自身 post-simulation metrics 构造风险标签，例如 target error、y error、disturbance、velocity。
- 这些 post-simulation metrics 只作为监督标签，不作为网络输入。

主飞轮 PoseRiskNet 指标：

| 指标 | 数值 |
|---|---:|
| test accuracy | 0.692 |
| test F1 | 0.786 |
| group top1 safe | 0.682 |
| group top3 safe | 0.958 |

v2 CUDA PoseRiskNet 指标：

| 指标 | 数值 |
|---|---:|
| test accuracy | 0.700 |
| test F1 | 0.790 |
| group top1 safe | 0.708 |
| group top3 safe | 1.000 |

判断：PoseRiskNet 是本轮最可靠的小网络。它不适合作为唯一 top1 贪心选择器，但非常适合作为 Top-3 安全候选保留/风险惩罚器。

### support-map pose ranker

输出：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260629_highwall_return_profile_v2_pose_ranker_structure`

输入：

- MuJoCo support-map / depth-map 张量，形状 `10 x 64 x 64`。
- 数值特征包括 course、target、candidate pose、candidate_score、石头几何、support overlap、support contact、balance error、bearing pressure proxy。
- 使用 `--exclude-postsim-features`，避免把测试后验作为输入。

测试指标：

| 指标 | 数值 |
|---|---:|
| test top1 hit | 0.304 |
| test top3 hit | 0.904 |
| moon 4-course top1 | 0.257 |
| moon 4-course top3 | 0.952 |

判断：support-map ranker 的 top3 很有价值，top1 仍不稳定。当前策略应继续使用 `candidate_pose_top_k=3`，而不是让网络独断输出唯一位姿。

### WallStateCritic

输出：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260629_highwall_return_profile_v2_wall_state_critic`

测试指标：

| 指标 | 数值 |
|---|---:|
| classification accuracy | 0.888 |
| classification F1 | 0.940 |
| group top1 | 0.328 |
| group top3 | 0.903 |

注意：正类比例约 `0.984`，分类指标容易虚高。当前 WallStateCritic 更适合作为辅助解释器或二级风险特征，不能直接作为主控制器。

## 当前闭环评估状态

主飞轮闭环评估：

- 输出：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260629_highwall_return_profile_v2_closed_loop_eval`
- 当前已经开始 `single_face_wall_3course_v1 | moon`。
- 前两个 3 层 trial 严格成功为 0。
- 主要失败类型：`missed_target + post_hold_drift`。

前两个 3 层 trial 摘要：

| trial | stable_count | failure_count | skipped_slot_count | visible_courses | strict success |
|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 2 | 3 | 3 | 0 |
| 1 | 9 | 4 | 2 | 3 | 0 |

代表性失败：

- middle: `tie_bridge_clast`, `buttress_clast`, `bearing_block_clast` 出现较大 drift。
- cap: `bearing_block_clast`, `wall_block_clast` 出现 missed target 和 post-hold drift。

当前判断：

- 网络已经能参与候选筛选，但没有完全解决月面低重力下的落点漂移。
- 继续单纯增高没有意义，下一步应先提高 3/4 层稳定率，尤其是 middle/cap 的目标误差和 hold 后漂移。

## 失败记录

两个训练入口发生 Windows 原生崩溃，没有 Python traceback：

| 入口 | return code | 处理 |
|---|---:|---|
| `train_candidate_pose_group_ranker` | 3228369023 | 保留失败记录；当前改用已成功的 support-map pose ranker |
| `train_modular_stack_models` | 3228369023 | 保留失败记录；后续需要改写为 PyTorch 或避免 NumPy/BLAS 原生路径 |

这不是数据删除或实验失败，而是工程层面的负样本：当前 Windows + NumPy 小网络入口不稳定，后续应优先使用 PyTorch 版本或已有 safe matmul 实现。

## 自动后续任务

已启动等待型 watcher：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_highwall_followup_watcher_v1`

它会等待当前主飞轮完成，然后自动启动：

- `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260630_highwall_followup_4course_rolebalanced_eval_v1`

后续 4 层 focused eval 配置：

| 参数 | 值 |
|---|---|
| target | `single_face_wall_4course_v1` |
| gravity | moon |
| rocks | 180 |
| trials | 2 |
| candidates | 10 |
| stone selector | role-balanced StoneSlotNet，失败则回退主飞轮 StoneSlotNet |
| pose ranker | 主飞轮 support-map ranker |
| risk net | v2 CUDA PoseRiskNet，失败则回退主飞轮 PoseRiskNet |
| pose risk weight | 0.30 |
| stone top-k | 18 |
| pose top-k | 3 |
| low release | enabled |
| base support prior | 1.45 |
| base continuity prior | 0.50 |

## 阶段结论

1. `high_wall` 多面体石头生成方法已经恢复并进入新的数据飞轮。
2. 本轮新增数据足够训练小网络，但 StoneSlotNet 仍偏弱，说明只看石头和 slot 不够。
3. PoseRiskNet 和 support-map pose ranker 的 top3 指标明显更有用，应继续作为神经网络接入主线。
4. 当前失败集中在 middle/cap 的落点误差和 post-hold drift，下一轮应该围绕这两个问题做 4 层 focused eval 和数据采集。
5. 启发式不应立刻完全退出；正确策略是保留低释放、base support、base continuity 等物理先验，同时让网络逐步替代候选筛选和风险排序。
