# 低层单面墙预仿真几何网络实验记录 2026-06-20

## 目标

本轮目标不是继续盲目堆更高，而是先把 2-3 层单面墙的成功数据收集和神经网络替代启发式搜索跑通。网络输入必须是放置前可获得的信息：石头几何、目标槽位、候选姿态、当前支承高度图和重力；仿真后的成功率、误差、扰动、速度只能作为标签或评估指标。

## 新增预仿真几何特征

`moon_rock_stack/features.py` 增加了从 OBJ 网格直接计算的多面体几何先验：

- `major_face_count`: 面法向聚类后的主面数量。
- `largest_face_area_ratio`: 最大主面面积占比。
- `top3_face_area_ratio`: 前 3 个主面面积占比。
- `face_area_entropy`: 面面积分布熵。
- `normal_concentration`: 面法向/面积集中度。
- `support_face_count`: 可作为支承面的主面数量。
- `support_face_area_ratio`: 支承主面总面积占比。
- `opposing_face_pair_count`: 近似对置主面数量。
- `opposing_face_area_ratio`: 最强对置主面的面积占比。
- `face_planarity`: 主面内部法向一致性。
- `support_plane_quality`: 支承面质量的综合几何先验。

这些特征只来自石头网格本身，不来自任何测试结果。

## 数据与模型

### 训练数据

- 低层墙学习表：`batch_runs/20260620_learning_dataset_lowwall_presim_geometry_v1`
  - run 数：4
  - placement 样本：429
  - candidate-pose 样本：23,274
- CNN 张量：`batch_runs/20260620_local_support_maps_lowwall_presim_geometry_v1`
  - 样本：20,634
  - 目标：`wall_segment_v1` 和 `single_face_wall_2course_v1`
  - 局部图：8 通道，64x64
- 闭环回放合并学习表：`batch_runs/20260620_learning_dataset_lowwall_presim_geometry_plus_closedloop_v1`
  - placement 样本：645
  - candidate-pose 样本：28,458

### Pose CNN Ranker

路径：`batch_runs/20260620_torch_support_map_cnn_lowwall_presim_score_v1`

输入：

- 局部支承图 `[8, 64, 64]`：当前支承高度、占据、目标高斯、候选 footprint、候选高度、重力比例、层数比例。
- 预仿真数值特征：目标位置、候选姿态、候选编号、石头几何、主面/支承面几何。
- 缺失值 mask。

输出：

- 每个候选姿态一个标量 ranking score，用于在同一块石头的候选姿态组内排序。

训练标签：

- `target_mode=score`，用候选仿真质量 `candidate_score` 做监督目标。
- `candidate_score`、`support_overlap`、`support_contact_count`、`support_balance_error_m`、`bearing_pressure_proxy` 均通过 `--exclude-postsim-features` 从输入中剔除。

测试指标：

| 模型 | rows | groups | top-1 | top-3 | top-1 regret | top-3 regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score CNN | 20,634 | 2,525 | 0.354 | 0.669 | 10.324 | 4.220 |
| risk-adjusted CNN | 20,634 | 2,525 | 0.182 | 0.517 | 6.983 | 3.081 |

解释：score CNN 更适合当前闭环选择；risk-adjusted CNN 更保守，后续可作为失败过滤器而不是主排序器。

### Stone-Fit MLP

路径：

- 非均衡版：`batch_runs/20260620_modular_stonefit_lowwall_presim_geometry_v1`
- 均衡损失版：`batch_runs/20260620_modular_stonefit_lowwall_presim_geometry_balanced_v1`

输入：

- 槽位/角色/重力/目标信息。
- 石头几何、类别、主面/支承面几何。

输出：

- 该石头是否适合当前槽位的概率。

指标：

| 模型 | rows | accuracy | precision | recall | F1 | predicted positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stone-fit MLP | 645 | 0.806 | 0.833 | 0.952 | 0.889 | 0.930 |
| balanced stone-fit MLP | 645 | 0.767 | 0.906 | 0.806 | 0.853 | 0.744 |

解释：均衡版更会拒绝风险石头，但当前数据仍不足以直接替代石头池搜索。

## 闭环堆叠结果

目标均为 `wall_segment_v1` 三层短单面墙，策略为 `statics_wall`。

| run | 搜索方式 | 地球成功率 | 月球成功率 | 说明 |
| --- | --- | ---: | ---: | --- |
| `20260620_curriculum_wallsegment3course_full8_upperbound_seed95006_v1` | full 8 pose，无 pose 网络 | 4/20 = 20.0% | 8/12 = 66.7% | 该 run 因超时只完成 12 个 moon trial |
| `20260620_closed_loop_lowwall_presim_score_top3_seed96001_v1` | pose CNN top-3 | 4/12 = 33.3% | 10/12 = 83.3% | 当前最好折中，姿态候选仿真从 8 减到 3 |
| `20260620_closed_loop_lowwall_presim_score_top2_seed96004_v1` | pose CNN top-2 | 5/12 = 41.7% | 7/12 = 58.3% | 更省搜索，但月球成功率明显下降 |
| `20260620_closed_loop_lowwall_presim_score_top4_seed96005_v1` | pose CNN top-4 | 3/12 = 25.0% | 8/12 = 66.7% | 未显著优于 top-3 |
| `20260620_closed_loop_lowwall_presim_score_top3_stonefit_top2_seed96002_v1` | pose top-3 + stone-fit top-2 | 1/12 = 8.3% | 4/12 = 33.3% | stone-fit 过早缩小石头池，效果差 |
| `20260620_closed_loop_lowwall_presim_score_top3_stonefit_balanced_top3_seed96003_v1` | pose top-3 + balanced stone-fit top-3 | 2/12 = 16.7% | 1/12 = 8.3% | 均衡 stone-fit 仍不能直接替代石头池 |

## 失败模式

pose CNN top-3 的失败主要集中在上层：

- 地球 cap：`missed_target+post_hold_drift` 6 次。
- 地球 middle：`missed_target` 或 `missed_target+post_hold_drift` 4 次。
- 月球 cap：`missed_target+post_hold_drift` 2 次。
- 月球 base：`missed_target` 1 次。

说明：当前姿态网络已经能有效减少候选姿态搜索，但地球短墙的限制主要在 cap/middle 的选石、支承关系和保持期漂移。单纯缩小石头池会放大错误，必须先收集更多“槽位-石头”负样本和成功样本。

## 图像记录

最佳 pose CNN top-3 run 已保存成功和典型失败案例：

- 截图目录：`batch_runs/20260620_closed_loop_lowwall_presim_score_top3_seed96001_v1/captures_960x720`
- manifest：`batch_runs/20260620_closed_loop_lowwall_presim_score_top3_seed96001_v1/captures_960x720/capture_manifest.csv`
- 案例数量：3 个成功、3 个失败。
- 图像数量：84 张 PNG，42 个深度 `.npy`。
- 包含视角：front、wall_front、right、back、left、top、wall_top。
- 每个视角保存 RGB、depth PNG 和 depth NPY，其中 `wall_front_depth.*` 与 `wall_top_depth.*` 已显式输出。

## 当前结论

1. pose CNN top-3 是目前有效的小网络替代模块。
   - 每块候选石头的姿态物理试验从 8 个减少到 3 个，减少 62.5% 的候选姿态仿真。
   - 月球三层短墙达到 83.3% 成功率，满足 2-3 层课程的阶段目标。
   - 地球三层短墙仍只有 33.3%，需要改进选石和上层稳定性。

2. top-2 太窄，top-4 没有带来稳定收益。
   - top-2 月球降到 58.3%。
   - top-4 月球 66.7%、地球 25.0%，不如 top-3。

3. stone-fit 网络暂时不能替代石头池。
   - 直接把手工石头池从 8 块缩到 2-3 块会显著降低成功率。
   - 当前 stone-fit 需要更多跨 catalog 数据、角色特异标签和负样本挖掘。

4. 后续网络输入仍必须保持预仿真原则。
   - 可以加入：候选姿态下的底面/上表面几何先验、长轴是否沿墙向、支承槽位关系、当前局部支承图。
   - 不可以加入：某块石头历史成功率、当前 run 后验成功率、仿真后 target error、扰动、速度、接触数等。

## 放置方式记录

当前仿真不是直接把石头贴在目标点上，也不是从很高处砸落。流程是：

1. 为当前槽位生成若干候选释放位姿。
2. 将石头 freejoint 瞬移到释放位姿，并把 6 维速度清零。
3. 释放位姿的高度为 `support_top + 0.5 * bbox_z + drop_clearance`。
4. 对 `statics_wall`，`drop_clearance` 目前是底层 0.004 m、上层 0.014 m。
5. 之后由 MuJoCo 在当前重力下自由落下、碰撞、settle。
6. 每个候选 settle 后计算误差、支承、扰动和速度；选最好候选作为该石头的放置结果。
7. 全部石头放完后再执行整堆 hold，检查 post-hold drift、掉落和形状误差。

因此当前模式更接近“准静态低高度释放”，不是机械臂轨迹级操作。这个假设先保留，因为它有利于筛选石头和规则；但未来需要升级为更真实的 `servo placement`：

- 石头从上方缓慢下降，而不是瞬移到释放点。
- 接触后检测法向力、切向滑移和姿态变化。
- 满足接触稳定阈值后再释放自由度。
- 记录接触建立过程，作为后续真实机器人或端到端策略训练数据。

## 下一步

1. 保留 pose CNN top-3 作为当前闭环默认方案。
2. 继续收集 `wall_segment_v1` 地球/月球 full-candidate 和 pose top-3 闭环数据，重点增加不同 seed/catalog。
3. 为 stone-fit 改成角色特异模型：
   - `base_fit_net`
   - `middle_fit_net`
   - `cap_fit_net`
4. 增加候选姿态下的预仿真几何特征：
   - 近水平底面面积代理。
   - 近水平上表面面积代理。
   - 墙向长轴对齐度。
   - 候选 footprint 与下层支承 footprint 的几何交叠代理，必须在不调用 MuJoCo candidate simulation 的前提下计算。
5. 地球墙优先优化 cap/middle 的 post-hold drift，而不是继续追求更高层数。
