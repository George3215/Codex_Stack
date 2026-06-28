# 2026-06-28 训练石头分布切换：NASA-like 仅作测试

## 修正原则

用户明确指出：NASA-like 石头只能作为后续测试集使用，不能作为训练数据主分布。

因此从本记录开始，数据策略切换为：

- 训练/采样：使用通用凸多面体石头。
- 测试/诊断：保留 `nasa_like_wall*` 数据，用于 held-out 泛化评估。
- 数据集构建：默认排除 `rock_profile` 以 `nasa_like` 开头的运行目录。
- 严禁删除历史输出；已产生的 NASA-like 批次保留，但标记为测试/诊断用途。

## 新增训练 profile

代码位置：

`D:\MoonStack\experiments\moon_rock_stack\moon_rock_stack\fractal_rocks.py`

新增 profile：

- `convex_poly_wall_train`
- `convex_poly_diverse_train`

生成方法：

1. 从多面体方向采样开始。
2. 根据 `bearing_block_clast`、`course_block_clast`、`wall_block_clast`、`cap_block_clast` 等通用干砌角色生成不同长宽高比例。
3. 加入有限方向抖动、宽支撑面和轻微形状差异。
4. 使用 `scipy.spatial.ConvexHull` 重新包络，确保训练石头为凸多面体。
5. 执行非薄片约束，避免特别扁的石头。
6. 输出仍保持多棱角、多面体边界，不生成平滑椭球。

## NASA-like 隔离规则

修改脚本：

`D:\MoonStack\experiments\moon_rock_stack\scripts\build_learning_dataset.py`

新增行为：

- 默认读取每个 run 的 `PROTOCOL.md`。
- 如果发现 `rock_profile` 以 `nasa_like` 开头，则排除该 run。
- 排除记录写入：
  - `dataset_summary.json`
  - `README.md`
- 只有显式传入 `--include-nasa-like-test-runs` 时，才会包含 NASA-like 数据。这个参数只应用于构建测试/诊断集，不用于训练集。

验证 smoke：

输入 run：

- NASA-like：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_random_polyhedral_v3_mixed_3course_cmd_v1`
- 凸多面体：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_convex_poly_wall_train_smoke_v1`

输出：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_dataset_filter_smoke_v1`

验证结果：

| 项目 | 数量 |
| --- | ---: |
| 保留训练 run | 1 |
| 排除 NASA-like run | 1 |
| placement examples | 11 |
| candidate pose examples | 98 |
| assignment examples | 49 |

被排除项：

- `20260628_random_polyhedral_v3_mixed_3course_cmd_v1`: `nasa_like_wall_v3`

## 已启动的新实验

后台任务：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260628_cmd_convex_poly_wall_train_3course_v1`

输出目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260628_convex_poly_wall_train_3course_cmd_v1`

实验配置：

- `rock_profile=convex_poly_wall_train`
- `rocks=150`
- `clusters=12`
- `trials=2`
- `target=single_face_wall_3course_v1`
- `gravity=moon`
- `candidates=6`
- `low_release_search=true`
- `base_support_prior_weight=0.35`
- `base_continuity_prior_weight=0.45`
- `pose_risk_weight=0.55`
- `commit_best_rejected=true`

启动验证：

- `started_at.txt` 已产生。
- 输出目录已产生 `meshes`、`mjcf`、`features.csv`、`structured_progress.csv`。
- 当前任务正在推进，已进入 trial 0 的 middle course。

## 对已有 NASA-like 批次的处理

以下批次不进入训练集：

- `20260628_random_polyhedral_v3_3course_batch_fg_v1`
- `20260628_random_polyhedral_v3_continuity_3course_batch_fg_v1`
- `20260628_random_polyhedral_v3_mixed_3course_cmd_v1`

这些数据可以保留为：

- 测试集候选。
- 泛化诊断。
- 失败案例图像展示。
- 与训练分布凸多面体结果做对比。

## 下一步

1. 等待 `convex_poly_wall_train` 3 层墙批次完成。
2. 如果 3 层凸多面体结果比 NASA-like 稳定，继续用该 profile 扩大训练数据。
3. 之后引入 `convex_poly_diverse_train`，增加通用凸多面体形状多样性。
4. 训练集只用通用凸多面体；NASA-like 后续作为 held-out test，评估模型是否过拟合训练分布。
5. 成功率统计需要分开报告：
   - train-distribution success rate
   - held-out NASA-like test success rate
   - failure reason breakdown
   - wall shape metrics: `target_rmse_xy_m`、`wall_y_span_m`、`wall_outlier_count`
