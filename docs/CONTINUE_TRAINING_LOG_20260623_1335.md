# 2026-06-23 自动堆叠训练续跑记录

记录时间：2026-06-23 13:32:07 +08:00  
仓库路径：`D:\MoonStack\experiments\moon_rock_stack`  
原则：只追加和修改实验代码、日志、模型、图片、统计文件；不删除任何数据。

## 本轮实验目的

当前目标不是盲目冲 10 层，而是先把 3-4 层单面墙的数据飞轮跑稳定：

1. 继续收集 3 层和 4 层单面墙的正负样本。
2. 用神经网络逐步替代早期启发式筛选，尤其是石头-槽位选择和候选位姿排序。
3. 保留有效物理先验：主支撑面、墙线连续性、上下层错缝、水平漂移约束、PoseRisk 风险惩罚。
4. 如果 4 层严格墙成功率稳定达到阈值，再进入 5 层和更高墙；否则继续在 3-4 层刷成功率和负样本。

## 当前自动调度状态

本地 2080 Ti 主机仍在运行自动实验：

- 主控进程：`scripts\auto_wall_scale_scheduler.py`
- 当前活跃飞轮：`20260622_autonomous_wall_flywheel_master_v2_c12_flywheel_3to4`
- 当前采集子任务：`20260622_autonomous_wall_flywheel_master_v2_c12_flywheel_3to4_collect_exploit_00_seed206252708`
- 当前策略：月面 `single_face_wall_3course_v1,single_face_wall_4course_v1` 混合采集，`statics_wall`，`exploit` 模式。
- 当前网络参与：
  - StoneSlotNet：石头-槽位几何适配筛选，`stone_fit_top_k=14`。
  - SupportMap v19：候选落点/支撑图 CNN 排序，`candidate_pose_top_k=8`。
  - PoseRiskNet v18b：候选位姿风险惩罚，`pose_risk_weight=0.45`。

远端 1080 Ti 调度器仍在运行：

- 调度脚本：`scripts\remote_1080ti_async_scheduler.ps1`
- 本地调度进程：PID 4888
- 调度周期：15 分钟轮询一次
- 当前状态：`desktop-m57fdie.tail83f520.ts.net` 的 SSH 22 端口仍超时，远端离线或不可达。
- 处理方式：不手动阻塞本地实验；远端上线后由调度器自动上传轻量采样任务。

## 4 层严格墙当前可确认统计

按 c09-c12 严格 4 层墙结果直接统计：

| cycle | trials | success | success_rate | max_height_m | best_success_drift_m |
| --- | ---: | ---: | ---: | ---: | ---: |
| c09 | 2 | 0 | 0.000 | 0.2606 | - |
| c10 | 2 | 1 | 0.500 | 0.4562 | 0.0122 |
| c11 | 2 | 0 | 0.000 | 0.3934 | - |
| c12 | 2 | 0 | 0.000 | 0.3436 | - |

阶段判断：

- c09-c12 可确认严格 4 层为 `1/8`。
- 加上上一轮 `20260622_cmd_auto_strict_4course_221000` 的 `1/2`，局部可确认为 `2/10`。
- master 累计口径到 c12 记录为约 `2/14`，说明它包含更早失败轮次；后续汇报时需要区分“局部窗口统计”和“master 累计统计”。

## 当前最好 4 层正样本

目前最值得保留的 4 层正样本来自：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_autonomous_wall_flywheel_master_v2_c10_strict_4course`

关键结果：

- trial：0
- target：`single_face_wall_4course_v1`
- gravity：moon，`1.624 m/s^2`
- success：1
- visible_courses：4
- stable_count：20
- failure_count：0
- stack_height_m：0.4562066544
- max_horizontal_drift_m：0.0121900380
- target_rmse_xy_m：0.0309673486
- target_max_xy_error_m：0.0690784694
- structure_score：2.5160120886
- 使用网络：
  - SupportMap v19：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_supportmap_v19_recent_3to4_train`
  - PoseRisk v18b：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_poserisk_v18b_recent_3to4_train`
  - StoneSlotNet：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260621_wall_flywheel_3course_focus_v2_stone_slot_net_20260621_014556`

保存的可视化：

- RGB/深度图目录：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_autonomous_wall_flywheel_master_v2_c10_strict_4course\cap_c10_success_failure_20260623_1330`
- 过程 GIF：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_autonomous_wall_flywheel_master_v2_c10_strict_4course\process_video_c10_strict_4course_20260623_1330\single_face_wall_4course_v1_statics_wall_moon_trial_00_process\process.gif`

## SupportMap v19 训练结果

模型目录：

`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260622_supportmap_v19_recent_3to4_train`

训练设置：

- 架构：`CNN map encoder + numeric MLP + groupwise softmax ranking loss`
- device：cuda
- GPU：NVIDIA GeForce RTX 2080 Ti
- 数据：8741 行，1400 个 rankable group
- 输入：支撑图/深度代理图 + 几何和槽位数值特征
- 约束：排除 post-simulation features，避免把仿真后验当作输入

主要指标：

- test_top1_hit_rate：0.231
- test_top3_hit_rate：0.685
- 4 层月面墙 top1：0.269
- 4 层月面墙 top3：0.642
- test_mean_top1_quality_regret：13.561
- test_mean_top3_quality_regret：3.876

解释：

SupportMap v19 已经适合作为 top-k 候选过滤器，但还不足以直接用 top-1 完全替代搜索。当前合理用法是网络把候选空间缩小，后面仍保留少量物理启发式和短程 MuJoCo probe。

## StoneSlotNet 故障与修复

c10/c11 飞轮在数据集和深度导出完成后，StoneSlotNet 训练/评估阶段出现 Windows 原生崩溃：

- 复现位置：`scripts\train_torch_stone_slot_net.py`
- 崩溃点：`predict()` 中的 NumPy BLAS 矩阵乘 `x @ w`
- 错误特征：`Windows fatal exception: code 0xc06d007f`，stdout/stderr 为空
- 判断：和之前 `numpy.polyfit` 原生崩溃属于同类 Windows NumPy/MKL 风险，不是 MuJoCo 物理本体错误。

已做修复：

1. `predict()` 改成 `safe_linear()`，使用分块 elementwise multiply + sum，避开 NumPy BLAS matmul。
2. `scripts\run_wall_data_flywheel.py` 中 StoneSlotNet 训练默认加 `--device cpu`，把 GPU 留给 SupportMap CNN 或后续更重模型。
3. 已用 `py_compile` 验证：
   - `scripts\train_torch_stone_slot_net.py`
   - `scripts\run_wall_data_flywheel.py`
4. 已用 c11 数据集 CPU 重跑成功：
   - 输出目录：`D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260623_debug_c11_stone_slot_cpu_retry_fixed`
   - row_count：6498
   - positive_count：417
   - test accuracy：0.603
   - test precision：0.0757
   - test recall：0.659
   - test_group top1_hit_rate：0.150
   - test_group top3_hit_rate：0.350

解释：

StoneSlotNet 当前仍是早期小网络，召回比精度更重要。它的目的不是直接决定最终堆叠，而是在进入候选位姿搜索前尽量保留可能成功的石头，同时剔除明显不合适的石头。

## 目前经验

有效规则：

- 低层需要更宽、更高支撑质量、更低尖刺评分的石头，减少后续漂移。
- 上层不能只看单块石头稳定性，还要看落点周围支撑连续性和墙线偏差。
- 只用几何外观选石头不够，必须把当前墙面观测引入候选位姿排序。
- 网络 top-k + 少量物理 probe 比纯随机尝试更有意义，也比纯 top-1 网络更稳。

无效或风险较高的规则：

- 纯随机放置对真实任务开销太大，不能作为长期主策略。
- 只追求堆高但不约束墙线，会退化成石头堆，不符合单面墙目标。
- top-1 网络直接替代启发式，目前成功率和泛化都不够。
- 把仿真后验成功率作为网络输入没有科学意义；训练阶段可用作 label 或 loss 权重，但部署输入必须来自先验几何和现场观测。

## 下一步执行判据

短期继续：

1. 让 c12 飞轮完成：采集、数据集重建、深度/支撑图导出、StoneSlotNet/SupportMap/PoseRisk 相关训练或评估。
2. 若 4 层严格成功率仍低于 0.60，继续 3-4 层飞轮，重点增加负样本和 hard negative。
3. 若 4 层局部窗口成功率稳定超过 0.60，再进入 5 层墙。

模型方向：

1. StoneSlotNet 保持小网络，用作石头-槽位前筛。
2. SupportMap/深度图网络继续做候选位姿 top-k 排序。
3. PoseRiskNet 继续做风险惩罚，尤其关注上层漂移和 cap 层失败。
4. 后续再考虑更大的多模态网络，把石头点云/几何特征、top depth、墙面占据图一起输入，输出候选落点和姿态分布。

数据记录要求：

- 成功样本保留 RGB、top depth、front RGB、过程 GIF。
- 失败样本保留 failure_cases、candidate_pose_log、placement_log，用作 hard negative。
- 汇报时必须区分严格物理成功、形状成功、可见层数、最大漂移和高度。

