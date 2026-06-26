# 2026-06-26 有效数据清洗与网络训练记录

## 实验目的

本轮任务不是继续盲目堆高，而是把已经产生的大量墙体堆叠数据做一次筛选，把对训练真正有用的数据单独整理到一个新目录中。这样后续网络训练不再直接混用旧的、低相关的、噪声较大的数据，便于让神经网络逐步替代启发式搜索。

本轮遵守两个原则：

- 不删除任何旧数据，只新增清洗脚本、清洗后的数据目录和训练输出目录。
- 暂时只保留 3 层和 4 层单面墙相关数据，忽略旧的低墙、非主策略和明显异常样本。

## 新增脚本

脚本路径：

```text
D:\MoonStack\experiments\moon_rock_stack\scripts\clean_policy_dataset.py
```

主要功能：

- 读取原始策略替代数据集。
- 按目标结构、策略、重力、角色、几何完整性和物理指标进行过滤。
- 输出新的干净数据目录，保留 csv/jsonl 格式，便于 sklearn、PyTorch 或后续数据管线直接读取。
- 写出 `dataset_summary.json` 和中文 `README.md`，记录清洗条件、保留数量和剔除原因。

## 数据来源与输出

原始数据集：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_policy_replacement_dataset_v1
```

清洗后数据集：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_clean_policy_wall34_dataset_v1
```

清洗后目录包含：

```text
run_examples.csv
placement_examples.csv
placement_examples.jsonl
candidate_pose_examples.csv
candidate_pose_examples.jsonl
assignment_candidate_examples.csv
dataset_summary.json
README.md
```

## 清洗规则

保留条件：

- 目标结构：`single_face_wall_3course_v1`、`single_face_wall_4course_v1`
- 策略：`statics_wall`
- 重力：`earth`、`moon`
- 石头角色：`base`、`middle`、`cap`
- 必须具备完整的石头几何特征。
- 候选位姿必须具备完整的位姿、目标误差、扰动和速度等关键物理字段。

异常过滤阈值：

```text
max_target_error_m = 1.25
max_abs_y_error_m = 0.75
max_disturbance_m = 1.25
max_velocity = 80.0
min_rock_volume = 1e-7
```

暂时不用的数据：

- 旧的简单低墙：`single_face_wall_v1`
- 非主策略：例如 `wall_bonded`
- 明显异常的候选位姿和 placement
- 没有对应有效候选位姿的 run

## 清洗结果

清洗后保留：

| 数据类型 | 保留数量 |
|---|---:|
| run examples | 123 |
| placement examples | 2,364 |
| candidate pose examples | 121,679 |
| assignment candidate examples | 27,322 |

原始候选位姿数据：

| 项目 | 数量 |
|---|---:|
| 原始 candidate pose | 141,988 |
| 保留 candidate pose | 121,679 |
| 剔除 candidate pose | 20,309 |

候选位姿剔除原因：

| 原因 | 数量 |
|---|---:|
| `candidate_pose_metric_outlier` | 18,107 |
| `target_not_selected` | 1,395 |
| `strategy_not_selected` | 807 |

placement 数据：

| 项目 | 数量 |
|---|---:|
| 原始 placement | 2,515 |
| 保留 placement | 2,364 |
| 剔除 placement | 151 |

placement 剔除原因：

| 原因 | 数量 |
|---|---:|
| `placement_metric_outlier` | 48 |
| `strategy_not_selected` | 72 |
| `target_not_selected` | 31 |

assignment candidate 数据：

| 项目 | 数量 |
|---|---:|
| 原始 assignment candidate | 28,075 |
| 保留 assignment candidate | 27,322 |
| 剔除 assignment candidate | 753 |

assignment candidate 剔除原因：

| 原因 | 数量 |
|---|---:|
| `run_without_clean_candidate_pose` | 288 |
| `target_not_selected` | 465 |

## 数据难度观察

3 层、4 层墙的 base、middle、cap 数据都保留下来了。清洗后数据仍然显示，上层和中层明显更难：

| 目标/角色/重力 | candidate pose 数量 | 风险正例率 |
|---|---:|---:|
| 4 层墙 / middle / moon | 28,061 | 0.847 |
| 4 层墙 / cap / moon | 14,174 | 0.890 |
| 4 层墙 / base / moon | 17,431 | 0.709 |
| 3 层墙 / cap / earth | 2,777 | 0.964 |
| 4 层墙 / cap / earth | 2,517 | 0.952 |

这说明越靠上越不是简单的“找一个平面放上去”，而是要同时考虑下方支撑、左右连续性、候选位姿释放高度和扰动传播。这个结论支持后续引入墙面局部观测网络，而不是只训练石头本身的几何分类器。

## 已训练网络 1：Clean PoseRiskNet

输出目录：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_clean_wall34_pose_risk_net_v1
```

训练数据：

- 数据源：清洗后的 `candidate_pose_examples`
- 样本数：121,679
- 按 run 划分训练/测试，避免同一个实验 run 同时出现在训练集和测试集。
- 输入只使用先验可获得的信息：重力、目标位置、候选位姿、石头几何、角色、结构目标、策略、cluster label。
- 后验仿真指标只作为标签，不作为输入。

指标：

| 指标 | 数值 |
|---|---:|
| train rows | 96,302 |
| test rows | 25,377 |
| train accuracy | 0.8027 |
| train precision | 0.9580 |
| train recall | 0.7890 |
| train F1 | 0.8653 |
| test accuracy | 0.7084 |
| test precision | 0.8714 |
| test recall | 0.7414 |
| test F1 | 0.8011 |
| test group top1 safe rate | 0.6251 |
| test group top3 safe rate | 0.8898 |

阶段判断：

- PoseRiskNet 在清洗后数据上仍然有效，适合作为候选位姿风险排序器。
- top3 安全率接近 0.89，说明它可以帮助缩小候选搜索范围。
- top1 安全率约 0.625，说明如果完全只选网络第一名仍然偏冒险，需要与支撑先验、连续性先验或 WallCritic 联合使用。

## 已训练网络 2：Clean StoneSlotNet

输出目录：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_clean_wall34_stone_slot_net_v1
```

训练数据：

- 数据源：清洗后的 `assignment_candidate_examples`
- 样本数：27,322
- 输入为石头几何、目标槽位、角色、结构目标和 cluster label。
- 不输入 candidate rank、真实仿真结果等后验字段。

指标：

| 指标 | 数值 |
|---|---:|
| train rows | 21,916 |
| test rows | 5,406 |
| train accuracy | 0.7703 |
| train precision | 0.1769 |
| train recall | 0.8734 |
| train F1 | 0.2942 |
| test accuracy | 0.7358 |
| test precision | 0.0757 |
| test recall | 0.3954 |
| test F1 | 0.1271 |
| test group top1 hit rate | 0.1031 |
| test group top3 hit rate | 0.2835 |

阶段判断：

- 只看“石头几何 + 槽位”的 StoneSlotNet 不能可靠决定某块石头是否适合当前位置。
- 这个结果很重要：它说明石头选择不能脱离当前墙体局部状态。
- 该网络可以保留为弱先验或辅助特征，但不能作为主要策略替代启发式。

## 正在后台运行的清洗版 SupportMap/WallCritic 训练

异步任务目录：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\async_jobs\20260626_cmd_clean_wall34_supportmap_train_v1
```

训练会话目录：

```text
D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260626_clean_wall34_supportmap_train_v1
```

设计目的：

- 把清洗后的数据进一步转换成局部墙面支撑图、正视/俯视结构张量等输入。
- 让网络看到“当前墙体局部状态 + 候选石头 + 目标槽位”，而不只是看到单块石头。
- 训练 SupportMapRanker / WallStateCritic，用来判断某个候选石头和位姿是否会破坏墙体连续性、支撑面积和稳定性。

当前启动策略：

- 先检查 GPU 显存。
- 如果显存占用低于 6500 MiB，就启动清洗版 SupportMap/WallCritic 训练。
- 本次启动时显存占用约 2266 MiB，满足条件，任务已经启动。

## 本轮结论

1. 数据清洗已经完成，新的有效数据目录已经建立。
2. 3 层和 4 层单面墙数据被集中到一个干净数据集，后续训练应优先使用该目录。
3. PoseRiskNet 在干净数据上仍然可用，适合替代一部分候选位姿风险判断。
4. StoneSlotNet 指标较差，说明只凭石头几何无法可靠完成石头选择。
5. 下一步最有价值的是训练带局部墙面观测的 SupportMapRanker / WallStateCritic，让网络看到墙体当前状态，再决定石头和落点。
6. 没有删除任何旧数据，旧数据仍然保留，后续如有需要可以重新回收或重新定义清洗规则。

## 后续建议

- 后续实验默认使用 `20260626_clean_policy_wall34_dataset_v1` 作为主训练集。
- 继续保留启发式先验，但逐步让网络承担排序和风险判断工作。
- 对 StoneSlotNet 的失败样本做二次分析，重点看是否缺少“墙体局部状态”导致泛化失败。
- SupportMap/WallCritic 训练完成后，应在 3 层墙上先做闭环评估，再进入 4 层墙稳定性提升。
- 当 4 层墙成功率稳定后，再考虑 5 层或更高墙体，而不是直接盲目冲高。
