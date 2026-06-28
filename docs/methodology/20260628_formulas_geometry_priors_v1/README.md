# MoonStack 评价公式、网络公式与石头几何生成先验

日期: 2026-06-28  
状态: v1，可作为后续实验日志、PPT 和论文方法章节的公式底稿。  
适用代码: `moon_rock_stack/features.py`, `moon_rock_stack/fractal_rocks.py`, `moon_rock_stack/structured.py`, `scripts/train_*`

## 1. 设计目标

当前任务不是把石头随机堆成土堆，而是在地球重力和月球重力下，用干叠 dry stacking 的结构目标堆出单面墙、石柱和更高层结构。因此后续记录必须同时回答四个问题:

1. 生成的石头是否符合岩石几何先验: 多面体、棱角、圆化程度、块状程度、凹陷、粗糙度、非薄片。
2. 候选放置是否符合结构目标: 目标槽位误差、层数、高度、墙面厚度、离群石头、漂移和最终速度。
3. 神经网络是否只记住石头，还是使用了可观测几何先验和局部石堆观测。
4. 数据是否形成闭环: 失败样本、成功样本、候选样本、训练、评估、再采样。

本文档把这些问题写成公式。后续实验 README 需要引用这里的公式，而不是只写定性描述。

## 2. 参考依据

这些来源只作为生成和评价的几何先验，不被当作唯一真理。MoonStack 的最终标准仍然是 MuJoCo 物理稳定性和结构任务成功率。

- Encyclopaedia Britannica, Sedimentary rock: 沉积碎屑岩描述常关注颗粒大小、形状、分选和结构纹理。用于支持我们把 `shape/roundness/sphericity/sorting` 当作几何先验。<https://www.britannica.com/science/sedimentary-rock>
- Roundness (geology): 地质中的 roundness 用来描述颗粒边角尖锐到圆化的程度，常见类别包括 angular、subangular、subrounded、rounded。用于支持 `roundness_proxy` 和 angular/blocky 两类生成。<https://en.wikipedia.org/wiki/Roundness_(geology)>
- Sphericity: Wadell 球形度使用同体积球体表面积和颗粒表面积之比。用于 `sphericity` 特征。<https://en.wikipedia.org/wiki/Sphericity>
- Breccia / Conglomerate: breccia 强调棱角状碎屑，conglomerate 强调圆化砾石。用于支持“训练集同时包含棱角块状和较圆化块状”的先验。<https://en.wikipedia.org/wiki/Breccia>, <https://en.wikipedia.org/wiki/Conglomerate_(geology)>
- Fractional Brownian motion / self-affine roughness: 自相似或自仿射粗糙表面常用 Hurst exponent 和 fractal dimension 描述。用于多尺度半径扰动和表面粗糙度，不用于生成尖刺。<https://en.wikipedia.org/wiki/Fractional_Brownian_motion>
- Powers, M. C. 1953. A new roundness scale for sedimentary particles. Journal of Sedimentary Petrology. 用于支持 roundness 分级思想。

## 3. 符号

石头网格:

```text
V = {v_i in R^3}_{i=1..N_v}        顶点集合
F = {f_j}_{j=1..N_f}               三角面集合
A                                  表面积
Vol                                体积
H(V)                               凸包
Vol_h                              凸包体积
B = (b_x, b_y, b_z)                axis-aligned bounding box
a >= b >= c                        bounding box 三个轴按长到短排序
n_j                                第 j 个面的单位法向
r_i = ||v_i - mean(V)||            顶点到中心的半径
N(i)                               顶点 i 的网格邻居
```

结构目标:

```text
s_i = (x_i^*, y_i^*, z_i^*)        第 i 块石头的目标槽位
p_i^0                              投放前质心位置
p_i^T                              settle 后质心位置
u_i                                是否被放置
g                                  重力，地球约 9.81 m/s^2，月球约 1.62 m/s^2
```

## 4. 石头几何特征公式

### 4.1 面数

当前 `encyclopedic_poly_train` 从一次细分二十面体开始:

```text
N_f <= 20 * 4^s
N_v = 10 * 4^s + 2
```

当前实现固定 `s = 1`，因此:

```text
N_f <= 80
N_v = 42
```

如果先生成凸包再做凹陷，凹陷只移动顶点，不新增三角面。因此当前训练用百科先验多面体的三角面上限仍为 80。这个上限的目的有三个:

1. 避免过密网格让 MuJoCo 碰撞开销过大。
2. 保留明显多面体棱角，而不是变成平滑球。
3. 保证特征、网络输入和数据量增长时仍可控。

### 4.2 球形度 sphericity

采用 Wadell 类球形度:

```text
Psi = pi^(1/3) * (6 Vol)^(2/3) / A
```

含义:

- `Psi -> 1`: 越接近球体或圆化砾石。
- `Psi` 较低: 越偏离球形，可能更长、更扁、更块状或更棱角。

### 4.3 矩形度 / 块状度 rectangularity

当前实现用体积占 bounding box 的比例作为矩形度代理:

```text
Q_rect = Vol / (b_x b_y b_z)
```

含义:

- `Q_rect` 高: 石头更像块体、近似长方体或立方体。
- `Q_rect` 低: 石头更不规则、凹凸更多，或存在局部凹陷/缺口。

后续可把矩形度分解成更细的面先验:

```text
Q_face = (A_1 + A_2 + A_3) / A
```

其中 `A_1,A_2,A_3` 是三个最大支撑面的面积。`Q_face` 高说明石头有更明显的可承重主面。

### 4.4 伸长率和扁平率

```text
E_long = a / b
F_flat = b / c
```

含义:

- `E_long` 高: 长条形石头。
- `F_flat` 高: 薄片化倾向。
- 当前生成器显式限制 `c / b >= tau`，其中训练用多面体 `tau = 0.64`，避免“特别扁的石头”主导训练。

### 4.5 棱角度 angularity

相邻三角面共享边时，使用法向夹角:

```text
A_ang = mean_{(j,k) adjacent} arccos(clip(n_j dot n_k, -1, 1)) / pi
```

含义:

- `A_ang` 高: 棱角多，局部折线明显。
- `A_ang` 低: 表面更圆滑。

MoonStack 需要有棱角的外星/月面风格多面体，但不能生成尖刺。因此后续还要同时检查 spike score。

### 4.6 圆化代理 roundness_proxy

真实地质 roundness 通常基于边角曲率半径和内切圆等定义，当前网格阶段先用可计算代理:

```text
R_proxy = clip(Psi * (1 - min(A_ang / 0.35, 1)), 0, 1)
```

含义:

- `R_proxy` 高: 更接近圆化砾石。
- `R_proxy` 低: 更接近 angular/subangular 多面体。
- 这不是 Powers roundness 的严格复现，只是训练特征和聚类特征。

### 4.7 凹陷代理 concavity_proxy

凸包体积凹陷:

```text
C_hull = clip(1 - Vol / Vol_h, 0, 1)
```

局部顶点凹陷:

```text
C_local = max_i max(0, 1 - r_i / median_{j in N(i)} r_j)
```

最终凹陷代理:

```text
C = max(C_hull, C_local)
```

含义:

- `C_hull` 检测整体凸包外的体积缺口。
- `C_local` 检测局部半径低于邻域的凹坑。
- 当前凹陷是“局部内凹代理”，不是保证复杂非凸拓扑的严格几何证明。

### 4.8 尖刺分数 spike_score

```text
S_spike = max_i max(0, r_i / median_{j in N(i)} r_j - 1)
```

用途:

- 如果 `S_spike` 高，说明顶点相对邻域过度外突，接近尖刺。
- 生成器需要压低该值，因为用户明确指出尖刺不符合多凸几何体石头先验。

### 4.9 粗糙度 roughness

```text
R_rough = std(r_i) / mean(r_i)
```

用途:

- 作为分形/自仿射粗糙表面的低维统计代理。
- 不能单独使用。高粗糙度可能来自合理棱角，也可能来自不合理尖刺，所以必须和 `S_spike`、`A_ang`、`C` 联合判断。

## 5. 多面体石头生成公式

### 5.1 生成总体流程

当前新增训练 profile:

```text
rock_profile = "encyclopedic_poly_train"
```

其生成目标:

1. 约 50% 块状/矩形度更强的 angular 或 subangular 多面体。
2. 约 50% 更接近圆化/等轴砾石的多面体。
3. 约 35% 带局部凹陷代理。
4. 不生成特别扁的石头。
5. 不生成尖刺。
6. NASA-like 石头仍保留为测试集，不进入训练集。

### 5.2 初始方向

从细分二十面体得到单位方向:

```text
u_i = v_i / ||v_i||
```

当前 `s = 1`，所以最多 80 个三角面。方向扰动:

```text
u_i' = normalize(u_i + epsilon_i)
epsilon_i ~ N(0, sigma_dir^2 I)
```

块状分支使用较小方向扰动，圆化分支使用稍大方向扰动但更强平滑。

### 5.3 块状/圆化混合变量

```text
z ~ Bernoulli(0.5)
z = 1: blocky/angular branch
z = 0: rounded/equant branch
```

轴向缩放:

```text
D_z = diag(d_x, d_y, d_z) / (d_x d_y d_z)^(1/3)
```

这样做保持尺度扰动不会系统性改变体积。不同 rock kind 会调整 `d_x,d_y,d_z` 的范围，例如:

```text
bearing_block_clast: d_x,d_y 较大，d_z 稍小但不允许薄片化
course_block_clast: d_x 更长，用于墙体横向搭接
equant_clast: d_x,d_y,d_z 更接近 1
```

### 5.4 多尺度半径扰动

用多个宽 lobes 代替尖刺噪声:

```text
rho_i = clip(1 + sum_{k=1..K} alpha_k exp((u_i' dot c_k - 1) / sigma_k^2), rho_min, rho_max)
```

其中:

```text
c_k                            随机 lobe 中心方向
alpha_k                        lobe 幅值，可正可负
sigma_k                        lobe 宽度
K                              lobe 数
```

顶点初值:

```text
x_i = R0 D_z u_i' rho_i eta_i
eta_i ~ Uniform(eta_min, eta_max)
```

块状分支:

```text
roughness in [0.035, 0.080]
K in {4,5,6}
eta_i in [0.985, 1.015]
```

圆化分支:

```text
roughness in [0.018, 0.055]
K in {5,6,7,8}
eta_i in [0.975, 1.025]
```

这个设计相当于低频分形/自仿射粗糙度近似: 保留自然不规则性，但避免高频尖刺。

### 5.5 分形粗糙度与 Hurst 指数的后续扩展

更严格的分形表面可以写成谱形式:

```text
h(u) = sum_{m=0..M} beta_m phi_m(u)
beta_m ~ N(0, sigma_m^2)
sigma_m proportional to lambda_m^(-H)
```

其中 `H` 是 Hurst exponent。对二维表面图形，常用关系:

```text
D_f = 3 - H
```

这里 `D_f` 是表面分形维数。当前实现还没有直接拟合 `H`，但生成器已经采用“低频 lobes + 局部凹陷 + 尖刺限制”的保守近似。后续如果要更像真实岩面，需要从真实石头点云估计 `H` 或功率谱斜率，再反推 `sigma_m`。

### 5.6 支撑面生成

为了 dry stacking，需要有可承重面，而不是所有石头都像球。对给定方向 `n`，用分位数平面切出宽支撑面:

```text
q = quantile({n dot x_i}, q0)
delta_i = max(0, q - n dot x_i)
x_i <- x_i + gamma delta_i n
```

或对相反方向做同样操作。含义:

- `gamma` 越大，支撑面越明显。
- `q0` 控制有多少顶点被压到近似平面。
- 这是 dry stacking 的几何先验: 底层优先需要大支撑面、上层需要可搭接面。

### 5.7 非薄片约束

如果最短轴太短:

```text
if c / b < tau:
    scale_short_axis = tau b / c
```

当前训练用多面体:

```text
tau = 0.64
```

这回应“没有特别扁的石头”的约束，避免底层支撑面积不足、上层越堆越不稳。

### 5.8 凸多面体与局部凹陷

凸多面体分支:

```text
M_convex = ConvexHull({x_i})
```

凹陷分支先取凸包再移动局部顶点:

```text
c_d ~ Uniform(S^2)                         dent 中心方向
theta_i = arccos(u_i dot c_d)
w_i = clip((cos(theta_i) - cos(theta_d)) / (1 - cos(theta_d)), 0, 1)
x_i <- x_i - depth * w_i^2 * u_i
```

其中:

```text
theta_d in [0.32, 0.54] rad
depth in [0.085, 0.185] * mean_radius
dent_count in {2,3,4}
```

这样生成的是“有局部凹坑的多面体”，不是尖刺状外突。凹陷后再次调用局部突出限制:

```text
S_spike <= S_max
```

当前实现用迭代限制局部半径相对邻域的过大外突。

## 6. 结构评价公式

### 6.1 槽位误差

对每个被放置石头:

```text
e_i = ||p_i^T[xy] - s_i[xy]||_2
RMSE = sqrt(mean_i e_i^2)
E_max = max_i e_i
```

### 6.2 漂移和速度

```text
d_i = ||p_i^T[xy] - p_i^0[xy]||_2
D_max = max_i d_i
v_inf = max_i ||v_i^T||_2
```

其中 `v_inf` 在代码中记录为 settle 后速度指标 `speed`。

### 6.3 单石头稳定判别

```text
stable_i = 1[
    z_i^T > 0.35 bbox_z_i
    and e_i < tau_target
    and ||p_i^T[xy]||_2 < tau_radial
    and |y_i^T| < tau_wall_y
]
```

稳定比例:

```text
P_stable = sum_i stable_i / max(N_placed, 1)
```

### 6.4 层数和高度

```text
C_visible = visible_course_count({p_i^T}, slots)
Z_top = max_i (z_i^T + bbox_z_i / 2)
```

墙体需要至少达到目标层数:

```text
C_visible >= C_required
Z_top >= Z_min(target)
```

### 6.5 单面墙形状指标

```text
W_y = max_i y_i^T - min_i y_i^T
W_x = max_i x_i^T - min_i x_i^T
Aspect_xy = W_x / max(W_y, eps)
O_wall = count_i[stone i is outside wall band]
```

单面墙不是石堆，因此需要:

```text
W_y <= W_y,max(target)
O_wall <= O_max(target)
Aspect_xy >= Aspect_min(target)
```

### 6.6 结构分数

当前 `moon_rock_stack/structured.py` 使用:

```text
Score =
    P_stable
    + 0.20 C_visible
    - 2.00 RMSE
    - 0.80 D_max
    - 0.12 v_inf
```

如果目标是单面墙，再加:

```text
Score_wall =
    Score
    + 0.20 min(Aspect_xy, 4.0)
    - 0.90 max(0, W_y - W_y,max)
    - 0.18 O_wall
```

解释:

- 层数和稳定比例是正奖励。
- 目标误差、漂移、残余速度是负奖励。
- 单面墙额外奖励“长而薄”，惩罚厚成土堆和离群石头。

### 6.7 成功判别

形状成功:

```text
ShapeSuccess = 1[
    N_placed >= N_slots - N_allowed_unstable
    and N_stable >= N_placed - N_allowed_unstable
    and RMSE < tau_rmse(target)
    and E_max < tau_error(target)
    and C_visible >= C_required(target)
    and Z_top >= Z_min(target)
    and R_max <= tau_radial(target)
    and WallShapeSuccess(target)
]
```

最终成功:

```text
Success = ShapeSuccess and D_max < 0.15 and v_inf < 0.25
```

这些阈值随目标变化，例如 3 层、4 层、5 层单面墙使用不同 `tau_rmse` 和 `tau_error`。后续报告必须把目标名写清楚，否则成功率不可比较。

## 7. 神经网络公式

当前策略不是立即端到端，而是多个小网络逐步替代启发式最终决策。启发式负责产生物理可行候选，网络负责排序、风险过滤和石头-槽位匹配。

### 7.1 StoneSlotNet: 石头-槽位匹配网络

输入:

```text
x = [g_rock, g_slot, g_target, onehot(role), onehot(target), onehot(gravity)]
```

其中 `g_rock` 包含:

```text
volume, surface_area, face_count, bbox, elongation, flatness,
sphericity, roughness, angularity, spike_score, rectangularity,
roundness_proxy, concavity_proxy, support_face features
```

每个数值特征同时加入缺失标记:

```text
x_num = [value_1, missing_1, value_2, missing_2, ...]
```

网络:

```text
h = ReLU(W1 x + b1)
logit = W2 h + b2
p_select = sigmoid(logit)
```

默认隐藏维度:

```text
hidden = 96
```

标签:

```text
y = 1[selected_count_in_placement_log > 0]
```

损失:

```text
L_slot = - w_y * alpha_sample * [
    y log p_select + (1-y) log(1-p_select)
]
```

其中 `w_y` 是正负样本不平衡权重，`alpha_sample` 是角色平衡/样本权重。

设计目的:

- 不是让网络记住某个石头编号。
- 让网络学习“什么几何特征的石头适合底层、搭接层、封顶层或柱体层”。

### 7.2 PoseRiskNet: 候选位姿风险网络

输入:

```text
x = [g_rock, pose_candidate, target_slot, gravity, role, target]
```

网络:

```text
h = ReLU(W1 x + b1)
logit_risk = W2 h + b2
p_risk = sigmoid(logit_risk)
```

默认隐藏维度:

```text
hidden = 128
```

风险标签:

```text
y_risk = 1[
    e_target > tau_e
    or |e_y| > tau_y
    or disturbance > tau_d
    or velocity > tau_v
]
```

损失:

```text
L_risk = - w_y [y_risk log p_risk + (1-y_risk) log(1-p_risk)]
```

运行时使用:

```text
Score_candidate = Score_heuristic - lambda_r p_risk
```

设计目的:

- 先让网络学会排除明显会冲散墙面、偏离墙线、残余速度过大的候选。
- 这是替代启发式的第一步，因为失败风险比完整端到端策略更容易学习。

### 7.3 CandidatePoseGroupRanker: 候选位姿排序网络

同一块石头、同一个槽位、同一个局部状态下，会生成一组候选:

```text
G = {x_k}_{k=1..K}
```

网络打分:

```text
h_k = ReLU(W1 x_k + b1)
s_k = W2 h_k + b2
P(k | G) = exp(s_k) / sum_{j in G} exp(s_j)
```

如果一个 group 内有多个成功候选:

```text
q_k = 1[y_k=1] / sum_j 1[y_j=1]
```

损失:

```text
L_rank = - sum_{k in G} q_k log P(k | G)
```

默认隐藏维度:

```text
hidden = 64
```

设计目的:

- 让网络在同一局部状态下选择更好的释放点、横向偏移和姿态。
- 这比二分类更接近实际执行: 机器人/仿真最终必须选一个候选，而不是只判断好坏。

### 7.4 SupportMapCNNRanker: 局部石堆观测排序网络

输入包含两部分:

```text
M_k in R^{C x H x W}       局部 support/depth map
z_k in R^d                 数值特征
```

地图编码器:

```text
f_map(M) =
    Conv( C -> 32, 5x5 ) + BN + SiLU + Pool
    Conv(32 -> 64, 3x3 ) + BN + SiLU + Pool
    Conv(64 ->128, 3x3 ) + BN + SiLU + AdaptiveAvgPool
```

数值编码器:

```text
f_num(z) = Dropout(SiLU(LayerNorm(W_z z + b_z)))
```

融合打分:

```text
s_k = W_o concat(f_map(M_k), f_num(z_k)) + b_o
```

groupwise softmax:

```text
P(k | G) = softmax({s_k}_{k in G})
```

质量软标签:

```text
q_k = softmax(label_k / T)
```

或二值正样本归一化:

```text
q_k = y_k / sum_j y_j
```

损失:

```text
L_support = - sum_{k in G} q_k log P(k | G)
```

默认隐藏维度:

```text
hidden = 128
```

设计目的:

- 回应“网络应结合自身石头几何信息和石堆区域观测”的要求。
- top-depth / support map 是最有价值的观测；其他角度 depth 如果全黄或无有效梯度，不进入核心训练。

### 7.5 总决策公式

当前仍保留启发式作为候选生成器，网络逐步替代候选排序:

```text
k* = argmax_{k in G} [
    lambda_h Score_heuristic(k)
    + lambda_slot p_select(k)
    + lambda_rank s_rank(k)
    + lambda_support s_support(k)
    - lambda_risk p_risk(k)
    + lambda_prior Phi_prior(k)
]
```

长期目标是逐步降低 `lambda_h`，让网络承担更多排序工作:

```text
lambda_h(t+1) <= lambda_h(t)
lambda_network(t+1) >= lambda_network(t)
```

但启发式不会完全删除，它会作为:

1. 候选生成约束。
2. 物理先验正则项。
3. 失败安全边界。

## 8. 数据集与训练/测试边界

训练集可以使用:

```text
convex_poly_wall_train
convex_poly_diverse_train
encyclopedic_poly_train
```

NASA-like 石头:

```text
nasa_like_wall
nasa_like_wall_v2
nasa_like_wall_v3
```

只能作为测试集或泛化评估，不能参与训练。允许从 NASA 石头提取“统计特征分布”生成相似但不相同的训练石头，但这些生成石头必须记录为 `derived_prior`，且不能与 NASA 测试 mesh 相同。

数据记录至少包含:

```text
run_id, target_name, gravity, rock_profile, strategy
stone_geometry_features
slot_features
candidate_pose_features
support_map_path
selected_candidate
pre_settle_pose
post_settle_pose
target_error
drift
velocity
shape_success
success
failure_reason
```

## 9. 后续报告必须给出的统计

每个实验 README 至少给出:

```text
N_trials
N_success
N_shape_success
SuccessRate = N_success / N_trials
ShapeSuccessRate = N_shape_success / N_trials
NegativePerPositive = (N_trials - N_success) / max(N_success, 1)
DeltaSuccessRate = SuccessRate_new - SuccessRate_baseline
```

按目标分组:

```text
single_face_wall_3course_v1
single_face_wall_4course_v1
single_face_wall_5course_v1
single_column_v*
```

按石头类别分组:

```text
blocky/angular
rounded/equant
concave_proxy_high
support_face_high
rectangularity_high
spike_score_high
```

按放置层级分组:

```text
course_1 foundation
course_2 bridging/interlock
course_3 wall continuity
course_4+ accumulated error and disturbance
```

## 10. 当前版本的局限

1. `roundness_proxy` 是 mesh 代理，不是严格 Powers roundness。
2. `concavity_proxy` 是局部凹陷/凸包缺口代理，不保证复杂非凸拓扑。
3. 分形生成当前使用低频 lobes 和粗糙度代理，还没有从真实石头拟合 Hurst exponent。
4. 网络输入已经加入几何先验，但完整端到端策略尚未完成；现阶段目标是先提高 3 层和 4 层单面墙成功率。
5. 深度观测优先使用 top_depth/support map；侧视 depth 如果渲染无信息，不作为核心训练输入。

## 11. 实验解释模板

后续每次报告可以使用下面的解释顺序:

```text
目的:
    本轮要提升哪个结构任务的成功率。

几何先验:
    使用哪些 rock_profile，面数上限是多少，是否包含局部凹陷，是否禁止薄片/尖刺。

候选生成:
    释放高度、候选数量、重力、摩擦、低高度搜索设置。

网络参与:
    哪些网络参与，输入是什么，输出是什么，lambda 权重是多少。

评价公式:
    引用本 README 的 RMSE、ShapeSuccess、Success、Score_wall。

数据结果:
    N_trials, N_success, success_rate, negative_per_positive, 按层级/石头类别/策略分组统计。

失败分析:
    是 foundation 不稳、course_2 搭接失败、course_3 累积误差、course_4 扰动放大，还是墙厚变成石堆。
```
