# 异步硬件调度经验记录 2026-06-22

## 目标

当前实验目标是把 3/4 层单面墙的成功率拉高，再进入 5 层和更高层。为了让数据飞轮跑起来，需要同时使用：

- 本机 RTX 2080 Ti：主训练、support-map 导出、PoseRisk/小网络训练、主调度。
- 远端 GTX 1080 Ti：轻量采样、独立验证批次、小模型/消融任务。

原则仍然是：所有实验追加保存，严禁删除。

## 当前本机 2080 Ti 分工

本机继续承担更重的任务：

- `20260622_231000_cmd_autonomous_wall_flywheel_master_v2`
  - 主调度器。
  - 4 层 strict success rate 未达到 `60%` 前继续 3/4 层数据飞轮。
  - 达到 `60%` 后才进入 5 层探测。
- `20260622_221000_cmd_auto_strict_4course`
  - 严格 4 层评估。
  - 已出现一个高质量 4 层成功样例：4/4 可见层，高度约 `0.410 m`，水平漂移约 `0.0128 m`。
- `20260622_224800_cmd_poserisk_v18b_train_eval4`
  - PoseRiskNet v18b 训练和 4 层评估。
- `20260622_225000_cmd_supportmap_v19_train_eval4`
  - support-map CNN v19 的地图导出和训练。
- `20260622_221000_cmd_auto_flywheel_3to4`
  - 3/4 层数据飞轮，保存 hard negative 和 near-miss。

当前策略是不再盲目加本机任务。本机 2080 Ti 已有约 7-8GB 显存占用，继续塞重任务会增加卡死风险，收益不如让 master v2 接管节奏。

## 远端 1080 Ti 分工

远端主机：

- Host: `desktop-m57fdie.tail83f520.ts.net`
- GPU: GTX 1080 Ti
- MoonStack 环境：`D:\ResearchManagerData\conda_envs\moon-rock-stack`
- 远端 legacy 代码：`D:\ResearchManager\moon_rock_stack`
- 远端追加输出：`D:\ResearchManagerData\runs`

远端当前不可达：

- SSH 端口超时。
- Tailscale 显示 `desktop-m57fdie` 最近约 11 小时未在线。

因此已启动本地异步远端调度器：

- 脚本：`scripts/remote_1080ti_async_scheduler.ps1`
- 本地日志：`batch_runs/remote_1080ti_scheduler/20260622_remote_1080ti_wall_worker/scheduler.log`
- 轮询周期：每 15 分钟一次，共 96 轮。
- 远端恢复在线后自动上传 worker 到：
  - `D:\ResearchManagerData\incoming\run_20260622_remote_1080ti_wall_worker.ps1`
- 远端输出目录：
  - `D:\ResearchManagerData\runs\20260622_remote_1080ti_wall_worker`

远端 worker 的第一阶段任务是轻量 3/4 层墙体采样：

- rocks: `96`
- targets: `single_face_wall_3course_v1,single_face_wall_4course_v1`
- gravity: `moon`
- candidates: `6`
- steps per rock: `240`
- hold steps: `900`
- trials: `1`

这个配置比本机主任务轻，目的不是替代主训练，而是补充独立样本和 sanity check。远端恢复后，先让 1080 Ti 产生小批量可靠数据，再根据结果决定是否让它承担更重的 support-map 或小网络消融。

## 异步调度经验

1. 不要让两台机器写同一个输出目录。

每个 worker 必须有独立 session 和独立输出路径。这样即使某个 worker 崩溃，也不会破坏其他结果。

2. 主训练和轻量采样分离。

2080 Ti 更适合做重训练和主调度；1080 Ti 更适合跑轻量采样、独立验证、消融任务。这样数据来源更多，同时避免本机单点拥塞。

3. 远端不可达时不要阻塞主流程。

远端 1080 Ti 当前离线，所以不能把主实验依赖远端。正确做法是本机 master v2 继续跑，远端调度器轮询等待；远端恢复后自动加入。

4. 缺模型时等待，不盲目发任务。

本机 master v2 已加入 `--wait-for-model-dirs`。support-map 或 PoseRiskNet 模型未产出时，调度器只记录缺失项，不启动必然失败的评估。

5. 删除记录不进入自动同步。

GitHub 自动同步脚本只提交新增和修改，不提交删除。这是为了防止误删在定时同步中扩散到远端仓库。

## 下一步

- 等本机 strict 4 层 trial 1 完成，更新 4 层累计成功率。
- 等 support-map v19 导出和训练结束，让 master v2 自动发下一轮 4 层改进任务。
- 等远端 1080 Ti 恢复在线，自动启动 3/4 层轻量采样。
- 远端采样完成后，把结果汇总进本机经验库，用作下一轮 StoneSlotNet / PoseRisk / support-map ranker 的训练数据候选。

