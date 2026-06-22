# GitHub 自动同步策略

## 目标

把当前月面石墙实验代码、脚本、文档和轻量级审计材料定时推送到：

`https://github.com/George3215/Codex_Stack.git`

这个同步是灾备和阶段记录用途，不替代本地大数据保存。

## 严禁删除策略

自动同步脚本 `scripts/auto_git_snapshot.ps1` 遵守以下规则：

- 只 stage 新增文件和修改文件。
- 不 stage 删除记录。
- 如果本地文件被误删，脚本会记录 `skip_missing_or_deleted`，但不会把删除提交到 GitHub。
- 不执行 `git reset --hard`、`git clean`、`Remove-Item` 等删除性操作。
- 超过 `90 MB` 的单文件默认跳过，避免 GitHub 推送失败。

## 不进入 GitHub 的内容

以下内容保留在本地，不自动提交：

- `batch_runs/` 原始实验数据、图片、视频、模型权重和大规模 CSV。
- `generated*/` 生成的 mesh / MJCF / 中间数据。
- `reports/` 本地报告输出。
- `*.npz`、`*.npy`、`*.pt`、`*.png`、`*.jpg`、`*.mp4`、`*.avi`、`*.zip`。

如果后续要备份完整原始数据，应使用 Git LFS、DVC、对象存储或硬盘级备份；不要直接把几十 GB 数据塞进普通 Git 仓库。

## 定时任务

Codex app 里会创建一个定时自动化任务，周期性运行安全同步流程：

1. 检查 git 状态。
2. stage 新增/修改且未被忽略的小文件。
3. 生成 `auto snapshot <timestamp>` 提交。
4. 推送到 `origin/main`。
5. 如果认证失败或远端拒绝，保留本地提交和日志，不删除任何内容。

