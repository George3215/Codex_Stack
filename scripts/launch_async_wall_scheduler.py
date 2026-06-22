from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BATCH_ROOT = REPO / "batch_runs"
PYTHON = Path(sys.executable)

DEFAULT_DATASET = BATCH_ROOT / "20260621_wall_flywheel_3course_pose_risk_augmented_learning_dataset"
DEFAULT_STONE_FIT = BATCH_ROOT / "20260621_wall_flywheel_3course_focus_v2_stone_slot_net_20260621_014556"
DEFAULT_POSE_RANKER = BATCH_ROOT / "20260621_wall_flywheel_3course_focus_v1_resume_train_pose_ranker_structure"
DEFAULT_POSE_RISK = BATCH_ROOT / "20260621_pose_risk_net_3course_v1"


def main() -> int:
    args = parse_args()
    session = args.session or datetime.now().strftime("%Y%m%d_%H%M%S_wall_4to5_async")
    session_dir = unique_dir(BATCH_ROOT / "async_scheduler" / session)
    session_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "created_at": now(),
        "session": session,
        "session_dir": str(session_dir),
        "policy": {
            "no_delete": True,
            "local_role": "RTX 2080 Ti: primary neural training plus bounded 4/5-course wall flywheel.",
            "remote_role": "GTX 1080 Ti: light training, sampling, and auxiliary data collection.",
            "cpu_policy": "Keep MuJoCo workers low while existing local simulations are running; use GPU for training in parallel.",
            "stage_goal": "Raise 3-course reliability, collect 4-course curriculum/evaluation data, and start 5-course exploration without mixing curriculum results into strict success metrics.",
        },
        "resource_snapshot": resource_snapshot(),
        "args": json_safe(vars(args)),
        "jobs": [],
    }
    write_json(session_dir / "manifest.json", manifest)

    jobs = build_jobs(args, session)
    write_json(session_dir / "planned_jobs.json", {"jobs": jobs})
    if args.dry_run:
        manifest["jobs"] = [{**job, "dry_run": True} for job in jobs]
        write_json(session_dir / "manifest.json", manifest)
        write_readme(session_dir, manifest)
        print(session_dir)
        return 0

    for job in jobs:
        result = start_local_job(job, session_dir)
        manifest["jobs"].append(result)
        write_json(session_dir / "manifest.json", manifest)

    write_readme(session_dir, manifest)
    print(session_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch append-only async MoonStack wall curriculum jobs.")
    parser.add_argument("--session", default="")
    parser.add_argument("--seed", type=int, default=206211501)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-pose-risk-train", action="store_true")
    parser.add_argument("--skip-wall-flywheel", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--stone-fit-ranker-dir", type=Path, default=DEFAULT_STONE_FIT)
    parser.add_argument("--candidate-pose-ranker-dir", type=Path, default=DEFAULT_POSE_RANKER)
    parser.add_argument("--pose-risk-ranker-dir", type=Path, default=DEFAULT_POSE_RISK)
    parser.add_argument("--targets", default="single_face_wall_4course_v1,single_face_wall_v1")
    parser.add_argument("--gravities", default="moon")
    parser.add_argument("--rocks", type=int, default=170)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--eval-trials", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--steps-per-rock", type=int, default=380)
    parser.add_argument("--hold-steps", type=int, default=1600)
    return parser.parse_args()


def build_jobs(args: argparse.Namespace, session: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    if not args.skip_pose_risk_train:
        output = BATCH_ROOT / f"{session}_local_pose_risk_net"
        jobs.append(
            {
                "name": "local_pose_risk_train",
                "role": "local_gpu_training",
                "purpose_zh": "用本机 2080Ti 训练候选位姿风险网络，目标是减少 4/5 层墙的无效位姿搜索。",
                "output": str(output),
                "command": [
                    str(PYTHON),
                    "-m",
                    "scripts.train_torch_pose_risk_net",
                    "--dataset",
                    str(args.dataset.resolve()),
                    "--output",
                    str(output),
                    "--target-contains",
                    "single_face_wall",
                    "--epochs",
                    "120",
                    "--batch-size",
                    "512",
                    "--hidden",
                    "192",
                    "--dropout",
                    "0.18",
                    "--lr",
                    "0.0008",
                    "--weight-decay",
                    "0.00025",
                    "--test-fraction",
                    "0.25",
                    "--split-by-run",
                    "--seed",
                    str(args.seed + 11),
                    "--device",
                    "auto",
                    "--target-error-limit",
                    "0.16",
                    "--target-y-error-limit",
                    "0.075",
                    "--disturbance-limit",
                    "0.08",
                    "--velocity-limit",
                    "0.22",
                ],
            }
        )

    if not args.skip_wall_flywheel:
        flywheel_session = f"{session}_local_4to5_flywheel"
        output = BATCH_ROOT / flywheel_session
        jobs.append(
            {
                "name": "local_4to5_wall_flywheel",
                "role": "local_mixed_data_training_eval",
                "purpose_zh": "采集 4/5 层墙 curriculum 数据，重建数据集，训练 StoneSlot/姿态/墙状态小网络，并做严格闭环评估。",
                "output": str(output),
                "command": [
                    str(PYTHON),
                    "-m",
                    "scripts.run_wall_data_flywheel",
                    "--session",
                    flywheel_session,
                    "--seed",
                    str(args.seed + 1000),
                    "--collect-batches",
                    "1",
                    "--collect-mode",
                    "exploit",
                    "--parallel-data-jobs",
                    "1",
                    "--mujoco-workers",
                    "1",
                    "--rocks",
                    str(args.rocks),
                    "--eval-rocks",
                    str(args.rocks),
                    "--rock-profile",
                    "high_wall",
                    "--clusters",
                    "10",
                    "--trials",
                    str(args.trials),
                    "--eval-trials",
                    str(args.eval_trials),
                    "--targets",
                    args.targets,
                    "--gravities",
                    args.gravities,
                    "--candidates",
                    str(args.candidates),
                    "--steps-per-rock",
                    str(args.steps_per_rock),
                    "--hold-steps",
                    str(args.hold_steps),
                    "--candidate-pose-top-k",
                    "3",
                    "--stone-fit-top-k",
                    "15",
                    "--max-groups",
                    "512",
                    "--grid-size",
                    "64",
                    "--window-m",
                    "0.9",
                    "--front-height-m",
                    "0.58",
                    "--shard-size",
                    "1000",
                    "--stone-epochs",
                    "80",
                    "--pose-epochs",
                    "60",
                    "--critic-epochs",
                    "60",
                    "--stone-batch-size",
                    "128",
                    "--pose-batch-size",
                    "128",
                    "--critic-batch-size",
                    "160",
                    "--stone-hidden",
                    "128",
                    "--pose-hidden",
                    "192",
                    "--critic-hidden",
                    "192",
                    "--exploit-stone-ranker-dir",
                    str(args.stone_fit_ranker_dir.resolve()),
                    "--exploit-pose-ranker-dir",
                    str(args.candidate_pose_ranker_dir.resolve()),
                    "--exploit-pose-risk-ranker-dir",
                    str(args.pose_risk_ranker_dir.resolve()),
                    "--eval-pose-risk-ranker-dir",
                    str(args.pose_risk_ranker_dir.resolve()),
                    "--pose-risk-weight",
                    "0.35",
                    "--collect-commit-best-rejected",
                    "--dataset-target-contains",
                    "single_face_wall",
                ],
            }
        )
    return jobs


def start_local_job(job: dict[str, Any], session_dir: Path) -> dict[str, Any]:
    start_cmd = [
        str(PYTHON),
        "scripts/async_process.py",
        "start",
        "--job-root",
        str(BATCH_ROOT / "async_jobs"),
        "--job-name",
        job["name"],
        "--cwd",
        str(REPO),
        "--env",
        "KMP_DUPLICATE_LIB_OK=TRUE",
        "--env",
        "PYTHONUTF8=1",
        "--",
        *job["command"],
    ]
    started_at = now()
    result = subprocess.run(start_cmd, cwd=REPO, capture_output=True, text=True, check=False)
    record = {
        **job,
        "launcher_command": start_cmd,
        "started_at": started_at,
        "returncode": result.returncode,
        "launcher_stdout": result.stdout,
        "launcher_stderr": result.stderr,
    }
    try:
        record["job_json"] = json.loads(result.stdout)
    except json.JSONDecodeError:
        record["job_json"] = None
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in job["name"])
    write_json(session_dir / f"{safe}_launcher_result.json", record)
    return record


def resource_snapshot() -> dict[str, Any]:
    return {
        "python": str(PYTHON),
        "gpu": run_text(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv"]),
        "python_processes": run_text(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU,WorkingSet,StartTime,Path | ConvertTo-Json -Depth 3",
            ]
        ),
    }


def run_text(command: list[str]) -> str:
    try:
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False, timeout=20)
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"{type(exc).__name__}: {exc}"
    return (result.stdout + result.stderr).strip()


def unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 2
    while True:
        candidate = path.with_name(f"{path.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_readme(session_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# 2026-06-21 异步调度记录",
        "",
        "## 调度目的",
        "",
        "- 本机 RTX 2080 Ti 负责主训练，优先训练候选位姿风险网络，并运行 4/5 层墙的数据飞轮。",
        "- 远端 GTX 1080 Ti 只承担轻量训练、采样和辅助数据收集，不再作为主训练端。",
        "- 所有任务只追加新目录和日志，不删除、不覆盖已有实验结果。",
        "",
        "## 当前阶段",
        "",
        "- 3 层墙已经达到进入 4 层阶段的门槛；当前重点是收集 4 层 curriculum/strict 数据。",
        "- 5 层墙作为探索任务启动，但不把 commit-best curriculum 结果当成 strict success。",
        "- 后续经验统计会从 placement/candidate logs 中抽取：哪些石头适合 base/middle/cap，哪些候选位姿容易失败，哪些槽位最缺可行石头。",
        "",
        "## 已启动本机任务",
        "",
    ]
    for job in manifest.get("jobs", []):
        job_json = job.get("job_json") or {}
        lines.extend(
            [
                f"- `{job.get('name', '')}`",
                f"  - 目的：{job.get('purpose_zh', '')}",
                f"  - 输出：`{job.get('output', '')}`",
                f"  - PID：`{job_json.get('pid', '')}`",
                f"  - job.json：`{job_json.get('job_dir', '')}\\job.json`" if job_json.get("job_dir") else "  - job.json：启动器未返回",
                f"  - stdout：`{job_json.get('stdout', '')}`",
                f"  - stderr：`{job_json.get('stderr', '')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 查看方式",
            "",
            "- 本机异步任务：`conda run -n moon-rock-stack python scripts/async_process.py list --job-root batch_runs/async_jobs`",
            "- 调度 manifest：`manifest.json`",
            "- 单个任务日志：查看对应 job 目录下的 `stdout.log` 和 `stderr.log`。",
        ]
    )
    (session_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
