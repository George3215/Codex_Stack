from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BATCH_ROOT = REPO / "batch_runs"
ASYNC_ROOT = BATCH_ROOT / "async_jobs"
CONDA_PYTHON = Path(r"C:\Users\all\miniconda3\envs\moon-rock-stack\python.exe")
PYTHON = CONDA_PYTHON if CONDA_PYTHON.exists() else Path(sys.executable)

DEFAULT_STONE_FIT = BATCH_ROOT / "20260621_wall_flywheel_3course_focus_v2_stone_slot_net_20260621_014556"
DEFAULT_POSE_RANKER = BATCH_ROOT / "20260621_support_map_cnn_negative_mining_v4_highwall_foreground"
DEFAULT_POSE_RISK = BATCH_ROOT / "20260622_resume_poserisk_strictdrift_hardnegative_100955_100ep"
DEFAULT_OBSERVED = BATCH_ROOT / "20260622_top8_probe40_course3net_3to4_moon_trials2_20260622_193500"


@dataclass
class PlannedJob:
    name: str
    purpose: str
    output: Path
    command: list[str]


def main() -> int:
    args = parse_args()
    session = args.session or datetime.now().strftime("%Y%m%d_%H%M%S_auto_wall_scale")
    session_dir = unique_dir(BATCH_ROOT / "auto_wall_scale" / session)
    session_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = session_dir / "LEDGER.md"
    manifest_path = session_dir / "manifest.json"
    launched: list[dict[str, Any]] = []
    observed = observed_paths(args)
    manifest: dict[str, Any] = {
        "created_at": now(),
        "session": session,
        "session_dir": str(session_dir),
        "repo": str(REPO),
        "python": str(PYTHON),
        "policy": {
            "no_delete": True,
            "stage_gate": "If four-course strict success is observed, launch five-course jobs; otherwise keep improving four-course reliability.",
            "neural_policy": "Use StoneSlotNet, support-map candidate ranker, and PoseRiskNet through course 3 for four-course walls.",
            "curriculum_policy": "Strict eval keeps commit-best off; data flywheel can keep best rejected placements as curriculum and hard negatives.",
            "action_policy": "Use low-release contact search for upper courses so execution impact is not mistaken for stone/pose failure.",
        },
        "args": json_safe(vars(args)),
        "observed": [str(path) for path in observed],
        "iterations": [],
    }
    write_json(manifest_path, manifest)
    append_ledger(
        ledger_path,
        [
            f"# 自动墙体升阶调度记录 {session}",
            "",
            "## 目标",
            "",
            "- 4 层严格单面墙未稳定前，继续采样、训练和评估 4 层。",
            "- 一旦观察到 4 层 strict success，就自动进入 5 层单面墙。",
            "- 严格评估不使用 commit-best；数据飞轮允许保存 best rejected 作为课程样本和负样本。",
            "- 全部输出追加到新目录，不删除、不覆盖历史实验。",
            "",
        ],
    )

    active_jobs: list[dict[str, Any]] = []
    for cycle in range(max(1, int(args.monitor_cycles))):
        active_jobs = [job for job in active_jobs if launched_job_is_active(job)]
        external_active = active_external_jobs(args)
        stats = summarize_observed(observed + launched_outputs(launched))
        stage = choose_stage(stats, args.four_success_threshold)
        iteration = {
            "cycle": cycle,
            "time": now(),
            "stage": stage,
            "active_job_count": len(active_jobs),
            "external_active_job_count": len(external_active),
            "external_active_jobs": external_active,
            "stats": stats,
        }
        manifest["iterations"].append(iteration)
        write_json(manifest_path, manifest)
        append_ledger(
            ledger_path,
            [
                f"## cycle {cycle} {iteration['time']}",
                "",
                f"- 当前阶段: `{stage}`",
                f"- 活跃调度任务数: `{len(active_jobs)}`",
                f"- 4 层 strict: `{stats.get('four_success_count', 0)}/{stats.get('four_trial_count', 0)}`",
                f"- 4 层 shape: `{stats.get('four_shape_success_count', 0)}/{stats.get('four_trial_count', 0)}`",
                "",
            ],
        )
        if active_jobs or external_active:
            if cycle < args.monitor_cycles - 1:
                time.sleep(max(1, int(args.poll_seconds)))
            continue

        dependency_report = model_dependency_report(args)
        iteration["model_dependencies"] = dependency_report
        if args.wait_for_model_dirs and dependency_report["missing"]:
            write_json(manifest_path, manifest)
            append_ledger(
                ledger_path,
                [
                    "### 模型依赖未就绪，等待下一轮",
                    "",
                    *[f"- `{item}`" for item in dependency_report["missing"]],
                    "",
                ],
            )
            if cycle < args.monitor_cycles - 1:
                time.sleep(max(1, int(args.poll_seconds)))
            continue

        jobs = build_jobs(args, session, cycle, stage)
        iteration["planned_jobs"] = [job_to_json(job) for job in jobs]
        write_json(manifest_path, manifest)
        append_ledger(ledger_path, ["### 本轮启动任务", ""])
        if args.dry_run:
            for job in jobs:
                append_ledger(ledger_path, [f"- DRY RUN `{job.name}` -> `{job.output}`"])
            break

        for job in jobs:
            record = start_job(job, args.launch_method)
            launched.append(record)
            if launched_job_is_active(record):
                active_jobs.append(record)
            append_ledger(
                ledger_path,
                [
                    f"- `{job.name}`",
                    f"  - 目的: {job.purpose}",
                    f"  - 输出: `{job.output}`",
                    f"  - PID: `{record.get('pid', '')}`",
                    f"  - job_dir: `{record.get('job_dir', '')}`",
                    f"  - returncode: `{record.get('launcher_returncode', '')}`",
                    "",
                ],
            )
        manifest["launched"] = launched
        write_json(manifest_path, manifest)
        if args.once:
            break
        if cycle < args.monitor_cycles - 1:
            time.sleep(max(1, int(args.poll_seconds)))

    write_readme(session_dir, manifest, ledger_path)
    print(session_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append-only scheduler for 4-to-5-course MoonStack wall experiments.")
    parser.add_argument("--session", default="")
    parser.add_argument("--observe-run", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--monitor-cycles", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=900)
    parser.add_argument("--launch-method", choices=["async", "cmd"], default="async")
    parser.add_argument("--respect-active-job-pattern", action="append", default=None)
    parser.add_argument(
        "--wait-for-model-dirs",
        action="store_true",
        help="Wait instead of launching jobs when configured runtime model directories are incomplete.",
    )
    parser.add_argument("--seed", type=int, default=206240100)
    parser.add_argument("--four-success-threshold", type=float, default=0.25)
    parser.add_argument("--rocks", type=int, default=120)
    parser.add_argument("--eval-rocks", type=int, default=120)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--eval-trials", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--candidate-probe-steps", type=int, default=40)
    parser.add_argument("--steps-per-rock", type=int, default=280)
    parser.add_argument("--hold-steps", type=int, default=1100)
    parser.add_argument(
        "--low-release-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use MuJoCo contact scanning to release upper-course rocks from the lowest collision-free height.",
    )
    parser.add_argument("--release-search-step-m", type=float, default=0.004)
    parser.add_argument("--release-extra-clearance-m", type=float, default=0.003)
    parser.add_argument(
        "--base-support-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bias the first wall course toward larger support footprint/volume and better bearing-face geometry.",
    )
    parser.add_argument("--base-support-prior-weight", type=float, default=1.0)
    parser.add_argument("--stone-fit-ranker-dir", type=Path, default=DEFAULT_STONE_FIT)
    parser.add_argument("--candidate-pose-ranker-dir", type=Path, default=DEFAULT_POSE_RANKER)
    parser.add_argument("--pose-risk-ranker-dir", type=Path, default=DEFAULT_POSE_RISK)
    parser.add_argument("--pose-risk-weight", type=float, default=0.35)
    parser.add_argument("--stone-fit-top-k", type=int, default=12)
    parser.add_argument("--candidate-pose-top-k", type=int, default=8)
    parser.add_argument("--ranker-max-course", type=int, default=3)
    parser.add_argument("--skip-flywheel", action="store_true")
    parser.add_argument("--skip-strict-eval", action="store_true")
    return parser.parse_args()


def observed_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(item).resolve() for item in args.observe_run if item]
    if DEFAULT_OBSERVED.exists():
        paths.append(DEFAULT_OBSERVED.resolve())
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def choose_stage(stats: dict[str, Any], threshold: float) -> str:
    trials = int(stats.get("four_trial_count", 0))
    strict = int(stats.get("four_success_count", 0))
    rate = strict / max(trials, 1)
    if strict > 0 and rate >= float(threshold):
        return "five_course_probe"
    return "four_course_improve"


def build_jobs(args: argparse.Namespace, session: str, cycle: int, stage: str) -> list[PlannedJob]:
    seed = int(args.seed) + 1009 * cycle
    jobs: list[PlannedJob] = []
    if stage == "five_course_probe":
        strict_targets = "single_face_wall_4course_v1,single_face_wall_5course_v1"
        flywheel_targets = strict_targets
        strict_name = f"{session}_c{cycle:02d}_strict_4to5"
        flywheel_name = f"{session}_c{cycle:02d}_flywheel_4to5"
    else:
        strict_targets = "single_face_wall_4course_v1"
        flywheel_targets = "single_face_wall_3course_v1,single_face_wall_4course_v1"
        strict_name = f"{session}_c{cycle:02d}_strict_4course"
        flywheel_name = f"{session}_c{cycle:02d}_flywheel_3to4"

    if not args.skip_strict_eval:
        strict_output = BATCH_ROOT / strict_name
        jobs.append(
            PlannedJob(
                name=f"{strict_name}_eval",
                purpose=(
                    "4 layer strict success-rate evaluation"
                    if stage == "four_course_improve"
                    else "5 layer strict probe after passing the 4 layer gate"
                ),
                output=strict_output,
                command=structured_command(
                    args=args,
                    output=strict_output,
                    targets=strict_targets,
                    seed=seed,
                    trials=args.trials,
                    rocks=args.rocks,
                    workers=args.workers,
                ),
            )
        )

    if not args.skip_flywheel:
        flywheel_output = BATCH_ROOT / flywheel_name
        jobs.append(
            PlannedJob(
                name=f"{flywheel_name}_data_train",
                purpose=(
                    "Collect 3/4 layer curriculum and hard-negative data, then train modular rankers"
                    if stage == "four_course_improve"
                    else "Collect 4/5 layer curriculum data while probing the next height"
                ),
                output=flywheel_output,
                command=flywheel_command(
                    args=args,
                    session=flywheel_name,
                    targets=flywheel_targets,
                    seed=seed + 500,
                ),
            )
        )
    return jobs


def structured_command(
    args: argparse.Namespace,
    output: Path,
    targets: str,
    seed: int,
    trials: int,
    rocks: int,
    workers: int,
) -> list[str]:
    command = [
        str(PYTHON),
        "-m",
        "moon_rock_stack.run_structured_experiment",
        "--rocks",
        str(rocks),
        "--rock-profile",
        "high_wall",
        "--clusters",
        "10",
        "--trials",
        str(trials),
        "--targets",
        targets,
        "--strategies",
        "statics_wall",
        "--gravities",
        "moon",
        "--candidates",
        str(args.candidates),
        "--steps-per-rock",
        str(args.steps_per_rock),
        "--hold-steps",
        str(args.hold_steps),
        "--candidate-probe-steps",
        str(args.candidate_probe_steps),
        "--workers",
        str(workers),
        "--seed",
        str(seed),
        "--output",
        str(output),
        "--stone-fit-ranker-dir",
        str(args.stone_fit_ranker_dir.resolve()),
        "--stone-fit-top-k",
        str(args.stone_fit_top_k),
        "--stone-fit-ranker-max-course",
        str(args.ranker_max_course),
        "--candidate-pose-ranker-dir",
        str(args.candidate_pose_ranker_dir.resolve()),
        "--candidate-pose-top-k",
        str(args.candidate_pose_top_k),
        "--candidate-pose-ranker-max-course",
        str(args.ranker_max_course),
        "--pose-risk-ranker-dir",
        str(args.pose_risk_ranker_dir.resolve()),
        "--pose-risk-weight",
        str(args.pose_risk_weight),
        "--pose-risk-ranker-max-course",
        str(args.ranker_max_course),
    ]
    if args.low_release_search:
        command.extend(
            [
                "--low-release-search",
                "--release-search-step-m",
                str(args.release_search_step_m),
                "--release-extra-clearance-m",
                str(args.release_extra_clearance_m),
            ]
        )
    if args.base_support_prior:
        command.extend(
            [
                "--base-support-prior",
                "--base-support-prior-weight",
                str(args.base_support_prior_weight),
            ]
        )
    return command


def flywheel_command(args: argparse.Namespace, session: str, targets: str, seed: int) -> list[str]:
    command = [
        str(PYTHON),
        "scripts/run_wall_data_flywheel.py",
        "--session",
        session,
        "--seed",
        str(seed),
        "--collect-batches",
        "1",
        "--collect-mode",
        "exploit",
        "--parallel-data-jobs",
        "1",
        "--mujoco-workers",
        str(args.workers),
        "--rocks",
        str(args.rocks),
        "--eval-rocks",
        str(args.eval_rocks),
        "--rock-profile",
        "high_wall",
        "--clusters",
        "10",
        "--trials",
        "1",
        "--eval-trials",
        str(args.eval_trials),
        "--targets",
        targets,
        "--gravities",
        "moon",
        "--candidates",
        str(args.candidates),
        "--steps-per-rock",
        str(args.steps_per_rock),
        "--hold-steps",
        str(args.hold_steps),
        "--candidate-probe-steps",
        str(args.candidate_probe_steps),
        "--candidate-pose-top-k",
        str(args.candidate_pose_top_k),
        "--candidate-pose-ranker-max-course",
        str(args.ranker_max_course),
        "--stone-fit-top-k",
        str(args.stone_fit_top_k),
        "--stone-fit-ranker-max-course",
        str(args.ranker_max_course),
        "--pose-risk-weight",
        str(args.pose_risk_weight),
        "--pose-risk-ranker-max-course",
        str(args.ranker_max_course),
        "--release-search-step-m",
        str(args.release_search_step_m),
        "--release-extra-clearance-m",
        str(args.release_extra_clearance_m),
        "--base-support-prior-weight",
        str(args.base_support_prior_weight),
        "--collect-commit-best-rejected",
        "--dataset-target-contains",
        "single_face_wall",
        "--max-groups",
        "768",
        "--grid-size",
        "64",
        "--window-m",
        "0.9",
        "--front-height-m",
        "0.60",
        "--shard-size",
        "1000",
        "--stone-epochs",
        "80",
        "--pose-epochs",
        "70",
        "--critic-epochs",
        "70",
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
        "--eval-stone-ranker-dir",
        str(args.stone_fit_ranker_dir.resolve()),
        "--eval-pose-ranker-dir",
        str(args.candidate_pose_ranker_dir.resolve()),
        "--eval-pose-risk-ranker-dir",
        str(args.pose_risk_ranker_dir.resolve()),
    ]
    if args.low_release_search:
        command.append("--low-release-search")
    if args.base_support_prior:
        command.append("--base-support-prior")
    return command


def start_job(job: PlannedJob, method: str) -> dict[str, Any]:
    if method == "cmd":
        return start_cmd_job(job)
    return start_async_job(job)


def start_async_job(job: PlannedJob) -> dict[str, Any]:
    command = [
        str(PYTHON),
        "scripts/async_process.py",
        "start",
        "--job-root",
        str(ASYNC_ROOT),
        "--job-name",
        job.name,
        "--cwd",
        str(REPO),
        "--env",
        "KMP_DUPLICATE_LIB_OK=TRUE",
        "--env",
        "PYTHONUTF8=1",
        "--",
        *job.command,
    ]
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
    record: dict[str, Any] = {
        "name": job.name,
        "purpose": job.purpose,
        "output": str(job.output),
        "command": job.command,
        "launcher_command": command,
        "launcher_returncode": int(result.returncode),
        "launcher_stdout": result.stdout,
        "launcher_stderr": result.stderr,
        "started_at": now(),
    }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {}
    record.update(payload)
    return record


def start_cmd_job(job: PlannedJob) -> dict[str, Any]:
    job_dir = unique_dir(ASYNC_ROOT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name(job.name)}")
    job_dir.mkdir(parents=True, exist_ok=False)
    stdout = job_dir / "stdout.txt"
    stderr = job_dir / "stderr.txt"
    run_cmd = job_dir / "run.cmd"
    run_cmd.write_text(cmd_script(job, job_dir, stdout, stderr), encoding="utf-8")
    result = subprocess.run(
        [r"C:\Windows\System32\cmd.exe", "/c", "start", "", "/B", str(run_cmd)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "name": job.name,
        "purpose": job.purpose,
        "output": str(job.output),
        "command": job.command,
        "job_dir": str(job_dir),
        "run_cmd": str(run_cmd),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "pid": 0,
        "launcher_returncode": int(result.returncode),
        "launcher_stdout": result.stdout,
        "launcher_stderr": result.stderr,
        "started_at": now(),
        "running": result.returncode == 0,
    }


def cmd_script(job: PlannedJob, job_dir: Path, stdout: Path, stderr: Path) -> str:
    command = " ^\n  ".join(cmd_quote(arg) for arg in job.command)
    return "\n".join(
        [
            "@echo off",
            "setlocal",
            f'set "JOB_DIR={job_dir}"',
            'set "PYTHONUTF8=1"',
            'set "KMP_DUPLICATE_LIB_OK=TRUE"',
            "",
            'echo %date% %time% > "%JOB_DIR%\\started_at.txt"',
            f'cd /d "{REPO}"',
            "",
            f"{command} > {cmd_quote(str(stdout))} 2> {cmd_quote(str(stderr))}",
            "",
            'set "EC=%ERRORLEVEL%"',
            'echo %EC% > "%JOB_DIR%\\exit_code.txt"',
            'echo %date% %time% > "%JOB_DIR%\\finished_at.txt"',
            "exit /b %EC%",
            "",
        ]
    )


def cmd_quote(value: Any) -> str:
    text = str(value).replace('"', '""')
    return f'"{text}"'


def summarize_observed(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for path in paths:
        result_path = path / "results.csv"
        if not result_path.exists():
            continue
        try:
            with result_path.open("r", encoding="utf-8", newline="") as handle:
                rows.extend(dict(row, _source=str(path)) for row in csv.DictReader(handle))
        except OSError:
            continue
    four = [row for row in rows if row.get("target_name") == "single_face_wall_4course_v1" and row.get("gravity") == "moon"]
    five = [row for row in rows if row.get("target_name") == "single_face_wall_5course_v1" and row.get("gravity") == "moon"]
    return {
        "observed_result_rows": len(rows),
        "four_trial_count": len(four),
        "four_success_count": sum(to_int(row.get("success")) for row in four),
        "four_shape_success_count": sum(to_int(row.get("shape_success")) for row in four),
        "four_mean_visible_courses": mean(to_float(row.get("visible_courses")) for row in four),
        "four_mean_height_m": mean(to_float(row.get("stack_height_m")) for row in four),
        "four_mean_drift_m": mean(to_float(row.get("max_horizontal_drift_m")) for row in four),
        "five_trial_count": len(five),
        "five_success_count": sum(to_int(row.get("success")) for row in five),
        "five_shape_success_count": sum(to_int(row.get("shape_success")) for row in five),
    }


def launched_outputs(launched: list[dict[str, Any]]) -> list[Path]:
    output: list[Path] = []
    for item in launched:
        path_text = str(item.get("output", ""))
        if path_text:
            output.append(Path(path_text))
    return output


def launched_job_is_active(job: dict[str, Any]) -> bool:
    pid = int(job.get("pid", 0) or 0)
    if pid > 0:
        return is_pid_running(pid)
    job_dir_text = str(job.get("job_dir", ""))
    if not job_dir_text:
        return False
    job_dir = Path(job_dir_text)
    return (job_dir / "started_at.txt").exists() and not (job_dir / "finished_at.txt").exists()


def active_external_jobs(args: argparse.Namespace) -> list[str]:
    patterns = args.respect_active_job_pattern
    if patterns is None:
        patterns = ["*cmd_auto_strict_4course*", "*cmd_auto_flywheel_3to4*"]
    active: list[str] = []
    for pattern in patterns:
        for job_dir in ASYNC_ROOT.glob(pattern):
            if (job_dir / "started_at.txt").exists() and not (job_dir / "finished_at.txt").exists():
                active.append(str(job_dir))
    return sorted(set(active))


def model_dependency_report(args: argparse.Namespace) -> dict[str, Any]:
    checks = [
        {
            "name": "stone_fit_ranker",
            "path": Path(args.stone_fit_ranker_dir),
            "alternatives": [[("stone_fit_net.npz"), ("stone_fit_net_schema.json")]],
        },
        {
            "name": "candidate_pose_ranker",
            "path": Path(args.candidate_pose_ranker_dir),
            "alternatives": [
                [("support_map_cnn_ranker.pt"), ("schema.json")],
                [("candidate_pose_rank_net.npz"), ("candidate_pose_rank_net_schema.json")],
            ],
        },
        {
            "name": "pose_risk_ranker",
            "path": Path(args.pose_risk_ranker_dir),
            "alternatives": [[("pose_risk_net.npz"), ("pose_risk_net_schema.json")]],
        },
    ]
    ready: list[dict[str, str]] = []
    missing: list[str] = []
    for check in checks:
        path = check["path"]
        if not path.exists():
            missing.append(f"{check['name']}: directory missing: {path}")
            continue
        matched = False
        for files in check["alternatives"]:
            if all((path / filename).exists() for filename in files):
                matched = True
                ready.append({"name": check["name"], "path": str(path), "files": ",".join(files)})
                break
        if not matched:
            options = [" + ".join(files) for files in check["alternatives"]]
            missing.append(f"{check['name']}: missing runtime files under {path}; expected one of: {' OR '.join(options)}")
    return {"ready": ready, "missing": missing}


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    return f'"{pid}"' in result.stdout or f",{pid}," in result.stdout


def write_readme(session_dir: Path, manifest: dict[str, Any], ledger_path: Path) -> None:
    launched = manifest.get("launched", [])
    lines = [
        "# 自动墙体升阶调度 README",
        "",
        "## 实验目的",
        "",
        "本调度器把当前阶段固定为数据飞轮：优先提升 4 层单面墙成功率；只有在观察到 4 层 strict success 后，才自动进入 5 层。",
        "",
        "## 网络参与方式",
        "",
        "- StoneSlotNet: 用石头几何先验选择更适合当前槽位的候选石头。",
        "- Support-map candidate ranker: 用局部墙体/深度支撑图对候选落点排序。",
        "- PoseRiskNet: 用几何、目标、重力和候选位姿预测漂移风险。",
        "- 4 层阶段设置 `ranker_max_course=3`，也就是网络参与到第 4 层 cap。",
        "",
        "## 输出",
        "",
        f"- manifest: `{session_dir / 'manifest.json'}`",
        f"- ledger: `{ledger_path}`",
    ]
    for job in launched:
        lines.append(f"- `{job.get('name', '')}` -> `{job.get('output', '')}`")
    (session_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_ledger(path: Path, lines: list[str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def job_to_json(job: PlannedJob) -> dict[str, Any]:
    return {
        "name": job.name,
        "purpose": job.purpose,
        "output": str(job.output),
        "command": job.command,
    }


def unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 2
    while True:
        candidate = path.with_name(f"{path.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value.strip())
    return cleaned[:80] or "job"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
