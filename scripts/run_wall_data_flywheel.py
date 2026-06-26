from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BATCH_ROOT = REPO / "batch_runs"
PYTHON = Path(sys.executable)

DEFAULT_PRIOR_RUNS = [
    "20260619_closed_loop_wall4_baseline_full6_v1",
    "20260619_closed_loop_wall4_neural_presim_top3_v1",
    "20260620_highcourse_4course_presim_pose_top3_seed97001_v1",
    "20260620_highcourse_4course_expanded_quality_top3_seed97002_v1",
    "20260620_highcourse_4course_mujoco_depth_structure_proxyonline_top3_seed97007_v1",
    "20260620_highcourse_4course_stoneslot_structure_top3_seed97008_v1",
    "20260620_highcourse_4course_stoneslot_structure_strictline_seed97009_v2",
    "20260620_highcourse_4course_stoneslot_structure_physicsgate_seed97011_v4",
]

DEFAULT_STONE_FIT = "20260620_torch_stone_slot_net_4course_assignment_v1_20260620_213352"
DEFAULT_POSE_RANKER = "20260620_torch_mujoco_depth_cnn_4course_groups256_structure_v1"


@dataclass
class Job:
    name: str
    kind: str
    command: list[str]
    output: Path | None = None


@dataclass
class JobResult:
    name: str
    kind: str
    command: list[str]
    output: Path | None
    stdout_path: Path
    stderr_path: Path
    returncode: int
    started_at: str
    finished_at: str
    elapsed_seconds: float
    parsed_output: Path | None


def main() -> int:
    args = parse_args()
    session = args.session or datetime.now().strftime("%Y%m%d_wall_flywheel_%H%M%S")
    session_dir = (BATCH_ROOT / session).resolve()
    log_dir = session_dir / "logs"
    session_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at": now(),
        "session": session,
        "session_dir": str(session_dir),
        "cwd": str(REPO),
        "python": str(PYTHON),
        "policy": (
        "3-4 course wall data flywheel: mixed explore/exploit collection, "
        "dataset rebuild, depth tensor export, modular network training, bounded closed-loop evaluation. "
        "Optional PoseRiskNet, course-gated rankers, and commit-best curriculum collection are used only when explicitly requested."
        ),
        "args": json_safe(vars(args)),
        "phases": [],
    }
    manifest_path = session_dir / "flywheel_manifest.json"
    write_json(manifest_path, manifest)

    all_results: list[JobResult] = []
    new_run_names: list[str] = []

    if not args.skip_collect:
        data_jobs = build_data_jobs(args, session, session_dir)
        phase = {"name": "collect", "jobs": [job_to_json(job) for job in data_jobs], "started_at": now()}
        manifest["phases"].append(phase)
        write_json(manifest_path, manifest)
        results = run_jobs(data_jobs, log_dir, max_workers=args.parallel_data_jobs, dry_run=args.dry_run)
        all_results.extend(results)
        phase["finished_at"] = now()
        phase["results"] = [result_to_json(result) for result in results]
        phase["failed_count"] = sum(result.returncode != 0 for result in results)
        new_run_names.extend(run_name_from_output(result.parsed_output or result.output) for result in results if result.returncode == 0)
        write_json(manifest_path, manifest)
        if phase["failed_count"]:
            write_summary(session_dir, manifest, all_results)
            return 1

    prior_runs = [] if args.no_default_prior_runs else DEFAULT_PRIOR_RUNS
    dataset_runs = existing_run_names(prior_runs + args.prior_run + new_run_names)
    if args.require_new_data and not new_run_names and not args.skip_collect:
        raise SystemExit("No new data runs were created.")
    if not dataset_runs and not (args.skip_dataset and args.dataset):
        raise SystemExit("No dataset runs are available.")

    dataset_dir: Path | None = None
    tensor_dir: Path | None = None
    stone_model_dir: Path | None = None
    pose_model_dir: Path | None = None
    pose_risk_model_dir: Path | None = None
    critic_model_dir: Path | None = None

    if not args.skip_dataset:
        dataset_output = BATCH_ROOT / f"{session}_learning_dataset"
        dataset_job = Job(
            name="build_learning_dataset",
            kind="dataset",
            output=dataset_output,
            command=[
                str(PYTHON),
                "-m",
                "scripts.build_learning_dataset",
                "--output",
                str(dataset_output),
                *flatten_run_args(dataset_runs),
            ],
        )
        result = run_job(dataset_job, log_dir, dry_run=args.dry_run)
        all_results.append(result)
        dataset_dir = result.parsed_output or result.output
        manifest["phases"].append({"name": "dataset", "results": [result_to_json(result)]})
        write_json(manifest_path, manifest)
        if result.returncode != 0:
            write_summary(session_dir, manifest, all_results)
            return 1

    if args.skip_training:
        write_summary(session_dir, manifest, all_results)
        return 0

    if dataset_dir is None:
        dataset_dir = args.dataset.resolve() if args.dataset else None
    if dataset_dir is None or (not args.dry_run and not dataset_dir.exists()):
        raise SystemExit("A dataset is required for training. Run without --skip-dataset or pass --dataset.")

    dataset_target_filters = args.dataset_target_contains or ["single_face_wall"]

    if not args.skip_depth_export:
        tensor_output = BATCH_ROOT / f"{session}_mujoco_depth_maps"
        tensor_job = Job(
            name="export_mujoco_depth_maps",
            kind="depth_export",
            output=tensor_output,
            command=[
                str(PYTHON),
                "-m",
                "scripts.export_mujoco_depth_observation_maps",
                "--dataset",
                str(dataset_dir),
                "--output",
                str(tensor_output),
                "--source",
                "candidate",
                "--grid-size",
                str(args.grid_size),
                "--window-m",
                str(args.window_m),
                "--front-height-m",
                str(args.front_height_m),
                "--shard-size",
                str(args.shard_size),
                "--max-groups",
                str(args.max_groups),
                "--sample-mode",
                "candidate-groups",
                "--sample-seed",
                str(args.seed + 91),
                "--dtype",
                "float16",
                *flatten_repeat_args("--target-contains", dataset_target_filters),
                "--strategy-contains",
                "statics_wall",
            ],
        )
        result = run_job(tensor_job, log_dir, dry_run=args.dry_run)
        all_results.append(result)
        tensor_dir = result.parsed_output or result.output
        manifest["phases"].append({"name": "depth_export", "results": [result_to_json(result)]})
        write_json(manifest_path, manifest)
        if result.returncode != 0:
            write_summary(session_dir, manifest, all_results)
            return 1

    if tensor_dir is None:
        tensor_dir = args.tensor_dir.resolve() if args.tensor_dir else None
    if tensor_dir is None or (not args.dry_run and not tensor_dir.exists()):
        raise SystemExit("A tensor directory is required for training. Run without --skip-depth-export or pass --tensor-dir.")

    train_jobs: list[Job] = []
    train_results: list[JobResult] = []
    assignment_candidate_count = dataset_assignment_candidate_count(dataset_dir)
    if assignment_candidate_count <= 0:
        stone_model_dir = resolve_model_fallback(args.exploit_stone_ranker_dir, args.eval_stone_ranker_dir, DEFAULT_STONE_FIT)
        skip_result = write_skip_result(
            log_dir=log_dir,
            name="skip_stone_slot_net_no_assignment_candidates",
            kind="train_stone_skip",
            output=stone_model_dir,
            message=(
                "Skipped StoneSlotNet training because the learning dataset has no assignment_candidate examples. "
                "Reusing the configured/default StoneSlotNet."
            ),
        )
        train_results.append(skip_result)
        all_results.append(skip_result)
        manifest.setdefault("warnings", []).append(
            "StoneSlotNet training skipped because assignment_candidate_example_count is 0; fallback StoneSlotNet will be reused."
        )
        write_json(manifest_path, manifest)
    else:
        train_jobs.append(
            Job(
                name="train_stone_slot_net",
                kind="train_stone",
                output=BATCH_ROOT / f"{session}_stone_slot_net",
                command=[
                    str(PYTHON),
                    "-m",
                    "scripts.train_torch_stone_slot_net",
                    "--dataset",
                    str(dataset_dir),
                    "--output",
                    str(BATCH_ROOT / f"{session}_stone_slot_net"),
                    *flatten_repeat_args("--target-contains", dataset_target_filters),
                    "--epochs",
                    str(args.stone_epochs),
                    "--batch-size",
                    str(args.stone_batch_size),
                    "--hidden",
                    str(args.stone_hidden),
                    "--dropout",
                    "0.12",
                    "--lr",
                    "0.001",
                    "--weight-decay",
                    "0.0002",
                    "--test-fraction",
                    "0.2",
                    "--split-by-run",
                    "--device",
                    "cpu",
                    "--seed",
                    str(args.seed + 101),
                ],
            )
        )
    train_jobs.extend([
        Job(
            name="train_pose_risk_net",
            kind="train_pose_risk",
            output=BATCH_ROOT / f"{session}_pose_risk_net",
            command=[
                str(PYTHON),
                "-m",
                "scripts.train_torch_pose_risk_net",
                "--dataset",
                str(dataset_dir),
                "--output",
                str(BATCH_ROOT / f"{session}_pose_risk_net"),
                *flatten_repeat_args("--target-contains", dataset_target_filters),
                "--epochs",
                str(args.pose_risk_epochs),
                "--batch-size",
                str(args.pose_risk_batch_size),
                "--hidden",
                str(args.pose_risk_hidden),
                "--dropout",
                "0.14",
                "--lr",
                "0.001",
                "--weight-decay",
                "0.0002",
                "--test-fraction",
                "0.2",
                "--split-by-run",
                "--seed",
                str(args.seed + 104),
            ]
            + (["--candidate-metric-labels"] if args.pose_risk_candidate_metric_labels else []),
        ),
        Job(
            name="train_pose_ranker_structure",
            kind="train_pose",
            output=BATCH_ROOT / f"{session}_pose_ranker_structure",
            command=[
                str(PYTHON),
                "-m",
                "scripts.train_torch_support_map_ranker",
                "--tensor-dir",
                str(tensor_dir),
                "--output",
                str(BATCH_ROOT / f"{session}_pose_ranker_structure"),
                "--epochs",
                str(args.pose_epochs),
                "--batch-size",
                str(args.pose_batch_size),
                "--hidden",
                str(args.pose_hidden),
                "--dropout",
                "0.18",
                "--lr",
                "0.0008",
                "--weight-decay",
                "0.0004",
                "--test-fraction",
                "0.2",
                "--split-by-run",
                "--seed",
                str(args.seed + 102),
                "--amp",
                "--target-mode",
                "structure_aware",
                "--exclude-postsim-features",
            ],
        ),
        Job(
            name="train_wall_state_critic",
            kind="train_critic",
            output=BATCH_ROOT / f"{session}_wall_state_critic",
            command=[
                str(PYTHON),
                "-m",
                "scripts.train_torch_wall_state_critic",
                "--tensor-dir",
                str(tensor_dir),
                "--output",
                str(BATCH_ROOT / f"{session}_wall_state_critic"),
                "--epochs",
                str(args.critic_epochs),
                "--batch-size",
                str(args.critic_batch_size),
                "--hidden",
                str(args.critic_hidden),
                "--dropout",
                "0.22",
                "--lr",
                "0.0008",
                "--weight-decay",
                "0.0004",
                "--test-fraction",
                "0.2",
                "--split-by-run",
                "--seed",
                str(args.seed + 103),
                "--amp",
                "--exclude-postsim-features",
            ],
        ),
    ])
    for job in train_jobs:
        result = run_job(job, log_dir, dry_run=args.dry_run)
        train_results.append(result)
        all_results.append(result)
        if job.kind == "train_stone":
            stone_model_dir = result.parsed_output or result.output
        elif job.kind == "train_pose_risk":
            pose_risk_model_dir = result.parsed_output or result.output
        elif job.kind == "train_pose":
            pose_model_dir = result.parsed_output or result.output
        elif job.kind == "train_critic":
            critic_model_dir = result.parsed_output or result.output
        if result.returncode != 0:
            manifest["phases"].append({"name": "training", "results": [result_to_json(item) for item in train_results]})
            write_json(manifest_path, manifest)
            write_summary(session_dir, manifest, all_results)
            return 1
    manifest["phases"].append({"name": "training", "results": [result_to_json(item) for item in train_results]})
    write_json(manifest_path, manifest)

    if not args.skip_eval:
        eval_output = BATCH_ROOT / f"{session}_closed_loop_eval"
        eval_job = Job(
            name="closed_loop_eval_latest_models",
            kind="eval",
            output=eval_output,
            command=[
                str(PYTHON),
                "-m",
                "moon_rock_stack.run_structured_experiment",
                "--rocks",
                str(args.eval_rocks),
                "--rock-profile",
                args.rock_profile,
                "--clusters",
                str(args.clusters),
                "--trials",
                str(args.eval_trials),
                "--targets",
                args.targets,
                "--strategies",
                "statics_wall",
                "--gravities",
                args.gravities,
                "--candidates",
                str(args.candidates),
                "--steps-per-rock",
                str(args.steps_per_rock),
                "--hold-steps",
                str(args.hold_steps),
                "--candidate-probe-steps",
                str(args.candidate_probe_steps),
                "--workers",
                str(args.mujoco_workers),
                "--seed",
                str(args.seed + 5000),
                "--stone-fit-ranker-dir",
                str(stone_model_dir or args.eval_stone_ranker_dir or latest_dir(DEFAULT_STONE_FIT)),
                "--stone-fit-top-k",
                str(args.stone_fit_top_k),
                "--stone-fit-ranker-max-course",
                str(args.stone_fit_ranker_max_course),
                "--candidate-pose-ranker-dir",
                str(pose_model_dir or args.eval_pose_ranker_dir or latest_dir(DEFAULT_POSE_RANKER)),
                "--candidate-pose-top-k",
                str(args.candidate_pose_top_k),
                "--candidate-pose-ranker-max-course",
                str(args.candidate_pose_ranker_max_course),
                "--output",
                str(eval_output),
            ],
        )
        if args.candidate_probe_hard_gate:
            eval_job.command.append("--candidate-probe-hard-gate")
        if args.moon_gate_strict:
            eval_job.command.append("--moon-gate-strict")
        if args.low_release_search:
            eval_job.command.extend(
                [
                    "--low-release-search",
                    "--release-search-step-m",
                    str(args.release_search_step_m),
                    "--release-extra-clearance-m",
                    str(args.release_extra_clearance_m),
                ]
            )
        if args.base_support_prior:
            eval_job.command.extend(
                [
                    "--base-support-prior",
                    "--base-support-prior-weight",
                    str(args.base_support_prior_weight),
                ]
            )
        if args.base_continuity_prior:
            eval_job.command.extend(
                [
                    "--base-continuity-prior",
                    "--base-continuity-prior-weight",
                    str(args.base_continuity_prior_weight),
                ]
            )
        eval_risk_dir = pose_risk_model_dir or (args.eval_pose_risk_ranker_dir.resolve() if args.eval_pose_risk_ranker_dir else None)
        if eval_risk_dir and eval_risk_dir.exists():
            eval_job.command.extend(
                [
                    "--pose-risk-ranker-dir",
                    str(eval_risk_dir),
                    "--pose-risk-weight",
                    str(args.pose_risk_weight),
                    "--pose-risk-ranker-max-course",
                    str(args.pose_risk_ranker_max_course),
                ]
            )
        if args.eval_commit_best_rejected:
            eval_job.command.append("--commit-best-rejected")
        result = run_job(eval_job, log_dir, dry_run=args.dry_run)
        all_results.append(result)
        manifest["phases"].append({"name": "eval", "results": [result_to_json(result)]})
        write_json(manifest_path, manifest)
        if result.returncode == 0 and not args.skip_capture:
            capture_job = Job(
                name="capture_eval_cases",
                kind="capture",
                output=(result.parsed_output or result.output),
                command=[
                    str(PYTHON),
                    "scripts/capture_cases.py",
                    "--output",
                    str(result.parsed_output or result.output),
                    "--max-success",
                    "2",
                    "--max-failure",
                    "4",
                    "--width",
                    "960",
                    "--height",
                    "720",
                    "--capture-dir-name",
                    "captures_960x720",
                ],
            )
            capture_result = run_job(capture_job, log_dir, dry_run=args.dry_run)
            all_results.append(capture_result)
            manifest["phases"].append({"name": "capture", "results": [result_to_json(capture_result)]})
            write_json(manifest_path, manifest)

    manifest["finished_at"] = now()
    manifest["outputs"] = {
        "new_runs": new_run_names,
        "dataset_dir": str(dataset_dir) if dataset_dir else "",
        "tensor_dir": str(tensor_dir) if tensor_dir else "",
        "stone_model_dir": str(stone_model_dir) if stone_model_dir else "",
        "pose_model_dir": str(pose_model_dir) if pose_model_dir else "",
        "pose_risk_model_dir": str(pose_risk_model_dir) if pose_risk_model_dir else "",
        "critic_model_dir": str(critic_model_dir) if critic_model_dir else "",
    }
    write_json(manifest_path, manifest)
    write_summary(session_dir, manifest, all_results)
    print(session_dir)
    return 0 if all(result.returncode == 0 for result in all_results) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 3-4 course wall data flywheel for modular neural stacking.")
    parser.add_argument("--session", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=98001)
    parser.add_argument("--collect-batches", type=int, default=1)
    parser.add_argument("--collect-mode", choices=["mixed", "explore", "exploit"], default="mixed")
    parser.add_argument("--parallel-data-jobs", type=int, default=2)
    parser.add_argument("--mujoco-workers", type=int, default=2)
    parser.add_argument("--rocks", type=int, default=140)
    parser.add_argument("--eval-rocks", type=int, default=140)
    parser.add_argument("--rock-profile", default="high_wall")
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--eval-trials", type=int, default=1)
    parser.add_argument("--targets", default="single_face_wall_3course_v1,single_face_wall_4course_v1")
    parser.add_argument("--gravities", default="earth,moon")
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--steps-per-rock", type=int, default=420)
    parser.add_argument("--hold-steps", type=int, default=1800)
    parser.add_argument("--candidate-probe-steps", type=int, default=0)
    parser.add_argument("--candidate-probe-hard-gate", action="store_true")
    parser.add_argument("--moon-gate-strict", action="store_true")
    parser.add_argument(
        "--low-release-search",
        action="store_true",
        help="Pass-through to structured experiments: lower upper-course rocks to the lowest contact-free release height.",
    )
    parser.add_argument("--release-search-step-m", type=float, default=0.004)
    parser.add_argument("--release-extra-clearance-m", type=float, default=0.003)
    parser.add_argument(
        "--base-support-prior",
        action="store_true",
        help="Pass-through to structured experiments: bias the first wall course toward larger bearing stones.",
    )
    parser.add_argument("--base-support-prior-weight", type=float, default=1.0)
    parser.add_argument(
        "--base-continuity-prior",
        action="store_true",
        help="Pass-through to structured experiments: penalize base stones that block future wall slots.",
    )
    parser.add_argument("--base-continuity-prior-weight", type=float, default=1.0)
    parser.add_argument("--candidate-pose-top-k", type=int, default=3)
    parser.add_argument(
        "--candidate-pose-ranker-max-course",
        type=int,
        default=-1,
        help="Pass-through to structured experiments. -1 keeps candidate-pose ranker active for all courses.",
    )
    parser.add_argument("--stone-fit-top-k", type=int, default=10)
    parser.add_argument(
        "--stone-fit-ranker-max-course",
        type=int,
        default=-1,
        help="Pass-through to structured experiments. -1 keeps stone-fit ranker active for all courses.",
    )
    parser.add_argument(
        "--pose-risk-weight",
        type=float,
        default=0.0,
        help="Optional PoseRiskNet penalty weight passed to structured collection/evaluation.",
    )
    parser.add_argument(
        "--pose-risk-ranker-max-course",
        type=int,
        default=-1,
        help="Pass-through to structured experiments. -1 keeps PoseRiskNet active for all courses.",
    )
    parser.add_argument(
        "--collect-commit-best-rejected",
        action="store_true",
        help="Collection-only curriculum mode: commit the best rejected candidate instead of skipping empty slots.",
    )
    parser.add_argument(
        "--eval-commit-best-rejected",
        action="store_true",
        help="Evaluation curriculum mode. Keep off for strict success-rate reporting.",
    )
    parser.add_argument("--max-groups", type=int, default=768)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--window-m", type=float, default=0.9)
    parser.add_argument("--front-height-m", type=float, default=0.55)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--stone-epochs", type=int, default=90)
    parser.add_argument("--pose-risk-epochs", type=int, default=90)
    parser.add_argument("--pose-epochs", type=int, default=70)
    parser.add_argument("--critic-epochs", type=int, default=80)
    parser.add_argument("--stone-batch-size", type=int, default=96)
    parser.add_argument("--pose-risk-batch-size", type=int, default=160)
    parser.add_argument("--pose-batch-size", type=int, default=96)
    parser.add_argument("--critic-batch-size", type=int, default=128)
    parser.add_argument("--stone-hidden", type=int, default=112)
    parser.add_argument("--pose-risk-hidden", type=int, default=144)
    parser.add_argument("--pose-hidden", type=int, default=160)
    parser.add_argument("--critic-hidden", type=int, default=160)
    parser.add_argument(
        "--pose-risk-candidate-metric-labels",
        action="store_true",
        help="Train PoseRiskNet from each candidate pose's own disturbance/velocity/error metrics.",
    )
    parser.add_argument("--prior-run", action="append", default=[])
    parser.add_argument("--no-default-prior-runs", action="store_true")
    parser.add_argument("--dataset-target-contains", action="append", default=[])
    parser.add_argument("--exploit-stone-ranker-dir", type=Path)
    parser.add_argument("--exploit-pose-ranker-dir", type=Path)
    parser.add_argument("--exploit-pose-risk-ranker-dir", type=Path)
    parser.add_argument("--eval-stone-ranker-dir", type=Path)
    parser.add_argument("--eval-pose-ranker-dir", type=Path)
    parser.add_argument("--eval-pose-risk-ranker-dir", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--tensor-dir", type=Path)
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-depth-export", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--require-new-data", action="store_true")
    return parser.parse_args()


def build_data_jobs(args: argparse.Namespace, session: str, session_dir: Path) -> list[Job]:
    modes = ["explore", "exploit"] if args.collect_mode == "mixed" else [args.collect_mode]
    jobs: list[Job] = []
    for batch_id in range(max(0, args.collect_batches)):
        for mode_index, mode in enumerate(modes):
            seed = args.seed + 137 * batch_id + 9001 * mode_index
            output = BATCH_ROOT / f"{session}_collect_{mode}_{batch_id:02d}_seed{seed}"
            command = [
                str(PYTHON),
                "-m",
                "moon_rock_stack.run_structured_experiment",
                "--rocks",
                str(args.rocks),
                "--rock-profile",
                args.rock_profile,
                "--clusters",
                str(args.clusters),
                "--trials",
                str(args.trials),
                "--targets",
                args.targets,
                "--strategies",
                "statics_wall",
                "--gravities",
                args.gravities,
                "--candidates",
                str(args.candidates),
                "--steps-per-rock",
                str(args.steps_per_rock),
                "--hold-steps",
                str(args.hold_steps),
                "--candidate-probe-steps",
                str(args.candidate_probe_steps),
                "--workers",
                str(args.mujoco_workers),
                "--seed",
                str(seed),
                "--output",
                str(output),
            ]
            if args.candidate_probe_hard_gate:
                command.append("--candidate-probe-hard-gate")
            if args.moon_gate_strict:
                command.append("--moon-gate-strict")
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
            if args.base_continuity_prior:
                command.extend(
                    [
                        "--base-continuity-prior",
                        "--base-continuity-prior-weight",
                        str(args.base_continuity_prior_weight),
                    ]
                )
            if mode == "exploit":
                stone_dir = args.exploit_stone_ranker_dir.resolve() if args.exploit_stone_ranker_dir else latest_dir(DEFAULT_STONE_FIT)
                pose_dir = args.exploit_pose_ranker_dir.resolve() if args.exploit_pose_ranker_dir else latest_dir(DEFAULT_POSE_RANKER)
                risk_dir = args.exploit_pose_risk_ranker_dir.resolve() if args.exploit_pose_risk_ranker_dir else None
                if stone_dir.exists():
                    command.extend(
                        [
                            "--stone-fit-ranker-dir",
                            str(stone_dir),
                            "--stone-fit-top-k",
                            str(args.stone_fit_top_k),
                            "--stone-fit-ranker-max-course",
                            str(args.stone_fit_ranker_max_course),
                        ]
                    )
                if pose_dir.exists():
                    command.extend(
                        [
                            "--candidate-pose-ranker-dir",
                            str(pose_dir),
                            "--candidate-pose-top-k",
                            str(args.candidate_pose_top_k),
                            "--candidate-pose-ranker-max-course",
                            str(args.candidate_pose_ranker_max_course),
                        ]
                    )
                if risk_dir and risk_dir.exists():
                    command.extend(
                        [
                            "--pose-risk-ranker-dir",
                            str(risk_dir),
                            "--pose-risk-weight",
                            str(args.pose_risk_weight),
                            "--pose-risk-ranker-max-course",
                            str(args.pose_risk_ranker_max_course),
                        ]
                    )
            if args.collect_commit_best_rejected:
                command.append("--commit-best-rejected")
            jobs.append(Job(name=f"collect_{mode}_{batch_id:02d}", kind=f"collect_{mode}", command=command, output=output))
    write_json(session_dir / "collect_jobs.json", {"jobs": [job_to_json(job) for job in jobs]})
    return jobs


def run_jobs(jobs: list[Job], log_dir: Path, max_workers: int, dry_run: bool) -> list[JobResult]:
    if not jobs:
        return []
    if dry_run:
        return [dry_run_result(job, log_dir) for job in jobs]
    max_workers = max(1, min(int(max_workers), len(jobs)))
    results: list[JobResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_job, job, log_dir, False) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item.name)


def run_job(job: Job, log_dir: Path, dry_run: bool = False) -> JobResult:
    stdout_path = log_dir / f"{job.name}.stdout.log"
    stderr_path = log_dir / f"{job.name}.stderr.log"
    started_at = now()
    start = time.time()
    if dry_run:
        stdout_path.write_text("DRY RUN\n" + " ".join(job.command) + "\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return JobResult(
            name=job.name,
            kind=job.kind,
            command=job.command,
            output=job.output,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            returncode=0,
            started_at=started_at,
            finished_at=now(),
            elapsed_seconds=0.0,
            parsed_output=job.output,
        )
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONUTF8"] = "1"
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.run(job.command, cwd=REPO, env=env, stdout=stdout, stderr=stderr, check=False)
    finished_at = now()
    parsed = parse_printed_output_dir(stdout_path)
    return JobResult(
        name=job.name,
        kind=job.kind,
        command=job.command,
        output=job.output,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        returncode=int(process.returncode),
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=time.time() - start,
        parsed_output=parsed,
    )


def dry_run_result(job: Job, log_dir: Path) -> JobResult:
    return run_job(job, log_dir, dry_run=True)


def parse_printed_output_dir(stdout_path: Path) -> Path | None:
    if not stdout_path.exists():
        return None
    lines = [line.strip() for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    for line in reversed(lines):
        candidate = Path(line)
        if candidate.exists():
            return candidate.resolve()
        if line.startswith("{"):
            continue
    return None


def latest_dir(name: str) -> Path:
    path = BATCH_ROOT / name
    if path.exists():
        return path.resolve()
    matches = sorted(BATCH_ROOT.glob(f"{name}*"), key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0].resolve() if matches else path.resolve()


def existing_run_names(names: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not name:
            continue
        run_name = Path(name).name
        if run_name in seen:
            continue
        if (BATCH_ROOT / run_name / "results.csv").exists():
            output.append(run_name)
            seen.add(run_name)
    return output


def run_name_from_output(path: Path | None) -> str:
    return path.name if path is not None else ""


def flatten_run_args(run_names: list[str]) -> list[str]:
    args: list[str] = []
    for run_name in run_names:
        args.extend(["--run", run_name])
    return args


def flatten_repeat_args(flag: str, values: list[str]) -> list[str]:
    args: list[str] = []
    for value in values:
        if value:
            args.extend([flag, value])
    return args


def dataset_assignment_candidate_count(dataset_dir: Path) -> int:
    summary_path = dataset_dir / "dataset_summary.json"
    if not summary_path.exists():
        return -1
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return -1
    return int(data.get("assignment_candidate_example_count", -1))


def resolve_model_fallback(*paths_or_names: Path | str | None) -> Path:
    for item in paths_or_names:
        if not item:
            continue
        path = Path(item)
        if not path.is_absolute():
            path = (REPO / path).resolve()
        if path.exists():
            return path
        candidate = latest_dir(str(item))
        if candidate.exists():
            return candidate
    return latest_dir(DEFAULT_STONE_FIT)


def write_skip_result(log_dir: Path, name: str, kind: str, output: Path, message: str) -> JobResult:
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    started_at = now()
    stdout_path.write_text(message + "\n" + str(output) + "\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    return JobResult(
        name=name,
        kind=kind,
        command=[],
        output=output,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        returncode=0,
        started_at=started_at,
        finished_at=now(),
        elapsed_seconds=0.0,
        parsed_output=output,
    )


def job_to_json(job: Job) -> dict[str, Any]:
    return {
        "name": job.name,
        "kind": job.kind,
        "command": job.command,
        "output": str(job.output) if job.output else "",
    }


def result_to_json(result: JobResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "kind": result.kind,
        "command": result.command,
        "output": str(result.output) if result.output else "",
        "parsed_output": str(result.parsed_output) if result.parsed_output else "",
        "stdout": str(result.stdout_path),
        "stderr": str(result.stderr_path),
        "returncode": result.returncode,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
    }


def write_summary(session_dir: Path, manifest: dict[str, Any], results: list[JobResult]) -> None:
    lines = [
        "# Wall Data Flywheel Session",
        "",
        f"- session: `{manifest.get('session', '')}`",
        f"- created_at: `{manifest.get('created_at', '')}`",
        f"- finished_at: `{manifest.get('finished_at', now())}`",
        f"- failed jobs: `{sum(result.returncode != 0 for result in results)}`",
        "",
        "## Outputs",
        "",
    ]
    outputs = manifest.get("outputs", {})
    for key, value in outputs.items():
        if isinstance(value, list):
            lines.append(f"- {key}: {', '.join(f'`{item}`' for item in value)}")
        else:
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Jobs", ""])
    for result in results:
        lines.append(
            f"- `{result.name}` {result.kind}: returncode={result.returncode}, "
            f"elapsed={result.elapsed_seconds:.1f}s, output=`{result.parsed_output or result.output}`"
        )
    (session_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
