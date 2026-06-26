from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"C:\Users\all\miniconda3\envs\moon-rock-stack\python.exe")
BATCH_ROOT = ROOT / "batch_runs"


def main() -> int:
    args = parse_args()
    session = args.train_session
    eval_session = args.eval_session
    log_path = args.log.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    train_manifest = BATCH_ROOT / session / "flywheel_manifest.json"
    stone_dir = BATCH_ROOT / f"{session}_stone_slot_net"
    pose_dir = BATCH_ROOT / f"{session}_pose_ranker_structure"
    risk_dir = BATCH_ROOT / f"{session}_pose_risk_net"

    required = [
        train_manifest,
        stone_dir / "stone_fit_net.npz",
        pose_dir / "support_map_cnn_ranker.pt",
        risk_dir / "pose_risk_net.npz",
    ]

    log(log_path, f"watch_start train_session={session} eval_session={eval_session}")
    for attempt in range(1, args.max_checks + 1):
        missing = [str(path) for path in required if not path.exists()]
        manifest_finished = manifest_has_finished(train_manifest)
        gpu_used = gpu_memory_used_mib()
        active_train = active_process_contains(session)
        log(
            log_path,
            "check "
            f"attempt={attempt} missing={len(missing)} manifest_finished={manifest_finished} "
            f"gpu_used_mib={gpu_used} active_train={active_train}",
        )

        ready = not missing and manifest_finished and gpu_used >= 0 and gpu_used <= args.gpu_threshold_mib
        if ready:
            break
        if attempt == args.max_checks:
            log(log_path, "max_checks_reached no_eval_started")
            return 2
        time.sleep(args.poll_seconds)

    output = BATCH_ROOT / eval_session
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
        str(args.workers),
        "--seed",
        str(args.seed),
        "--stone-fit-ranker-dir",
        str(stone_dir),
        "--stone-fit-top-k",
        str(args.stone_fit_top_k),
        "--stone-fit-ranker-max-course",
        str(args.ranker_max_course),
        "--candidate-pose-ranker-dir",
        str(pose_dir),
        "--candidate-pose-top-k",
        str(args.candidate_pose_top_k),
        "--candidate-pose-ranker-max-course",
        str(args.ranker_max_course),
        "--pose-risk-ranker-dir",
        str(risk_dir),
        "--pose-risk-weight",
        str(args.pose_risk_weight),
        "--pose-risk-ranker-max-course",
        str(args.ranker_max_course),
        "--output",
        str(output),
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
        command.extend(["--base-support-prior", "--base-support-prior-weight", str(args.base_support_prior_weight)])
    if args.base_continuity_prior:
        command.extend(
            ["--base-continuity-prior", "--base-continuity-prior-weight", str(args.base_continuity_prior_weight)]
        )

    log(log_path, "eval_start " + " ".join(command))
    result = subprocess.run(command, cwd=ROOT)
    log(log_path, f"eval_finished returncode={result.returncode} output={output}")
    return int(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for clean wall models, then run closed-loop wall evaluation.")
    parser.add_argument("--train-session", default="20260626_clean_wall34_supportmap_train_v1")
    parser.add_argument("--eval-session", default="20260626_clean_wall34_supportmap_eval_v1")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-checks", type=int, default=96)
    parser.add_argument("--gpu-threshold-mib", type=int, default=6500)
    parser.add_argument("--rocks", type=int, default=128)
    parser.add_argument("--rock-profile", default="high_wall")
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--targets", default="single_face_wall_3course_v1,single_face_wall_4course_v1")
    parser.add_argument("--gravities", default="moon")
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--steps-per-rock", type=int, default=420)
    parser.add_argument("--hold-steps", type=int, default=1800)
    parser.add_argument("--candidate-probe-steps", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=206266201)
    parser.add_argument("--stone-fit-top-k", type=int, default=6)
    parser.add_argument("--candidate-pose-top-k", type=int, default=1)
    parser.add_argument("--ranker-max-course", type=int, default=-1)
    parser.add_argument("--pose-risk-weight", type=float, default=0.65)
    parser.add_argument("--low-release-search", action="store_true")
    parser.add_argument("--release-search-step-m", type=float, default=0.004)
    parser.add_argument("--release-extra-clearance-m", type=float, default=0.003)
    parser.add_argument("--base-support-prior", action="store_true")
    parser.add_argument("--base-support-prior-weight", type=float, default=1.0)
    parser.add_argument("--base-continuity-prior", action="store_true")
    parser.add_argument("--base-continuity-prior-weight", type=float, default=0.35)
    return parser.parse_args()


def manifest_has_finished(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(data.get("finished_at"))


def gpu_memory_used_mib() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return -1
    if result.returncode != 0:
        return -1
    text = result.stdout.strip().splitlines()
    if not text:
        return -1
    try:
        return int(float(text[0].strip()))
    except ValueError:
        return -1


def active_process_contains(text: str) -> bool:
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | Select-Object -ExpandProperty CommandLine",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return any(text in line for line in result.stdout.splitlines())


def log(path: Path, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")


if __name__ == "__main__":
    raise SystemExit(main())
