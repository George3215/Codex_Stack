from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BATCH_ROOT = REPO / "batch_runs"
PYTHON = Path(sys.executable)


def main() -> int:
    args = parse_args()
    session_dir = (BATCH_ROOT / args.session).resolve()
    log_dir = session_dir / "logs"
    session_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs()
    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "session_dir": str(session_dir),
        "python": str(PYTHON),
        "cwd": str(REPO),
        "policy": "two MuJoCo CPU data jobs plus three CUDA training sweeps; keep system reserve instead of intentionally exhausting RAM/VRAM",
        "jobs": [],
    }
    if args.foreground:
        manifest_path = session_dir / "jobs_manifest.json"
        return run_foreground(jobs, session_dir, log_dir, manifest_path, poll_seconds=args.poll_seconds)

    for job in jobs:
        stdout_path = log_dir / f"{job['name']}.stdout.log"
        stderr_path = log_dir / f"{job['name']}.stderr.log"
        command = [str(PYTHON), *job["args"]]
        entry = {
            "name": job["name"],
            "kind": job["kind"],
            "command": command,
            "output": job.get("output", ""),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        if args.dry_run:
            entry["pid"] = None
            entry["status"] = "dry_run"
        else:
            stdout = stdout_path.open("ab")
            stderr = stderr_path.open("ab")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    stdout=stdout,
                    stderr=stderr,
                    stdin=subprocess.DEVNULL,
                    creationflags=detached_creationflags(),
                )
            finally:
                stdout.close()
                stderr.close()
            entry["pid"] = int(process.pid)
            entry["status"] = "started"
        manifest["jobs"].append(entry)

    manifest_path = session_dir / "jobs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch bounded parallel MoonStack jobs for modular-network experiments.")
    parser.add_argument("--session", default="parallel_jobs_20260619_modular_highwall_v1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--foreground", action="store_true", help="Run a foreground supervisor that keeps child jobs alive and writes status snapshots.")
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def run_foreground(
    jobs: list[dict[str, Any]],
    session_dir: Path,
    log_dir: Path,
    manifest_path: Path,
    poll_seconds: int,
) -> int:
    processes: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "session_dir": str(session_dir),
        "python": str(PYTHON),
        "cwd": str(REPO),
        "policy": "foreground supervisor for two MuJoCo CPU data jobs plus three CUDA training sweeps",
        "jobs": [],
    }
    for job in jobs:
        stdout_path = log_dir / f"{job['name']}.stdout.log"
        stderr_path = log_dir / f"{job['name']}.stderr.log"
        command = [str(PYTHON), *job["args"]]
        stdout = stdout_path.open("ab")
        stderr = stderr_path.open("ab")
        process = subprocess.Popen(command, cwd=REPO, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL)
        entry = {
            "name": job["name"],
            "kind": job["kind"],
            "command": command,
            "output": job.get("output", ""),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "pid": int(process.pid),
            "status": "started",
        }
        manifest["jobs"].append(entry)
        processes.append({"job": job, "entry": entry, "process": process, "stdout": stdout, "stderr": stderr})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    poll_seconds = max(5, int(poll_seconds))
    status_path = session_dir / "status.jsonl"
    print(manifest_path, flush=True)
    while True:
        snapshot = status_snapshot(processes)
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=True) + "\n")
        print(
            f"{snapshot['created_at']} running={snapshot['running_count']} done={snapshot['done_count']} "
            f"failed={snapshot['failed_count']} gpu='{snapshot['gpu']}'",
            flush=True,
        )
        if snapshot["running_count"] == 0:
            break
        time.sleep(poll_seconds)

    for item in processes:
        item["stdout"].close()
        item["stderr"].close()
    return 0 if status_snapshot(processes)["failed_count"] == 0 else 1


def status_snapshot(processes: list[dict[str, Any]]) -> dict[str, Any]:
    jobs = []
    running = 0
    done = 0
    failed = 0
    for item in processes:
        process: subprocess.Popen[Any] = item["process"]
        returncode = process.poll()
        if returncode is None:
            status = "running"
            running += 1
        elif returncode == 0:
            status = "done"
            done += 1
        else:
            status = "failed"
            failed += 1
        entry = item["entry"]
        jobs.append(
            {
                "name": entry["name"],
                "kind": entry["kind"],
                "pid": entry["pid"],
                "status": status,
                "returncode": returncode,
                "output_exists": bool(entry.get("output") and Path(entry["output"]).exists()),
                "stdout_bytes": file_size(entry["stdout"]),
                "stderr_bytes": file_size(entry["stderr"]),
            }
        )
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "running_count": running,
        "done_count": done,
        "failed_count": failed,
        "gpu": gpu_status(),
        "jobs": jobs,
    }


def file_size(path: str) -> int:
    candidate = Path(path)
    return candidate.stat().st_size if candidate.exists() else 0


def gpu_status() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            cwd=REPO,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return f"nvidia-smi-error:{exc}"
    return result.stdout.strip().replace("\n", " | ")


def build_jobs() -> list[dict[str, Any]]:
    return [
        {
            "name": "cpu_fullcandidate_highwall_seed93014",
            "kind": "mujoco_fullcandidate",
            "output": str(BATCH_ROOT / "20260619_newrocks_highwall_seed93014_fullcandidates_highonly_parallel_v1"),
            "args": [
                "-m",
                "moon_rock_stack.run_structured_experiment",
                "--rocks",
                "220",
                "--rock-profile",
                "high_wall",
                "--clusters",
                "12",
                "--trials",
                "1",
                "--seed",
                "93014",
                "--targets",
                "single_face_wall_high_v1",
                "--strategies",
                "statics_wall",
                "--gravities",
                "earth,moon",
                "--candidates",
                "5",
                "--steps-per-rock",
                "280",
                "--hold-steps",
                "700",
                "--workers",
                "2",
                "--output",
                str(BATCH_ROOT / "20260619_newrocks_highwall_seed93014_fullcandidates_highonly_parallel_v1"),
            ],
        },
        {
            "name": "cpu_fullcandidate_screening_seed93015",
            "kind": "mujoco_fullcandidate",
            "output": str(BATCH_ROOT / "20260619_newrocks_screening_seed93015_fullcandidates_highonly_parallel_v1"),
            "args": [
                "-m",
                "moon_rock_stack.run_structured_experiment",
                "--rocks",
                "220",
                "--rock-profile",
                "screening_stress",
                "--clusters",
                "14",
                "--trials",
                "1",
                "--seed",
                "93015",
                "--targets",
                "single_face_wall_high_v1",
                "--strategies",
                "statics_wall",
                "--gravities",
                "earth,moon",
                "--candidates",
                "5",
                "--steps-per-rock",
                "280",
                "--hold-steps",
                "700",
                "--workers",
                "2",
                "--output",
                str(BATCH_ROOT / "20260619_newrocks_screening_seed93015_fullcandidates_highonly_parallel_v1"),
            ],
        },
        {
            "name": "gpu_quality_temp20_dropout30",
            "kind": "torch_train",
            "output": str(BATCH_ROOT / "20260619_torch_support_map_cnn_quality_ranker_temp20_dropout30_holdout93013_v1"),
            "args": [
                "scripts/train_torch_support_map_ranker.py",
                "--tensor-dir",
                str(BATCH_ROOT / "20260619_local_support_maps_multicatalog_wall_plus93013_v1"),
                "--output",
                str(BATCH_ROOT / "20260619_torch_support_map_cnn_quality_ranker_temp20_dropout30_holdout93013_v1"),
                "--epochs",
                "90",
                "--batch-size",
                "96",
                "--hidden",
                "160",
                "--dropout",
                "0.30",
                "--lr",
                "0.001",
                "--weight-decay",
                "0.0004",
                "--device",
                "cuda",
                "--amp",
                "--exclude-postsim-features",
                "--target-mode",
                "score",
                "--quality-temperature",
                "20",
                "--test-run-name",
                "20260619_newrocks_highwall_seed93013_fullcandidates_highonly_fg_v1",
            ],
        },
        {
            "name": "gpu_quality_temp55_dropout20",
            "kind": "torch_train",
            "output": str(BATCH_ROOT / "20260619_torch_support_map_cnn_quality_ranker_temp55_dropout20_holdout93013_v1"),
            "args": [
                "scripts/train_torch_support_map_ranker.py",
                "--tensor-dir",
                str(BATCH_ROOT / "20260619_local_support_maps_multicatalog_wall_plus93013_v1"),
                "--output",
                str(BATCH_ROOT / "20260619_torch_support_map_cnn_quality_ranker_temp55_dropout20_holdout93013_v1"),
                "--epochs",
                "90",
                "--batch-size",
                "96",
                "--hidden",
                "160",
                "--dropout",
                "0.20",
                "--lr",
                "0.001",
                "--weight-decay",
                "0.00025",
                "--device",
                "cuda",
                "--amp",
                "--exclude-postsim-features",
                "--target-mode",
                "score",
                "--quality-temperature",
                "55",
                "--test-run-name",
                "20260619_newrocks_highwall_seed93013_fullcandidates_highonly_fg_v1",
            ],
        },
        {
            "name": "gpu_selected_dropout30_hidden160",
            "kind": "torch_train",
            "output": str(BATCH_ROOT / "20260619_torch_support_map_cnn_selected_ranker_dropout30_hidden160_holdout93013_v1"),
            "args": [
                "scripts/train_torch_support_map_ranker.py",
                "--tensor-dir",
                str(BATCH_ROOT / "20260619_local_support_maps_multicatalog_wall_plus93013_v1"),
                "--output",
                str(BATCH_ROOT / "20260619_torch_support_map_cnn_selected_ranker_dropout30_hidden160_holdout93013_v1"),
                "--epochs",
                "90",
                "--batch-size",
                "96",
                "--hidden",
                "160",
                "--dropout",
                "0.30",
                "--lr",
                "0.001",
                "--weight-decay",
                "0.0004",
                "--device",
                "cuda",
                "--amp",
                "--exclude-postsim-features",
                "--target-mode",
                "selected",
                "--test-run-name",
                "20260619_newrocks_highwall_seed93013_fullcandidates_highonly_fg_v1",
            ],
        },
    ]


def detached_creationflags() -> int:
    if sys.platform != "win32":
        return 0
    detached_process = 0x00000008
    create_new_process_group = 0x00000200
    create_no_window = 0x08000000
    return detached_process | create_new_process_group | create_no_window


if __name__ == "__main__":
    raise SystemExit(main())
