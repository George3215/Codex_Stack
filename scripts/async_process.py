from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def main() -> int:
    args = parse_args()
    if args.command == "start":
        return start_job(args)
    if args.command == "status":
        return status_job(args)
    if args.command == "list":
        return list_jobs(args)
    raise ValueError(args.command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start and inspect detached MoonStack jobs.")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Start a detached process and write job metadata.")
    start.add_argument("--job-root", type=Path, default=Path("batch_runs/async_jobs"))
    start.add_argument("--job-name", required=True)
    start.add_argument("--cwd", type=Path, default=Path.cwd())
    start.add_argument("--env", action="append", default=[], help="Extra KEY=VALUE environment entries.")
    start.add_argument("argv", nargs=argparse.REMAINDER, help="Command to run after --.")

    status = sub.add_parser("status", help="Print one job status JSON.")
    status.add_argument("--job-json", type=Path, required=True)

    list_cmd = sub.add_parser("list", help="List jobs under a root.")
    list_cmd.add_argument("--job-root", type=Path, default=Path("batch_runs/async_jobs"))
    return parser.parse_args()


def start_job(args: argparse.Namespace) -> int:
    argv = normalize_argv(args.argv)
    if not argv:
        raise SystemExit("No command provided. Use: start ... -- <command> <args>")
    job_root = args.job_root.resolve()
    job_dir = unique_job_dir(job_root, args.job_name)
    job_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    env = os.environ.copy()
    for entry in args.env:
        if "=" not in entry:
            raise SystemExit(f"Invalid --env entry: {entry}")
        key, value = entry.split("=", 1)
        env[key] = value

    stdout_handle = stdout_path.open("wb")
    stderr_handle = stderr_path.open("wb")
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    proc = subprocess.Popen(
        argv,
        cwd=str(args.cwd.resolve()),
        stdout=stdout_handle,
        stderr=stderr_handle,
        stdin=subprocess.DEVNULL,
        env=env,
        creationflags=creationflags,
        close_fds=True,
    )
    stdout_handle.close()
    stderr_handle.close()
    metadata = {
        "job_name": args.job_name,
        "job_dir": str(job_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pid": int(proc.pid),
        "cwd": str(args.cwd.resolve()),
        "argv": argv,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "status": "running",
    }
    write_json(job_dir / "job.json", metadata)
    print(json.dumps({**metadata, "running": True}, indent=2))
    return 0


def status_job(args: argparse.Namespace) -> int:
    metadata = json.loads(args.job_json.read_text(encoding="utf-8-sig"))
    pid = metadata.get("pid")
    metadata["running"] = is_pid_running(int(pid)) if pid is not None else False
    metadata["status"] = "running" if metadata["running"] else "exited_or_unknown"
    if pid is None:
        metadata["status_note"] = "missing_pid_in_legacy_job_json"
    print(json.dumps(metadata, indent=2))
    return 0


def list_jobs(args: argparse.Namespace) -> int:
    jobs: list[dict[str, Any]] = []
    for job_json in sorted(args.job_root.resolve().glob("*/job.json")):
        metadata = json.loads(job_json.read_text(encoding="utf-8-sig"))
        pid = metadata.get("pid")
        metadata["running"] = is_pid_running(int(pid)) if pid is not None else False
        metadata["status"] = "running" if metadata["running"] else "exited_or_unknown"
        if pid is None:
            metadata["status_note"] = "missing_pid_in_legacy_job_json"
        jobs.append(metadata)
    print(json.dumps(jobs, indent=2))
    return 0


def normalize_argv(argv: list[str]) -> list[str]:
    if argv and argv[0] == "--":
        return argv[1:]
    return argv


def unique_job_dir(root: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = root / f"{stamp}_{safe_name(name)}"
    if not base.exists():
        return base
    suffix = 2
    while True:
        candidate = root / f"{stamp}_{safe_name(name)}_{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value.strip())
    return cleaned[:80] or "job"


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        if f'"{pid}"' in result.stdout or f",{pid}," in result.stdout:
            return True
        ps = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in ps.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
