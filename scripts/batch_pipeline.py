from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RunConfig:
    name: str
    rocks: int
    clusters: int
    trials: int
    candidates: int
    steps_per_rock: int
    hold_steps: int
    stack_rocks: int
    strategies: str


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    run_root = root / "batch_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    configs = [
        RunConfig(
            name="baseline_vs_filter_18r",
            rocks=18,
            clusters=5,
            trials=4,
            candidates=8,
            steps_per_rock=1800,
            hold_steps=3200,
            stack_rocks=14,
            strategies="paper_baseline,physics_filter,risk_aware",
        ),
        RunConfig(
            name="more_candidates_18r",
            rocks=18,
            clusters=5,
            trials=4,
            candidates=14,
            steps_per_rock=1800,
            hold_steps=3200,
            stack_rocks=14,
            strategies="paper_baseline,physics_filter,risk_aware",
        ),
        RunConfig(
            name="larger_library_24r",
            rocks=24,
            clusters=6,
            trials=4,
            candidates=10,
            steps_per_rock=1800,
            hold_steps=3400,
            stack_rocks=16,
            strategies="paper_baseline,physics_filter,risk_aware",
        ),
        RunConfig(
            name="stress_stack_30r",
            rocks=30,
            clusters=6,
            trials=3,
            candidates=12,
            steps_per_rock=2200,
            hold_steps=4200,
            stack_rocks=20,
            strategies="paper_baseline,physics_filter,risk_aware",
        ),
    ]

    manifest_path = run_root / "manifest.csv"
    manifest_fields = [
        "name",
        "output",
        "start_time",
        "end_time",
        "elapsed_sec",
        "returncode",
        "command",
        "config_json",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=manifest_fields)
        writer.writeheader()
        for config in configs:
            output = run_root / config.name
            command = [
                sys.executable,
                "-m",
                "moon_rock_stack.run_experiment",
                "--rocks",
                str(config.rocks),
                "--clusters",
                str(config.clusters),
                "--trials",
                str(config.trials),
                "--candidates",
                str(config.candidates),
                "--steps-per-rock",
                str(config.steps_per_rock),
                "--hold-steps",
                str(config.hold_steps),
                "--stack-rocks",
                str(config.stack_rocks),
                "--strategies",
                config.strategies,
                "--output",
                str(output),
            ]
            start = datetime.now()
            t0 = time.perf_counter()
            stdout_path = run_root / f"{config.name}.stdout.txt"
            stderr_path = run_root / f"{config.name}.stderr.txt"
            with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_file:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    shell=False,
                    check=False,
                )
            end = datetime.now()
            elapsed = time.perf_counter() - t0
            writer.writerow(
                {
                    "name": config.name,
                    "output": str(output),
                    "start_time": start.isoformat(timespec="seconds"),
                    "end_time": end.isoformat(timespec="seconds"),
                    "elapsed_sec": f"{elapsed:.3f}",
                    "returncode": completed.returncode,
                    "command": " ".join(command),
                    "config_json": json.dumps(asdict(config), sort_keys=True),
                }
            )
            manifest_file.flush()

    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
