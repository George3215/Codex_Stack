from __future__ import annotations

import argparse
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
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run_root = root / "batch_runs" / args.name
    if run_root.exists():
        run_root = root / "batch_runs" / f"{args.name}_{datetime.now().strftime('%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)

    configs = quick_configs() if args.quick else standard_configs()
    write_protocol(run_root / "PROTOCOL.md", configs)
    manifest_path = run_root / "manifest.csv"
    fields = [
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
        writer = csv.DictWriter(manifest_file, fieldnames=fields)
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
            stdout_path = run_root / f"{config.name}.stdout.txt"
            stderr_path = run_root / f"{config.name}.stderr.txt"
            start = datetime.now()
            t0 = time.perf_counter()
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
            elapsed = time.perf_counter() - t0
            end = datetime.now()
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
            if completed.returncode != 0:
                print(f"Run failed: {config.name}; see {stderr_path}", file=sys.stderr)
                return completed.returncode

    print(run_root)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run angular polyhedral rock stacking experiments.")
    parser.add_argument("--name", default=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_angular_poly")
    parser.add_argument("--quick", action="store_true", help="Run a short data-collection pass.")
    return parser.parse_args()


def quick_configs() -> list[RunConfig]:
    return [
        RunConfig(
            name="angular_quick_18r",
            rocks=18,
            clusters=5,
            trials=2,
            candidates=6,
            steps_per_rock=1000,
            hold_steps=2200,
            stack_rocks=10,
            strategies="paper_baseline,physics_filter,support_first,risk_aware",
        )
    ]


def standard_configs() -> list[RunConfig]:
    return [
        RunConfig(
            name="angular_core_20r",
            rocks=20,
            clusters=5,
            trials=3,
            candidates=8,
            steps_per_rock=1400,
            hold_steps=3000,
            stack_rocks=12,
            strategies="paper_baseline,physics_filter,support_first,risk_aware",
        ),
        RunConfig(
            name="angular_candidate_sweep_24r",
            rocks=24,
            clusters=6,
            trials=2,
            candidates=14,
            steps_per_rock=1500,
            hold_steps=3200,
            stack_rocks=14,
            strategies="physics_filter,support_first,risk_aware",
        ),
    ]


def write_protocol(path: Path, configs: list[RunConfig]) -> None:
    lines = [
        "# Angular Polyhedral Moon/Earth Stacking Protocol",
        "",
        "This batch uses non-slab angular polyhedral rocks. It follows the literature path summarized in `D:/MoonStack/Asset/Papers/README.md`: geometry-aware ordering, physics-based candidate pose filtering, stability-aware sequence adjustment, and matched Earth/Moon gravity comparisons.",
        "",
        "## Literature-Derived Rules",
        "",
        "- Use the same rock library and random seeds for Earth and Moon gravity so the comparison isolates gravity effects.",
        "- Compare interpretable geometry heuristics against MuJoCo candidate-pose filtering.",
        "- Use sequence risk feedback from earlier placements/failures for `risk_aware` planning.",
        "- Record per-rock geometric features and per-placement stability proxies for later data-driven strategy updates.",
        "",
        "## Configs",
        "",
    ]
    for config in configs:
        lines.append(f"- `{config.name}`: `{json.dumps(asdict(config), sort_keys=True)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
