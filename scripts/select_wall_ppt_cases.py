from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(r"D:\MoonStack\experiments\moon_rock_stack\batch_runs")


def as_float(value: str | None, default: float = -1.0) -> float:
    try:
        return float(value or "")
    except ValueError:
        return default


def as_int(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value or ""))
    except ValueError:
        return default


def find_captures(
    run_dir: Path,
    stem: str,
    target: str,
    strategy: str,
    gravity: str,
    trial: str,
) -> list[str]:
    matches: list[str] = []
    trial_suffix = f"trial_{as_int(trial):02d}" if trial != "" else ""
    tokens = [target, strategy, gravity, trial_suffix]
    tokens = [token.lower() for token in tokens if token]
    if not stem and not tokens:
        return matches
    for capture_dir in run_dir.rglob("*"):
        if not capture_dir.is_dir():
            continue
        name = capture_dir.name.lower()
        if stem and stem.lower() in name:
            pass
        elif tokens and all(token in name for token in tokens):
            pass
        else:
            continue
        if (
            (capture_dir / "wall_front_rgb.png").exists()
            or (capture_dir / "front_rgb.png").exists()
        ):
            matches.append(str(capture_dir))
    return matches


def main() -> None:
    rows: list[dict[str, object]] = []
    for results_path in ROOT.rglob("results.csv"):
        with results_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                target = row.get("target_name", "")
                strategy = row.get("strategy", "")
                if "wall" not in target.lower() and "wall" not in strategy.lower():
                    continue
                stem = Path(row.get("state_path") or row.get("xml") or "").stem
                captures = find_captures(
                    results_path.parent,
                    stem,
                    target,
                    strategy,
                    row.get("gravity", ""),
                    row.get("trial", ""),
                )
                rows.append(
                    {
                        "run": str(results_path.parent),
                        "gravity": row.get("gravity", ""),
                        "strategy": strategy,
                        "target": target,
                        "success": row.get("success", ""),
                        "shape_success": row.get("shape_success", ""),
                        "visible_courses": row.get("visible_courses", ""),
                        "height_m": row.get("stack_height_m", ""),
                        "stable_count": row.get("stable_count", ""),
                        "failure_count": row.get("failure_count", ""),
                        "rock_count": row.get("rock_count", ""),
                        "rmse_xy_m": row.get("target_rmse_xy_m", ""),
                        "structure_score": row.get("structure_score", ""),
                        "stem": stem,
                        "captures": captures[:4],
                    }
                )

    def rank(item: dict[str, object]) -> tuple[int, int, float, float]:
        return (
            1 if item["captures"] else 0,
            as_int(str(item["success"])),
            as_float(str(item["visible_courses"])),
            as_float(str(item["height_m"])),
        )

    rows.sort(key=rank, reverse=True)
    print(f"wall_rows={len(rows)}")
    for item in rows[:100]:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
