from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    batch_root = root / "batch_runs"
    rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    output_dirs = sorted({path.parent for path in batch_root.glob("**/results.csv")} | {path.parent for path in batch_root.glob("**/summary.json")})
    for output_dir in output_dirs:
        run_group = output_dir.parent.name
        run_name = output_dir.name
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = summarize_results_csv(output_dir / "results.csv")
        else:
            summary = summarize_results_csv(output_dir / "results.csv")
        for strategy, gravity_data in summary.items():
            for gravity, metrics in gravity_data.items():
                rows.append(
                    {
                        "run_group": run_group,
                        "run_name": run_name,
                        "strategy": strategy,
                        "gravity": gravity,
                        **metrics,
                        "summary_path": str(summary_path) if summary_path.exists() else "",
                    }
                )

        failure_path = output_dir / "failure_by_cluster.csv"
        if failure_path.exists():
            with failure_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    row["run_group"] = run_group
                    row["run_name"] = run_name
                    failure_rows.append(row)

    aggregate_dir = batch_root / "_aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    write_csv(aggregate_dir / "summary_all.csv", rows)
    write_csv(aggregate_dir / "failure_by_cluster_all.csv", failure_rows)
    write_markdown(aggregate_dir / "README.md", rows, failure_rows)
    print(aggregate_dir)
    return 0


def summarize_results_csv(results_path: Path) -> dict[str, dict[str, dict[str, float]]]:
    if not results_path.exists():
        return {}
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for strategy in sorted({row["strategy"] for row in rows}):
        summary[strategy] = {}
        for gravity in sorted({row["gravity"] for row in rows if row["strategy"] == strategy}):
            subset = [row for row in rows if row["strategy"] == strategy and row["gravity"] == gravity]
            if not subset:
                continue
            summary[strategy][gravity] = {
                "trials": float(len(subset)),
                "success_rate": sum(float(row["success"]) for row in subset) / len(subset),
                "mean_stable_count": sum(float(row["stable_count"]) for row in subset) / len(subset),
                "mean_failure_count": sum(float(row["failure_count"]) for row in subset) / len(subset),
                "mean_stack_height_m": sum(float(row["stack_height_m"]) for row in subset) / len(subset),
                "mean_max_drift_m": sum(float(row["max_horizontal_drift_m"]) for row in subset) / len(subset),
                "mean_velocity_inf_norm": sum(float(row["velocity_inf_norm"]) for row in subset) / len(subset),
            }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Moon Rock Stack Batch Aggregate",
        "",
        "This file is generated without deleting or modifying individual experiment outputs.",
        "",
        "## Best Runs By Stable Count",
        "",
    ]
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row.get("mean_stable_count", 0.0)),
            float(row.get("mean_failure_count", 999.0)),
            -float(row.get("success_rate", 0.0)),
        ),
    )
    for row in ranked[:12]:
        lines.append(
            f"- `{row['run_group']}/{row['run_name']}` `{row['strategy']}` `{row['gravity']}`: "
            f"stable={float(row['mean_stable_count']):.2f}, failures={float(row['mean_failure_count']):.2f}, "
            f"success={float(row['success_rate']):.2f}, height={float(row['mean_stack_height_m']):.3f} m, "
            f"drift={float(row['mean_max_drift_m']):.3f} m, velocity={float(row['mean_velocity_inf_norm']):.3f}"
        )

    lines.extend(["", "## Failure-Prone Clusters", ""])
    grouped: dict[tuple[str, str, str], tuple[float, float]] = {}
    for row in failure_rows:
        key = (row["run_name"], row["strategy"], row["cluster_label"])
        placed = float(row.get("placed_count", 0.0))
        failed = float(row.get("failure_count", 0.0))
        old_failed, old_placed = grouped.get(key, (0.0, 0.0))
        grouped[key] = (old_failed + failed, old_placed + placed)
    ranked_failures = sorted(
        ((key, failed, placed, failed / placed if placed else 0.0) for key, (failed, placed) in grouped.items()),
        key=lambda item: (-item[3], -item[1], item[0]),
    )
    for (run_name, strategy, cluster), failed, placed, rate in ranked_failures[:16]:
        lines.append(f"- `{run_name}` `{strategy}` `{cluster}`: {failed:.0f}/{placed:.0f} failed, rate={rate:.2f}")

    lines.extend(
        [
            "",
            "## Current Interpretation",
            "",
            "- Candidate-pose physics filtering generally reduces left-stack-radius failures.",
            "- Rounded rocks repeatedly appear as high-risk late-stack elements.",
            "- Moon gravity often needs longer settling; low drift can still pair with nonzero residual velocity.",
            "- Risk-aware ordering helps when its risk estimate is gravity-specific and includes geometry priors.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
