from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .clustering import cluster_features
from .features import FEATURE_COLUMNS, extract_features
from .fractal_rocks import generate_rocks, write_all_objs
from .mjcf import write_world_xml
from .simulate import GRAVITIES, run_trial_detailed, stacking_order


def main() -> None:
    args = parse_args()
    output_dir = args.output.resolve()
    mesh_dir = output_dir / "meshes"
    mjcf_dir = output_dir / "mjcf"
    output_dir.mkdir(parents=True, exist_ok=True)

    rocks = generate_rocks(args.rocks, seed=args.seed)
    write_all_objs(mesh_dir, rocks)

    rows = [extract_features(rock) for rock in rocks]
    labels, names = cluster_features(rows, clusters=args.clusters, seed=args.seed)
    for row, label in zip(rows, labels):
        row["cluster_id"] = int(label)
        row["cluster_label"] = names[int(label)]

    write_csv(output_dir / "features.csv", rows)
    write_csv(output_dir / "cluster_summary.csv", cluster_summary(rows))
    results: list[dict[str, Any]] = []
    placements: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    strategy_sequence = parse_strategy_list(args.strategies)
    risk_by_cluster: dict[str, float] = {}

    for strategy in strategy_sequence:
        strategy_rng_seed = args.seed + 1009 * (strategy_sequence.index(strategy) + 1)
        for gravity_label, gravity in GRAVITIES.items():
            if strategy == "risk_aware":
                risk_by_cluster = estimate_cluster_risk(placements, failures, gravity_label=gravity_label)
            for trial_id in range(args.trials):
                rng_seed = strategy_rng_seed + trial_id * 37 + int(gravity * 10)
                order = stacking_order(
                    rows,
                    strategy=strategy,
                    rng=None,
                    risk_by_cluster=risk_by_cluster,
                    stack_rocks=args.stack_rocks,
                )
                xml_path = mjcf_dir / f"{strategy}_{gravity_label}_trial_{trial_id:02d}.xml"
                write_world_xml(xml_path, rows, gravity=gravity, trial_id=trial_id)
                if not args.skip_sim:
                    detailed = run_trial_detailed(
                        xml_path=xml_path,
                        rows=rows,
                        order=order,
                        gravity_label=gravity_label,
                        trial_id=trial_id,
                        seed=rng_seed,
                        steps_per_rock=args.steps_per_rock,
                        hold_steps=args.hold_steps,
                        strategy=strategy,
                        candidate_count=candidates_for_strategy(strategy, args.candidates),
                        stack_rocks=args.stack_rocks,
                    )
                    state_path = output_dir / "states" / f"{strategy}_{gravity_label}_trial_{trial_id:02d}.npz"
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        state_path,
                        qpos=detailed["state"]["qpos"],
                        qvel=detailed["state"]["qvel"],
                        strategy=strategy,
                        gravity=gravity_label,
                        trial=trial_id,
                    )
                    detailed["summary"]["state_path"] = str(state_path)
                    results.append(detailed["summary"])
                    placements.extend(detailed["placements"])
                    failures.extend(detailed["failures"])
        if results and not args.skip_sim:
            write_pipeline_outputs(output_dir, results, placements, failures, rows)

    if results:
        write_pipeline_outputs(output_dir, results, placements, failures, rows)
        summary = summarize(results)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_strategy_report(output_dir / "strategy_report.md", summary, failures, rows)
        print(json.dumps(summary, indent=2))
    else:
        print(f"Generated meshes, features, and MJCF under {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and simulate lunar rock stacking in MuJoCo.")
    parser.add_argument("--rocks", type=int, default=18, help="Number of generated rocks.")
    parser.add_argument("--clusters", type=int, default=5, help="K-means geometry clusters.")
    parser.add_argument("--trials", type=int, default=4, help="Trials per gravity condition.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--steps-per-rock", type=int, default=1600, help="MuJoCo settling steps after each placed rock.")
    parser.add_argument("--hold-steps", type=int, default=2600, help="Final hold steps for stability measurement.")
    parser.add_argument("--candidates", type=int, default=8, help="Candidate poses per rock for physics-filter strategies.")
    parser.add_argument("--stack-rocks", type=int, default=None, help="Only stack the top N selected rocks.")
    parser.add_argument(
        "--strategies",
        default="paper_baseline,physics_filter,risk_aware",
        help="Comma-separated strategies: paper_baseline, physics_filter, support_first, strength, random, risk_aware.",
    )
    parser.add_argument("--skip-sim", action="store_true", help="Only generate rocks, features, and MJCF XML.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated"),
        help="Output directory.",
    )
    return parser.parse_args()


def parse_strategy_list(value: str) -> list[str]:
    strategies = [item.strip() for item in value.split(",") if item.strip()]
    if not strategies:
        raise ValueError("At least one strategy is required.")
    valid = {"paper_baseline", "physics_filter", "support_first", "strength", "random", "risk_aware"}
    unknown = sorted(set(strategies) - valid)
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(unknown)}")
    return strategies


def candidates_for_strategy(strategy: str, requested: int) -> int:
    if strategy in {"paper_baseline", "random"}:
        return 1
    return max(1, requested)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    extra_keys = sorted({key for row in rows for key in row.keys()} - set(fieldnames))
    fieldnames.extend(extra_keys)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_pipeline_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    placements: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    write_csv(output_dir / "results.csv", results)
    write_csv(output_dir / "placement_log.csv", placements)
    write_csv(output_dir / "failure_cases.csv", failures)
    write_csv(output_dir / "failure_by_cluster.csv", failure_by_cluster(placements, failures))
    write_csv(output_dir / "failure_by_feature.csv", failure_by_feature(rows, failures))


def cluster_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cluster_ids = sorted({int(row["cluster_id"]) for row in rows})
    summary_rows: list[dict[str, Any]] = []
    for cluster_id in cluster_ids:
        subset = [row for row in rows if int(row["cluster_id"]) == cluster_id]
        label = str(subset[0]["cluster_label"])
        source_counts: dict[str, int] = {}
        for row in subset:
            source = str(row["source_kind"])
            source_counts[source] = source_counts.get(source, 0) + 1
        summary: dict[str, Any] = {
            "cluster_id": cluster_id,
            "cluster_label": label,
            "count": len(subset),
            "source_kind_counts": json.dumps(source_counts, sort_keys=True),
            "members": " ".join(f'{int(row["index"]):03d}' for row in subset),
        }
        for column in FEATURE_COLUMNS:
            summary[f"mean_{column}"] = sum(float(row[column]) for row in subset) / len(subset)
        summary_rows.append(summary)
    return summary_rows


def estimate_cluster_risk(
    placements: list[dict[str, Any]], failures: list[dict[str, Any]], gravity_label: str | None = None
) -> dict[str, float]:
    placed_by_cluster: dict[str, int] = {}
    failed_by_cluster: dict[str, int] = {}
    for row in placements:
        if gravity_label is not None and str(row["gravity"]) != gravity_label:
            continue
        cluster = str(row["cluster_label"])
        placed_by_cluster[cluster] = placed_by_cluster.get(cluster, 0) + 1
    for row in failures:
        if gravity_label is not None and str(row["gravity"]) != gravity_label:
            continue
        cluster = str(row["cluster_label"])
        failed_by_cluster[cluster] = failed_by_cluster.get(cluster, 0) + 1
    risk: dict[str, float] = {}
    for cluster, count in placed_by_cluster.items():
        risk[cluster] = failed_by_cluster.get(cluster, 0) / max(count, 1)
    return risk


def failure_by_cluster(
    placements: list[dict[str, Any]], failures: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    placed: dict[tuple[str, str], int] = {}
    failed: dict[tuple[str, str], int] = {}
    for row in placements:
        key = (str(row["strategy"]), str(row["cluster_label"]))
        placed[key] = placed.get(key, 0) + 1
    for row in failures:
        key = (str(row["strategy"]), str(row["cluster_label"]))
        failed[key] = failed.get(key, 0) + 1
    rows: list[dict[str, Any]] = []
    for key in sorted(placed):
        strategy, cluster = key
        rows.append(
            {
                "strategy": strategy,
                "cluster_label": cluster,
                "placed_count": placed[key],
                "failure_count": failed.get(key, 0),
                "failure_rate": failed.get(key, 0) / max(placed[key], 1),
            }
        )
    return rows


def failure_by_feature(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failed_ids = {int(row["rock_index"]) for row in failures}
    output: list[dict[str, Any]] = []
    for source in sorted({str(row["source_kind"]) for row in rows}):
        subset = [row for row in rows if str(row["source_kind"]) == source]
        fail_subset = [row for row in subset if int(row["index"]) in failed_ids]
        output.append(_feature_summary("source_kind", source, subset, fail_subset))
    for cluster in sorted({str(row["cluster_label"]) for row in rows}):
        subset = [row for row in rows if str(row["cluster_label"]) == cluster]
        fail_subset = [row for row in subset if int(row["index"]) in failed_ids]
        output.append(_feature_summary("cluster_label", cluster, subset, fail_subset))
    return output


def _feature_summary(
    group_type: str,
    group_value: str,
    subset: list[dict[str, Any]],
    fail_subset: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "group_type": group_type,
        "group_value": group_value,
        "rock_count": len(subset),
        "failed_unique_rock_count": len(fail_subset),
        "failed_unique_rate": len(fail_subset) / max(len(subset), 1),
    }
    for column in (
        "volume",
        "roughness",
        "angularity",
        "spike_score",
        "compactness",
        "flatness",
        "elongation",
        "sphericity",
        "stability_score",
    ):
        summary[f"mean_{column}"] = sum(float(row[column]) for row in subset) / max(len(subset), 1)
        summary[f"failed_mean_{column}"] = (
            sum(float(row[column]) for row in fail_subset) / len(fail_subset) if fail_subset else ""
        )
    return summary


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for strategy in sorted({str(row["strategy"]) for row in results}):
        summary[strategy] = {}
        for gravity_label in GRAVITIES:
            subset = [row for row in results if row["gravity"] == gravity_label and row["strategy"] == strategy]
            if not subset:
                continue
            summary[strategy][gravity_label] = {
                "trials": len(subset),
                "success_rate": sum(int(row["success"]) for row in subset) / len(subset),
                "mean_stable_count": sum(float(row["stable_count"]) for row in subset) / len(subset),
                "mean_failure_count": sum(float(row["failure_count"]) for row in subset) / len(subset),
                "mean_stack_height_m": sum(float(row["stack_height_m"]) for row in subset) / len(subset),
                "mean_max_drift_m": sum(float(row["max_horizontal_drift_m"]) for row in subset) / len(subset),
                "mean_velocity_inf_norm": sum(float(row["velocity_inf_norm"]) for row in subset) / len(subset),
            }
    return summary


def write_strategy_report(
    path: Path,
    summary: dict[str, Any],
    failures: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Strategy Report",
        "",
        "This report follows the literature path in `D:/MoonStack/Asset/Papers`: geometry heuristics first, physics-based candidate filtering second, and data-driven risk adjustment third.",
        "",
        "## Strategy Summary",
        "",
    ]
    for strategy, gravity_data in summary.items():
        lines.append(f"### {strategy}")
        for gravity, metrics in gravity_data.items():
            lines.append(
                f"- `{gravity}`: stable={metrics['mean_stable_count']:.2f}, failures={metrics['mean_failure_count']:.2f}, "
                f"height={metrics['mean_stack_height_m']:.3f} m, drift={metrics['mean_max_drift_m']:.3f} m, "
                f"velocity={metrics['mean_velocity_inf_norm']:.3f}"
            )
        lines.append("")

    lines.extend(["## Failure Patterns", ""])
    if failures:
        by_cluster: dict[str, int] = {}
        by_reason: dict[str, int] = {}
        for failure in failures:
            by_cluster[str(failure["cluster_label"])] = by_cluster.get(str(failure["cluster_label"]), 0) + 1
            by_reason[str(failure["failure_reason"])] = by_reason.get(str(failure["failure_reason"]), 0) + 1
        lines.append("Cluster failure counts:")
        for cluster, count in sorted(by_cluster.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{cluster}`: {count}")
        lines.append("")
        lines.append("Failure reasons:")
        for reason, count in sorted(by_reason.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("No failed rocks were recorded.")

    lines.extend(
        [
            "",
            "## Data-Driven Improvement Rules",
            "",
            "- If wedge/broad or subangular clasts succeed disproportionately, reserve them for the base and early support layers.",
            "- If fractured or elongated clasts dominate failures, delay them or require more candidate pose validation.",
            "- If any `spiky_reject` cluster appears, reduce generator roughness and discard those rocks before simulation.",
            "- If Moon gravity has higher residual velocity, increase settle steps and add a disturbance test before accepting a pose.",
            "- If a strategy reduces drift but not failures, use it as a candidate filter and add sequence search on top.",
            "- Use `failure_by_cluster.csv` to update `risk_aware` ordering; high-risk clusters are delayed or skipped when `--stack-rocks` is used.",
            "",
            "## Recommended Next Run",
            "",
            "```powershell",
            "conda activate moon-rock-stack",
            "python -m moon_rock_stack.run_experiment --rocks 30 --clusters 6 --trials 8 --candidates 16 --stack-rocks 20 --output generated_pipeline",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
