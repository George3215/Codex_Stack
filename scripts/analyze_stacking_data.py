from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


NUMERIC_FEATURES = (
    "volume",
    "surface_area",
    "bbox_x",
    "bbox_y",
    "bbox_z",
    "elongation",
    "flatness",
    "sphericity",
    "roughness",
    "angularity",
    "spike_score",
    "compactness",
    "stability_score",
    "mass",
)


def main() -> int:
    args = parse_args()
    batch_root = args.batch_root.resolve()
    output_dirs = sorted(path.parent for path in batch_root.glob("**/results.csv"))
    if not output_dirs:
        raise SystemExit(f"No results.csv found under {batch_root}")

    analysis_dir = batch_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, Any]] = []
    success_detail_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    rock_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    failure_reason_rows: list[dict[str, Any]] = []
    category_role_rows: list[dict[str, Any]] = []
    for output_dir in output_dirs:
        run_results = add_run_id(read_csv(output_dir / "results.csv"), output_dir)
        result_rows.extend(run_results)
        success_detail_rows.extend(summarize_success_details_from_rows(run_results, output_dir.parent.name, output_dir.name))
        method_rows.extend(summarize_methods(output_dir))
        rock_rows.extend(summarize_rocks(output_dir))
        group_rows.extend(summarize_groups(output_dir))
        failure_reason_rows.extend(summarize_failure_reasons(output_dir))
        category_role_rows.extend(summarize_category_roles(output_dir))

    overall_success_detail_rows = summarize_success_details_from_rows(result_rows, "ALL", "ALL")
    overall_method_rows = summarize_methods_from_rows(result_rows, "ALL", "ALL")
    overall_group_rows = summarize_groups_from_rock_rows(rock_rows)
    overall_failure_reason_rows = summarize_failure_reason_rows(
        add_run_id(read_all_failure_rows(output_dirs), Path("ALL") / "ALL"),
        "ALL",
        "ALL",
    )
    overall_category_role_rows = summarize_category_role_rows(read_all_placement_rows(output_dirs), read_all_failure_rows(output_dirs), "ALL", "ALL")
    write_csv(analysis_dir / "success_rate_detail_by_run.csv", success_detail_rows)
    write_csv(analysis_dir / "success_rate_detail_overall.csv", overall_success_detail_rows)
    write_csv(analysis_dir / "method_summary_by_run.csv", method_rows)
    write_csv(analysis_dir / "method_summary_overall.csv", overall_method_rows)
    write_csv(analysis_dir / "rock_outcomes.csv", rock_rows)
    write_csv(analysis_dir / "geometry_group_summary_by_run.csv", group_rows)
    write_csv(analysis_dir / "geometry_group_summary_overall.csv", overall_group_rows)
    write_csv(analysis_dir / "failure_reason_summary_by_run.csv", failure_reason_rows)
    write_csv(analysis_dir / "failure_reason_summary_overall.csv", overall_failure_reason_rows)
    write_csv(analysis_dir / "category_role_summary_by_run.csv", category_role_rows)
    write_csv(analysis_dir / "category_role_summary_overall.csv", overall_category_role_rows)
    write_report(
        analysis_dir / "README.md",
        batch_root=batch_root,
        success_detail_rows=overall_success_detail_rows,
        method_rows=overall_method_rows,
        group_rows=overall_group_rows,
        run_method_rows=method_rows,
        failure_reason_rows=overall_failure_reason_rows,
        category_role_rows=overall_category_role_rows,
    )
    print(analysis_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze angular rock stacking experiment data.")
    parser.add_argument("--batch-root", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_run_id(rows: list[dict[str, str]], output_dir: Path) -> list[dict[str, Any]]:
    return [{**row, "run_name": output_dir.name, "run_group": output_dir.parent.name} for row in rows]


def read_all_failure_rows(output_dirs: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for output_dir in output_dirs:
        rows.extend(read_csv(output_dir / "failure_cases.csv"))
    return rows


def read_all_placement_rows(output_dirs: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for output_dir in output_dirs:
        rows.extend(read_csv(output_dir / "placement_log.csv"))
    return rows


def summarize_success_details_from_rows(rows: list[dict[str, Any]], run_group: str, run_name: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(row.get("target_name", "stack"), row["strategy"], row["gravity"]) for row in rows})
    for target_name, strategy, gravity in keys:
        subset = [
            row
            for row in rows
            if row.get("target_name", "stack") == target_name and row["strategy"] == strategy and row["gravity"] == gravity
        ]
        trials = len(subset)
        success_count = sum(int(float(row.get("success", 0))) for row in subset)
        shape_success_count = sum(int(float(row.get("shape_success", row.get("success", 0)))) for row in subset)
        stable_total = sum(float(row.get("stable_count", 0.0)) for row in subset)
        rock_total = sum(float(row.get("rock_count", 0.0)) for row in subset)
        output.append(
            {
                "run_group": run_group,
                "run_name": run_name,
                "target_name": target_name,
                "strategy": strategy,
                "gravity": gravity,
                "trials": trials,
                "success_count": success_count,
                "success_rate": success_count / max(trials, 1),
                "shape_success_count": shape_success_count,
                "shape_success_rate": shape_success_count / max(trials, 1),
                "stable_total": stable_total,
                "rock_total": rock_total,
                "stable_fraction": stable_total / max(rock_total, 1.0),
                "mean_structure_score": mean(float(row.get("structure_score", 0.0)) for row in subset),
                "mean_rmse_xy_m": mean(float(row.get("target_rmse_xy_m", 0.0)) for row in subset),
                "mean_max_error_xy_m": mean(float(row.get("target_max_xy_error_m", 0.0)) for row in subset),
                "mean_height_m": mean(float(row.get("stack_height_m", 0.0)) for row in subset),
                "mean_drift_m": mean(float(row.get("max_horizontal_drift_m", 0.0)) for row in subset),
                "mean_velocity_inf_norm": mean(float(row.get("velocity_inf_norm", 0.0)) for row in subset),
            }
        )
    return output


def summarize_methods(output_dir: Path) -> list[dict[str, Any]]:
    rows = add_run_id(read_csv(output_dir / "results.csv"), output_dir)
    return summarize_methods_from_rows(rows, output_dir.parent.name, output_dir.name)


def summarize_methods_from_rows(rows: list[dict[str, Any]], run_group: str, run_name: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(row.get("target_name", "stack"), row["strategy"], row["gravity"]) for row in rows})
    for target_name, strategy, gravity in keys:
        subset = [
            row
            for row in rows
            if row.get("target_name", "stack") == target_name and row["strategy"] == strategy and row["gravity"] == gravity
        ]
        if not subset:
            continue
        mean_rocks = mean(float(row["rock_count"]) for row in subset)
        mean_stable = mean(float(row["stable_count"]) for row in subset)
        output.append(
            {
                "run_group": run_group,
                "run_name": run_name,
                "target_name": target_name,
                "strategy": strategy,
                "gravity": gravity,
                "trials": len(subset),
                "success_rate": mean(float(row["success"]) for row in subset),
                "mean_rock_count": mean_rocks,
                "mean_stable_count": mean_stable,
                "mean_stable_fraction": mean_stable / max(mean_rocks, 1e-9),
                "mean_failure_count": mean(float(row["failure_count"]) for row in subset),
                "mean_stack_height_m": mean(float(row["stack_height_m"]) for row in subset),
                "mean_max_drift_m": mean(float(row["max_horizontal_drift_m"]) for row in subset),
                "mean_radial_distance_m": mean(float(row["max_radial_distance_m"]) for row in subset),
                "mean_velocity_inf_norm": mean(float(row["velocity_inf_norm"]) for row in subset),
                "method_score": method_score(subset),
            }
        )
    return output


def summarize_rocks(output_dir: Path) -> list[dict[str, Any]]:
    features = {int(row["index"]): row for row in read_csv(output_dir / "features.csv")}
    placements = read_csv(output_dir / "placement_log.csv")
    failures = read_csv(output_dir / "failure_cases.csv")
    failure_counts: dict[tuple[str, str, str, int], int] = defaultdict(int)
    for row in failures:
        key = (row.get("target_name", "stack"), row["strategy"], row["gravity"], row["trial"], int(row["rock_index"]))
        failure_counts[key] += 1

    grouped: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in placements:
        rock_index = int(row["rock_index"])
        if rock_index < 0:
            continue
        grouped[(row.get("target_name", "stack"), row["strategy"], row["gravity"], rock_index)].append(row)

    output: list[dict[str, Any]] = []
    for (target_name, strategy, gravity, rock_index), rows in sorted(grouped.items()):
        feature = features[rock_index]
        fail_count = sum(
            failure_counts.get((row.get("target_name", "stack"), row["strategy"], row["gravity"], row["trial"], rock_index), 0)
            for row in rows
        )
        output.append(
            {
                "run_group": output_dir.parent.name,
                "run_name": output_dir.name,
                "target_name": target_name,
                "strategy": strategy,
                "gravity": gravity,
                "rock_index": rock_index,
                "source_kind": feature["source_kind"],
                "cluster_label": feature["cluster_label"],
                "placed_count": len(rows),
                "failure_count": fail_count,
                "failure_rate": fail_count / max(len(rows), 1),
                "mean_stack_level": mean(placement_level(row) for row in rows),
                "mean_support_overlap": mean(row_float(row, "support_overlap") for row in rows),
                "mean_radial_distance_m": mean(placement_radial_distance(row) for row in rows),
                "mean_velocity_after_place": mean(row_float(row, "velocity_inf_norm_after_place") for row in rows),
                "mean_height_gain_m": mean(row_float(row, "height_gain_m") for row in rows),
                **{key: feature[key] for key in NUMERIC_FEATURES if key in feature},
            }
        )
    return output


def summarize_groups(output_dir: Path) -> list[dict[str, Any]]:
    return summarize_groups_from_rock_rows(summarize_rocks(output_dir), output_dir.parent.name, output_dir.name)


def summarize_failure_reasons(output_dir: Path) -> list[dict[str, Any]]:
    failures = add_run_id(read_csv(output_dir / "failure_cases.csv"), output_dir)
    return summarize_failure_reason_rows(failures, output_dir.parent.name, output_dir.name)


def summarize_failure_reason_rows(rows: list[dict[str, Any]], run_group: str, run_name: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted(
        {
            (
                row.get("target_name", "stack"),
                row.get("strategy", ""),
                row.get("gravity", ""),
                row.get("course", ""),
                row.get("role", ""),
                row.get("failure_reason", ""),
                row.get("source_kind", ""),
                row.get("cluster_label", ""),
            )
            for row in rows
        }
    )
    for target_name, strategy, gravity, course, role, reason, source_kind, cluster_label in keys:
        subset = [
            row
            for row in rows
            if row.get("target_name", "stack") == target_name
            and row.get("strategy", "") == strategy
            and row.get("gravity", "") == gravity
            and row.get("course", "") == course
            and row.get("role", "") == role
            and row.get("failure_reason", "") == reason
            and row.get("source_kind", "") == source_kind
            and row.get("cluster_label", "") == cluster_label
        ]
        output.append(
            {
                "run_group": run_group,
                "run_name": run_name,
                "target_name": target_name,
                "strategy": strategy,
                "gravity": gravity,
                "course": course,
                "role": role,
                "failure_reason": reason,
                "source_kind": source_kind,
                "cluster_label": cluster_label,
                "failure_count": len(subset),
                "mean_target_error_xy_m": mean(row_float(row, "target_error_xy_m") for row in subset),
                "mean_horizontal_drift_m": mean(row_float(row, "horizontal_drift_m") for row in subset),
                "mean_flatness": mean(row_float(row, "flatness") for row in subset),
                "mean_elongation": mean(row_float(row, "elongation") for row in subset),
                "mean_spike_score": mean(row_float(row, "spike_score") for row in subset),
                "mean_stability_score": mean(row_float(row, "stability_score") for row in subset),
            }
        )
    return output


def summarize_category_roles(output_dir: Path) -> list[dict[str, Any]]:
    placements = add_run_id(read_csv(output_dir / "placement_log.csv"), output_dir)
    failures = add_run_id(read_csv(output_dir / "failure_cases.csv"), output_dir)
    return summarize_category_role_rows(placements, failures, output_dir.parent.name, output_dir.name)


def summarize_category_role_rows(
    placements: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    run_group: str,
    run_name: str,
) -> list[dict[str, Any]]:
    failure_keys = {
        (
            row.get("target_name", "stack"),
            row.get("strategy", ""),
            row.get("gravity", ""),
            row.get("trial", ""),
            row.get("slot_id", ""),
            row.get("rock_index", ""),
        )
        for row in failures
    }
    accum: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in placements:
        failed = (
            row.get("target_name", "stack"),
            row.get("strategy", ""),
            row.get("gravity", ""),
            row.get("trial", ""),
            row.get("slot_id", ""),
            row.get("rock_index", ""),
        ) in failure_keys
        for group_type, group_value in (
            ("source_kind", row.get("source_kind", "")),
            ("cluster_label", row.get("cluster_label", "")),
        ):
            key = (
                row.get("target_name", "stack"),
                row.get("strategy", ""),
                row.get("gravity", ""),
                row.get("course", ""),
                row.get("role", ""),
                group_type,
                group_value,
            )
            item = accum.setdefault(
                key,
                {
                    "run_group": run_group,
                    "run_name": run_name,
                    "target_name": key[0],
                    "strategy": key[1],
                    "gravity": key[2],
                    "course": key[3],
                    "role": key[4],
                    "group_type": key[5],
                    "group_value": key[6],
                    "placed_count": 0,
                    "failure_count": 0,
                    "_support_sum": 0.0,
                    "_target_error_sum": 0.0,
                    "_velocity_sum": 0.0,
                    "_height_gain_sum": 0.0,
                },
            )
            item["placed_count"] += 1
            item["failure_count"] += int(failed)
            item["_support_sum"] += row_float(row, "support_overlap")
            item["_target_error_sum"] += row_float(row, "target_error_xy_m")
            item["_velocity_sum"] += row_float(row, "velocity_inf_norm_after_place")
            item["_height_gain_sum"] += row_float(row, "height_gain_m")

    output: list[dict[str, Any]] = []
    for item in accum.values():
        placed = max(float(item["placed_count"]), 1.0)
        output.append(
            {
                "run_group": item["run_group"],
                "run_name": item["run_name"],
                "target_name": item["target_name"],
                "strategy": item["strategy"],
                "gravity": item["gravity"],
                "course": item["course"],
                "role": item["role"],
                "group_type": item["group_type"],
                "group_value": item["group_value"],
                "placed_count": item["placed_count"],
                "failure_count": item["failure_count"],
                "failure_rate": float(item["failure_count"]) / placed,
                "mean_support_overlap": item["_support_sum"] / placed,
                "mean_target_error_xy_m": item["_target_error_sum"] / placed,
                "mean_velocity_after_place": item["_velocity_sum"] / placed,
                "mean_height_gain_m": item["_height_gain_sum"] / placed,
            }
        )
    return sorted(
        output,
        key=lambda row: (
            str(row["target_name"]),
            str(row["strategy"]),
            str(row["gravity"]),
            int(float(row["course"])) if str(row["course"]) not in {"", "nan"} else 0,
            str(row["role"]),
            str(row["group_type"]),
            str(row["group_value"]),
        ),
    )


def summarize_groups_from_rock_rows(
    rows: list[dict[str, Any]], run_group: str = "ALL", run_name: str = "ALL"
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group_type in ("source_kind", "cluster_label"):
        keys = sorted({str(row[group_type]) for row in rows})
        for key in keys:
            subset = [row for row in rows if str(row[group_type]) == key]
            placed = sum(float(row["placed_count"]) for row in subset)
            failures = sum(float(row["failure_count"]) for row in subset)
            output.append(
                {
                    "run_group": run_group,
                    "run_name": run_name,
                    "group_type": group_type,
                    "group_value": key,
                    "rock_strategy_gravity_rows": len(subset),
                    "placed_count": placed,
                    "failure_count": failures,
                    "failure_rate": failures / max(placed, 1.0),
                    "mean_support_overlap": weighted_mean(subset, "mean_support_overlap", "placed_count"),
                    "mean_radial_distance_m": weighted_mean(subset, "mean_radial_distance_m", "placed_count"),
                    "mean_velocity_after_place": weighted_mean(subset, "mean_velocity_after_place", "placed_count"),
                    "mean_height_gain_m": weighted_mean(subset, "mean_height_gain_m", "placed_count"),
                    "mean_roughness": mean(float(row["roughness"]) for row in subset),
                    "mean_angularity": mean(float(row["angularity"]) for row in subset),
                    "mean_spike_score": mean(float(row.get("spike_score", 0.0)) for row in subset),
                    "mean_compactness": mean(float(row["compactness"]) for row in subset),
                    "mean_flatness": mean(float(row["flatness"]) for row in subset),
                    "mean_elongation": mean(float(row["elongation"]) for row in subset),
                    "mean_stability_score": mean(float(row["stability_score"]) for row in subset),
                }
            )
    return output


def method_score(rows: list[dict[str, Any]]) -> float:
    stable_fraction = mean(float(row["stable_count"]) / max(float(row["rock_count"]), 1.0) for row in rows)
    success_rate = mean(float(row["success"]) for row in rows)
    drift = mean(float(row["max_horizontal_drift_m"]) for row in rows)
    velocity = mean(float(row["velocity_inf_norm"]) for row in rows)
    return stable_fraction + 0.35 * success_rate - 0.9 * drift - 0.12 * velocity


def weighted_mean(rows: list[dict[str, Any]], value_key: str, weight_key: str) -> float:
    total_weight = sum(float(row[weight_key]) for row in rows)
    if total_weight <= 0:
        return 0.0
    return sum(float(row[value_key]) * float(row[weight_key]) for row in rows) / total_weight


def mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def row_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def placement_level(row: dict[str, Any]) -> float:
    if row.get("stack_level", "") not in {"", None}:
        return float(row["stack_level"])
    return row_float(row, "course")


def placement_radial_distance(row: dict[str, Any]) -> float:
    if row.get("radial_distance_m", "") not in {"", None}:
        return float(row["radial_distance_m"])
    return (row_float(row, "settled_x") ** 2 + row_float(row, "settled_y") ** 2) ** 0.5


def write_report(
    path: Path,
    batch_root: Path,
    success_detail_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    run_method_rows: list[dict[str, Any]],
    failure_reason_rows: list[dict[str, Any]],
    category_role_rows: list[dict[str, Any]],
) -> None:
    ranked_methods = sorted(
        method_rows,
        key=lambda row: (-float(row["method_score"]), -float(row["mean_stable_fraction"]), float(row["mean_max_drift_m"])),
    )
    ranked_clusters = sorted(
        [row for row in group_rows if row["group_type"] == "cluster_label"],
        key=lambda row: (float(row["failure_rate"]), -float(row["mean_support_overlap"]), -float(row["mean_stability_score"])),
    )
    risk_clusters = sorted(
        [row for row in group_rows if row["group_type"] == "cluster_label"],
        key=lambda row: (-float(row["failure_rate"]), float(row["mean_support_overlap"])),
    )
    ranked_sources = sorted(
        [row for row in group_rows if row["group_type"] == "source_kind"],
        key=lambda row: (float(row["failure_rate"]), -float(row["mean_support_overlap"]), -float(row["mean_stability_score"])),
    )
    risk_sources = sorted(
        [row for row in group_rows if row["group_type"] == "source_kind"],
        key=lambda row: (-float(row["failure_rate"]), float(row["mean_support_overlap"])),
    )

    lines = [
        "# Angular Rock Stacking Data Analysis",
        "",
        f"Batch root: `{batch_root}`",
        "",
        "## Experiment Logic",
        "",
        "The experiment follows the papers in `D:/MoonStack/Asset/Papers`: use geometry-aware stone selection, validate candidate poses in MuJoCo, compare Earth and Moon gravity with the same stone libraries, and feed failure statistics back into risk-aware sequencing.",
        "",
    ]
    lines.extend(["## Detailed Success Rates", ""])
    for row in sorted(
        success_detail_rows,
        key=lambda item: (
            str(item["target_name"]),
            -float(item["success_rate"]),
            -float(item["shape_success_rate"]),
            str(item["strategy"]),
            str(item["gravity"]),
        ),
    ):
        lines.append(
            f"- `{row['target_name']}` `{row['strategy']}` `{row['gravity']}`: "
            f"{int(row['success_count'])}/{int(row['trials'])} success "
            f"({float(row['success_rate']):.3f}), shape={int(row['shape_success_count'])}/{int(row['trials'])} "
            f"({float(row['shape_success_rate']):.3f}), stable_fraction={float(row['stable_fraction']):.3f}, "
            f"rmse={float(row['mean_rmse_xy_m']):.3f} m, max_error={float(row['mean_max_error_xy_m']):.3f} m"
        )

    lines.extend(["", "## Methods That Currently Stack Better", ""])
    for row in ranked_methods:
        lines.append(
            f"- `{row.get('target_name', 'stack')}` `{row['strategy']}` under `{row['gravity']}`: score={float(row['method_score']):.3f}, "
            f"stable={float(row['mean_stable_count']):.2f}/{float(row['mean_rock_count']):.2f}, "
            f"success={float(row['success_rate']):.2f}, height={float(row['mean_stack_height_m']):.3f} m, "
            f"drift={float(row['mean_max_drift_m']):.3f} m, velocity={float(row['mean_velocity_inf_norm']):.3f}"
        )

    lines.extend(["", "## Source Kinds That Currently Help", ""])
    for row in ranked_sources[:10]:
        lines.append(
            f"- `{row['group_value']}`: failure_rate={float(row['failure_rate']):.3f}, "
            f"support={float(row['mean_support_overlap']):.3f}, angularity={float(row['mean_angularity']):.3f}, "
            f"spike={float(row.get('mean_spike_score', 0.0)):.3f}, compactness={float(row['mean_compactness']):.3f}, "
            f"stability_score={float(row['mean_stability_score']):.3f}"
        )

    lines.extend(["", "## Cluster Labels That Currently Help", ""])
    for row in ranked_clusters[:10]:
        lines.append(
            f"- `{row['group_value']}`: failure_rate={float(row['failure_rate']):.3f}, "
            f"support={float(row['mean_support_overlap']):.3f}, angularity={float(row['mean_angularity']):.3f}, "
            f"spike={float(row.get('mean_spike_score', 0.0)):.3f}, compactness={float(row['mean_compactness']):.3f}, "
            f"stability_score={float(row['mean_stability_score']):.3f}"
        )

    lines.extend(["", "## Source Kinds To Treat Carefully", ""])
    for row in risk_sources[:10]:
        lines.append(
            f"- `{row['group_value']}`: failure_rate={float(row['failure_rate']):.3f}, "
            f"support={float(row['mean_support_overlap']):.3f}, roughness={float(row['mean_roughness']):.3f}, "
            f"elongation={float(row['mean_elongation']):.3f}, angularity={float(row['mean_angularity']):.3f}, "
            f"spike={float(row.get('mean_spike_score', 0.0)):.3f}"
        )

    lines.extend(["", "## Cluster Labels To Treat Carefully", ""])
    for row in risk_clusters[:10]:
        lines.append(
            f"- `{row['group_value']}`: failure_rate={float(row['failure_rate']):.3f}, "
            f"support={float(row['mean_support_overlap']):.3f}, roughness={float(row['mean_roughness']):.3f}, "
            f"elongation={float(row['mean_elongation']):.3f}, angularity={float(row['mean_angularity']):.3f}, "
            f"spike={float(row.get('mean_spike_score', 0.0)):.3f}"
        )

    lines.extend(["", "## Dominant Failure Details", ""])
    if failure_reason_rows:
        for row in sorted(failure_reason_rows, key=lambda item: (-int(item["failure_count"]), str(item["target_name"])))[:24]:
            lines.append(
                f"- `{row['target_name']}` `{row['strategy']}` `{row['gravity']}` course={row['course']} role={row['role']} "
                f"`{row['source_kind']}`/`{row['cluster_label']}` reason=`{row['failure_reason']}`: "
                f"n={row['failure_count']}, error={float(row['mean_target_error_xy_m']):.3f} m, "
                f"drift={float(row['mean_horizontal_drift_m']):.3f} m"
            )
    else:
        lines.append("No failures recorded.")

    lines.extend(["", "## Best Category-Role Combinations", ""])
    for row in sorted(
        category_role_rows,
        key=lambda item: (float(item["failure_rate"]), -float(item["placed_count"]), -float(item["mean_support_overlap"])),
    )[:24]:
        lines.append(
            f"- `{row['target_name']}` `{row['strategy']}` `{row['gravity']}` course={row['course']} role={row['role']} "
            f"{row['group_type']}=`{row['group_value']}`: fail={int(row['failure_count'])}/{int(row['placed_count'])} "
            f"({float(row['failure_rate']):.3f}), support={float(row['mean_support_overlap']):.3f}, "
            f"target_error={float(row['mean_target_error_xy_m']):.3f} m"
        )

    lines.extend(["", "## Per-Run Method Summary", ""])
    for row in sorted(run_method_rows, key=lambda item: (item["run_name"], item.get("target_name", "stack"), item["strategy"], item["gravity"])):
        lines.append(
            f"- `{row['run_name']}` `{row.get('target_name', 'stack')}` `{row['strategy']}` `{row['gravity']}`: "
            f"stable_fraction={float(row['mean_stable_fraction']):.3f}, success={float(row['success_rate']):.2f}, "
            f"drift={float(row['mean_max_drift_m']):.3f}, velocity={float(row['mean_velocity_inf_norm']):.3f}"
        )

    lines.extend(
        [
            "",
            "## Data Files",
            "",
            "- `method_summary_by_run.csv`: strategy and gravity comparison per run.",
            "- `method_summary_overall.csv`: aggregated strategy and gravity comparison.",
            "- `success_rate_detail_by_run.csv`: exact success counts and rates per target, strategy, and gravity.",
            "- `success_rate_detail_overall.csv`: aggregated exact success counts and rates.",
            "- `rock_outcomes.csv`: per-rock placement outcomes joined with geometry features.",
            "- `geometry_group_summary_by_run.csv`: source/cluster failure and support summaries per run.",
            "- `geometry_group_summary_overall.csv`: aggregated source/cluster summaries.",
            "- `failure_reason_summary_by_run.csv`: failure reason details crossed with target, role, source kind, and cluster.",
            "- `failure_reason_summary_overall.csv`: aggregated failure reason details.",
            "- `category_role_summary_by_run.csv`: source/cluster performance by course and role.",
            "- `category_role_summary_overall.csv`: aggregated source/cluster performance by course and role.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
