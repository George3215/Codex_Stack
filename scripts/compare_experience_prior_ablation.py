from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


def main() -> int:
    args = parse_args()
    baseline = args.baseline.resolve()
    prior = args.prior.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    baseline_results = read_csv(baseline / "results.csv")
    prior_results = read_csv(prior / "results.csv")
    baseline_placements = read_csv(baseline / "placement_log.csv")
    prior_placements = read_csv(prior / "placement_log.csv")

    result_rows = compare_results(baseline_results, prior_results)
    skip_rows = compare_skips(baseline_placements, prior_placements)
    prior_score_rows = summarize_prior_scores(prior_placements)

    write_csv(output / "result_comparison.csv", result_rows)
    write_csv(output / "skip_comparison_by_role.csv", skip_rows)
    write_csv(output / "prior_score_summary.csv", prior_score_rows)
    write_readme(
        output / "README.md",
        baseline=baseline,
        prior=prior,
        output=output,
        baseline_results=baseline_results,
        prior_results=prior_results,
        result_rows=result_rows,
        skip_rows=skip_rows,
        prior_score_rows=prior_score_rows,
    )
    print(output)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs experience-prior wall stacking runs.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def compare_results(baseline_rows: list[dict[str, str]], prior_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "metric": "trials",
            "baseline": len(baseline_rows),
            "prior": len(prior_rows),
            "delta": len(prior_rows) - len(baseline_rows),
        },
        metric_delta("strict_success_rate", rate(baseline_rows, "success"), rate(prior_rows, "success")),
        metric_delta("shape_success_rate", rate(baseline_rows, "shape_success"), rate(prior_rows, "shape_success")),
        metric_delta("mean_visible_courses", avg(baseline_rows, "visible_courses"), avg(prior_rows, "visible_courses")),
        metric_delta("mean_stable_count", avg(baseline_rows, "stable_count"), avg(prior_rows, "stable_count")),
        metric_delta("mean_failure_count", avg(baseline_rows, "failure_count"), avg(prior_rows, "failure_count")),
        metric_delta("mean_skipped_slot_count", avg(baseline_rows, "skipped_slot_count"), avg(prior_rows, "skipped_slot_count")),
        metric_delta("mean_rmse_xy_m", avg(baseline_rows, "target_rmse_xy_m"), avg(prior_rows, "target_rmse_xy_m")),
        metric_delta("mean_max_drift_m", avg(baseline_rows, "max_horizontal_drift_m"), avg(prior_rows, "max_horizontal_drift_m")),
        metric_delta("mean_stack_height_m", avg(baseline_rows, "stack_height_m"), avg(prior_rows, "stack_height_m")),
    ]


def compare_skips(baseline_rows: list[dict[str, str]], prior_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    baseline_groups = group_placements(baseline_rows)
    prior_groups = group_placements(prior_rows)
    keys = sorted(set(baseline_groups).union(prior_groups))
    output: list[dict[str, Any]] = []
    for key in keys:
        base_items = baseline_groups.get(key, [])
        prior_items = prior_groups.get(key, [])
        base_skip = skip_rate(base_items)
        prior_skip = skip_rate(prior_items)
        output.append(
            {
                "target_name": key[0],
                "gravity": key[1],
                "role": key[2],
                "course": key[3],
                "baseline_rows": len(base_items),
                "baseline_skipped": skipped_count(base_items),
                "baseline_skip_rate": base_skip,
                "prior_rows": len(prior_items),
                "prior_skipped": skipped_count(prior_items),
                "prior_skip_rate": prior_skip,
                "skip_rate_delta": prior_skip - base_skip,
                "baseline_mean_disturbance_non_skipped": avg([row for row in base_items if not is_skipped(row)], "placed_disturbance_xy_m"),
                "prior_mean_disturbance_non_skipped": avg([row for row in prior_items if not is_skipped(row)], "placed_disturbance_xy_m"),
                "baseline_mean_target_error_non_skipped": avg([row for row in base_items if not is_skipped(row)], "target_error_xy_m"),
                "prior_mean_target_error_non_skipped": avg([row for row in prior_items if not is_skipped(row)], "target_error_xy_m"),
            }
        )
    output.sort(key=lambda row: (row["target_name"], row["gravity"], int_or_999(row["course"]), row["role"]))
    return output


def summarize_prior_scores(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("target_name", "")),
                str(row.get("gravity", "")),
                str(row.get("role", "")),
                str(row.get("course", "")),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (target, gravity, role, course), items in sorted(grouped.items()):
        output.append(
            {
                "target_name": target,
                "gravity": gravity,
                "role": role,
                "course": course,
                "rows": len(items),
                "mean_experience_prior_score": avg(items, "experience_prior_score"),
                "mean_source_weight": avg(items, "experience_prior_source_weight"),
                "mean_cluster_weight": avg(items, "experience_prior_cluster_weight"),
                "skip_rate": skip_rate(items),
            }
        )
    return output


def group_placements(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("target_name", "")),
                str(row.get("gravity", "")),
                str(row.get("role", "")),
                str(row.get("course", "")),
            )
        ].append(row)
    return grouped


def write_readme(
    path: Path,
    *,
    baseline: Path,
    prior: Path,
    output: Path,
    baseline_results: list[dict[str, str]],
    prior_results: list[dict[str, str]],
    result_rows: list[dict[str, Any]],
    skip_rows: list[dict[str, Any]],
    prior_score_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Experience Prior A/B 对比",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- baseline: `{baseline}`",
        f"- prior: `{prior}`",
        f"- output: `{output}`",
        "",
        "## 样本状态",
        "",
        f"- baseline results rows: `{len(baseline_results)}`",
        f"- prior results rows: `{len(prior_results)}`",
    ]
    if not prior_results:
        lines.extend(
            [
                "",
                "prior 组还没有完成 `results.csv`，当前报告只用于确认任务是否已经开始。等 prior 输出 results 后重新运行本脚本。",
            ]
        )
    else:
        lines.extend(["", "## 结果指标", ""])
        lines.extend(result_table(result_rows))
        lines.extend(["", "## Skip 变化", ""])
        lines.extend(skip_table(skip_rows))
        lines.extend(["", "## Prior 分数", ""])
        lines.extend(prior_table(prior_score_rows))
        lines.extend(["", "## 判读标准", ""])
        lines.extend(
            [
                "- 如果 cap/course=3 和 middle/course=2 的 skip_rate_delta 小于 0，说明经验先验减少了可行位姿缺失。",
                "- 如果 skip_rate 下降但 failure_count 或 drift 上升，说明候选更容易被放上去，但物理扰动仍需要 PoseRisk/低释放继续约束。",
                "- 如果 skip_rate、failure_count、drift 同时下降，说明 source_kind/cluster_label 经验先验有继续神经网络化的价值。",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def result_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| metric | baseline | prior | delta |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| `{row['metric']}` | {fmt(row['baseline'])} | {fmt(row['prior'])} | {fmt(row['delta'])} |")
    return lines


def skip_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| target | gravity | role | course | baseline skip | prior skip | delta |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['target_name']}` | `{row['gravity']}` | `{row['role']}` | {row['course']} | "
            f"{fmt(row['baseline_skip_rate'])} | {fmt(row['prior_skip_rate'])} | {fmt(row['skip_rate_delta'])} |"
        )
    return lines


def prior_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| target | gravity | role | course | rows | prior score | source weight | cluster weight | skip |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['target_name']}` | `{row['gravity']}` | `{row['role']}` | {row['course']} | {row['rows']} | "
            f"{fmt(row['mean_experience_prior_score'])} | {fmt(row['mean_source_weight'])} | "
            f"{fmt(row['mean_cluster_weight'])} | {fmt(row['skip_rate'])} |"
        )
    return lines


def metric_delta(name: str, baseline: float, prior: float) -> dict[str, Any]:
    return {"metric": name, "baseline": baseline, "prior": prior, "delta": prior - baseline}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rate(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(to_float(row.get(key)) for row in rows) / len(rows)


def avg(rows: list[dict[str, str]], key: str) -> float:
    values = [to_float(row.get(key)) for row in rows if str(row.get(key, "")).strip() != ""]
    return mean(values) if values else 0.0


def skip_rate(rows: list[dict[str, str]]) -> float:
    return skipped_count(rows) / max(len(rows), 1)


def skipped_count(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if is_skipped(row))


def is_skipped(row: dict[str, str]) -> bool:
    return (
        int(round(to_float(row.get("placement_skipped")))) == 1
        or str(row.get("skip_reason", "")).strip() != ""
        or str(row.get("cluster_label", "")).strip().lower() == "skipped"
        or str(row.get("source_kind", "")).strip().lower() == "skipped"
    )


def to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def int_or_999(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 999


def fmt(value: Any) -> str:
    return f"{to_float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
