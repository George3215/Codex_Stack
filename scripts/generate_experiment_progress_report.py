from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


CRITERIA = [
    ("candidate_pose_ranker_used", "SupportMap / pose ranker"),
    ("pose_risk_ranker_used", "PoseRiskNet"),
    ("stone_fit_ranker_used", "StoneSlotNet"),
    ("base_support_prior", "base support prior"),
    ("base_continuity_prior", "base continuity prior"),
    ("candidate_probe_hard_gate", "candidate hard gate"),
    ("moon_gate_strict", "moon strict gate"),
    ("low_release_search_requested", "low release search"),
]

MODEL_METRIC_KEYS = [
    ("test_accuracy", ("test", "accuracy")),
    ("test_precision", ("test", "precision")),
    ("test_recall", ("test", "recall")),
    ("test_f1", ("test", "f1")),
    ("top1_hit_rate", ("test_group", "top1_hit_rate")),
    ("top3_hit_rate", ("test_group", "top3_hit_rate")),
    ("top1_safe_rate", ("test_group", "top1_safe_rate")),
    ("top3_safe_rate", ("test_group", "top3_safe_rate")),
    ("top1_success_rate", ("test_group", "top1_success_rate")),
    ("top3_success_rate", ("test_group", "top3_success_rate")),
]


def main() -> int:
    args = parse_args()
    batch_root = args.batch_root.resolve()
    output = args.output.resolve()
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    result_rows = read_result_rows(batch_root, args.target_contains)
    run_rows = summarize_runs(result_rows)
    task_rows = summarize_task_growth(run_rows)
    criterion_rows = summarize_criteria(result_rows)
    dataset_rows = read_dataset_flow(batch_root, args.dataset)
    model_rows = read_model_metrics(batch_root)

    write_csv(output / "success_by_run.csv", run_rows)
    write_csv(output / "task_growth.csv", task_rows)
    write_csv(output / "criterion_effectiveness.csv", criterion_rows)
    write_csv(output / "dataset_flow.csv", dataset_rows)
    write_csv(output / "model_metrics.csv", model_rows)

    chart_paths = make_charts(figures, run_rows, task_rows, criterion_rows, dataset_rows, model_rows)
    write_readme(
        output / "README.md",
        batch_root=batch_root,
        output=output,
        result_rows=result_rows,
        run_rows=run_rows,
        task_rows=task_rows,
        criterion_rows=criterion_rows,
        dataset_rows=dataset_rows,
        model_rows=model_rows,
        chart_paths=chart_paths,
        target_contains=args.target_contains,
    )
    print(output)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Chinese data-flow and progress report for MoonStack wall-stacking experiments."
    )
    parser.add_argument("--batch-root", type=Path, default=Path("batch_runs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-contains", default="single_face_wall")
    parser.add_argument(
        "--dataset",
        action="append",
        type=Path,
        default=[],
        help="Optional dataset directories to force into the data-flow table.",
    )
    return parser.parse_args()


def read_result_rows(batch_root: Path, target_contains: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(batch_root.glob("**/results.csv")):
        if is_under_analysis_dir(path):
            continue
        run_dir = path.parent
        for row in read_csv(path):
            target_name = row.get("target_name", "")
            if target_contains and target_contains not in target_name:
                continue
            row = dict(row)
            row["run_name"] = run_dir.name
            row["run_dir"] = str(run_dir)
            row["run_group"] = run_dir.parent.name
            row["run_mtime"] = run_dir.stat().st_mtime
            row["run_order"] = run_order_key(run_dir)
            rows.append(row)
    return rows


def summarize_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("run_dir", ""),
            row.get("target_name", "unknown"),
            row.get("strategy", "unknown"),
            row.get("gravity", "unknown"),
        )
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for (run_dir, target_name, strategy, gravity), items in grouped.items():
        first = items[0]
        out.append(
            {
                "run_group": first.get("run_group", ""),
                "run_name": first.get("run_name", ""),
                "run_dir": run_dir,
                "run_order": first.get("run_order", ""),
                "run_mtime": first.get("run_mtime", 0.0),
                "target_name": target_name,
                "strategy": strategy,
                "gravity": gravity,
                "task": f"{target_name}|{gravity}",
                "trials": len(items),
                "strict_success_count": sum(to_int(row.get("success")) for row in items),
                "shape_success_count": sum(to_int(row.get("shape_success", row.get("success"))) for row in items),
                "strict_success_rate": mean(to_float(row.get("success")) for row in items),
                "shape_success_rate": mean(to_float(row.get("shape_success", row.get("success"))) for row in items),
                "stable_fraction": stable_fraction(items),
                "mean_stable_count": mean(to_float(row.get("stable_count")) for row in items),
                "mean_failure_count": mean(to_float(row.get("failure_count")) for row in items),
                "mean_height_m": mean(to_float(row.get("stack_height_m")) for row in items),
                "mean_rmse_xy_m": mean(to_float(row.get("target_rmse_xy_m")) for row in items),
                "mean_drift_m": mean(to_float(row.get("max_horizontal_drift_m")) for row in items),
                "mean_velocity_inf_norm": mean(to_float(row.get("velocity_inf_norm")) for row in items),
                "candidate_pose_ranker_used_rate": mean(to_bool(row.get("candidate_pose_ranker_used")) for row in items),
                "pose_risk_ranker_used_rate": mean(to_bool(row.get("pose_risk_ranker_used")) for row in items),
                "stone_fit_ranker_used_rate": mean(to_bool(row.get("stone_fit_ranker_used")) for row in items),
                "base_support_prior_rate": mean(to_bool(row.get("base_support_prior")) for row in items),
                "base_continuity_prior_rate": mean(to_bool(row.get("base_continuity_prior")) for row in items),
                "low_release_search_requested_rate": mean(
                    to_bool(row.get("low_release_search_requested")) for row in items
                ),
            }
        )

    out.sort(key=lambda row: (float(row["run_mtime"]), row["run_name"], row["task"]))
    return out


def summarize_task_growth(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["task"])].append(row)

    out: list[dict[str, Any]] = []
    for task, items in grouped.items():
        items = sorted(items, key=lambda row: (float(row["run_mtime"]), str(row["run_name"])))
        if not items:
            continue
        latest = items[-1]
        previous = items[-2] if len(items) >= 2 else None
        previous_best = max(items[:-1], key=lambda row: float(row["strict_success_rate"]), default=None)
        first = items[0]
        out.append(
            {
                "task": task,
                "runs": len(items),
                "total_trials": sum(int(row["trials"]) for row in items),
                "first_run": first["run_name"],
                "latest_run": latest["run_name"],
                "latest_trials": latest["trials"],
                "latest_strict_success_rate": latest["strict_success_rate"],
                "latest_shape_success_rate": latest["shape_success_rate"],
                "latest_stable_fraction": latest["stable_fraction"],
                "previous_run": previous["run_name"] if previous else "",
                "previous_strict_success_rate": previous["strict_success_rate"] if previous else "",
                "growth_vs_previous_strict": delta(latest, previous, "strict_success_rate"),
                "growth_vs_previous_shape": delta(latest, previous, "shape_success_rate"),
                "growth_vs_previous_stable": delta(latest, previous, "stable_fraction"),
                "best_previous_run": previous_best["run_name"] if previous_best else "",
                "best_previous_strict_success_rate": previous_best["strict_success_rate"] if previous_best else "",
                "growth_vs_best_previous_strict": delta(latest, previous_best, "strict_success_rate"),
                "trend_label": trend_label(items),
                "bottleneck_hint": bottleneck_hint(items),
            }
        )
    out.sort(key=lambda row: (-int(row["total_trials"]), str(row["task"])))
    return out


def summarize_criteria(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, label in CRITERIA:
        if not any(key in row for row in rows):
            continue
        enabled = [row for row in rows if to_bool(row.get(key))]
        disabled = [row for row in rows if key in row and not to_bool(row.get(key))]
        if not enabled or not disabled:
            out.append(
                {
                    "criterion": key,
                    "label": label,
                    "enabled_trials": len(enabled),
                    "disabled_trials": len(disabled),
                    "enabled_strict_success_rate": rate(enabled, "success"),
                    "disabled_strict_success_rate": rate(disabled, "success"),
                    "delta_strict_success_rate": "",
                    "enabled_shape_success_rate": rate(enabled, "shape_success"),
                    "disabled_shape_success_rate": rate(disabled, "shape_success"),
                    "delta_shape_success_rate": "",
                    "enabled_stable_fraction": stable_fraction(enabled),
                    "disabled_stable_fraction": stable_fraction(disabled),
                    "delta_stable_fraction": "",
                    "enabled_mean_drift_m": mean(to_float(row.get("max_horizontal_drift_m")) for row in enabled),
                    "disabled_mean_drift_m": mean(to_float(row.get("max_horizontal_drift_m")) for row in disabled),
                    "interpretation": "evidence_insufficient_single_state",
                }
            )
            continue
        enabled_strict = rate(enabled, "success")
        disabled_strict = rate(disabled, "success")
        enabled_shape = rate(enabled, "shape_success")
        disabled_shape = rate(disabled, "shape_success")
        enabled_stable = stable_fraction(enabled)
        disabled_stable = stable_fraction(disabled)
        out.append(
            {
                "criterion": key,
                "label": label,
                "enabled_trials": len(enabled),
                "disabled_trials": len(disabled),
                "enabled_strict_success_rate": enabled_strict,
                "disabled_strict_success_rate": disabled_strict,
                "delta_strict_success_rate": enabled_strict - disabled_strict,
                "enabled_shape_success_rate": enabled_shape,
                "disabled_shape_success_rate": disabled_shape,
                "delta_shape_success_rate": enabled_shape - disabled_shape,
                "enabled_stable_fraction": enabled_stable,
                "disabled_stable_fraction": disabled_stable,
                "delta_stable_fraction": enabled_stable - disabled_stable,
                "enabled_mean_drift_m": mean(to_float(row.get("max_horizontal_drift_m")) for row in enabled),
                "disabled_mean_drift_m": mean(to_float(row.get("max_horizontal_drift_m")) for row in disabled),
                "interpretation": criterion_label(enabled_strict - disabled_strict, enabled_stable - disabled_stable),
            }
        )
    out.sort(
        key=lambda row: (
            row.get("delta_strict_success_rate") == "",
            -abs(to_float(row.get("delta_strict_success_rate"))),
            str(row["criterion"]),
        )
    )
    return out


def read_dataset_flow(batch_root: Path, forced: list[Path]) -> list[dict[str, Any]]:
    paths = {path.resolve() for path in batch_root.glob("**/dataset_summary.json")}
    for item in forced:
        path = item / "dataset_summary.json" if item.is_dir() else item
        paths.add(path.resolve())

    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        output_dir = Path(data.get("output_dir") or path.parent)
        rows.append(
            {
                "dataset_name": output_dir.name,
                "dataset_dir": str(output_dir),
                "created_at": data.get("created_at", ""),
                "source_dataset": data.get("source_dataset", ""),
                "run_examples": data.get("run_example_count", ""),
                "placement_examples": data.get("placement_example_count", ""),
                "candidate_pose_examples": data.get("candidate_pose_example_count", ""),
                "assignment_candidate_examples": data.get("assignment_candidate_example_count", ""),
                "candidate_pose_rejected": nested(data, ["file_stats", "candidate_pose", "rejected"]),
                "placement_rejected": nested(data, ["file_stats", "placement", "rejected"]),
                "assignment_rejected": nested(data, ["file_stats", "assignment", "rejected"]),
                "filters": json.dumps(data.get("filters", {}), ensure_ascii=False, sort_keys=True),
            }
        )
    rows.sort(key=lambda row: (str(row["created_at"]), str(row["dataset_name"])))
    return rows


def read_model_metrics(batch_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(batch_root.glob("**/*metrics.json")):
        if is_under_analysis_dir(path):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        row: dict[str, Any] = {
            "model_name": model_name_from_path(path),
            "metrics_path": str(path),
            "output_dir": data.get("output_dir", str(path.parent)),
            "dataset_dir": data.get("dataset_dir", ""),
            "created_at": data.get("created_at", ""),
            "row_count": data.get("row_count", ""),
            "train_rows": data.get("train_rows", ""),
            "test_rows": data.get("test_rows", ""),
            "epochs": data.get("epochs", ""),
            "batch_size": data.get("batch_size", ""),
            "hidden": data.get("hidden", ""),
            "purpose": data.get("purpose", ""),
            "input_policy": data.get("input_policy", ""),
        }
        for out_key, key_path in MODEL_METRIC_KEYS:
            row[out_key] = nested(data, list(key_path))
        rows.append(row)
    rows.sort(key=lambda row: (str(row["created_at"]), str(row["model_name"])))
    return rows


def make_charts(
    figures: Path,
    run_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    criterion_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    chart_paths: dict[str, Path] = {}
    if run_rows:
        path = figures / "success_rate_timeline.svg"
        write_success_timeline_svg(run_rows, path)
        chart_paths["success_timeline"] = path

    if task_rows:
        path = figures / "task_latest_success.svg"
        write_task_success_svg(task_rows, path)
        chart_paths["task_success"] = path

    if criterion_rows:
        path = figures / "criterion_delta.svg"
        write_criterion_delta_svg(criterion_rows, path)
        chart_paths["criterion_delta"] = path

    if dataset_rows:
        path = figures / "dataset_volume.svg"
        write_dataset_volume_svg(dataset_rows, path)
        chart_paths["dataset_volume"] = path

    if model_rows:
        path = figures / "model_topk_metrics.svg"
        write_model_metrics_svg(model_rows, path)
        chart_paths["model_metrics"] = path

    return chart_paths


def write_success_timeline_svg(run_rows: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["task"])].append(row)
    ranked_tasks = sorted(grouped, key=lambda task: -sum(int(row["trials"]) for row in grouped[task]))[:6]
    series: list[dict[str, Any]] = []
    for index, task in enumerate(ranked_tasks):
        rows = sorted(grouped[task], key=lambda row: (float(row["run_mtime"]), str(row["run_name"])))
        series.append(
            {
                "label": short_task(task),
                "color": palette(index),
                "values": [float(row["strict_success_rate"]) for row in rows],
            }
        )
    write_line_svg(path, "Strict success rate timeline", "Comparable run index", "Strict success rate", series, 0.0, 1.0)


def write_task_success_svg(task_rows: list[dict[str, Any]], path: Path) -> None:
    rows = sorted(task_rows, key=lambda row: (-to_float(row["latest_strict_success_rate"]), -int(row["total_trials"])))[:12]
    labels = [short_task(str(row["task"])) for row in rows]
    groups = [
        ("strict", "#2563eb", [to_float(row["latest_strict_success_rate"]) for row in rows]),
        ("shape", "#16a34a", [to_float(row["latest_shape_success_rate"]) for row in rows]),
    ]
    write_grouped_bar_svg(path, "Latest task success rates", labels, groups, 0.0, 1.0)


def write_criterion_delta_svg(criterion_rows: list[dict[str, Any]], path: Path) -> None:
    rows = [row for row in criterion_rows if row.get("delta_strict_success_rate") != ""]
    rows = sorted(rows, key=lambda row: abs(to_float(row["delta_strict_success_rate"])), reverse=True)[:10]
    labels = [str(row["label"]) for row in rows]
    values = [to_float(row["delta_strict_success_rate"]) for row in rows] + [
        to_float(row["delta_stable_fraction"]) for row in rows
    ]
    ymin = min(-0.05, min(values, default=0.0))
    ymax = max(0.05, max(values, default=0.0))
    pad = max(0.05, (ymax - ymin) * 0.12)
    groups = [
        ("strict delta", "#dc2626", [to_float(row["delta_strict_success_rate"]) for row in rows]),
        ("stable delta", "#0d9488", [to_float(row["delta_stable_fraction"]) for row in rows]),
    ]
    write_grouped_bar_svg(path, "Criterion enabled-minus-disabled deltas", labels, groups, ymin - pad, ymax + pad)


def write_dataset_volume_svg(dataset_rows: list[dict[str, Any]], path: Path) -> None:
    rows = dataset_rows[-10:]
    labels = [str(row["dataset_name"]) for row in rows]
    groups = [
        ("placement log10", "#2563eb", [math.log10(1.0 + to_float(row.get("placement_examples"))) for row in rows]),
        ("candidate log10", "#16a34a", [math.log10(1.0 + to_float(row.get("candidate_pose_examples"))) for row in rows]),
        ("assignment log10", "#f59e0b", [math.log10(1.0 + to_float(row.get("assignment_candidate_examples"))) for row in rows]),
    ]
    ymax = max([value for _, _, values in groups for value in values], default=1.0)
    write_grouped_bar_svg(path, "Dataset volume, log10 rows", labels, groups, 0.0, max(1.0, ymax * 1.08))


def write_model_metrics_svg(model_rows: list[dict[str, Any]], path: Path) -> None:
    rows = [
        row
        for row in model_rows
        if any(row.get(key) != "" for key in ("top1_hit_rate", "top3_hit_rate", "top1_safe_rate", "top3_safe_rate"))
    ][-14:]
    if not rows:
        rows = [row for row in model_rows if row.get("test_f1") != ""][-14:]
    labels = [short_model(str(row["model_name"])) for row in rows]
    groups = [
        ("top1/F1", "#7c3aed", [first_float(row, ["top1_hit_rate", "top1_safe_rate", "top1_success_rate", "test_f1"]) for row in rows]),
        (
            "top3/precision",
            "#0891b2",
            [first_float(row, ["top3_hit_rate", "top3_safe_rate", "top3_success_rate", "test_precision"]) for row in rows],
        ),
    ]
    write_grouped_bar_svg(path, "Model ranking metrics", labels, groups, 0.0, 1.0)


def write_line_svg(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: list[dict[str, Any]],
    ymin: float,
    ymax: float,
) -> None:
    width, height = 1040, 560
    left, top, right, bottom = 70, 54, 260, 84
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_len = max((len(item["values"]) for item in series), default=1)
    max_len = max(max_len, 2)

    def sx(index: int) -> float:
        return left + (index / (max_len - 1)) * plot_w

    def sy(value: float) -> float:
        value = min(max(value, ymin), ymax)
        return top + (ymax - value) / max(ymax - ymin, 1e-9) * plot_h

    parts = svg_header(width, height, title)
    parts.extend(svg_axes(left, top, plot_w, plot_h, ymin, ymax, xlabel, ylabel))
    for index, item in enumerate(series):
        points = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(item["values"]))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{item["color"]}" stroke-width="2.4" stroke-linejoin="round"/>'
        )
        for i, value in enumerate(item["values"]):
            parts.append(f'<circle cx="{sx(i):.1f}" cy="{sy(value):.1f}" r="3.2" fill="{item["color"]}"/>')
        ly = top + 22 + index * 23
        parts.append(f'<rect x="{width - right + 28}" y="{ly - 10}" width="14" height="14" fill="{item["color"]}"/>')
        parts.append(svg_text(width - right + 50, ly + 1, item["label"], 12, "start"))
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_grouped_bar_svg(
    path: Path,
    title: str,
    labels: list[str],
    groups: list[tuple[str, str, list[float]]],
    ymin: float,
    ymax: float,
) -> None:
    width, height = 1100, 610
    left, top, right, bottom = 76, 54, 220, 150
    plot_w = width - left - right
    plot_h = height - top - bottom
    count = max(len(labels), 1)
    group_count = max(len(groups), 1)
    slot_w = plot_w / count
    bar_w = min(24.0, slot_w * 0.72 / group_count)

    def sx(label_index: int, group_index: int) -> float:
        center = left + slot_w * (label_index + 0.5)
        offset = (group_index - (group_count - 1) / 2) * bar_w * 1.18
        return center + offset - bar_w / 2

    def sy(value: float) -> float:
        value = min(max(value, ymin), ymax)
        return top + (ymax - value) / max(ymax - ymin, 1e-9) * plot_h

    zero_y = sy(0.0)
    parts = svg_header(width, height, title)
    parts.extend(svg_axes(left, top, plot_w, plot_h, ymin, ymax, "", "value"))
    parts.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}" stroke="#111827" stroke-width="1"/>')
    for gi, (_, color, values) in enumerate(groups):
        for li, value in enumerate(values):
            x = sx(li, gi)
            y = sy(max(value, 0.0))
            y0 = sy(min(value, 0.0))
            h = abs(y0 - y)
            parts.append(f'<rect x="{x:.1f}" y="{min(y, y0):.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
    for li, label in enumerate(labels):
        x = left + slot_w * (li + 0.5)
        parts.append(
            f'<text x="{x:.1f}" y="{height - bottom + 38}" font-size="11" fill="#374151" '
            f'text-anchor="end" transform="rotate(-36 {x:.1f},{height - bottom + 38})">{svg_escape(label[:34])}</text>'
        )
    for gi, (label, color, _) in enumerate(groups):
        ly = top + 22 + gi * 24
        parts.append(f'<rect x="{width - right + 28}" y="{ly - 10}" width="14" height="14" fill="{color}"/>')
        parts.append(svg_text(width - right + 50, ly + 1, label, 12, "start"))
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 28, title, 18, "middle", weight="700"),
    ]


def svg_axes(left: int, top: int, plot_w: int, plot_h: int, ymin: float, ymax: float, xlabel: str, ylabel: str) -> list[str]:
    parts = [
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f9fafb" stroke="#d1d5db"/>',
    ]
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        y = top + plot_h - plot_h * i / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(svg_text(left - 10, y + 4, f"{value:.2f}", 11, "end"))
    if xlabel:
        parts.append(svg_text(left + plot_w / 2, top + plot_h + 60, xlabel, 12, "middle"))
    if ylabel:
        parts.append(
            f'<text x="20" y="{top + plot_h / 2:.1f}" font-size="12" fill="#374151" text-anchor="middle" '
            f'transform="rotate(-90 20,{top + plot_h / 2:.1f})">{svg_escape(ylabel)}</text>'
        )
    return parts


def svg_text(x: float, y: float, text: str, size: int, anchor: str, weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-family="Arial, sans-serif" '
        f'font-weight="{weight}" fill="#111827" text-anchor="{anchor}">{svg_escape(text)}</text>'
    )


def svg_escape(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def palette(index: int) -> str:
    colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#f59e0b", "#0891b2", "#4b5563"]
    return colors[index % len(colors)]


def plot_success_timeline(plt: Any, run_rows: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["task"])].append(row)
    ranked_tasks = sorted(grouped, key=lambda task: -sum(int(row["trials"]) for row in grouped[task]))[:6]

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    for task in ranked_tasks:
        rows = sorted(grouped[task], key=lambda row: (float(row["run_mtime"]), str(row["run_name"])))
        y = [float(row["strict_success_rate"]) for row in rows]
        x = list(range(1, len(y) + 1))
        ax.plot(x, y, marker="o", linewidth=1.8, label=short_task(task))
    ax.set_title("Strict success rate timeline")
    ax.set_xlabel("Comparable run index")
    ax.set_ylabel("Strict success rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_task_success(plt: Any, task_rows: list[dict[str, Any]], path: Path) -> None:
    rows = sorted(task_rows, key=lambda row: (-to_float(row["latest_strict_success_rate"]), -int(row["total_trials"])))[:12]
    labels = [short_task(str(row["task"])) for row in rows]
    strict = [to_float(row["latest_strict_success_rate"]) for row in rows]
    shape = [to_float(row["latest_shape_success_rate"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    x = list(range(len(rows)))
    ax.bar([i - 0.18 for i in x], strict, width=0.36, label="strict", color="#2563eb")
    ax.bar([i + 0.18 for i in x], shape, width=0.36, label="shape", color="#16a34a")
    ax.set_title("Latest task success rates")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_criterion_delta(plt: Any, criterion_rows: list[dict[str, Any]], path: Path) -> None:
    rows = [row for row in criterion_rows if row.get("delta_strict_success_rate") != ""]
    rows = sorted(rows, key=lambda row: abs(to_float(row["delta_strict_success_rate"])), reverse=True)[:10]
    labels = [str(row["label"]) for row in rows]
    strict_delta = [to_float(row["delta_strict_success_rate"]) for row in rows]
    stable_delta = [to_float(row["delta_stable_fraction"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    x = list(range(len(rows)))
    ax.bar([i - 0.18 for i in x], strict_delta, width=0.36, label="strict delta", color="#dc2626")
    ax.bar([i + 0.18 for i in x], stable_delta, width=0.36, label="stable delta", color="#0d9488")
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("Criterion enabled-minus-disabled deltas")
    ax.set_ylabel("Delta")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_dataset_volume(plt: Any, dataset_rows: list[dict[str, Any]], path: Path) -> None:
    rows = dataset_rows[-10:]
    labels = [str(row["dataset_name"]) for row in rows]
    placement = [to_float(row.get("placement_examples")) for row in rows]
    candidate = [to_float(row.get("candidate_pose_examples")) for row in rows]
    assignment = [to_float(row.get("assignment_candidate_examples")) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=160)
    x = list(range(len(rows)))
    ax.bar([i - 0.25 for i in x], placement, width=0.25, label="placement", color="#2563eb")
    ax.bar(x, candidate, width=0.25, label="candidate pose", color="#16a34a")
    ax.bar([i + 0.25 for i in x], assignment, width=0.25, label="assignment", color="#f59e0b")
    ax.set_title("Dataset volume")
    ax.set_ylabel("Rows")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_model_metrics(plt: Any, model_rows: list[dict[str, Any]], path: Path) -> None:
    rows = [
        row
        for row in model_rows
        if any(row.get(key) != "" for key in ("top1_hit_rate", "top3_hit_rate", "top1_safe_rate", "top3_safe_rate"))
    ][-14:]
    if not rows:
        rows = [row for row in model_rows if row.get("test_f1") != ""][-14:]
    labels = [short_model(str(row["model_name"])) for row in rows]
    top1 = [first_float(row, ["top1_hit_rate", "top1_safe_rate", "top1_success_rate", "test_f1"]) for row in rows]
    top3 = [first_float(row, ["top3_hit_rate", "top3_safe_rate", "top3_success_rate", "test_precision"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=160)
    x = list(range(len(rows)))
    ax.bar([i - 0.18 for i in x], top1, width=0.36, label="top1/F1", color="#7c3aed")
    ax.bar([i + 0.18 for i in x], top3, width=0.36, label="top3/precision", color="#0891b2")
    ax.set_title("Model ranking metrics")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_readme(
    path: Path,
    *,
    batch_root: Path,
    output: Path,
    result_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    criterion_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    chart_paths: dict[str, Path],
    target_contains: str,
) -> None:
    lines: list[str] = [
        "# 石头堆叠数据流与效果增长报告",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- batch_root: `{batch_root}`",
        f"- report_dir: `{output}`",
        f"- target_filter: `{target_contains}`",
        "",
        "## 一眼判断",
        "",
    ]
    lines.extend(quick_judgement(task_rows, criterion_rows, model_rows))
    lines.extend(
        [
            "",
            "## 本报告产物",
            "",
            "- `success_by_run.csv`: 每个 run / target / gravity 的成功率、稳定比例、漂移和网络启用情况。",
            "- `task_growth.csv`: 每个任务的最新成功率、相对上一可比 run 的增长率、瓶颈提示。",
            "- `criterion_effectiveness.csv`: 每个判别标准启用/未启用的统计对照。",
            "- `dataset_flow.csv`: 数据集来源、清洗后样本量和剔除量。",
            "- `model_metrics.csv`: 各网络训练样本量、测试指标、top-k 指标。",
            "- `figures/`: 自动生成的折线图和柱状图。",
            "",
            "## 数据流动",
            "",
        ]
    )
    lines.extend(data_flow_table(dataset_rows, model_rows))
    lines.extend(["", "## 任务成功率与增长率", ""])
    lines.extend(task_growth_table(task_rows))
    lines.extend(["", "## 判别标准统计对照", ""])
    lines.append("下面的启用/未启用差值是观察性统计，不是严格因果证明。严格因果需要后续同 seed、同目标、同数据的 A/B。")
    lines.append("")
    lines.extend(criterion_table(criterion_rows))
    lines.extend(["", "## 网络训练与参与程度", ""])
    lines.extend(model_table(model_rows))
    lines.extend(["", "## 图表", ""])
    if chart_paths:
        for name, chart_path in chart_paths.items():
            rel = chart_path.relative_to(output).as_posix()
            lines.append(f"![{name}]({rel})")
            lines.append("")
    else:
        lines.append("- 未生成图表，通常是当前环境缺少 matplotlib。CSV 统计仍然可用。")
    lines.extend(["", "## 瓶颈判断规则", ""])
    lines.extend(
        [
            "- 如果最近 3 个可比 run 的 strict success 增长都小于 5 个百分点，并且 shape/stable 也不涨，视为疑似瓶颈。",
            "- 如果 shape success 上升但 strict success 不升，优先检查释放高度、settling、漂移和残余速度。",
            "- 如果 top3 指标高但 top1 指标低，说明网络能筛出候选池，但还不能单独决策第一名。",
            "- 如果 StoneSlotNet 低而 SupportMap/WallCritic 高，说明石头选择必须结合墙体局部状态。",
            "- 如果 base 成功率高但 middle/cap 低，说明主要问题是上层支撑、互锁和误差传播。",
            "",
            "## 原始规模",
            "",
            f"- results rows: `{len(result_rows)}`",
            f"- run/task rows: `{len(run_rows)}`",
            f"- task groups: `{len(task_rows)}`",
            f"- model metric files: `{len(model_rows)}`",
            f"- dataset summaries: `{len(dataset_rows)}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def quick_judgement(
    task_rows: list[dict[str, Any]], criterion_rows: list[dict[str, Any]], model_rows: list[dict[str, Any]]
) -> list[str]:
    lines: list[str] = []
    bottlenecks = [row for row in task_rows if "bottleneck" in str(row.get("bottleneck_hint", ""))]
    latest_four = [
        row
        for row in task_rows
        if "4course" in str(row.get("task", "")) and "moon" in str(row.get("task", ""))
    ]
    if latest_four:
        row = sorted(latest_four, key=lambda item: -int(item["total_trials"]))[0]
        lines.append(
            f"- 4 层月面最新 strict success rate: `{fmt(row['latest_strict_success_rate'])}`，"
            f"shape: `{fmt(row['latest_shape_success_rate'])}`，stable: `{fmt(row['latest_stable_fraction'])}`。"
        )
    if bottlenecks:
        lines.append(f"- 有 `{len(bottlenecks)}` 个任务被自动标记为疑似瓶颈，需要看 `task_growth.csv`。")
    else:
        lines.append("- 暂未按最近 3 个可比 run 自动判定明显瓶颈，但这不代表没有局部瓶颈。")

    effective = [
        row
        for row in criterion_rows
        if row.get("delta_strict_success_rate") != "" and row.get("interpretation") == "promising_observational"
    ]
    if not effective:
        effective = [
            row
            for row in criterion_rows
            if row.get("delta_strict_success_rate") != "" and to_float(row["delta_strict_success_rate"]) > 0
        ]
    if effective:
        best = max(effective, key=lambda row: to_float(row["delta_strict_success_rate"]))
        lines.append(
            f"- 观察性统计中 strict success 差值最大的判别标准是 `{best['label']}`，"
            f"enabled-disabled = `{fmt(best['delta_strict_success_rate'])}`。"
        )

    ranked_models = [
        row
        for row in model_rows
        if any(row.get(key) != "" for key in ("top1_hit_rate", "top1_safe_rate", "test_f1"))
    ]
    if ranked_models:
        latest = ranked_models[-1]
        lines.append(
            f"- 最新可读模型指标来自 `{latest['model_name']}`，训练行数 `{latest.get('row_count', '')}`，"
            f"test F1 `{fmt(latest.get('test_f1', ''))}`。"
        )
    return lines


def data_flow_table(dataset_rows: list[dict[str, Any]], model_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| 阶段 | 名称 | 输入/来源 | 输出规模 | 用途 |",
        "|---|---|---|---:|---|",
    ]
    for row in dataset_rows[-8:]:
        size = (
            f"placement={row.get('placement_examples', '')}, "
            f"candidate={row.get('candidate_pose_examples', '')}, "
            f"assignment={row.get('assignment_candidate_examples', '')}"
        )
        source = row.get("source_dataset", "") or "raw experiment logs"
        lines.append(f"| dataset | `{row['dataset_name']}` | `{source}` | {size} | training / analysis |")
    for row in model_rows[-12:]:
        size = f"rows={row.get('row_count', '')}, epochs={row.get('epochs', '')}"
        source = row.get("dataset_dir", "") or row.get("metrics_path", "")
        lines.append(f"| model | `{row['model_name']}` | `{source}` | {size} | ranking / risk / critic |")
    return lines


def task_growth_table(task_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| 任务 | runs | trials | 最新 run | strict | shape | stable | vs previous strict | 瓶颈提示 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in task_rows[:18]:
        lines.append(
            f"| `{row['task']}` | {row['runs']} | {row['total_trials']} | `{row['latest_run']}` | "
            f"{fmt(row['latest_strict_success_rate'])} | {fmt(row['latest_shape_success_rate'])} | "
            f"{fmt(row['latest_stable_fraction'])} | {fmt(row['growth_vs_previous_strict'])} | "
            f"{row['bottleneck_hint']} |"
        )
    return lines


def criterion_table(criterion_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| 判别标准 | enabled trials | disabled trials | strict 差值 | shape 差值 | stable 差值 | 解释 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in criterion_rows:
        lines.append(
            f"| `{row['label']}` | {row['enabled_trials']} | {row['disabled_trials']} | "
            f"{fmt(row['delta_strict_success_rate'])} | {fmt(row['delta_shape_success_rate'])} | "
            f"{fmt(row['delta_stable_fraction'])} | {row['interpretation']} |"
        )
    return lines


def model_table(model_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| 模型 | rows | epochs | test F1 | top1 | top3 | 数据集 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in model_rows[-20:]:
        top1 = first_float(row, ["top1_hit_rate", "top1_safe_rate", "top1_success_rate", "test_accuracy"])
        top3 = first_float(row, ["top3_hit_rate", "top3_safe_rate", "top3_success_rate", "test_precision"])
        dataset = Path(str(row.get("dataset_dir", ""))).name if row.get("dataset_dir") else ""
        lines.append(
            f"| `{row['model_name']}` | {row.get('row_count', '')} | {row.get('epochs', '')} | "
            f"{fmt(row.get('test_f1', ''))} | {fmt(top1)} | {fmt(top3)} | `{dataset}` |"
        )
    return lines


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


def is_under_analysis_dir(path: Path) -> bool:
    parts = set(path.parts)
    return "_aggregate" in parts or "analysis" in parts or "progress_reports" in parts


def run_order_key(run_dir: Path) -> str:
    text = f"{run_dir.parent.name}_{run_dir.name}"
    match = re.search(r"(20\d{6})(?:[_-]?(\d{6}))?", text)
    if match:
        return "".join(part for part in match.groups() if part)
    return f"{run_dir.stat().st_mtime:.0f}"


def model_name_from_path(path: Path) -> str:
    name = path.name
    if name == "metrics.json":
        return path.parent.name
    base = name.replace("_metrics.json", "").replace(".json", "")
    if base in {"stone_fit_net", "pose_risk_net", "candidate_pose_rank_net", "support_map_ranker", "wall_state_critic"}:
        return f"{path.parent.name}/{base}"
    return base


def nested(data: dict[str, Any], path: list[str]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return ""
        current = current[key]
    return current


def to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result


def to_int(value: Any) -> int:
    return int(round(to_float(value)))


def to_bool(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return 1
    if text in {"0", "false", "no", "n", "off", ""}:
        return 0
    return int(to_float(value) != 0.0)


def mean(values: Any) -> float:
    seq = [to_float(value) for value in values]
    return sum(seq) / len(seq) if seq else 0.0


def rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return mean(to_float(row.get(key, row.get("success"))) for row in rows)


def stable_fraction(rows: list[dict[str, Any]]) -> float:
    stable = sum(to_float(row.get("stable_count")) for row in rows)
    rocks = sum(to_float(row.get("rock_count")) for row in rows)
    return stable / rocks if rocks > 0 else 0.0


def delta(latest: dict[str, Any], previous: dict[str, Any] | None, key: str) -> float | str:
    if not previous:
        return ""
    return to_float(latest.get(key)) - to_float(previous.get(key))


def trend_label(items: list[dict[str, Any]]) -> str:
    if len(items) < 2:
        return "single_run"
    d = to_float(items[-1]["strict_success_rate"]) - to_float(items[-2]["strict_success_rate"])
    if d > 0.05:
        return "improving"
    if d < -0.05:
        return "regressing"
    return "flat"


def bottleneck_hint(items: list[dict[str, Any]]) -> str:
    if len(items) < 3:
        return "insufficient_history"
    last3 = items[-3:]
    strict_growth = to_float(last3[-1]["strict_success_rate"]) - to_float(last3[0]["strict_success_rate"])
    shape_growth = to_float(last3[-1]["shape_success_rate"]) - to_float(last3[0]["shape_success_rate"])
    stable_growth = to_float(last3[-1]["stable_fraction"]) - to_float(last3[0]["stable_fraction"])
    latest = last3[-1]
    if abs(strict_growth) < 0.05 and abs(shape_growth) < 0.05 and abs(stable_growth) < 0.03:
        return "possible_bottleneck"
    if shape_growth > 0.05 and strict_growth < 0.03:
        return "dynamic_or_settling_bottleneck"
    if to_float(latest["stable_fraction"]) > 0.8 and to_float(latest["strict_success_rate"]) < 0.2:
        return "shape_metric_or_drift_bottleneck"
    return "still_moving"


def criterion_label(strict_delta: float, stable_delta: float) -> str:
    if strict_delta > 0.05 and stable_delta >= -0.03:
        return "promising_observational"
    if strict_delta < -0.05 and stable_delta <= 0.03:
        return "possibly_harmful_observational"
    if stable_delta > 0.05 and strict_delta <= 0.03:
        return "stability_only_gain"
    return "mixed_or_unclear"


def first_float(row: dict[str, Any], keys: list[str]) -> float:
    for key in keys:
        value = row.get(key, "")
        if value != "":
            return to_float(value)
    return 0.0


def short_task(task: str) -> str:
    task = task.replace("single_face_wall_", "wall_")
    task = task.replace("_v1", "")
    return task[:42]


def short_model(name: str) -> str:
    name = name.replace("20260626_", "").replace("20260622_", "")
    name = name.replace("_flywheel_3to4", "")
    name = name.replace("_metrics", "")
    return name[:36]


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
    return f"{to_float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
