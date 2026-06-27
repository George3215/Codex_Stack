from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


DEFAULT_TARGETS = {"single_face_wall_3course_v1", "single_face_wall_4course_v1"}
DEFAULT_STRATEGIES = {"statics_wall"}
DEFAULT_GRAVITIES = {"earth", "moon"}
DEFAULT_ROLES = {"base", "middle", "cap"}

REQUIRED_GEOMETRY = [
    "rock_volume",
    "rock_surface_area",
    "rock_bbox_x",
    "rock_bbox_y",
    "rock_bbox_z",
    "rock_angularity",
    "rock_spike_score",
    "rock_major_face_count",
    "rock_support_face_count",
    "rock_support_face_area_ratio",
    "rock_mass",
]

REQUIRED_CANDIDATE_POSE = [
    "pose_x",
    "pose_y",
    "pose_z",
    "pose_qw",
    "pose_qx",
    "pose_qy",
    "pose_qz",
    "target_error_xy_m",
    "target_y_error_m",
    "placed_disturbance_xy_m",
    "velocity_inf_norm_after_place",
    "height_gain_m",
]

REQUIRED_PLACEMENT = [
    "target_error_xy_m",
    "target_y_error_m",
    "placed_disturbance_xy_m",
    "velocity_inf_norm_after_place",
    "height_gain_m",
]


def main() -> int:
    args = parse_args()
    source = args.dataset.resolve()
    output = unique_dir(args.output.resolve())
    output.mkdir(parents=True, exist_ok=False)

    targets = set(args.target)
    strategies = set(args.strategy)
    gravities = set(args.gravity)
    roles = set(args.role)

    candidate_rows, candidate_stats, allowed_runs = filter_csv(
        source / "candidate_pose_examples.csv",
        output / "candidate_pose_examples.csv",
        lambda row: keep_candidate_pose(row, targets, strategies, gravities, roles, args),
    )
    write_jsonl(output / "candidate_pose_examples.jsonl", candidate_rows)

    placement_rows, placement_stats, placement_runs = filter_csv(
        source / "placement_examples.csv",
        output / "placement_examples.csv",
        lambda row: keep_placement(row, targets, strategies, gravities, roles, allowed_runs, args),
    )
    write_jsonl(output / "placement_examples.jsonl", placement_rows)
    allowed_runs.update(placement_runs)

    assignment_rows, assignment_stats, assignment_runs = filter_csv(
        source / "assignment_candidate_examples.csv",
        output / "assignment_candidate_examples.csv",
        lambda row: keep_assignment(row, targets, roles, allowed_runs, args),
    )
    allowed_runs.update(assignment_runs)

    run_rows, run_stats, _ = filter_csv(
        source / "run_examples.csv",
        output / "run_examples.csv",
        lambda row: keep_run(row, targets, strategies, gravities, allowed_runs),
    )

    summary = build_summary(
        source=source,
        output=output,
        args=args,
        run_rows=run_rows,
        placement_rows=placement_rows,
        candidate_rows=candidate_rows,
        assignment_rows=assignment_rows,
        stats={
            "candidate_pose": candidate_stats,
            "placement": placement_stats,
            "assignment": assignment_stats,
            "run": run_stats,
        },
    )
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_readme(output, summary)
    print(output)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean MoonStack policy-replacement training data.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", action="append", default=None)
    parser.add_argument("--strategy", action="append", default=None)
    parser.add_argument("--gravity", action="append", default=None)
    parser.add_argument("--role", action="append", default=None)
    parser.add_argument("--max-target-error-m", type=float, default=1.25)
    parser.add_argument("--max-abs-y-error-m", type=float, default=0.75)
    parser.add_argument("--max-disturbance-m", type=float, default=1.25)
    parser.add_argument("--max-velocity", type=float, default=80.0)
    parser.add_argument("--min-rock-volume", type=float, default=1e-7)
    parser.add_argument("--drop-skipped-placement", action="store_true")
    args = parser.parse_args()
    if args.target is None:
        args.target = sorted(DEFAULT_TARGETS)
    if args.strategy is None:
        args.strategy = sorted(DEFAULT_STRATEGIES)
    if args.gravity is None:
        args.gravity = sorted(DEFAULT_GRAVITIES)
    if args.role is None:
        args.role = sorted(DEFAULT_ROLES)
    return args


def filter_csv(
    source: Path,
    target: Path,
    keep: Callable[[dict[str, str]], tuple[bool, str]],
) -> tuple[list[dict[str, str]], dict[str, Any], set[str]]:
    rows_out: list[dict[str, str]] = []
    allowed_runs: set[str] = set()
    reasons: Counter[str] = Counter()
    total = 0
    fieldnames: list[str] = []
    if not source.exists() or source.stat().st_size == 0:
        target.write_text("", encoding="utf-8")
        return [], {"source": str(source), "total": 0, "kept": 0, "rejected": 0, "reasons": {}}, set()
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            total += 1
            ok, reason = keep(row)
            if ok:
                rows_out.append(row)
                if row.get("run_name"):
                    allowed_runs.add(str(row["run_name"]))
            else:
                reasons[reason] += 1
    write_csv(target, rows_out, fieldnames)
    return rows_out, {
        "source": str(source),
        "total": total,
        "kept": len(rows_out),
        "rejected": total - len(rows_out),
        "reasons": dict(sorted(reasons.items())),
    }, allowed_runs


def keep_candidate_pose(
    row: dict[str, str],
    targets: set[str],
    strategies: set[str],
    gravities: set[str],
    roles: set[str],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    common_ok, reason = keep_common(row, targets, roles, args)
    if not common_ok:
        return False, reason
    if row.get("strategy", "") not in strategies:
        return False, "strategy_not_selected"
    if row.get("gravity", "") not in gravities:
        return False, "gravity_not_selected"
    if not has_required(row, REQUIRED_CANDIDATE_POSE):
        return False, "missing_candidate_pose_metric"
    if not finite_limits_ok(row, args):
        return False, "candidate_pose_metric_outlier"
    if row.get("candidate_rock_index", "") in {"", "-1"}:
        return False, "missing_candidate_rock"
    if row.get("candidate_id", "") == "":
        return False, "missing_candidate_id"
    return True, "kept"


def keep_placement(
    row: dict[str, str],
    targets: set[str],
    strategies: set[str],
    gravities: set[str],
    roles: set[str],
    allowed_runs: set[str],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    common_ok, reason = keep_common(row, targets, roles, args)
    if not common_ok:
        return False, reason
    if row.get("strategy", "") not in strategies:
        return False, "strategy_not_selected"
    if row.get("gravity", "") not in gravities:
        return False, "gravity_not_selected"
    if allowed_runs and row.get("run_name", "") not in allowed_runs:
        return False, "run_without_clean_candidate_pose"
    if args.drop_skipped_placement and as_bool(row.get("is_skipped_slot", "")):
        return False, "skipped_placement"
    if not as_bool(row.get("is_skipped_slot", "")):
        if not has_required(row, REQUIRED_PLACEMENT):
            return False, "missing_placement_metric"
        if not finite_limits_ok(row, args):
            return False, "placement_metric_outlier"
    return True, "kept"


def keep_assignment(
    row: dict[str, str],
    targets: set[str],
    roles: set[str],
    allowed_runs: set[str],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    common_ok, reason = keep_common(row, targets, roles, args)
    if not common_ok:
        return False, reason
    if allowed_runs and row.get("run_name", "") not in allowed_runs:
        return False, "run_without_clean_candidate_pose"
    if row.get("candidate_rock_index", "") in {"", "-1"}:
        return False, "missing_candidate_rock"
    return True, "kept"


def keep_run(
    row: dict[str, str],
    targets: set[str],
    strategies: set[str],
    gravities: set[str],
    allowed_runs: set[str],
) -> tuple[bool, str]:
    if allowed_runs and row.get("run_name", "") not in allowed_runs:
        return False, "run_without_clean_rows"
    if row.get("target_name", "") not in targets:
        return False, "target_not_selected"
    if row.get("strategy", "") not in strategies:
        return False, "strategy_not_selected"
    if row.get("gravity", "") not in gravities:
        return False, "gravity_not_selected"
    return True, "kept"


def keep_common(
    row: dict[str, str],
    targets: set[str],
    roles: set[str],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if row.get("target_name", "") not in targets:
        return False, "target_not_selected"
    if row.get("role", "") not in roles:
        return False, "role_not_selected"
    if not has_required(row, REQUIRED_GEOMETRY):
        return False, "missing_geometry"
    volume = parse_float(row.get("rock_volume", ""))
    if volume is None or volume < args.min_rock_volume:
        return False, "invalid_rock_volume"
    return True, "kept"


def has_required(row: dict[str, str], columns: list[str]) -> bool:
    return all(is_finite(row.get(column, "")) for column in columns)


def finite_limits_ok(row: dict[str, str], args: argparse.Namespace) -> bool:
    target_error = parse_float(row.get("target_error_xy_m", ""))
    y_error = parse_float(row.get("target_y_error_m", ""))
    disturbance = parse_float(row.get("placed_disturbance_xy_m", ""))
    velocity = parse_float(row.get("velocity_inf_norm_after_place", ""))
    if target_error is not None and abs(target_error) > args.max_target_error_m:
        return False
    if y_error is not None and abs(y_error) > args.max_abs_y_error_m:
        return False
    if disturbance is not None and abs(disturbance) > args.max_disturbance_m:
        return False
    if velocity is not None and abs(velocity) > args.max_velocity:
        return False
    return True


def build_summary(
    source: Path,
    output: Path,
    args: argparse.Namespace,
    run_rows: list[dict[str, str]],
    placement_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    assignment_rows: list[dict[str, str]],
    stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": str(source),
        "output_dir": str(output),
        "filters": {
            "targets": args.target,
            "strategies": args.strategy,
            "gravities": args.gravity,
            "roles": args.role,
            "max_target_error_m": args.max_target_error_m,
            "max_abs_y_error_m": args.max_abs_y_error_m,
            "max_disturbance_m": args.max_disturbance_m,
            "max_velocity": args.max_velocity,
            "min_rock_volume": args.min_rock_volume,
            "drop_skipped_placement": bool(args.drop_skipped_placement),
        },
        "run_example_count": len(run_rows),
        "placement_example_count": len(placement_rows),
        "candidate_pose_example_count": len(candidate_rows),
        "assignment_candidate_example_count": len(assignment_rows),
        "placement_by_target_role_gravity": summarize_rows(placement_rows, ["target_name", "role", "gravity"]),
        "candidate_pose_by_target_role_gravity": summarize_candidate_rows(candidate_rows),
        "assignment_by_target_role": summarize_rows(assignment_rows, ["target_name", "role"]),
        "file_stats": stats,
    }


def summarize_rows(rows: list[dict[str, str]], keys: list[str]) -> dict[str, dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key, "") for key in keys)].append(row)
    output: dict[str, dict[str, Any]] = {}
    for key, items in sorted(buckets.items()):
        skipped = sum(as_bool(row.get("is_skipped_slot", "")) for row in items)
        success = sum(as_bool(row.get("label_success", "")) for row in items)
        failure = sum(as_bool(row.get("is_failure_case", "")) for row in items)
        label_selected = sum(parse_float(row.get("selected_count_in_placement_log", "")) and 1 or 0 for row in items)
        output["|".join(key)] = {
            "rows": len(items),
            "success": int(success),
            "failure": int(failure),
            "skipped": int(skipped),
            "selected_candidates": int(label_selected),
        }
    return output


def summarize_candidate_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("target_name", ""), row.get("role", ""), row.get("gravity", ""))].append(row)
    output: dict[str, dict[str, Any]] = {}
    for key, items in sorted(buckets.items()):
        selected = sum(as_bool(row.get("label_selected_by_pose_search", "")) for row in items)
        committed_success = sum(as_bool(row.get("label_committed_success", "")) for row in items)
        risky = sum(candidate_is_risky(row) for row in items)
        output["|".join(key)] = {
            "rows": len(items),
            "selected_by_pose_search": int(selected),
            "committed_success": int(committed_success),
            "risk_positive": int(risky),
            "risk_positive_rate": risky / max(len(items), 1),
        }
    return output


def candidate_is_risky(row: dict[str, str]) -> int:
    target_error = parse_float(row.get("target_error_xy_m", "")) or 0.0
    y_error = abs(parse_float(row.get("target_y_error_m", "")) or 0.0)
    disturbance = parse_float(row.get("placed_disturbance_xy_m", "")) or 0.0
    velocity = parse_float(row.get("velocity_inf_norm_after_place", "")) or 0.0
    return int(target_error > 0.16 or y_error > 0.075 or disturbance > 0.080 or velocity > 0.22)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_readme(output: Path, summary: dict[str, Any]) -> None:
    filters = summary.get("filters", {})
    targets = ", ".join(f"`{item}`" for item in filters.get("targets", []))
    strategies = ", ".join(f"`{item}`" for item in filters.get("strategies", []))
    gravities = ", ".join(f"`{item}`" for item in filters.get("gravities", []))
    roles = ", ".join(f"`{item}`" for item in filters.get("roles", []))
    lines = [
        "# Cleaned Policy Replacement Dataset",
        "",
        "这是用于 3/4 层单面墙神经策略替代的清洗数据集。原始数据没有删除，本目录只保存筛选后的训练数据。",
        "",
        f"- source: `{summary['source_dataset']}`",
        f"- run_examples: `{summary['run_example_count']}`",
        f"- placement_examples: `{summary['placement_example_count']}`",
        f"- candidate_pose_examples: `{summary['candidate_pose_example_count']}`",
        f"- assignment_candidate_examples: `{summary['assignment_candidate_example_count']}`",
        "",
        "## 清洗规则",
        "",
        f"- targets: {targets}",
        f"- strategies: {strategies}",
        f"- gravities: {gravities}",
        f"- roles: {roles}",
        f"- max_target_error_m: `{filters.get('max_target_error_m')}`",
        f"- max_abs_y_error_m: `{filters.get('max_abs_y_error_m')}`",
        f"- max_disturbance_m: `{filters.get('max_disturbance_m')}`",
        f"- max_velocity: `{filters.get('max_velocity')}`",
        f"- min_rock_volume: `{filters.get('min_rock_volume')}`",
        f"- drop_skipped_placement: `{filters.get('drop_skipped_placement')}`",
        "- 候选位姿必须有完整 pose、几何特征和物理后验指标。",
        "- 明显异常的候选会被隔离：过大落点误差、过大 y 偏差、过大扰动或速度。",
        "- strict 失败、失败 placement、skipped slot 会保留为负样本，除非显式设置 `--drop-skipped-placement`。",
        "",
        "## 用途",
        "",
        "- `candidate_pose_examples.csv`: 训练 PoseRiskNet / SupportMapRanker / WallStateCritic。",
        "- `assignment_candidate_examples.csv`: 训练 StoneSlotNet 粗筛。",
        "- `placement_examples.csv`: 统计成功/失败经验和结构层级瓶颈。",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return
    lines = [
        "# Cleaned Policy Replacement Dataset",
        "",
        "这是用于 3/4 层单面墙神经策略替代的清洗数据集。原始数据没有删除，本目录只保存筛选后的训练数据。",
        "",
        f"- source: `{summary['source_dataset']}`",
        f"- run_examples: `{summary['run_example_count']}`",
        f"- placement_examples: `{summary['placement_example_count']}`",
        f"- candidate_pose_examples: `{summary['candidate_pose_example_count']}`",
        f"- assignment_candidate_examples: `{summary['assignment_candidate_example_count']}`",
        "",
        "## 清洗规则",
        "",
        "- 只保留 `single_face_wall_3course_v1` 和 `single_face_wall_4course_v1`。",
        "- 只保留 `statics_wall` 策略。",
        "- 保留 `earth` 与 `moon`，让网络继续学习重力差异。",
        "- 只保留 `base/middle/cap` 角色。",
        "- 候选位姿必须有完整 pose、几何特征和物理后验指标。",
        "- 删除明显异常的候选：过大落点误差、过大 y 偏差、过大扰动或速度。",
        "- 旧烟测、`single_face_wall_v1`、`wall_bonded` 和缺 candidate pose 的 run 暂不进入主训练。",
        "",
        "## 用途",
        "",
        "- `candidate_pose_examples.csv`: 训练 PoseRiskNet / SupportMapRanker / WallStateCritic。",
        "- `assignment_candidate_examples.csv`: 训练 StoneSlotNet 粗筛。",
        "- `placement_examples.csv`: 统计成功/失败经验和结构层级瓶颈。",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_float(value: str | None) -> float | None:
    if value in {"", None}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def is_finite(value: str | None) -> bool:
    return parse_float(value) is not None


def as_bool(value: str | None) -> bool:
    if value in {"", None}:
        return False
    try:
        return bool(int(float(value)))
    except ValueError:
        return str(value).strip().lower() in {"true", "yes"}


def unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.name}_{stamp}")
    suffix = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{stamp}_{suffix:02d}")
        suffix += 1
    return candidate


if __name__ == "__main__":
    raise SystemExit(main())
