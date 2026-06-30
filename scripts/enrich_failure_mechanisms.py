from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


FILES = [
    "run_examples.csv",
    "placement_examples.csv",
    "candidate_pose_examples.csv",
    "assignment_candidate_examples.csv",
]

SOURCE_PLACEMENT_COLUMNS = [
    "same_course_placed_count",
    "left_neighbor_present",
    "right_neighbor_present",
    "neighbor_gap_left_m",
    "neighbor_gap_right_m",
    "neighbor_gap_max_positive_m",
    "course_height_std_after_m",
    "course_y_std_after_m",
    "course_y_abs_max_after_m",
    "direct_support_count_course_below",
    "support_load_path_count",
    "support_span_x_m",
    "support_span_cover_ratio",
    "support_underhang_left_m",
    "support_underhang_right_m",
    "support_underhang_max_m",
    "candidate_probe_steps",
    "candidate_probe_hard_gate",
    "candidate_probe_rock_drift_m",
    "candidate_probe_placed_disturbance_m",
    "candidate_probe_speed",
    "low_release_failed",
    "low_release_search",
    "release_contact_clearance_m",
    "release_contact_z",
    "release_drop_reduction_m",
    "release_original_z",
    "release_search_checks",
    "release_z",
    "skip_reason",
    "stone_fit_prob",
    "stone_fit_rank",
    "stone_fit_hybrid_score",
    "stone_fit_top_k",
    "base_support_prior_score",
    "base_continuity_prior_score",
    "experience_prior_score",
]

SOURCE_CANDIDATE_COLUMNS = [
    "base_candidate_score",
    "ranker_prob",
    "ranker_rank",
    "ranker_top_k",
    "pose_risk_prob",
    "pose_risk_weight",
    "pose_risk_penalty",
    "pose_rank_score",
    "low_release_search",
    "low_release_failed",
    "release_original_z",
    "release_z",
    "release_drop_reduction_m",
    "release_search_checks",
    "release_contact_z",
    "release_contact_clearance_m",
    "gravity_m_s2",
    "stone_pool_size",
    "stone_pool_rock_index",
    "stone_fit_prob",
    "stone_fit_rank",
    "stone_fit_hybrid_score",
    "stone_fit_top_k",
    "base_support_prior_score",
    "base_continuity_prior_score",
    "experience_prior_score",
    "same_course_placed_count",
    "left_neighbor_present",
    "right_neighbor_present",
    "neighbor_gap_left_m",
    "neighbor_gap_right_m",
    "neighbor_gap_max_positive_m",
    "course_height_std_after_m",
    "course_y_std_after_m",
    "course_y_abs_max_after_m",
    "direct_support_count_course_below",
    "support_load_path_count",
    "support_span_x_m",
    "support_span_cover_ratio",
    "support_underhang_left_m",
    "support_underhang_right_m",
    "support_underhang_max_m",
    "candidate_probe_steps",
    "candidate_probe_hard_gate",
    "candidate_probe_rock_drift_m",
    "candidate_probe_placed_disturbance_m",
    "candidate_probe_speed",
]

MECHANISM_COLUMNS = [
    "mechanism_no_feasible_pose",
    "mechanism_bottom_support_insufficient",
    "mechanism_upper_contact_too_few",
    "mechanism_release_disturbance_excessive",
    "mechanism_geometry_mismatch",
    "mechanism_neighbor_gap_too_large",
    "mechanism_target_miss",
    "mechanism_post_hold_drift",
    "mechanism_low_or_fallen",
    "failure_mechanism_primary",
    "failure_mechanism_count",
    "mechanism_rule_version",
]

RULE_VERSION = "20260630_v1"


def main() -> int:
    args = parse_args()
    batch_root = args.batch_root.resolve()
    output = unique_dir(args.output.resolve())
    output.mkdir(parents=True, exist_ok=False)

    datasets = discover_datasets(args, batch_root)
    if not datasets:
        raise SystemExit("No learning dataset directories found.")

    placement_cache: dict[Path, dict[tuple[str, ...], dict[str, str]]] = {}
    candidate_cache: dict[Path, dict[tuple[str, ...], dict[str, str]]] = {}

    merged: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in FILES}
    dataset_stats: list[dict[str, Any]] = []
    for dataset in datasets:
        stat: dict[str, Any] = {"dataset": str(dataset), "files": {}}
        for filename in FILES:
            path = dataset / filename
            rows = read_csv(path)
            stat["files"][filename] = len(rows)
            for row in rows:
                enriched = dict(row)
                enriched.setdefault("source_dataset", dataset.name)
                if filename == "placement_examples.csv":
                    enrich_from_source_placement(enriched, placement_cache)
                    add_mechanisms(enriched, table="placement")
                elif filename == "candidate_pose_examples.csv":
                    enrich_from_source_candidate(enriched, candidate_cache)
                    add_mechanisms(enriched, table="candidate_pose")
                key = row_key(filename, enriched)
                if key in merged[filename]:
                    merge_non_empty(merged[filename][key], enriched)
                else:
                    merged[filename][key] = enriched
        dataset_stats.append(stat)

    outputs: dict[str, list[dict[str, Any]]] = {
        filename: list(rows_by_key.values()) for filename, rows_by_key in merged.items()
    }
    for filename in FILES:
        write_csv(output / filename, outputs[filename])
    write_jsonl(output / "placement_examples.jsonl", outputs["placement_examples.csv"])
    write_jsonl(output / "candidate_pose_examples.jsonl", outputs["candidate_pose_examples.csv"])

    summary = build_summary(output, datasets, dataset_stats, outputs)
    write_json(output / "dataset_summary.json", summary)
    write_readme(output, summary)
    print(output)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge learning datasets and add post-hoc failure mechanism labels without modifying source data."
    )
    parser.add_argument("--batch-root", type=Path, default=Path("batch_runs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", action="append", type=Path, default=[])
    parser.add_argument("--dataset-glob", default="*learning_dataset*")
    parser.add_argument("--exclude-name-contains", action="append", default=["nasa", "NASA"])
    parser.add_argument("--target-contains", action="append", default=[])
    return parser.parse_args()


def discover_datasets(args: argparse.Namespace, batch_root: Path) -> list[Path]:
    if args.dataset:
        candidates = [path.resolve() for path in args.dataset]
    else:
        candidates = sorted(path.resolve() for path in batch_root.glob(args.dataset_glob) if path.is_dir())
    output: list[Path] = []
    for path in candidates:
        name = path.name
        if any(token and token in name for token in args.exclude_name_contains):
            continue
        if args.target_contains and not dataset_mentions_target(path, args.target_contains):
            continue
        if any((path / filename).exists() for filename in FILES):
            output.append(path)
    return output


def dataset_mentions_target(path: Path, tokens: list[str]) -> bool:
    for filename in ("run_examples.csv", "placement_examples.csv", "candidate_pose_examples.csv"):
        rows = read_csv(path / filename)
        for row in rows[:2000]:
            if any(token in row.get("target_name", "") for token in tokens):
                return True
    return False


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def row_key(filename: str, row: dict[str, Any]) -> str:
    example_id = str(row.get("example_id", "")).strip()
    if example_id:
        return example_id
    if filename == "run_examples.csv":
        parts = ["run_name", "target_name", "strategy", "gravity", "trial"]
    elif filename == "placement_examples.csv":
        parts = ["run_name", "target_name", "strategy", "gravity", "trial", "slot_id", "candidate_rock_index"]
    elif filename == "candidate_pose_examples.csv":
        parts = [
            "run_name",
            "target_name",
            "strategy",
            "gravity",
            "trial",
            "slot_id",
            "candidate_rock_index",
            "candidate_id",
        ]
    else:
        parts = ["run_name", "target_name", "slot_id", "candidate_rock_index", "candidate_rank"]
    return "|".join(str(row.get(part, "")) for part in parts)


def merge_non_empty(base: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if value in {"", None}:
            continue
        if base.get(key, "") in {"", None}:
            base[key] = value


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


def source_run_path(row: dict[str, Any]) -> Path | None:
    raw = str(row.get("run_path", "")).strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None


def enrich_from_source_placement(row: dict[str, Any], cache: dict[Path, dict[tuple[str, ...], dict[str, str]]]) -> None:
    run_path = source_run_path(row)
    if run_path is None:
        return
    if run_path not in cache:
        cache[run_path] = index_source_rows(run_path / "placement_log.csv", placement_source_keys)
    source = find_source(cache[run_path], placement_lookup_keys(row))
    if not source:
        return
    for column in SOURCE_PLACEMENT_COLUMNS:
        if row.get(column, "") in {"", None} and source.get(column, "") not in {"", None}:
            row[column] = source[column]


def enrich_from_source_candidate(row: dict[str, Any], cache: dict[Path, dict[tuple[str, ...], dict[str, str]]]) -> None:
    run_path = source_run_path(row)
    if run_path is None:
        return
    if run_path not in cache:
        cache[run_path] = index_source_rows(run_path / "candidate_pose_log.csv", candidate_source_keys)
    source = find_source(cache[run_path], candidate_lookup_keys(row))
    if not source:
        return
    for column in SOURCE_CANDIDATE_COLUMNS:
        if row.get(column, "") in {"", None} and source.get(column, "") not in {"", None}:
            row[column] = source[column]


def index_source_rows(path: Path, key_fn: Any) -> dict[tuple[str, ...], dict[str, str]]:
    output: dict[tuple[str, ...], dict[str, str]] = {}
    for source in read_csv(path):
        for key in key_fn(source):
            if key not in output:
                output[key] = source
    return output


def find_source(index: dict[tuple[str, ...], dict[str, str]], keys: list[tuple[str, ...]]) -> dict[str, str] | None:
    for key in keys:
        if key in index:
            return index[key]
    return None


def placement_source_keys(row: dict[str, str]) -> list[tuple[str, ...]]:
    return placement_lookup_keys(
        {
            "target_name": row.get("target_name", ""),
            "strategy": row.get("strategy", ""),
            "gravity": row.get("gravity", ""),
            "trial": row.get("trial", ""),
            "slot_id": row.get("slot_id", ""),
            "candidate_rock_index": row.get("rock_index", ""),
            "committed_rock_index": row.get("rock_index", ""),
            "candidate_id": row.get("candidate_id", ""),
        }
    )


def placement_lookup_keys(row: dict[str, Any]) -> list[tuple[str, ...]]:
    base = (
        str(row.get("target_name", "")),
        str(row.get("strategy", "")),
        str(row.get("gravity", "")),
        str(row.get("trial", "")),
        str(row.get("slot_id", "")),
    )
    rock_ids = [
        str(row.get("committed_rock_index", "")),
        str(row.get("candidate_rock_index", "")),
        str(row.get("rock_index", "")),
    ]
    candidate_id = str(row.get("candidate_id", ""))
    keys: list[tuple[str, ...]] = []
    for rock_id in rock_ids:
        if rock_id:
            keys.append(base + (rock_id, candidate_id))
            keys.append(base + (rock_id, ""))
    keys.append(base + ("", candidate_id))
    return keys


def candidate_source_keys(row: dict[str, str]) -> list[tuple[str, ...]]:
    return candidate_lookup_keys(
        {
            "target_name": row.get("target_name", ""),
            "strategy": row.get("strategy", ""),
            "gravity": row.get("gravity", ""),
            "trial": row.get("trial", ""),
            "slot_id": row.get("slot_id", ""),
            "candidate_rock_index": row.get("rock_index", ""),
            "candidate_id": row.get("candidate_id", ""),
        }
    )


def candidate_lookup_keys(row: dict[str, Any]) -> list[tuple[str, ...]]:
    return [
        (
            str(row.get("target_name", "")),
            str(row.get("strategy", "")),
            str(row.get("gravity", "")),
            str(row.get("trial", "")),
            str(row.get("slot_id", "")),
            str(row.get("candidate_rock_index", "")),
            str(row.get("candidate_id", "")),
        )
    ]


def add_mechanisms(row: dict[str, Any], table: str) -> None:
    failed = row_failed(row, table)
    reason = str(row.get("failure_reason", "")).strip()
    course = parse_float(row.get("course", ""))
    support_overlap = parse_float(row.get("support_overlap", ""))
    support_contacts = parse_float(row.get("support_contact_count", ""))
    direct_support = parse_float(row.get("direct_support_count_course_below", ""))
    support_cover = parse_float(row.get("support_span_cover_ratio", ""))
    underhang = abs(parse_float(row.get("support_underhang_max_m", "")))
    neighbor_gap = parse_float(row.get("neighbor_gap_max_positive_m", ""))
    disturbance = max(
        parse_float(row.get("placed_disturbance_xy_m", "")),
        parse_float(row.get("candidate_probe_placed_disturbance_m", "")),
        parse_float(row.get("candidate_probe_rock_drift_m", "")),
    )
    velocity = max(parse_float(row.get("velocity_inf_norm_after_place", "")), parse_float(row.get("candidate_probe_speed", "")))
    target_error = parse_float(row.get("target_error_xy_m", ""))
    target_y_error = abs(parse_float(row.get("target_y_error_m", "")))
    rock_spike = parse_float(row.get("rock_spike_score", ""))
    rock_elongation = parse_float(row.get("rock_elongation", ""))
    rock_flatness = parse_float(row.get("rock_flatness", ""))
    support_faces = parse_float(row.get("rock_support_face_count", ""))
    support_face_ratio = parse_float(row.get("rock_support_face_area_ratio", ""))
    support_plane = parse_float(row.get("rock_support_plane_quality", ""))
    low_release_failed = parse_bool(row.get("low_release_failed", ""))

    mechanism = {
        "mechanism_no_feasible_pose": failed and (
            "no_feasible" in reason
            or "assignment_gate_no_feasible" in reason
            or str(row.get("skip_reason", "")).strip() in {"no_feasible_pose", "assignment_gate_no_feasible_fallback"}
        ),
        "mechanism_bottom_support_insufficient": failed
        and (
            (course <= 0 and (support_faces < 1 or support_face_ratio < 0.04 or support_plane < 0.04))
            or (course > 0 and (support_overlap < 0.55 or support_cover < 0.50 or underhang > 0.035))
            or reason in {"unstable_structure", "low_or_fallen"}
        ),
        "mechanism_upper_contact_too_few": failed
        and course > 0
        and (support_contacts < 2 or direct_support < 2 or support_overlap < 0.60),
        "mechanism_release_disturbance_excessive": failed
        and (low_release_failed or disturbance > 0.06 or velocity > 0.22 or "post_hold_drift" in reason),
        "mechanism_geometry_mismatch": failed
        and (
            "no_feasible" in reason
            or rock_spike > 0.12
            or rock_elongation > 1.85
            or rock_flatness > 1.65
            or support_faces < 1
            or support_plane < 0.05
        ),
        "mechanism_neighbor_gap_too_large": failed and course > 0 and neighbor_gap > 0.045,
        "mechanism_target_miss": failed and ("missed_target" in reason or target_error > 0.16 or target_y_error > 0.075),
        "mechanism_post_hold_drift": failed and ("post_hold_drift" in reason or velocity > 0.25),
        "mechanism_low_or_fallen": failed and ("low_or_fallen" in reason),
    }
    for key, value in mechanism.items():
        row[key] = int(bool(value))
    row["failure_mechanism_count"] = int(sum(int(bool(value)) for value in mechanism.values()))
    row["failure_mechanism_primary"] = primary_mechanism(row, failed)
    row["mechanism_rule_version"] = RULE_VERSION


def row_failed(row: dict[str, Any], table: str) -> bool:
    if table == "placement":
        return parse_bool(row.get("label_success", "")) is False
    if table == "candidate_pose":
        metric_bad = (
            parse_float(row.get("target_error_xy_m", "")) > 0.16
            or abs(parse_float(row.get("target_y_error_m", ""))) > 0.075
            or parse_float(row.get("placed_disturbance_xy_m", "")) > 0.08
            or parse_float(row.get("velocity_inf_norm_after_place", "")) > 0.22
            or parse_float(row.get("candidate_probe_speed", "")) > 0.22
        )
        return (parse_bool(row.get("label_selected_by_pose_search", "")) and not parse_bool(row.get("label_committed_success", ""))) or metric_bad
    return False


def primary_mechanism(row: dict[str, Any], failed: bool) -> str:
    if not failed:
        return "success_or_safe"
    order = [
        ("mechanism_no_feasible_pose", "no_feasible_pose"),
        ("mechanism_upper_contact_too_few", "upper_contact_too_few"),
        ("mechanism_bottom_support_insufficient", "bottom_support_insufficient"),
        ("mechanism_neighbor_gap_too_large", "neighbor_gap_too_large"),
        ("mechanism_release_disturbance_excessive", "release_disturbance_excessive"),
        ("mechanism_target_miss", "target_miss"),
        ("mechanism_geometry_mismatch", "geometry_mismatch"),
        ("mechanism_post_hold_drift", "post_hold_drift"),
        ("mechanism_low_or_fallen", "low_or_fallen"),
    ]
    for column, name in order:
        if parse_bool(row.get(column, "")):
            return name
    if parse_bool(row.get("is_skipped_slot", "")):
        return "skipped_slot"
    return "unknown_failure"


def parse_float(raw: Any) -> float:
    try:
        if raw in {"", None}:
            return 0.0
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def parse_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    return False


def build_summary(
    output: Path,
    datasets: list[Path],
    dataset_stats: list[dict[str, Any]],
    outputs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    placement = outputs["placement_examples.csv"]
    candidate = outputs["candidate_pose_examples.csv"]
    mechanism_counts = Counter(str(row.get("failure_mechanism_primary", "")) for row in placement)
    candidate_mechanism_counts = Counter(str(row.get("failure_mechanism_primary", "")) for row in candidate)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output),
        "input_dataset_count": len(datasets),
        "input_datasets": [str(path) for path in datasets],
        "input_dataset_stats": dataset_stats,
        "row_counts": {filename: len(rows) for filename, rows in outputs.items()},
        "placement_label_success": {
            "success": sum(1 for row in placement if parse_bool(row.get("label_success", ""))),
            "negative": sum(1 for row in placement if not parse_bool(row.get("label_success", ""))),
        },
        "candidate_pose_labels": {
            "selected": sum(1 for row in candidate if parse_bool(row.get("label_selected_by_pose_search", ""))),
            "committed_success": sum(1 for row in candidate if parse_bool(row.get("label_committed_success", ""))),
        },
        "placement_failure_mechanism_primary": dict(mechanism_counts),
        "candidate_failure_mechanism_primary": dict(candidate_mechanism_counts),
        "mechanism_rule_version": RULE_VERSION,
        "field_policy": {
            "pre_action_inputs": [
                "target_name",
                "strategy",
                "gravity",
                "course",
                "role",
                "target_x",
                "target_y",
                "pose_x",
                "pose_y",
                "pose_z",
                "pose_quaternion",
                "rock geometry features",
                "rock source kind and cluster",
                "top/front/depth/support-map observations when exported separately",
            ],
            "post_action_supervision_only": [
                "settled pose",
                "target error",
                "support/contact metrics after probing",
                "disturbance/velocity",
                "failure_reason",
                "mechanism_* labels",
                "label_success",
                "label_committed_success",
            ],
        },
        "known_limitations": [
            "Mechanism labels are rule-derived weak labels, not ground-truth human annotations.",
            "Old datasets may not contain dense candidate-pose logs, so candidate-level mechanisms are more complete for recent runs.",
            "Mechanism labels must be used as supervision or diagnostics, not as action-network inputs at inference time.",
        ],
    }


def write_readme(output: Path, summary: dict[str, Any]) -> None:
    rows = summary["row_counts"]
    mechanisms = summary["placement_failure_mechanism_primary"]
    lines = [
        "# Enriched Failure Mechanism Dataset",
        "",
        "目的：合并旧学习数据集，去重后追加更细的失败机制弱标签，供中间网络训练、经验分析和未来端到端模型的辅助监督使用。",
        "",
        "## Row Counts",
        "",
        f"- run_examples: `{rows.get('run_examples.csv', 0)}`",
        f"- placement_examples: `{rows.get('placement_examples.csv', 0)}`",
        f"- candidate_pose_examples: `{rows.get('candidate_pose_examples.csv', 0)}`",
        f"- assignment_candidate_examples: `{rows.get('assignment_candidate_examples.csv', 0)}`",
        "",
        "## Failure Mechanism Labels",
        "",
        "- `mechanism_bottom_support_insufficient`: 底层或下层支撑不足。",
        "- `mechanism_upper_contact_too_few`: 上层接触点/直接支撑数量不足。",
        "- `mechanism_release_disturbance_excessive`: 释放或探测后扰动速度过大。",
        "- `mechanism_geometry_mismatch`: 石头几何与槽位/姿态不匹配。",
        "- `mechanism_neighbor_gap_too_large`: 邻接缝隙或侧向连续性不足。",
        "- `mechanism_target_miss`: 落点偏离目标。",
        "- `mechanism_no_feasible_pose`: 未找到可行姿态。",
        "- `mechanism_post_hold_drift`: hold 后发生漂移。",
        "- `mechanism_low_or_fallen`: 低矮或掉落。",
        "",
        "## Primary Mechanism Counts",
        "",
    ]
    for key, count in sorted(mechanisms.items(), key=lambda item: (-int(item[1]), str(item[0]))):
        lines.append(f"- `{key}`: `{count}`")
    lines.extend(
        [
            "",
            "## Data Policy",
            "",
            "机制标签是后验弱标签，只能作为监督目标或诊断统计，不能作为动作网络推理输入。",
            "动作输入应限制为已知几何、目标槽位、候选位姿、重力和实际观测图。",
            "",
            f"rule version: `{summary['mechanism_rule_version']}`",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
