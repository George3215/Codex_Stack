from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


def main() -> int:
    args = parse_args()
    batch_root = args.batch_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    result_rows = read_all_results(batch_root, args.target_contains)
    success_rows = [row for row in result_rows if to_int(row.get("success")) == 1]
    shape_rows = [row for row in result_rows if to_int(row.get("shape_success", row.get("success"))) == 1]
    near_rows = [row for row in result_rows if is_near_success(row)]

    placement_rows = read_all_placements(batch_root, args.target_contains)
    success_keys = {case_key(row) for row in success_rows}
    shape_keys = {case_key(row) for row in shape_rows}
    near_keys = {case_key(row) for row in near_rows}

    success_placements = [row for row in placement_rows if placement_case_key(row) in success_keys]
    shape_placements = [row for row in placement_rows if placement_case_key(row) in shape_keys]
    near_placements = [row for row in placement_rows if placement_case_key(row) in near_keys]

    role_rows = summarize_role_experience(success_placements, shape_placements, near_placements)
    skip_rows = summarize_skips(placement_rows)
    case_summary_rows = summarize_cases(result_rows)
    selected_case_rows = summarize_selected_cases(success_rows, shape_rows, near_rows)
    experience_priors = build_experience_priors(result_rows, role_rows, skip_rows)

    write_csv(output / "success_cases.csv", success_rows)
    write_csv(output / "shape_success_cases.csv", shape_rows)
    write_csv(output / "near_success_cases.csv", near_rows)
    write_csv(output / "selected_case_summary.csv", selected_case_rows)
    write_csv(output / "case_summary_by_task.csv", case_summary_rows)
    write_csv(output / "role_experience.csv", role_rows)
    write_csv(output / "skip_by_role.csv", skip_rows)
    write_json(output / "experience_priors.json", experience_priors)
    write_readme(
        output / "README.md",
        batch_root=batch_root,
        output=output,
        result_rows=result_rows,
        success_rows=success_rows,
        shape_rows=shape_rows,
        near_rows=near_rows,
        role_rows=role_rows,
        skip_rows=skip_rows,
        selected_case_rows=selected_case_rows,
    )
    print(output)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract successful wall-stacking experience from previous runs.")
    parser.add_argument("--batch-root", type=Path, default=Path("batch_runs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-contains", default="single_face_wall")
    return parser.parse_args()


def read_all_results(batch_root: Path, target_contains: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(batch_root.rglob("results.csv")):
        if skip_path(path):
            continue
        run_dir = path.parent
        for row in read_csv(path):
            target_name = row.get("target_name", "")
            if target_contains and target_contains not in target_name:
                continue
            rows.append(
                {
                    **row,
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir),
                    "run_group": run_dir.parent.name,
                    "run_mtime": run_dir.stat().st_mtime,
                }
            )
    rows.sort(key=lambda row: (float(row["run_mtime"]), row["run_name"], row.get("target_name", ""), row.get("trial", "")))
    return rows


def read_all_placements(batch_root: Path, target_contains: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(batch_root.rglob("placement_log.csv")):
        if skip_path(path):
            continue
        run_dir = path.parent
        for row in read_csv(path):
            target_name = row.get("target_name", "")
            if target_contains and target_contains not in target_name:
                continue
            rows.append(
                {
                    **row,
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir),
                    "run_group": run_dir.parent.name,
                    "run_mtime": run_dir.stat().st_mtime,
                }
            )
    return rows


def skip_path(path: Path) -> bool:
    ignored = {"analysis", "_aggregate", "progress_reports"}
    return any(part in ignored for part in path.parts)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8-sig")


def case_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("run_dir", "")),
        str(row.get("target_name", "")),
        str(row.get("strategy", "")),
        str(row.get("gravity", "")),
        str(row.get("trial", "")),
    )


def placement_case_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("run_dir", "")),
        str(row.get("target_name", "")),
        str(row.get("strategy", "")),
        str(row.get("gravity", "")),
        str(row.get("trial", "")),
    )


def is_near_success(row: dict[str, Any]) -> bool:
    target = str(row.get("target_name", ""))
    if "4course" not in target:
        return False
    if to_int(row.get("success")) == 1:
        return True
    visible = to_float(row.get("visible_courses"))
    stable = to_float(row.get("stable_count"))
    rocks = max(to_float(row.get("rock_count")), 1.0)
    failure = to_float(row.get("failure_count"))
    skipped = to_float(row.get("skipped_slot_count"))
    drift = to_float(row.get("max_horizontal_drift_m"))
    rmse = to_float(row.get("target_rmse_xy_m"))
    shape = to_int(row.get("shape_success", row.get("success"))) == 1
    return bool(
        shape
        or visible >= 4
        or (stable / rocks >= 0.85 and failure <= 1 and skipped <= 6)
        or (visible >= 3 and drift <= 0.03 and rmse <= 0.08)
    )


def summarize_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("target_name", "")), str(row.get("gravity", "")))].append(row)
    output: list[dict[str, Any]] = []
    for (target, gravity), items in sorted(grouped.items()):
        output.append(
            {
                "target_name": target,
                "gravity": gravity,
                "trials": len(items),
                "strict_success": sum(to_int(row.get("success")) for row in items),
                "shape_success": sum(to_int(row.get("shape_success", row.get("success"))) for row in items),
                "strict_success_rate": safe_mean(to_float(row.get("success")) for row in items),
                "shape_success_rate": safe_mean(to_float(row.get("shape_success", row.get("success"))) for row in items),
                "mean_visible_courses": safe_mean(to_float(row.get("visible_courses")) for row in items),
                "mean_stable_fraction": stable_fraction(items),
                "mean_skipped_slots": safe_mean(to_float(row.get("skipped_slot_count")) for row in items),
                "mean_failure_count": safe_mean(to_float(row.get("failure_count")) for row in items),
                "mean_rmse_xy_m": safe_mean(to_float(row.get("target_rmse_xy_m")) for row in items),
                "mean_drift_m": safe_mean(to_float(row.get("max_horizontal_drift_m")) for row in items),
            }
        )
    return output


def summarize_selected_cases(
    success_rows: list[dict[str, Any]], shape_rows: list[dict[str, Any]], near_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    tagged: list[tuple[str, dict[str, Any]]] = []
    tagged.extend(("strict_success", row) for row in success_rows)
    tagged.extend(("shape_success", row) for row in shape_rows if case_key(row) not in {case_key(item) for item in success_rows})
    tagged.extend(
        ("near_success", row)
        for row in near_rows
        if case_key(row) not in {case_key(item) for item in success_rows} and case_key(row) not in {case_key(item) for item in shape_rows}
    )
    output: list[dict[str, Any]] = []
    for tag, row in tagged:
        output.append(
            {
                "tag": tag,
                "run_name": row.get("run_name", ""),
                "run_dir": row.get("run_dir", ""),
                "target_name": row.get("target_name", ""),
                "gravity": row.get("gravity", ""),
                "trial": row.get("trial", ""),
                "success": row.get("success", ""),
                "shape_success": row.get("shape_success", ""),
                "visible_courses": row.get("visible_courses", ""),
                "rock_count": row.get("rock_count", ""),
                "stable_count": row.get("stable_count", ""),
                "failure_count": row.get("failure_count", ""),
                "skipped_slot_count": row.get("skipped_slot_count", ""),
                "target_rmse_xy_m": row.get("target_rmse_xy_m", ""),
                "max_horizontal_drift_m": row.get("max_horizontal_drift_m", ""),
                "stack_height_m": row.get("stack_height_m", ""),
                "candidate_pose_top_k": row.get("candidate_pose_top_k", ""),
                "stone_fit_top_k": row.get("stone_fit_top_k", ""),
                "pose_risk_weight": row.get("pose_risk_weight", ""),
                "low_release_search_requested": row.get("low_release_search_requested", ""),
                "order": row.get("order", ""),
            }
        )
    output.sort(key=lambda row: (row["tag"], row["target_name"], row["run_name"], str(row["trial"])))
    return output


def summarize_role_experience(
    success_placements: list[dict[str, Any]],
    shape_placements: list[dict[str, Any]],
    near_placements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tagged: list[tuple[str, dict[str, Any]]] = []
    tagged.extend(("strict_success_case", row) for row in success_placements)
    tagged.extend(("shape_success_case", row) for row in shape_placements)
    tagged.extend(("near_success_case", row) for row in near_placements)

    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for tag, row in tagged:
        if is_skipped(row):
            continue
        key = (
            tag,
            str(row.get("target_name", "")),
            str(row.get("gravity", "")),
            str(row.get("role", "")),
            str(row.get("course", "")),
            str(row.get("source_kind", "")),
        )
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for (tag, target, gravity, role, course, source_kind), items in sorted(grouped.items()):
        cluster_counter = Counter(str(row.get("cluster_label", "")) for row in items)
        output.append(
            {
                "tag": tag,
                "target_name": target,
                "gravity": gravity,
                "role": role,
                "course": course,
                "source_kind": source_kind,
                "placements": len(items),
                "top_clusters": ";".join(f"{name}:{count}" for name, count in cluster_counter.most_common(5)),
                "mean_target_error_xy_m": safe_mean(to_float(row.get("target_error_xy_m")) for row in items),
                "mean_support_overlap": safe_mean(to_float(row.get("support_overlap")) for row in items),
                "mean_support_contact_count": safe_mean(to_float(row.get("support_contact_count")) for row in items),
                "mean_direct_support_count": safe_mean(to_float(row.get("direct_support_count_course_below")) for row in items),
                "mean_support_load_path_count": safe_mean(to_float(row.get("support_load_path_count")) for row in items),
                "mean_disturbance_xy_m": safe_mean(to_float(row.get("placed_disturbance_xy_m")) for row in items),
                "mean_speed_after_place": safe_mean(to_float(row.get("velocity_inf_norm_after_place")) for row in items),
                "mean_stone_fit_rank": safe_mean(to_float(row.get("stone_fit_rank")) for row in items),
                "mean_stone_fit_prob": safe_mean(to_float(row.get("stone_fit_prob")) for row in items),
                "mean_release_drop_reduction_m": safe_mean(to_float(row.get("release_drop_reduction_m")) for row in items),
            }
        )
    output.sort(key=lambda row: (row["tag"], row["target_name"], row["role"], int_or_999(row["course"]), -int(row["placements"])))
    return output


def summarize_skips(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target = str(row.get("target_name", ""))
        if "4course" not in target:
            continue
        grouped[(target, str(row.get("gravity", "")), str(row.get("role", "")), str(row.get("course", "")))].append(row)

    output: list[dict[str, Any]] = []
    for (target, gravity, role, course), items in sorted(grouped.items()):
        skipped = [row for row in items if is_skipped(row)]
        reasons = Counter(str(row.get("skip_reason", "")) or "unknown" for row in skipped)
        output.append(
            {
                "target_name": target,
                "gravity": gravity,
                "role": role,
                "course": course,
                "rows": len(items),
                "skipped": len(skipped),
                "skip_rate": len(skipped) / max(len(items), 1),
                "top_skip_reasons": ";".join(f"{name}:{count}" for name, count in reasons.most_common(5)),
                "mean_target_error_non_skipped": safe_mean(
                    to_float(row.get("target_error_xy_m")) for row in items if not is_skipped(row)
                ),
                "mean_disturbance_non_skipped": safe_mean(
                    to_float(row.get("placed_disturbance_xy_m")) for row in items if not is_skipped(row)
                ),
            }
        )
    output.sort(key=lambda row: (-float(row["skip_rate"]), row["target_name"], row["role"], int_or_999(row["course"])))
    return output


def build_experience_priors(
    result_rows: list[dict[str, Any]],
    role_rows: list[dict[str, Any]],
    skip_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    role_priors: dict[str, Any] = {}
    for role in ("base", "middle", "cap"):
        entries = [
            row
            for row in role_rows
            if row.get("role") == role
            and row.get("source_kind")
            and row.get("source_kind") != "skipped"
            and row.get("tag") in {"strict_success_case", "shape_success_case", "near_success_4course"}
        ]
        source_scores: Counter[str] = Counter()
        cluster_scores: Counter[str] = Counter()
        metric_weights: list[float] = []
        target_errors: list[float] = []
        support_overlaps: list[float] = []
        support_contacts: list[float] = []
        disturbances: list[float] = []
        fit_ranks: list[float] = []
        for row in entries:
            placements = max(1, to_int(row.get("placements")))
            tag_multiplier = {
                "strict_success_case": 2.0,
                "shape_success_case": 1.6,
                "near_success_4course": 1.0,
            }.get(str(row.get("tag")), 1.0)
            support_overlap = min(max(to_float(row.get("mean_support_overlap")), 0.0), 1.0)
            support_contact = min(max(to_float(row.get("mean_support_contact_count")), 0.0), 3.0)
            disturbance = min(max(to_float(row.get("mean_disturbance_xy_m")), 0.0), 0.25)
            quality = max(0.35, 1.0 + 0.28 * support_overlap + 0.08 * support_contact - 0.9 * disturbance)
            weight = float(placements) * tag_multiplier * quality

            source_scores[str(row.get("source_kind", ""))] += weight
            for cluster, count in parse_count_pairs(str(row.get("top_clusters", ""))).items():
                cluster_scores[cluster] += tag_multiplier * quality * float(count)

            metric_weights.append(weight)
            target_errors.append(to_float(row.get("mean_target_error_xy_m")))
            support_overlaps.append(to_float(row.get("mean_support_overlap")))
            support_contacts.append(to_float(row.get("mean_support_contact_count")))
            disturbances.append(to_float(row.get("mean_disturbance_xy_m")))
            fit_ranks.append(to_float(row.get("mean_stone_fit_rank")))

        role_priors[role] = {
            "source_kind_weights": top_normalized_scores(source_scores, limit=10),
            "cluster_label_weights": top_normalized_scores(cluster_scores, limit=12),
            "metrics_from_successful_or_near_successful_placements": {
                "weighted_mean_target_error_xy_m": weighted_mean(target_errors, metric_weights),
                "weighted_mean_support_overlap": weighted_mean(support_overlaps, metric_weights),
                "weighted_mean_support_contact_count": weighted_mean(support_contacts, metric_weights),
                "weighted_mean_disturbance_xy_m": weighted_mean(disturbances, metric_weights),
                "weighted_mean_stone_fit_rank": weighted_mean(fit_ranks, metric_weights),
            },
            "training_note": (
                "Use as a weak prior for candidate-pool ranking only. "
                "Do not use these outcome-derived weights as direct neural-network input features."
            ),
        }

    four_rows = [row for row in result_rows if row.get("target_name") == "single_face_wall_4course_v1"]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "single_face_wall historical success and near-success placements",
        "intended_use": [
            "Bias candidate-pool ranking by role-level geometry/source/cluster priors.",
            "Keep MuJoCo feasibility, low-release search, SupportMap/PoseRisk, and final physical settling as gates.",
            "Log prior scores in placement_log.csv and compare success/skip deltas against no-prior baselines.",
        ],
        "sample_counts": {
            "result_rows": len(result_rows),
            "four_course_rows": len(four_rows),
            "strict_success_rows": sum(1 for row in result_rows if to_int(row.get("success")) == 1),
            "shape_success_rows": sum(1 for row in result_rows if to_int(row.get("shape_success", row.get("success"))) == 1),
        },
        "role_priors": role_priors,
        "bottleneck_priors": [
            {
                "target_name": row.get("target_name", ""),
                "gravity": row.get("gravity", ""),
                "role": row.get("role", ""),
                "course": to_int(row.get("course")),
                "skip_rate": to_float(row.get("skip_rate")),
                "skipped": to_int(row.get("skipped")),
                "rows": to_int(row.get("rows")),
                "top_skip_reasons": row.get("top_skip_reasons", ""),
                "mean_target_error_non_skipped": to_float(row.get("mean_target_error_non_skipped")),
                "mean_disturbance_non_skipped": to_float(row.get("mean_disturbance_non_skipped")),
            }
            for row in skip_rows[:8]
        ],
        "next_policy_hypotheses": [
            "For 4-course walls, expand the candidate pool mainly for course-2 middle and course-3 cap slots.",
            "Prefer tie_bridge/interlock/course-like stones in middle slots when support overlap remains high.",
            "Prefer bearing/cap/compact block-like stones in cap slots, but keep PoseRisk and low-release gates active.",
            "Treat no_feasible_pose as the dominant fourth-course bottleneck; reducing skipped slots is more important than further base tuning.",
        ],
    }


def parse_count_pairs(value: str) -> dict[str, int]:
    output: dict[str, int] = {}
    for item in value.split(";"):
        if ":" not in item:
            continue
        name, count = item.rsplit(":", 1)
        name = name.strip()
        if not name:
            continue
        output[name] = output.get(name, 0) + to_int(count)
    return output


def top_normalized_scores(scores: Counter[str], limit: int) -> dict[str, float]:
    if not scores:
        return {}
    top_items = [(name, score) for name, score in scores.most_common(limit) if score > 0.0]
    if not top_items:
        return {}
    max_score = max(score for _name, score in top_items)
    return {name: round(float(score / max_score), 6) for name, score in top_items}


def weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0.0 or not values:
        return 0.0
    return float(sum(value * weight for value, weight in zip(values, weights)) / total)


def write_readme(
    path: Path,
    *,
    batch_root: Path,
    output: Path,
    result_rows: list[dict[str, Any]],
    success_rows: list[dict[str, Any]],
    shape_rows: list[dict[str, Any]],
    near_rows: list[dict[str, Any]],
    role_rows: list[dict[str, Any]],
    skip_rows: list[dict[str, Any]],
    selected_case_rows: list[dict[str, Any]],
) -> None:
    four_success = [row for row in success_rows if "4course" in str(row.get("target_name", ""))]
    four_near = [row for row in near_rows if "4course" in str(row.get("target_name", ""))]
    three_success = [row for row in success_rows if "3course" in str(row.get("target_name", ""))]
    top_near = sorted(
        four_near,
        key=lambda row: (
            -to_float(row.get("visible_courses")),
            to_float(row.get("skipped_slot_count")),
            to_float(row.get("failure_count")),
            to_float(row.get("max_horizontal_drift_m")),
        ),
    )[:10]
    promising_roles = [
        row
        for row in role_rows
        if row["tag"] in {"strict_success_case", "shape_success_case"} and int(row["placements"]) >= 3
    ][:20]
    worst_skips = skip_rows[:12]

    lines = [
        "# 历史成功经验抽取报告",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- batch_root: `{batch_root}`",
        f"- output: `{output}`",
        "",
        "## 样本规模",
        "",
        f"- results rows: `{len(result_rows)}`",
        f"- strict success cases: `{len(success_rows)}`",
        f"- shape success cases: `{len(shape_rows)}`",
        f"- 3-course strict success cases: `{len(three_success)}`",
        f"- 4-course strict success cases: `{len(four_success)}`",
        f"- 4-course near-success cases: `{len(four_near)}`",
        "",
        "## 主要结论",
        "",
    ]
    lines.extend(main_lessons(four_success, four_near, three_success, skip_rows, role_rows))
    lines.extend(["", "## 4 层近成功案例", ""])
    lines.extend(case_table(top_near))
    lines.extend(["", "## 成功/近成功 placement 的角色经验", ""])
    lines.extend(role_table(promising_roles))
    lines.extend(["", "## 4 层 skip 瓶颈", ""])
    lines.extend(skip_table(worst_skips))
    lines.extend(
        [
            "",
            "## 生成文件",
            "",
            "- `success_cases.csv`: strict success run 级案例。",
            "- `shape_success_cases.csv`: shape success run 级案例。",
            "- `near_success_cases.csv`: 4 层近成功案例，包括可见 4 层、高稳定、低漂移但未 strict 成功的样本。",
            "- `selected_case_summary.csv`: 成功和近成功案例的关键字段摘要。",
            "- `case_summary_by_task.csv`: target/gravity 级成功率和稳定性统计。",
            "- `role_experience.csv`: 成功/近成功案例中按 role/course/source_kind 汇总的 placement 特征。",
            "- `skip_by_role.csv`: 4 层任务中按 role/course 的 skipped slot 统计。",
            "- `experience_priors.json`: 从成功/近成功案例抽取的角色级 source_kind、cluster_label 和瓶颈先验，可供下一轮候选池排序使用。",
            "",
            "## 下一轮策略建议",
            "",
            "1. 不要只扩大石头池。扩大石头池可以增加可能性，但已经观察到计算成本显著上升，需要配合更强的先筛规则。",
            "2. 4 层应优先降低 middle/cap 的 skipped slot，而不是继续优化 base。base 在成功和近成功案例中通常不是主要瓶颈。",
            "3. 对 4 层不要过早压到单一 top1/top-k。可以让 StoneSlotNet 给更宽的候选池，再由 SupportMap/PoseRisk 和局部支撑约束二次排序。",
            "4. 对 successful/near-success placement 中的 source_kind 和 cluster_label 做正样本重采样，优先给 middle/cap 提供相似的互锁块、承重块、cap 块。",
            "5. 继续记录失败，因为 4 层 strict success 仍少，near-success 和 hard negative 是当前最有学习价值的数据。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main_lessons(
    four_success: list[dict[str, Any]],
    four_near: list[dict[str, Any]],
    three_success: list[dict[str, Any]],
    skip_rows: list[dict[str, Any]],
    role_rows: list[dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    if four_success:
        lines.append(f"- 历史中存在 `{len(four_success)}` 个 4 层 strict success，可作为强正样本。")
    else:
        lines.append("- 当前扫描范围内 4 层 strict success 很少或没有，4 层学习应主要依赖 near-success 和 hard negative。")
    if three_success:
        lines.append(
            f"- 3 层 strict success 已有 `{len(three_success)}` 个，可作为稳定低层结构的主要经验来源。"
        )
    if four_near:
        best = sorted(
            four_near,
            key=lambda row: (
                -to_float(row.get("visible_courses")),
                to_float(row.get("skipped_slot_count")),
                to_float(row.get("failure_count")),
                to_float(row.get("max_horizontal_drift_m")),
            ),
        )[0]
        lines.append(
            "- 最好的 4 层近成功通常已经能达到 "
            f"`visible_courses={best.get('visible_courses')}`，但 strict 失败来自 skipped/failure/drift 的组合。"
        )
    if skip_rows:
        high_skip = skip_rows[0]
        lines.append(
            f"- 4 层 skip 最突出的分组是 gravity=`{high_skip['gravity']}` role=`{high_skip['role']}` "
            f"course=`{high_skip['course']}`，"
            f"skip_rate=`{to_float(high_skip['skip_rate']):.3f}`，主要原因 `{high_skip['top_skip_reasons']}`。"
        )
    cap_rows = [row for row in role_rows if row.get("role") == "cap" and row.get("tag") in {"strict_success_case", "shape_success_case"}]
    middle_rows = [row for row in role_rows if row.get("role") == "middle" and row.get("tag") in {"strict_success_case", "shape_success_case"}]
    if middle_rows:
        best_middle = max(middle_rows, key=lambda row: int(row["placements"]))
        lines.append(
            f"- middle 层成功 placement 中较常见 source_kind=`{best_middle['source_kind']}`，"
            f"top clusters `{best_middle['top_clusters']}`。"
        )
    if cap_rows:
        best_cap = max(cap_rows, key=lambda row: int(row["placements"]))
        lines.append(
            f"- cap 层成功 placement 中较常见 source_kind=`{best_cap['source_kind']}`，"
            f"top clusters `{best_cap['top_clusters']}`。"
        )
    return lines


def case_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| run | trial | target | visible | stable | failure | skipped | rmse | drift | height |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        stable = f"{row.get('stable_count', '')}/{row.get('rock_count', '')}"
        lines.append(
            f"| `{row.get('run_name', '')}` | {row.get('trial', '')} | `{row.get('target_name', '')}` | "
            f"{fmt(row.get('visible_courses'))} | {stable} | {fmt(row.get('failure_count'))} | "
            f"{fmt(row.get('skipped_slot_count'))} | {fmt(row.get('target_rmse_xy_m'))} | "
            f"{fmt(row.get('max_horizontal_drift_m'))} | {fmt(row.get('stack_height_m'))} |"
        )
    return lines


def role_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| tag | target | role | course | source | n | clusters | err | overlap | support | disturbance | rank |",
        "|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['tag']}` | `{row['target_name']}` | `{row['role']}` | {row['course']} | "
            f"`{row['source_kind']}` | {row['placements']} | `{row['top_clusters']}` | "
            f"{fmt(row['mean_target_error_xy_m'])} | {fmt(row['mean_support_overlap'])} | "
            f"{fmt(row['mean_support_contact_count'])} | {fmt(row['mean_disturbance_xy_m'])} | "
            f"{fmt(row['mean_stone_fit_rank'])} |"
        )
    return lines


def skip_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| target | gravity | role | course | rows | skipped | skip_rate | reasons |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['target_name']}` | `{row['gravity']}` | `{row['role']}` | {row['course']} | {row['rows']} | "
            f"{row['skipped']} | {fmt(row['skip_rate'])} | `{row['top_skip_reasons']}` |"
        )
    return lines


def stable_fraction(rows: list[dict[str, Any]]) -> float:
    stable = sum(to_float(row.get("stable_count")) for row in rows)
    rocks = sum(to_float(row.get("rock_count")) for row in rows)
    return stable / rocks if rocks > 0 else 0.0


def is_skipped(row: dict[str, Any]) -> bool:
    return (
        to_int(row.get("placement_skipped")) == 1
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


def to_int(value: Any) -> int:
    return int(round(to_float(value)))


def safe_mean(values: Any) -> float:
    seq = [float(value) for value in values]
    return mean(seq) if seq else 0.0


def int_or_999(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 999


def fmt(value: Any) -> str:
    return f"{to_float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
