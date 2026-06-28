from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


FEATURE_COLUMNS = [
    "rock_volume",
    "rock_surface_area",
    "rock_face_count",
    "rock_bbox_x",
    "rock_bbox_y",
    "rock_bbox_z",
    "rock_elongation",
    "rock_flatness",
    "rock_sphericity",
    "rock_roughness",
    "rock_angularity",
    "rock_spike_score",
    "rock_compactness",
    "rock_rectangularity",
    "rock_roundness_proxy",
    "rock_concavity_proxy",
    "rock_stability_score",
    "rock_major_face_count",
    "rock_largest_face_area_ratio",
    "rock_top3_face_area_ratio",
    "rock_support_face_count",
    "rock_support_face_area_ratio",
    "rock_opposing_face_pair_count",
    "rock_opposing_face_area_ratio",
    "rock_support_plane_quality",
]


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    output_dir = unique_dir(args.output.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)

    placements = read_csv(dataset_dir / "placement_examples.csv")
    if not placements:
        raise SystemExit(f"No placement_examples.csv rows found in {dataset_dir}")

    summaries = {
        "by_role": summarize_groups(placements, ["role"], args.min_count),
        "by_source_kind": summarize_groups(placements, ["rock_source_kind"], args.min_count),
        "by_role_source_kind": summarize_groups(placements, ["role", "rock_source_kind"], args.min_count),
        "by_target_role": summarize_groups(placements, ["target_name", "role"], args.min_count),
    }
    feature_rows = summarize_feature_signals(placements, args.min_count)
    write_csv(output_dir / "role_summary.csv", summaries["by_role"])
    write_csv(output_dir / "source_kind_summary.csv", summaries["by_source_kind"])
    write_csv(output_dir / "role_source_kind_summary.csv", summaries["by_role_source_kind"])
    write_csv(output_dir / "target_role_summary.csv", summaries["by_target_role"])
    write_csv(output_dir / "feature_signal_by_role.csv", feature_rows)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "placement_rows": len(placements),
        "min_count": args.min_count,
        "files": [
            "role_summary.csv",
            "source_kind_summary.csv",
            "role_source_kind_summary.csv",
            "target_role_summary.csv",
            "feature_signal_by_role.csv",
            "README.md",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_readme(output_dir, dataset_dir, summaries, feature_rows, report)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize role/source/geometry experience from a MoonStack learning dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=3)
    return parser.parse_args()


def summarize_groups(rows: list[dict[str, str]], keys: list[str], min_count: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key, "") for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for key, items in sorted(buckets.items()):
        if len(items) < min_count:
            continue
        skipped = sum(as_bool(row.get("is_skipped_slot", "")) for row in items)
        failures = sum(as_bool(row.get("is_failure_case", "")) for row in items)
        successes = sum(as_bool(row.get("label_success", "")) for row in items)
        committed = max(0, len(items) - skipped)
        row_out: dict[str, Any] = {name: value for name, value in zip(keys, key)}
        row_out.update(
            {
                "examples": len(items),
                "success": successes,
                "failure": failures,
                "skipped": skipped,
                "committed": committed,
                "success_rate_all": successes / max(len(items), 1),
                "success_rate_committed": successes / max(committed, 1),
                "failure_rate_committed": failures / max(committed, 1),
                "skip_rate": skipped / max(len(items), 1),
                "mean_target_error_xy_m": mean_float(items, "target_error_xy_m"),
                "mean_target_y_error_m": mean_abs_float(items, "target_y_error_m"),
                "mean_support_overlap": mean_float(items, "support_overlap"),
                "mean_support_balance_error_m": mean_float(items, "support_balance_error_m"),
                "mean_disturbance_xy_m": mean_float(items, "placed_disturbance_xy_m"),
                "mean_velocity_inf_norm": mean_float(items, "velocity_inf_norm_after_place"),
            }
        )
        output.append(row_out)
    return sorted(output, key=lambda row: (-float(row["examples"]), str(tuple(row.get(key, "") for key in keys))))


def summarize_feature_signals(rows: list[dict[str, str]], min_count: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    roles = sorted({row.get("role", "") for row in rows if row.get("role", "")})
    for role in roles:
        role_rows = [row for row in rows if row.get("role", "") == role and not as_bool(row.get("is_skipped_slot", ""))]
        success_rows = [row for row in role_rows if as_bool(row.get("label_success", ""))]
        fail_rows = [row for row in role_rows if not as_bool(row.get("label_success", ""))]
        if len(success_rows) < min_count or len(fail_rows) < min_count:
            continue
        for feature in FEATURE_COLUMNS:
            success_mean = mean_float(success_rows, feature)
            fail_mean = mean_float(fail_rows, feature)
            output.append(
                {
                    "role": role,
                    "feature": feature,
                    "success_count": len(success_rows),
                    "failure_count": len(fail_rows),
                    "success_mean": success_mean,
                    "failure_mean": fail_mean,
                    "success_minus_failure": success_mean - fail_mean,
                }
            )
    return sorted(output, key=lambda row: (row["role"], -abs(float(row["success_minus_failure"]))))


def write_readme(
    output_dir: Path,
    dataset_dir: Path,
    summaries: dict[str, list[dict[str, Any]]],
    feature_rows: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    lines = [
        "# 3/4层石墙数据经验统计",
        "",
        f"数据集：`{dataset_dir}`",
        f"placement rows：`{report['placement_rows']}`",
        "",
        "## 关键结论",
        "",
    ]
    role_rows = summaries["by_role"]
    for row in role_rows:
        lines.append(
            f"- `{row.get('role', '')}`：样本 `{row['examples']}`，成功 `{row['success']}`，失败 `{row['failure']}`，跳过 `{row['skipped']}`，"
            f"committed 成功率 `{float(row['success_rate_committed']):.3f}`，跳过率 `{float(row['skip_rate']):.3f}`。"
        )
    lines.extend(["", "## 每个角色当前较好的石头来源", ""])
    for role in sorted({row.get("role", "") for row in summaries["by_role_source_kind"]}):
        candidates = [
            row for row in summaries["by_role_source_kind"]
            if row.get("role", "") == role and int(row["examples"]) >= max(3, int(report["min_count"]))
        ]
        candidates.sort(key=lambda row: (float(row["success_rate_committed"]), -float(row["skip_rate"]), int(row["examples"])), reverse=True)
        for row in candidates[:5]:
            lines.append(
                f"- `{role}` / `{row.get('rock_source_kind', '')}`：样本 `{row['examples']}`，"
                f"committed 成功率 `{float(row['success_rate_committed']):.3f}`，跳过率 `{float(row['skip_rate']):.3f}`，"
                f"平均 y 误差 `{float(row['mean_target_y_error_m']):.4f}` m。"
            )
    lines.extend(["", "## 几何特征信号", ""])
    for role in sorted({row["role"] for row in feature_rows}):
        role_features = [row for row in feature_rows if row["role"] == role]
        for row in role_features[:6]:
            delta = float(row["success_minus_failure"])
            direction = "更高" if delta > 0 else "更低"
            lines.append(
                f"- `{role}` 成功样本的 `{row['feature']}` 平均值比失败样本{direction} `{abs(delta):.5f}`。"
            )
    lines.extend(
        [
            "",
            "## 使用建议",
            "",
            "- 这份统计是当前数据集的经验，不是物理定律；需要随着新 run 自动重算。",
            "- base 已相对稳定，middle/cap 的失败和跳过更多，应优先强化中高层石头选择、墙线约束和候选位姿风险筛选。",
            "- PointNet 直接拼 embedding 的消融已经下降，下一步点云网络应改成 role/affordance 监督，而不是直接替换几何先验。",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def mean_float(rows: list[dict[str, str]], key: str) -> float:
    values = [to_float(row.get(key, "")) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / max(len(values), 1)


def mean_abs_float(rows: list[dict[str, str]], key: str) -> float:
    values = [to_float(row.get(key, "")) for row in rows]
    values = [abs(value) for value in values if value is not None]
    return sum(values) / max(len(values), 1)


def to_float(value: str | None) -> float | None:
    if value in {"", None}:
        return None
    try:
        output = float(value)
    except ValueError:
        return None
    return output


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
