from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


TABLES = [
    "run_examples.csv",
    "placement_examples.csv",
    "candidate_pose_examples.csv",
    "assignment_candidate_examples.csv",
]


def main() -> int:
    args = parse_args()
    output = unique_dir(args.output.resolve())
    output.mkdir(parents=True, exist_ok=False)
    datasets = [path.resolve() for path in args.dataset]
    merged: dict[str, list[dict[str, str]]] = {}
    stats: dict[str, Any] = {}
    for table in TABLES:
        rows, table_stats = merge_table(datasets, table)
        merged[table] = rows
        stats[table] = table_stats
        write_csv(output / table, rows)
        if table in {"placement_examples.csv", "candidate_pose_examples.csv"}:
            write_jsonl(output / table.replace(".csv", ".jsonl"), rows)
    summary = build_summary(output, datasets, merged, stats)
    write_json(output / "dataset_summary.json", summary)
    write_readme(output, summary)
    print(output)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append-only merge of clean MoonStack policy datasets.")
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def merge_table(datasets: list[Path], table: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows_out: list[dict[str, str]] = []
    seen: set[str] = set()
    source_counts: dict[str, int] = {}
    duplicate_count = 0
    for dataset in datasets:
        path = dataset / table
        source_name = dataset.name
        count = 0
        for row in read_csv(path):
            key = row_key(table, row)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            enriched = dict(row)
            if "merged_source_dataset" not in enriched:
                enriched["merged_source_dataset"] = source_name
            elif not enriched["merged_source_dataset"]:
                enriched["merged_source_dataset"] = source_name
            rows_out.append(enriched)
            count += 1
        source_counts[source_name] = count
    return rows_out, {
        "rows": len(rows_out),
        "duplicates_skipped": duplicate_count,
        "source_rows_kept": source_counts,
    }


def row_key(table: str, row: dict[str, str]) -> str:
    example_id = row.get("example_id", "")
    if example_id:
        return f"{table}:example:{example_id}"
    if table == "run_examples.csv":
        return "|".join(
            [
                table,
                row.get("run_name", ""),
                row.get("target_name", ""),
                row.get("strategy", ""),
                row.get("gravity", ""),
                row.get("trial", ""),
            ]
        )
    return "|".join(
        [
            table,
            row.get("run_name", ""),
            row.get("target_name", ""),
            row.get("strategy", ""),
            row.get("gravity", ""),
            row.get("trial", ""),
            row.get("slot_id", ""),
            row.get("candidate_rock_index", ""),
            row.get("candidate_id", ""),
            row.get("candidate_rank", ""),
        ]
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def build_summary(
    output: Path,
    datasets: list[Path],
    merged: dict[str, list[dict[str, str]]],
    stats: dict[str, Any],
) -> dict[str, Any]:
    run_rows = merged["run_examples.csv"]
    placement_rows = merged["placement_examples.csv"]
    candidate_rows = merged["candidate_pose_examples.csv"]
    assignment_rows = merged["assignment_candidate_examples.csv"]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output),
        "input_datasets": [str(path) for path in datasets],
        "run_example_count": len(run_rows),
        "placement_example_count": len(placement_rows),
        "candidate_pose_example_count": len(candidate_rows),
        "assignment_candidate_example_count": len(assignment_rows),
        "run_success": summarize_bool(run_rows, "success"),
        "run_shape_success": summarize_bool(run_rows, "shape_success"),
        "placement_success": summarize_bool(placement_rows, "label_success"),
        "placement_by_target_role_gravity": summarize_rows(placement_rows, ["target_name", "role", "gravity"]),
        "candidate_pose_by_target_role_gravity": summarize_candidate_rows(candidate_rows),
        "assignment_by_target_role": summarize_rows(assignment_rows, ["target_name", "role"]),
        "table_stats": stats,
        "policy": "Append-only merge. Source datasets are not modified or deleted.",
    }


def summarize_bool(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    positives = sum(as_bool(row.get(column, "")) for row in rows)
    return {"positive": int(positives), "negative": int(len(rows) - positives), "rows": len(rows)}


def summarize_rows(rows: list[dict[str, str]], keys: list[str]) -> dict[str, dict[str, int]]:
    buckets: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key, "") for key in keys)].append(row)
    output: dict[str, dict[str, int]] = {}
    for key, items in sorted(buckets.items()):
        output["|".join(key)] = {
            "rows": len(items),
            "success": int(sum(as_bool(row.get("label_success", "")) for row in items)),
            "failure": int(sum(as_bool(row.get("is_failure_case", "")) for row in items)),
            "skipped": int(sum(as_bool(row.get("is_skipped_slot", "")) for row in items)),
            "selected_candidates": int(sum(as_bool(row.get("selected_count_in_placement_log", "")) for row in items)),
        }
    return output


def summarize_candidate_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, float | int]]:
    buckets: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("target_name", ""), row.get("role", ""), row.get("gravity", ""))].append(row)
    output: dict[str, dict[str, float | int]] = {}
    for key, items in sorted(buckets.items()):
        selected = sum(as_bool(row.get("label_selected_by_pose_search", "")) for row in items)
        committed_success = sum(as_bool(row.get("label_committed_success", "")) for row in items)
        mechanisms = Counter(row.get("failure_mechanism_primary", "") for row in items if row.get("failure_mechanism_primary", ""))
        output["|".join(key)] = {
            "rows": len(items),
            "selected_by_pose_search": int(selected),
            "committed_success": int(committed_success),
            "committed_success_rate": float(committed_success / max(len(items), 1)),
            "top_mechanism": mechanisms.most_common(1)[0][0] if mechanisms else "",
        }
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


def write_readme(output: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Merged Clean Policy Dataset",
        "",
        "目的：把已有清洗数据集与新采集的 controller 高成功率数据合并，形成后续 PoseRiskNet / SupportMapRanker 训练输入。",
        "",
        "策略：",
        "",
        "- 只创建新目录，不修改旧数据。",
        "- 按 `example_id` 或 run/slot/candidate 组合去重。",
        "- 保留 `merged_source_dataset` 字段，便于追踪样本来源。",
        "",
        "规模：",
        "",
        f"- run_examples: `{summary['run_example_count']}`",
        f"- placement_examples: `{summary['placement_example_count']}`",
        f"- candidate_pose_examples: `{summary['candidate_pose_example_count']}`",
        f"- assignment_candidate_examples: `{summary['assignment_candidate_example_count']}`",
        "",
        "输入数据集：",
        "",
    ]
    lines.extend(f"- `{path}`" for path in summary["input_datasets"])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
