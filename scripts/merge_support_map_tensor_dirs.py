from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ARRAY_KEYS = [
    "maps",
    "numeric_features",
    "numeric_feature_present",
    "label_selected_by_pose_search",
    "label_committed_success",
    "candidate_score",
]


def main() -> int:
    args = parse_args()
    output_dir = unique_dir(args.output.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)

    rows_out: list[dict[str, Any]] = []
    shard_buffers: dict[str, list[np.ndarray]] = {key: [] for key in ARRAY_KEYS}
    seen: set[str] = set()
    shard_count = 0
    input_summaries: list[dict[str, Any]] = []
    source_counts: list[dict[str, Any]] = []
    skipped_duplicate_count = 0

    for tensor_dir in [path.resolve() for path in args.tensor_dir]:
        rows = read_csv(tensor_dir / "examples_index.csv")
        if not rows:
            raise SystemExit(f"No examples_index.csv found in {tensor_dir}")
        summary = read_json(tensor_dir / "summary.json")
        input_summaries.append({"tensor_dir": str(tensor_dir), "summary": summary})
        shard_files = sorted({row.get("shard_file", "") for row in rows if row.get("shard_file", "")})
        kept_for_source = 0
        for shard_file in shard_files:
            shard_path = tensor_dir / shard_file
            data = np.load(shard_path)
            required_missing = [key for key in ARRAY_KEYS if key not in data.files]
            if required_missing:
                raise SystemExit(f"{shard_path} missing keys: {required_missing}")
            shard_rows = [row for row in rows if row.get("shard_file", "") == shard_file]
            for row in shard_rows:
                shard_row = int(float(row.get("shard_row", "0") or 0))
                dedupe_key = build_dedupe_key(row, args.dedupe_column)
                if dedupe_key in seen:
                    skipped_duplicate_count += 1
                    continue
                seen.add(dedupe_key)
                for key in ARRAY_KEYS:
                    shard_buffers[key].append(np.asarray(data[key][shard_row]))
                row_out = dict(row)
                row_out["source_tensor_dir"] = str(tensor_dir)
                row_out["source_shard_file"] = shard_file
                row_out["source_shard_row"] = shard_row
                row_out["global_row"] = len(rows_out)
                row_out["shard_file"] = f"support_maps_{shard_count + 1:04d}.npz"
                row_out["shard_row"] = len(shard_buffers["maps"]) - 1
                rows_out.append(row_out)
                kept_for_source += 1
                if len(shard_buffers["maps"]) >= args.shard_size:
                    shard_count += 1
                    flush_shard(output_dir, shard_count, rows_out, shard_buffers, compress=args.compress)
        source_counts.append(
            {
                "tensor_dir": str(tensor_dir),
                "input_rows": len(rows),
                "kept_rows": kept_for_source,
            }
        )

    if shard_buffers["maps"]:
        shard_count += 1
        flush_shard(output_dir, shard_count, rows_out, shard_buffers, compress=args.compress)

    write_csv(output_dir / "examples_index.csv", rows_out)
    first_summary = input_summaries[0]["summary"] if input_summaries else {}
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "input_tensor_dirs": [str(path.resolve()) for path in args.tensor_dir],
        "source_counts": source_counts,
        "row_count": len(rows_out),
        "shard_count": shard_count,
        "skipped_duplicate_count": skipped_duplicate_count,
        "dedupe_column": args.dedupe_column,
        "shard_size": args.shard_size,
        "compress": bool(args.compress),
        "channels": first_summary.get("channels", []),
        "numeric_features": first_summary.get("numeric_features", []),
        "grid_size": first_summary.get("grid_size", 64),
        "window_m": first_summary.get("window_m", 0.9),
        "front_height_m": first_summary.get("front_height_m", 0.6),
        "dtype": first_summary.get("dtype", "float16"),
        "source": "merged_tensor_dirs",
        "merge_policy": "append_only_new_output; original tensor dirs are not modified",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_dir / "input_summaries.json").write_text(
        json.dumps(input_summaries, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_readme(output_dir, summary)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge support-map tensor directories without re-rendering MuJoCo depth maps.")
    parser.add_argument("--tensor-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--dedupe-column", default="example_id")
    parser.add_argument("--compress", action="store_true", help="Use np.savez_compressed. Slower but smaller.")
    return parser.parse_args()


def build_dedupe_key(row: dict[str, str], column: str) -> str:
    value = row.get(column, "")
    if value:
        return f"{column}:{value}"
    parts = [
        row.get("run_name", ""),
        row.get("target_name", ""),
        row.get("strategy", ""),
        row.get("gravity", ""),
        row.get("trial", ""),
        row.get("slot_id", ""),
        row.get("candidate_rock_index", ""),
        row.get("candidate_id", ""),
    ]
    return "|".join(parts)


def flush_shard(
    output_dir: Path,
    shard_number: int,
    rows_out: list[dict[str, Any]],
    buffers: dict[str, list[np.ndarray]],
    *,
    compress: bool,
) -> None:
    if not buffers["maps"]:
        return
    shard_name = f"support_maps_{shard_number:04d}.npz"
    shard_start = len(rows_out) - len(buffers["maps"])
    for local_row, row in enumerate(rows_out[shard_start:]):
        row["shard_file"] = shard_name
        row["shard_row"] = local_row
    save_fn = np.savez_compressed if compress else np.savez
    save_fn(
        output_dir / shard_name,
        maps=np.stack(buffers["maps"], axis=0),
        numeric_features=np.stack(buffers["numeric_features"], axis=0),
        numeric_feature_present=np.stack(buffers["numeric_feature_present"], axis=0),
        label_selected_by_pose_search=np.asarray(buffers["label_selected_by_pose_search"], dtype=np.int8),
        label_committed_success=np.asarray(buffers["label_committed_success"], dtype=np.int8),
        candidate_score=np.asarray(buffers["candidate_score"], dtype=np.float32),
    )
    for key in ARRAY_KEYS:
        buffers[key].clear()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Merged Support-Map Tensor Dataset",
        "",
        "Purpose: merge already-rendered MuJoCo depth/support tensors without modifying original data.",
        "",
        f"- rows: {summary['row_count']}",
        f"- shards: {summary['shard_count']}",
        f"- skipped duplicates: {summary['skipped_duplicate_count']}",
        f"- dedupe column: `{summary['dedupe_column']}`",
        "",
        "This is append-only output. Original tensor directories are not modified.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
