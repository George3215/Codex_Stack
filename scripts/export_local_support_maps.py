from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


CHANNELS = [
    "height_before_m",
    "support_occupancy",
    "support_count_clipped",
    "target_gaussian",
    "candidate_footprint",
    "candidate_height_m",
    "gravity_ratio",
    "course_ratio",
]

SUPERVISION_FIELDS = [
    "target_error_xy_m",
    "target_x_error_m",
    "target_y_error_m",
    "radial_distance_m",
    "placed_disturbance_xy_m",
    "velocity_inf_norm_after_place",
    "height_gain_m",
]

NUMERIC_FEATURES = [
    "course",
    "target_x",
    "target_y",
    "pose_x",
    "pose_y",
    "pose_z",
    "pose_qw",
    "pose_qx",
    "pose_qy",
    "pose_qz",
    "candidate_id",
    "candidate_count",
    "candidate_score",
    "rock_volume",
    "rock_surface_area",
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
    "rock_stability_score",
    "rock_major_face_count",
    "rock_largest_face_area_ratio",
    "rock_top3_face_area_ratio",
    "rock_face_area_entropy",
    "rock_normal_concentration",
    "rock_support_face_count",
    "rock_support_face_area_ratio",
    "rock_opposing_face_pair_count",
    "rock_opposing_face_area_ratio",
    "rock_face_planarity",
    "rock_support_plane_quality",
    "rock_mass",
    "support_overlap",
    "support_contact_count",
    "support_balance_error_m",
    "bearing_pressure_proxy",
]


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    placement_rows = read_csv(dataset_dir / "placement_examples.csv")
    if not placement_rows:
        raise SystemExit(f"No placement_examples.csv found in {dataset_dir}")

    if args.source == "candidate":
        source_rows = read_csv(dataset_dir / "candidate_pose_examples.csv")
        source_name = "candidate_pose"
    else:
        source_rows = placement_rows
        source_name = "placement"
    if not source_rows:
        raise SystemExit(f"No {source_name} rows found in {dataset_dir}")

    filtered_rows = filter_rows(source_rows, args)
    if args.max_examples is not None:
        filtered_rows = filtered_rows[: args.max_examples]
    if not filtered_rows:
        raise SystemExit("No rows remained after filtering.")

    output_dir = unique_dir(args.output.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)

    support_by_group = build_support_index(placement_rows, args.include_failed_support)
    grid_x, grid_y = make_local_grid(args.grid_size, args.window_m)

    index_rows: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    shard_maps: list[np.ndarray] = []
    shard_numeric: list[np.ndarray] = []
    shard_present: list[np.ndarray] = []
    shard_selected: list[int] = []
    shard_success: list[int] = []
    shard_scores: list[float] = []
    shard_count = 0
    channel_sums = np.zeros(len(CHANNELS), dtype=np.float64)
    channel_sq_sums = np.zeros(len(CHANNELS), dtype=np.float64)
    channel_pixels = 0

    for example_index, row in enumerate(filtered_rows):
        group_supports = supports_before(row, support_by_group)
        map_tensor = render_local_map(row, group_supports, grid_x, grid_y, args)
        numeric, present = numeric_vector(row)
        selected_label = parse_bool(row.get("label_selected_by_pose_search", ""))
        success_label = parse_bool(row.get("label_committed_success", row.get("label_success", "")))
        score = parse_float(row.get("candidate_score", ""), 0.0)

        shard_row = build_index_row(row, source_name, example_index, len(group_supports))
        shard_rows.append(shard_row)
        shard_maps.append(map_tensor)
        shard_numeric.append(numeric)
        shard_present.append(present)
        shard_selected.append(selected_label)
        shard_success.append(success_label)
        shard_scores.append(score)

        channel_sums += map_tensor.sum(axis=(1, 2))
        channel_sq_sums += np.square(map_tensor).sum(axis=(1, 2))
        channel_pixels += map_tensor.shape[1] * map_tensor.shape[2]

        if len(shard_rows) >= args.shard_size:
            shard_count += 1
            index_rows.extend(
                flush_shard(
                    output_dir,
                    shard_count,
                    shard_rows,
                    shard_maps,
                    shard_numeric,
                    shard_present,
                    shard_selected,
                    shard_success,
                    shard_scores,
                    args.dtype,
                )
            )
            shard_rows, shard_maps, shard_numeric, shard_present = [], [], [], []
            shard_selected, shard_success, shard_scores = [], [], []

    if shard_rows:
        shard_count += 1
        index_rows.extend(
            flush_shard(
                output_dir,
                shard_count,
                shard_rows,
                shard_maps,
                shard_numeric,
                shard_present,
                shard_selected,
                shard_success,
                shard_scores,
                args.dtype,
            )
        )

    write_csv(output_dir / "examples_index.csv", index_rows)
    channel_stats = summarize_channels(channel_sums, channel_sq_sums, channel_pixels)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "source": source_name,
        "row_count": len(filtered_rows),
        "shard_count": shard_count,
        "grid_size": args.grid_size,
        "window_m": args.window_m,
        "dtype": args.dtype,
        "channels": CHANNELS,
        "numeric_features": NUMERIC_FEATURES,
        "target_contains": args.target_contains,
        "strategy_contains": args.strategy_contains,
        "support_state_assumption": (
            "Supports are reconstructed from prior successful committed placements in the same "
            "run/target/strategy/gravity/trial group. This is a deterministic learning proxy, "
            "not a full MuJoCo state rewind."
        ),
        "channel_stats": channel_stats,
    }
    write_json(output_dir / "summary.json", summary)
    write_readme(output_dir, summary)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export local support height-map tensors for stacking networks.")
    parser.add_argument("--dataset", type=Path, required=True, help="Directory containing placement/candidate CSVs.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=["candidate", "placement"], default="candidate")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--window-m", type=float, default=0.9)
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--target-contains", action="append", default=[], help="Keep rows whose target_name contains this text.")
    parser.add_argument("--strategy-contains", action="append", default=[], help="Keep rows whose strategy contains this text.")
    parser.add_argument(
        "--include-failed-support",
        action="store_true",
        help="Include earlier failed placements in support maps. Default uses successful committed stones only.",
    )
    return parser.parse_args()


def filter_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    target_filters = [item.lower() for item in args.target_contains]
    strategy_filters = [item.lower() for item in args.strategy_contains]
    for row in rows:
        target_name = row.get("target_name", "").lower()
        strategy = row.get("strategy", "").lower()
        if target_filters and not any(token in target_name for token in target_filters):
            continue
        if strategy_filters and not any(token in strategy for token in strategy_filters):
            continue
        selected.append(row)
    return selected


def build_support_index(rows: list[dict[str, str]], include_failed: bool) -> dict[tuple[str, str, str, str, str], list[dict[str, str]]]:
    indexed: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("is_skipped_slot", "0") in {"1", "true", "True"}:
            continue
        if not include_failed and parse_bool(row.get("label_success", "")) != 1:
            continue
        if parse_float_or_none(row.get("settled_x", "")) is None or parse_float_or_none(row.get("settled_y", "")) is None:
            continue
        indexed[group_key(row)].append(row)
    for support_rows in indexed.values():
        support_rows.sort(key=lambda item: (parse_int(item.get("slot_id", ""), 10**9), item.get("example_id", "")))
    return indexed


def supports_before(row: dict[str, str], support_by_group: dict[tuple[str, str, str, str, str], list[dict[str, str]]]) -> list[dict[str, str]]:
    slot_id = parse_int(row.get("slot_id", ""), 10**9)
    return [support for support in support_by_group.get(group_key(row), []) if parse_int(support.get("slot_id", ""), -1) < slot_id]


def group_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("run_name", ""),
        row.get("target_name", ""),
        row.get("strategy", ""),
        row.get("gravity", ""),
        row.get("trial", ""),
    )


def make_local_grid(grid_size: int, window_m: float) -> tuple[np.ndarray, np.ndarray]:
    cell = window_m / grid_size
    axis = np.linspace(-0.5 * window_m + 0.5 * cell, 0.5 * window_m - 0.5 * cell, grid_size, dtype=np.float32)
    return np.meshgrid(axis, axis)


def render_local_map(
    row: dict[str, str],
    supports: list[dict[str, str]],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    target_x = parse_float(row.get("target_x", ""), parse_float(row.get("pose_x", row.get("placed_x", "")), 0.0))
    target_y = parse_float(row.get("target_y", ""), parse_float(row.get("pose_y", row.get("placed_y", "")), 0.0))
    world_x = target_x + grid_x
    world_y = target_y + grid_y

    height = np.zeros_like(grid_x, dtype=np.float32)
    support_count = np.zeros_like(grid_x, dtype=np.float32)
    for support in supports:
        sx = parse_float(support.get("settled_x", support.get("placed_x", "")), 0.0)
        sy = parse_float(support.get("settled_y", support.get("placed_y", "")), 0.0)
        bbox_x = parse_float(support.get("rock_bbox_x", ""), 0.12)
        bbox_y = parse_float(support.get("rock_bbox_y", ""), 0.10)
        bbox_z = parse_float(support.get("rock_bbox_z", ""), 0.08)
        yaw = yaw_from_row(support, prefix="quat")
        mask = ellipse_mask(world_x, world_y, sx, sy, bbox_x, bbox_y, yaw)
        top_z = parse_float(support.get("settled_z", support.get("placed_z", "")), 0.0) + 0.5 * bbox_z
        support_count[mask] += 1.0
        height[mask] = np.maximum(height[mask], top_z)

    pose_x = parse_float(row.get("pose_x", row.get("placed_x", "")), target_x)
    pose_y = parse_float(row.get("pose_y", row.get("placed_y", "")), target_y)
    bbox_x = parse_float(row.get("rock_bbox_x", ""), 0.12)
    bbox_y = parse_float(row.get("rock_bbox_y", ""), 0.10)
    bbox_z = parse_float(row.get("rock_bbox_z", ""), 0.08)
    yaw = yaw_from_row(row, prefix="pose_q")
    if row.get("pose_qw", "") == "":
        yaw = yaw_from_row(row, prefix="quat")
    candidate_mask = ellipse_mask(world_x, world_y, pose_x, pose_y, bbox_x, bbox_y, yaw)

    sigma = max(args.window_m / args.grid_size * 1.5, 0.035)
    target_gaussian = np.exp(-0.5 * ((world_x - target_x) ** 2 + (world_y - target_y) ** 2) / (sigma * sigma)).astype(np.float32)
    gravity_ratio = gravity_value(row) / 9.80665
    course_ratio = parse_float(row.get("course", ""), 0.0) / 6.0

    tensor = np.zeros((len(CHANNELS), args.grid_size, args.grid_size), dtype=np.float32)
    tensor[0] = height
    tensor[1] = (support_count > 0.0).astype(np.float32)
    tensor[2] = np.clip(support_count, 0.0, 4.0) / 4.0
    tensor[3] = target_gaussian
    tensor[4] = candidate_mask.astype(np.float32)
    tensor[5] = candidate_mask.astype(np.float32) * bbox_z
    tensor[6].fill(gravity_ratio)
    tensor[7].fill(course_ratio)
    return tensor


def ellipse_mask(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    center_x: float,
    center_y: float,
    bbox_x: float,
    bbox_y: float,
    yaw: float,
) -> np.ndarray:
    rx = max(0.5 * bbox_x, 0.025)
    ry = max(0.5 * bbox_y, 0.025)
    dx = grid_x - center_x
    dy = grid_y - center_y
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    local_x = cos_yaw * dx - sin_yaw * dy
    local_y = sin_yaw * dx + cos_yaw * dy
    return (local_x / rx) ** 2 + (local_y / ry) ** 2 <= 1.0


def yaw_from_row(row: dict[str, str], prefix: str) -> float:
    if prefix == "pose_q":
        qw = parse_float(row.get("pose_qw", ""), 1.0)
        qx = parse_float(row.get("pose_qx", ""), 0.0)
        qy = parse_float(row.get("pose_qy", ""), 0.0)
        qz = parse_float(row.get("pose_qz", ""), 0.0)
    else:
        qw = parse_float(row.get(f"{prefix}_w", ""), 1.0)
        qx = parse_float(row.get(f"{prefix}_x", ""), 0.0)
        qy = parse_float(row.get(f"{prefix}_y", ""), 0.0)
        qz = parse_float(row.get(f"{prefix}_z", ""), 0.0)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def numeric_vector(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    values: list[float] = []
    present: list[int] = []
    for name in NUMERIC_FEATURES:
        value = parse_float_or_none(row.get(name, ""))
        if value is None:
            values.append(0.0)
            present.append(0)
        else:
            values.append(value)
            present.append(1)
    return np.asarray(values, dtype=np.float32), np.asarray(present, dtype=np.int8)


def build_index_row(row: dict[str, str], source_name: str, example_index: int, support_count: int) -> dict[str, Any]:
    output = {
        "global_row": example_index,
        "source": source_name,
        "example_id": row.get("example_id", ""),
        "run_name": row.get("run_name", ""),
        "target_name": row.get("target_name", ""),
        "strategy": row.get("strategy", ""),
        "gravity": row.get("gravity", ""),
        "trial": row.get("trial", ""),
        "slot_id": row.get("slot_id", ""),
        "course": row.get("course", ""),
        "role": row.get("role", ""),
        "candidate_rock_index": row.get("candidate_rock_index", row.get("committed_rock_index", "")),
        "candidate_id": row.get("candidate_id", ""),
        "label_selected_by_pose_search": row.get("label_selected_by_pose_search", ""),
        "label_committed_success": row.get("label_committed_success", row.get("label_success", "")),
        "candidate_score": row.get("candidate_score", ""),
        "prior_support_count": support_count,
    }
    for field in SUPERVISION_FIELDS:
        output[field] = row.get(field, "")
    return output


def flush_shard(
    output_dir: Path,
    shard_number: int,
    rows: list[dict[str, Any]],
    maps: list[np.ndarray],
    numeric: list[np.ndarray],
    present: list[np.ndarray],
    selected: list[int],
    success: list[int],
    scores: list[float],
    dtype_name: str,
) -> list[dict[str, Any]]:
    shard_name = f"support_maps_{shard_number:04d}.npz"
    shard_path = output_dir / shard_name
    dtype = np.float16 if dtype_name == "float16" else np.float32
    np.savez_compressed(
        shard_path,
        maps=np.asarray(maps, dtype=dtype),
        numeric_features=np.asarray(numeric, dtype=np.float32),
        numeric_feature_present=np.asarray(present, dtype=np.int8),
        label_selected_by_pose_search=np.asarray(selected, dtype=np.int8),
        label_committed_success=np.asarray(success, dtype=np.int8),
        candidate_score=np.asarray(scores, dtype=np.float32),
    )
    csv_name = f"support_maps_{shard_number:04d}.csv"
    for shard_row, row in enumerate(rows):
        row["shard_file"] = shard_name
        row["shard_row"] = shard_row
    write_csv(output_dir / csv_name, rows)
    return rows


def summarize_channels(channel_sums: np.ndarray, channel_sq_sums: np.ndarray, pixels_per_channel: int) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    if pixels_per_channel == 0:
        return stats
    for index, name in enumerate(CHANNELS):
        mean = channel_sums[index] / pixels_per_channel
        variance = max(channel_sq_sums[index] / pixels_per_channel - mean * mean, 0.0)
        stats[name] = {"mean": float(mean), "std": float(math.sqrt(variance))}
    return stats


def gravity_value(row: dict[str, str]) -> float:
    value = parse_float_or_none(row.get("gravity_m_s2", ""))
    if value is not None:
        return value
    gravity = row.get("gravity", "").lower()
    if "moon" in gravity or "lunar" in gravity:
        return 1.62
    return 9.80665


def parse_float(value: str | None, default: float) -> float:
    parsed = parse_float_or_none(value)
    return default if parsed is None else parsed


def parse_float_or_none(value: str | None) -> float | None:
    if value in {"", None}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def parse_int(value: str | None, default: int) -> int:
    parsed = parse_float_or_none(value)
    if parsed is None:
        return default
    return int(parsed)


def parse_bool(value: str | None) -> int:
    if value in {"1", "true", "True", "yes", "Yes"}:
        return 1
    return 0


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = collect_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames or ["empty"]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Local Support Map Tensor Export",
        "",
        "This directory contains local raster tensors for neural stack placement models.",
        "",
        f"- source: {summary['source']}",
        f"- examples: {summary['row_count']}",
        f"- shards: {summary['shard_count']}",
        f"- grid: {summary['grid_size']} x {summary['grid_size']}",
        f"- local window: {summary['window_m']} m",
        f"- dtype: {summary['dtype']}",
        "",
        "NPZ arrays per shard:",
        "",
        "- `maps`: `[N, C, H, W]`, channels listed in `summary.json`.",
        "- `numeric_features`: tabular candidate and rock features.",
        "- `numeric_feature_present`: missing-value mask for tabular features.",
        "- `label_selected_by_pose_search`: whether the heuristic pose search selected this candidate pose.",
        "- `label_committed_success`: whether the committed placement later satisfied the stability/shape label.",
        "- `candidate_score`: raw heuristic score from the candidate generator.",
        "",
        "State reconstruction assumption:",
        "",
        summary["support_state_assumption"],
        "",
        "Use this as a supervised-learning proxy for CNN/U-Net/Transformer experiments. It should later be replaced or augmented by exact MuJoCo state snapshots for high-fidelity world-model training.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
