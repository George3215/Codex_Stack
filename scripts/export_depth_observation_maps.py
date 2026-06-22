from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.export_local_support_maps import (
    NUMERIC_FEATURES,
    build_index_row,
    build_support_index,
    filter_rows,
    flush_shard,
    gravity_value,
    numeric_vector,
    parse_bool,
    parse_float,
    parse_float_or_none,
    read_csv,
    supports_before,
    unique_dir,
    write_csv,
    write_json,
)


CHANNELS = [
    "top_height_before_m",
    "top_support_occupancy",
    "top_support_count_clipped",
    "top_target_gaussian",
    "top_candidate_footprint",
    "top_candidate_height_m",
    "front_support_silhouette",
    "front_support_depth_proxy",
    "front_target_gaussian",
    "front_candidate_silhouette",
    "front_candidate_depth_proxy",
    "gravity_ratio",
    "course_ratio",
]


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    placement_rows = read_csv(dataset_dir / "placement_examples.csv")
    if not placement_rows:
        raise SystemExit(f"No placement_examples.csv found in {dataset_dir}")

    source_rows = read_csv(dataset_dir / "candidate_pose_examples.csv" if args.source == "candidate" else "placement_examples.csv")
    source_name = "candidate_pose" if args.source == "candidate" else "placement"
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
    top_x, top_y, front_x, front_z = make_observation_grids(args.grid_size, args.window_m, args.front_height_m)

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
        map_tensor = render_observation_map(row, group_supports, top_x, top_y, front_x, front_z, args)
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
        "front_height_m": args.front_height_m,
        "dtype": args.dtype,
        "channels": CHANNELS,
        "numeric_features": NUMERIC_FEATURES,
        "target_contains": args.target_contains,
        "strategy_contains": args.strategy_contains,
        "observation_model": (
            "Two-view depth proxy: top view is an x-y height map; front view is an x-z silhouette/depth proxy "
            "approximating wall-facing depth observation. It is generated from logged geometry, not raw camera pixels."
        ),
        "support_state_assumption": (
            "Supports are reconstructed from prior successful committed placements in the same "
            "run/target/strategy/gravity/trial group. This is a deterministic learning proxy for a future RGB-D pipeline."
        ),
        "channel_stats": channel_stats,
    }
    write_json(output_dir / "summary.json", summary)
    write_readme(output_dir, summary)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export two-view depth-proxy tensors for stacking networks.")
    parser.add_argument("--dataset", type=Path, required=True, help="Directory containing placement/candidate CSVs.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=["candidate", "placement"], default="candidate")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--window-m", type=float, default=0.9)
    parser.add_argument("--front-height-m", type=float, default=0.55)
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


def make_observation_grids(
    grid_size: int,
    window_m: float,
    front_height_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    top_cell = window_m / grid_size
    top_axis = np.linspace(-0.5 * window_m + 0.5 * top_cell, 0.5 * window_m - 0.5 * top_cell, grid_size, dtype=np.float32)
    top_x, top_y = np.meshgrid(top_axis, top_axis)
    front_z_axis = np.linspace(0.5 * front_height_m / grid_size, front_height_m - 0.5 * front_height_m / grid_size, grid_size, dtype=np.float32)
    front_x, front_z = np.meshgrid(top_axis, front_z_axis)
    return top_x, top_y, front_x, front_z


def render_observation_map(
    row: dict[str, str],
    supports: list[dict[str, str]],
    top_x: np.ndarray,
    top_y: np.ndarray,
    front_x: np.ndarray,
    front_z: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    target_x = parse_float(row.get("target_x", ""), parse_float(row.get("pose_x", row.get("placed_x", "")), 0.0))
    target_y = parse_float(row.get("target_y", ""), parse_float(row.get("pose_y", row.get("placed_y", "")), 0.0))
    world_top_x = target_x + top_x
    world_top_y = target_y + top_y
    world_front_x = target_x + front_x

    top_height = np.zeros_like(top_x, dtype=np.float32)
    top_support_count = np.zeros_like(top_x, dtype=np.float32)
    front_silhouette = np.zeros_like(front_x, dtype=np.float32)
    front_depth = np.zeros_like(front_x, dtype=np.float32)

    support_top = 0.0
    for support in supports:
        sx = parse_float(support.get("settled_x", support.get("placed_x", "")), 0.0)
        sy = parse_float(support.get("settled_y", support.get("placed_y", "")), 0.0)
        sz = parse_float(support.get("settled_z", support.get("placed_z", "")), 0.0)
        bbox_x = parse_float(support.get("rock_bbox_x", ""), 0.12)
        bbox_y = parse_float(support.get("rock_bbox_y", ""), 0.10)
        bbox_z = parse_float(support.get("rock_bbox_z", ""), 0.08)
        yaw = yaw_from_row(support)
        top_mask = ellipse_mask(world_top_x, world_top_y, sx, sy, bbox_x, bbox_y, yaw)
        top_z = sz + 0.5 * bbox_z
        support_top = max(support_top, top_z)
        top_support_count[top_mask] += 1.0
        top_height[top_mask] = np.maximum(top_height[top_mask], top_z)
        front_mask = front_rect_mask(world_front_x, front_z, sx, sz, bbox_x, bbox_z)
        front_silhouette[front_mask] = 1.0
        depth_value = front_depth_value(sy, bbox_y, target_y, args.window_m)
        front_depth[front_mask] = np.maximum(front_depth[front_mask], depth_value)

    pose_x = parse_float(row.get("pose_x", row.get("placed_x", "")), target_x)
    pose_y = parse_float(row.get("pose_y", row.get("placed_y", "")), target_y)
    pose_z = parse_float(row.get("pose_z", row.get("placed_z", "")), support_top)
    bbox_x = parse_float(row.get("rock_bbox_x", ""), 0.12)
    bbox_y = parse_float(row.get("rock_bbox_y", ""), 0.10)
    bbox_z = parse_float(row.get("rock_bbox_z", ""), 0.08)
    yaw = yaw_from_row(row, pose=True)
    top_candidate = ellipse_mask(world_top_x, world_top_y, pose_x, pose_y, bbox_x, bbox_y, yaw)
    front_candidate = front_rect_mask(world_front_x, front_z, pose_x, pose_z, bbox_x, bbox_z)
    candidate_depth = np.zeros_like(front_x, dtype=np.float32)
    candidate_depth[front_candidate] = front_depth_value(pose_y, bbox_y, target_y, args.window_m)

    sigma_top = max(args.window_m / args.grid_size * 1.5, 0.035)
    top_target = np.exp(-0.5 * ((world_top_x - target_x) ** 2 + (world_top_y - target_y) ** 2) / (sigma_top * sigma_top))
    target_z = support_top + 0.5 * bbox_z
    sigma_front_x = sigma_top
    sigma_front_z = max(args.front_height_m / args.grid_size * 2.0, 0.025)
    front_target = np.exp(
        -0.5
        * (
            ((world_front_x - target_x) / sigma_front_x) ** 2
            + ((front_z - target_z) / sigma_front_z) ** 2
        )
    )

    gravity_ratio = gravity_value(row) / 9.80665
    course_ratio = parse_float(row.get("course", ""), 0.0) / 6.0
    tensor = np.zeros((len(CHANNELS), args.grid_size, args.grid_size), dtype=np.float32)
    channel_map = {
        "top_height_before_m": top_height,
        "top_support_occupancy": (top_support_count > 0.0).astype(np.float32),
        "top_support_count_clipped": np.clip(top_support_count, 0.0, 4.0) / 4.0,
        "top_target_gaussian": top_target.astype(np.float32),
        "top_candidate_footprint": top_candidate.astype(np.float32),
        "top_candidate_height_m": top_candidate.astype(np.float32) * bbox_z,
        "front_support_silhouette": front_silhouette,
        "front_support_depth_proxy": front_depth,
        "front_target_gaussian": front_target.astype(np.float32),
        "front_candidate_silhouette": front_candidate.astype(np.float32),
        "front_candidate_depth_proxy": candidate_depth,
        "gravity_ratio": np.full_like(top_x, gravity_ratio, dtype=np.float32),
        "course_ratio": np.full_like(top_x, course_ratio, dtype=np.float32),
    }
    for index, name in enumerate(CHANNELS):
        tensor[index] = channel_map[name]
    return tensor


def yaw_from_row(row: dict[str, str], pose: bool = False) -> float:
    if pose and row.get("pose_qw", "") != "":
        qw = parse_float(row.get("pose_qw", ""), 1.0)
        qx = parse_float(row.get("pose_qx", ""), 0.0)
        qy = parse_float(row.get("pose_qy", ""), 0.0)
        qz = parse_float(row.get("pose_qz", ""), 0.0)
    else:
        qw = parse_float(row.get("quat_w", ""), 1.0)
        qx = parse_float(row.get("quat_x", ""), 0.0)
        qy = parse_float(row.get("quat_y", ""), 0.0)
        qz = parse_float(row.get("quat_z", ""), 0.0)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


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


def front_rect_mask(
    grid_x: np.ndarray,
    grid_z: np.ndarray,
    center_x: float,
    center_z: float,
    bbox_x: float,
    bbox_z: float,
) -> np.ndarray:
    x_half = max(0.5 * bbox_x, 0.025)
    z_half = max(0.5 * bbox_z, 0.025)
    return (np.abs(grid_x - center_x) <= x_half) & (np.abs(grid_z - center_z) <= z_half)


def front_depth_value(center_y: float, bbox_y: float, target_y: float, window_m: float) -> float:
    camera_y = target_y - 0.5 * window_m
    front_surface_y = center_y - 0.5 * bbox_y
    return float(np.clip((front_surface_y - camera_y) / max(window_m, 1e-6), 0.0, 1.0))


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Two-View Depth-Proxy Tensor Export",
        "",
        "This directory contains tensors for a perception-oriented stacking ranker.",
        "",
        f"- source: {summary['source']}",
        f"- examples: {summary['row_count']}",
        f"- shards: {summary['shard_count']}",
        f"- grid: {summary['grid_size']} x {summary['grid_size']}",
        f"- top local window: {summary['window_m']} m",
        f"- front height window: {summary['front_height_m']} m",
        f"- dtype: {summary['dtype']}",
        "",
        "Observation channels:",
        "",
        *[f"- `{name}`" for name in summary["channels"]],
        "",
        "Purpose:",
        "",
        "This is the first bridge from purely geometric/support-map ranking toward RGB-D conditioned ranking. The top view approximates a top depth/height observation, and the front view approximates a wall-facing depth observation. These channels are still generated from logs, not raw camera frames, so they should be treated as a proxy dataset.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_channels(channel_sums: np.ndarray, channel_sq_sums: np.ndarray, pixels_per_channel: int) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    if pixels_per_channel == 0:
        return stats
    for index, name in enumerate(CHANNELS):
        mean = channel_sums[index] / pixels_per_channel
        variance = max(channel_sq_sums[index] / pixels_per_channel - mean * mean, 0.0)
        stats[name] = {"mean": float(mean), "std": float(math.sqrt(variance))}
    return stats


if __name__ == "__main__":
    raise SystemExit(main())
