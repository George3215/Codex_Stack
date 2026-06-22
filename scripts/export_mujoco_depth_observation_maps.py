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

from scripts.export_depth_observation_maps import ellipse_mask, front_rect_mask, make_observation_grids
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
    parse_int,
    read_csv,
    supports_before,
    unique_dir,
    write_csv,
    write_json,
)


CHANNELS = [
    "render_front_depth_norm",
    "render_front_valid",
    "render_top_depth_norm",
    "render_top_valid",
    "top_target_gaussian",
    "top_candidate_footprint",
    "front_target_gaussian",
    "front_candidate_silhouette",
    "gravity_ratio",
    "course_ratio",
]

CANDIDATE_GROUP_COLUMNS = [
    "run_name",
    "target_name",
    "strategy",
    "gravity",
    "trial",
    "slot_id",
    "candidate_rock_index",
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
    filtered_rows, sampling_summary = sample_rows(filtered_rows, source_name, args)
    if not filtered_rows:
        raise SystemExit("No rows remained after filtering.")

    output_dir = unique_dir(args.output.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)
    support_by_group = build_support_index(placement_rows, args.include_failed_support)
    top_x, top_y, front_x, front_z = make_observation_grids(args.grid_size, args.window_m, args.front_height_m)

    render_cache: dict[Path, RenderContext] = {}
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
    skipped_rows: list[dict[str, Any]] = []

    try:
        for example_index, row in enumerate(filtered_rows):
            try:
                group_supports = supports_before(row, support_by_group)
                map_tensor = render_observation(row, group_supports, top_x, top_y, front_x, front_z, args, render_cache)
            except Exception as exc:  # noqa: BLE001 - record bad examples without stopping long exports
                skipped_rows.append(
                    {
                        "global_row": example_index,
                        "example_id": row.get("example_id", ""),
                        "run_name": row.get("run_name", ""),
                        "reason": str(exc),
                    }
                )
                continue
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
    finally:
        for context in render_cache.values():
            context.close()

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
    write_csv(output_dir / "skipped_render_rows.csv", skipped_rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "source": source_name,
        "source_row_count_after_filter": sampling_summary["source_row_count_after_filter"],
        "requested_row_count": len(filtered_rows),
        "row_count": len(index_rows),
        "skipped_row_count": len(skipped_rows),
        "shard_count": shard_count,
        "grid_size": args.grid_size,
        "window_m": args.window_m,
        "front_height_m": args.front_height_m,
        "dtype": args.dtype,
        "channels": CHANNELS,
        "numeric_features": NUMERIC_FEATURES,
        "target_contains": args.target_contains,
        "strategy_contains": args.strategy_contains,
        "sample_seed": args.sample_seed,
        "sampling": sampling_summary,
        "render_model": (
            "MuJoCo pre-placement RGB-D proxy rendered from reconstructed support state. "
            "Unused freejoint rocks are moved outside the camera view; prior successful support stones and the current candidate are rendered."
        ),
        "channel_stats": summarize_channels(channel_sums, channel_sq_sums, channel_pixels),
    }
    write_json(output_dir / "summary.json", summary)
    write_readme(output_dir, summary)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MuJoCo-rendered front/top depth tensors for stacking rankers.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=["candidate", "placement"], default="candidate")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--window-m", type=float, default=0.9)
    parser.add_argument("--front-height-m", type=float, default=0.55)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-groups", type=int, default=None, help="For candidate source, keep whole candidate-pose groups.")
    parser.add_argument(
        "--sample-mode",
        choices=["auto", "rows", "candidate-groups"],
        default="auto",
        help="Use candidate-groups for candidate-pose ranker data so each slot keeps multiple pose candidates.",
    )
    parser.add_argument("--sample-seed", type=int, default=None, help="Shuffle examples before optional max_examples cut.")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--target-contains", action="append", default=[])
    parser.add_argument("--strategy-contains", action="append", default=[])
    parser.add_argument("--include-failed-support", action="store_true")
    parser.add_argument("--depth-near-m", type=float, default=0.25)
    parser.add_argument("--depth-far-m", type=float, default=2.2)
    return parser.parse_args()


def sample_rows(rows: list[dict[str, str]], source_name: str, args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    source_count = len(rows)
    mode = args.sample_mode
    if mode == "auto":
        mode = "candidate-groups" if source_name == "candidate_pose" else "rows"
    if mode == "candidate-groups" and source_name != "candidate_pose":
        mode = "rows"

    if mode == "candidate-groups":
        groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            groups[candidate_group_key(row)].append(row)
        items = list(groups.items())
        if args.sample_seed is not None:
            rng = np.random.default_rng(args.sample_seed)
            order = rng.permutation(len(items))
            items = [items[int(index)] for index in order]
        if args.max_groups is not None:
            items = items[: args.max_groups]

        selected: list[dict[str, str]] = []
        selected_group_count = 0
        for _key, group_rows in items:
            if args.max_examples is not None and selected and len(selected) + len(group_rows) > args.max_examples:
                break
            selected.extend(group_rows)
            selected_group_count += 1
            if args.max_examples is not None and len(selected) >= args.max_examples:
                break
        return selected, {
            "mode": mode,
            "source_row_count_after_filter": source_count,
            "candidate_group_columns": CANDIDATE_GROUP_COLUMNS,
            "candidate_group_count_after_filter": len(groups),
            "selected_group_count": selected_group_count,
            "max_groups": args.max_groups,
            "max_examples": args.max_examples,
            "sample_seed": args.sample_seed,
            "note": "max_examples is group-aware; it stops at a group boundary except when the first group already exceeds it.",
        }

    selected = list(rows)
    if args.sample_seed is not None:
        rng = np.random.default_rng(args.sample_seed)
        order = rng.permutation(len(selected))
        selected = [selected[int(index)] for index in order]
    if args.max_examples is not None:
        selected = selected[: args.max_examples]
    return selected, {
        "mode": mode,
        "source_row_count_after_filter": source_count,
        "selected_group_count": None,
        "max_groups": args.max_groups,
        "max_examples": args.max_examples,
        "sample_seed": args.sample_seed,
    }


def candidate_group_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")) for column in CANDIDATE_GROUP_COLUMNS)


class RenderContext:
    def __init__(self, xml_path: Path, grid_size: int) -> None:
        import mujoco

        self.mujoco = mujoco
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, grid_size)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, grid_size)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=grid_size, width=grid_size)
        self.renderer.enable_depth_rendering()
        self.joint_addrs = self._joint_addresses()

    def _joint_addresses(self) -> dict[int, tuple[int, int]]:
        output: dict[int, tuple[int, int]] = {}
        for joint_id in range(self.model.njnt):
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if not name or not name.startswith("rock_") or not name.endswith("_free"):
                continue
            rock_index = int(name.removeprefix("rock_").removesuffix("_free"))
            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            qvel_addr = int(self.model.jnt_dofadr[joint_id])
            output[rock_index] = (qpos_addr, qvel_addr)
        return output

    def reset_hidden(self) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        for rock_index, (qpos_addr, qvel_addr) in self.joint_addrs.items():
            hidden_x = 10.0 + 0.08 * (rock_index % 20)
            hidden_y = 10.0 + 0.08 * (rock_index // 20)
            self.data.qpos[qpos_addr : qpos_addr + 3] = np.array([hidden_x, hidden_y, -1.5], dtype=np.float64)
            self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            self.data.qvel[qvel_addr : qvel_addr + 6] = 0.0

    def set_rock(self, rock_index: int, pos: np.ndarray, quat: np.ndarray) -> None:
        if rock_index not in self.joint_addrs:
            raise KeyError(f"rock_{rock_index:03d}_free not found in {self.xml_path}")
        qpos_addr, qvel_addr = self.joint_addrs[rock_index]
        self.data.qpos[qpos_addr : qpos_addr + 3] = pos
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat
        self.data.qvel[qvel_addr : qvel_addr + 6] = 0.0

    def forward(self) -> None:
        self.mujoco.mj_forward(self.model, self.data)

    def render_depth(self, lookat: np.ndarray, azimuth: float, elevation: float, distance: float) -> np.ndarray:
        cam = self.mujoco.MjvCamera()
        cam.type = self.mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        self.renderer.update_scene(self.data, camera=cam)
        return np.asarray(self.renderer.render(), dtype=np.float32)

    def close(self) -> None:
        self.renderer.close()


def render_observation(
    row: dict[str, str],
    supports: list[dict[str, str]],
    top_x: np.ndarray,
    top_y: np.ndarray,
    front_x: np.ndarray,
    front_z: np.ndarray,
    args: argparse.Namespace,
    render_cache: dict[Path, RenderContext],
) -> np.ndarray:
    xml_path = xml_path_for_row(row)
    context = render_cache.get(xml_path)
    if context is None:
        context = RenderContext(xml_path, args.grid_size)
        render_cache[xml_path] = context
    context.reset_hidden()

    support_top = 0.0
    for support in supports:
        rock_index = parse_int(support.get("committed_rock_index", support.get("candidate_rock_index", "")), -1)
        if rock_index < 0:
            continue
        sx = parse_float(support.get("settled_x", support.get("placed_x", "")), 0.0)
        sy = parse_float(support.get("settled_y", support.get("placed_y", "")), 0.0)
        sz = parse_float(support.get("settled_z", support.get("placed_z", "")), 0.0)
        qw = parse_float(support.get("quat_w", ""), 1.0)
        qx = parse_float(support.get("quat_x", ""), 0.0)
        qy = parse_float(support.get("quat_y", ""), 0.0)
        qz = parse_float(support.get("quat_z", ""), 0.0)
        context.set_rock(rock_index, np.array([sx, sy, sz], dtype=np.float64), normalized_quat(qw, qx, qy, qz))
        support_top = max(support_top, sz + 0.5 * parse_float(support.get("rock_bbox_z", ""), 0.08))

    candidate_index = parse_int(row.get("candidate_rock_index", row.get("rock_index", "")), -1)
    if candidate_index < 0:
        raise ValueError("candidate_rock_index is missing")
    pose = np.array(
        [
            parse_float(row.get("pose_x", row.get("placed_x", "")), 0.0),
            parse_float(row.get("pose_y", row.get("placed_y", "")), 0.0),
            parse_float(row.get("pose_z", row.get("placed_z", "")), 0.0),
        ],
        dtype=np.float64,
    )
    quat = normalized_quat(
        parse_float(row.get("pose_qw", row.get("quat_w", "")), 1.0),
        parse_float(row.get("pose_qx", row.get("quat_x", "")), 0.0),
        parse_float(row.get("pose_qy", row.get("quat_y", "")), 0.0),
        parse_float(row.get("pose_qz", row.get("quat_z", "")), 0.0),
    )
    context.set_rock(candidate_index, pose, quat)
    context.forward()

    target_x = parse_float(row.get("target_x", ""), pose[0])
    target_y = parse_float(row.get("target_y", ""), pose[1])
    target_z = support_top + 0.5 * parse_float(row.get("rock_bbox_z", ""), 0.08)
    lookat = np.array([target_x, target_y, max(0.16, min(0.34, target_z))], dtype=np.float64)
    front_depth = context.render_depth(lookat=lookat, azimuth=0.0, elevation=-12.0, distance=1.25)
    top_depth = context.render_depth(lookat=lookat, azimuth=0.0, elevation=-88.0, distance=1.05)

    front_norm, front_valid = normalize_depth(front_depth, args.depth_near_m, args.depth_far_m)
    top_norm, top_valid = normalize_depth(top_depth, args.depth_near_m, args.depth_far_m)
    top_target, top_candidate, front_target, front_candidate = analytic_guidance_maps(row, top_x, top_y, front_x, front_z, args)
    gravity_ratio = gravity_value(row) / 9.80665
    course_ratio = parse_float(row.get("course", ""), 0.0) / 6.0
    return np.stack(
        [
            front_norm,
            front_valid,
            top_norm,
            top_valid,
            top_target,
            top_candidate,
            front_target,
            front_candidate,
            np.full_like(front_norm, gravity_ratio, dtype=np.float32),
            np.full_like(front_norm, course_ratio, dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def xml_path_for_row(row: dict[str, str]) -> Path:
    run_path = Path(row.get("run_path", ""))
    target = row.get("target_name", "")
    strategy = row.get("strategy", "")
    gravity = row.get("gravity", "")
    trial = parse_int(row.get("trial", ""), 0)
    xml_path = run_path / "mjcf" / f"{target}_{strategy}_{gravity}_trial_{trial:02d}.xml"
    if xml_path.exists():
        return xml_path
    matches = sorted((run_path / "mjcf").glob(f"{target}_{strategy}_{gravity}_trial_*.xml"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot locate XML for row under {run_path}")


def normalized_quat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def normalize_depth(depth: np.ndarray, near: float, far: float) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(depth)
    valid = finite & (depth > 0.0) & (depth <= far)
    clipped = np.clip(np.where(valid, depth, far), near, far)
    norm = 1.0 - (clipped - near) / max(far - near, 1e-6)
    return norm.astype(np.float32), valid.astype(np.float32)


def analytic_guidance_maps(
    row: dict[str, str],
    top_x: np.ndarray,
    top_y: np.ndarray,
    front_x: np.ndarray,
    front_z: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_x = parse_float(row.get("target_x", ""), parse_float(row.get("pose_x", ""), 0.0))
    target_y = parse_float(row.get("target_y", ""), parse_float(row.get("pose_y", ""), 0.0))
    pose_x = parse_float(row.get("pose_x", row.get("placed_x", "")), target_x)
    pose_y = parse_float(row.get("pose_y", row.get("placed_y", "")), target_y)
    pose_z = parse_float(row.get("pose_z", row.get("placed_z", "")), 0.0)
    bbox_x = parse_float(row.get("rock_bbox_x", ""), 0.12)
    bbox_y = parse_float(row.get("rock_bbox_y", ""), 0.10)
    bbox_z = parse_float(row.get("rock_bbox_z", ""), 0.08)
    yaw = quat_yaw(
        parse_float(row.get("pose_qw", row.get("quat_w", "")), 1.0),
        parse_float(row.get("pose_qx", row.get("quat_x", "")), 0.0),
        parse_float(row.get("pose_qy", row.get("quat_y", "")), 0.0),
        parse_float(row.get("pose_qz", row.get("quat_z", "")), 0.0),
    )
    world_top_x = target_x + top_x
    world_top_y = target_y + top_y
    world_front_x = target_x + front_x
    sigma_top = max(args.window_m / args.grid_size * 1.5, 0.035)
    top_target = np.exp(-0.5 * ((world_top_x - target_x) ** 2 + (world_top_y - target_y) ** 2) / (sigma_top * sigma_top))
    top_candidate = ellipse_mask(world_top_x, world_top_y, pose_x, pose_y, bbox_x, bbox_y, yaw)
    target_z = pose_z
    sigma_front_z = max(args.front_height_m / args.grid_size * 2.0, 0.025)
    front_target = np.exp(
        -0.5
        * (
            ((world_front_x - target_x) / sigma_top) ** 2
            + ((front_z - target_z) / sigma_front_z) ** 2
        )
    )
    front_candidate = front_rect_mask(world_front_x, front_z, pose_x, pose_z, bbox_x, bbox_z)
    return (
        top_target.astype(np.float32),
        top_candidate.astype(np.float32),
        front_target.astype(np.float32),
        front_candidate.astype(np.float32),
    )


def quat_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def summarize_channels(channel_sums: np.ndarray, channel_sq_sums: np.ndarray, pixels_per_channel: int) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    if pixels_per_channel == 0:
        return stats
    for index, name in enumerate(CHANNELS):
        mean = channel_sums[index] / pixels_per_channel
        variance = max(channel_sq_sums[index] / pixels_per_channel - mean * mean, 0.0)
        stats[name] = {"mean": float(mean), "std": float(math.sqrt(variance))}
    return stats


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MuJoCo Rendered Depth Observation Export",
        "",
        "This dataset renders pre-placement candidate observations from MuJoCo instead of using only geometric proxy maps.",
        "",
        f"- requested examples: {summary['requested_row_count']}",
        f"- rendered examples: {summary['row_count']}",
        f"- skipped examples: {summary['skipped_row_count']}",
        f"- shards: {summary['shard_count']}",
        f"- grid: {summary['grid_size']} x {summary['grid_size']}",
        f"- dtype: {summary['dtype']}",
        "",
        "Channels:",
        "",
        *[f"- `{name}`" for name in summary["channels"]],
        "",
        "Purpose:",
        "",
        "This is the first true MuJoCo-rendered pre-placement depth dataset. It still uses logged support reconstruction, but the front/top depth channels come from MuJoCo rendering with unused stones hidden from view.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
