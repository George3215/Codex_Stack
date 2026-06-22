from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moon_rock_stack.simulate import _set_freejoint_pose
from moon_rock_stack.run_structured_experiment import (
    load_candidate_pose_ranker,
    load_pose_risk_ranker,
    load_stone_fit_ranker,
)
from moon_rock_stack.structured import run_structured_trial_detailed, slots_for_target, structured_order


def main() -> int:
    args = parse_args()
    output_dir = args.output.resolve()
    results = read_csv(output_dir / "results.csv")
    case = select_case(results, args)
    if case is None:
        raise SystemExit(f"No matching successful case found in {output_dir / 'results.csv'}")

    placement_rows = select_placements(read_csv(output_dir / "placement_log.csv"), case)
    if not placement_rows:
        raise SystemExit("No committed placement rows found for selected case.")

    case_name = (
        f"{case.get('target_name', 'target')}_{case['strategy']}_{case['gravity']}_"
        f"trial_{int(case['trial']):02d}_{'algorithm_keyframes' if args.algorithm_keyframes else 'process'}"
    )
    video_root = output_dir / args.video_dir_name
    case_dir = video_root / case_name
    frames_dir = case_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if args.algorithm_keyframes:
        frame_paths, manifest = render_algorithm_keyframes(
            output_dir=output_dir,
            case=case,
            frames_dir=frames_dir,
            width=args.width,
            height=args.height,
            fps=args.fps,
            save_every_frame=args.save_frames,
            combined_view=not args.front_only,
            base_seed=args.base_seed,
            target_index=args.target_index,
            strategy_index=args.strategy_index,
            steps_per_rock=args.steps_per_rock,
            hold_steps=args.hold_steps,
        )
    else:
        frame_paths, manifest = render_replay(
            xml_path=Path(case["xml"]),
            features=read_csv(output_dir / "features.csv"),
            placement_rows=placement_rows,
            case=case,
            frames_dir=frames_dir,
            width=args.width,
            height=args.height,
            frame_stride=args.frame_stride,
            settle_steps=args.settle_steps,
            final_hold_steps=args.final_hold_steps,
            fps=args.fps,
            save_every_frame=args.save_frames,
            combined_view=not args.front_only,
        )
    gif_path = case_dir / "process.gif"
    write_gif(frame_paths, gif_path, fps=args.fps)
    manifest.update(
        {
            "gif_path": str(gif_path),
            "frame_count": len(frame_paths),
            "frames_dir": str(frames_dir),
            "case_dir": str(case_dir),
        }
    )
    (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(case_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a dry-stacking process replay as PNG frames and an animated GIF.")
    parser.add_argument("--output", type=Path, required=True, help="Experiment output directory containing results.csv.")
    parser.add_argument("--gravity", default="moon", help="Case gravity to render.")
    parser.add_argument("--trial", type=int, default=None, help="Optional exact trial id.")
    parser.add_argument("--target", default="single_face_wall_3course_v1", help="Target name to render.")
    parser.add_argument("--strategy", default="statics_wall", help="Strategy name to render.")
    parser.add_argument("--video-dir-name", default="process_videos", help="Subdirectory for rendered process videos.")
    parser.add_argument("--width", type=int, default=480, help="Per-camera render width.")
    parser.add_argument("--height", type=int, default=360, help="Per-camera render height.")
    parser.add_argument("--fps", type=int, default=8, help="Animated GIF frame rate.")
    parser.add_argument("--frame-stride", type=int, default=72, help="MuJoCo steps between rendered process frames.")
    parser.add_argument("--settle-steps", type=int, default=0, help="Override per-rock settle steps; default uses result effective_steps_per_rock.")
    parser.add_argument("--final-hold-steps", type=int, default=720, help="Extra post-stack hold steps to show final stability.")
    parser.add_argument("--save-frames", action="store_true", help="Keep all rendered PNG frames; otherwise keep key frames plus GIF.")
    parser.add_argument("--front-only", action="store_true", help="Render only the front camera instead of front+top side by side.")
    parser.add_argument("--algorithm-keyframes", action="store_true", help="Rerun the placement algorithm and render exact after-placement keyframes.")
    parser.add_argument("--base-seed", type=int, help="Original experiment seed. Required for --algorithm-keyframes if PROTOCOL.md does not record seed.")
    parser.add_argument("--target-index", type=int, default=0, help="Target index in the original run.")
    parser.add_argument("--strategy-index", type=int, default=0, help="Strategy index in the original run.")
    parser.add_argument("--steps-per-rock", type=int, default=0, help="Requested steps per rock for --algorithm-keyframes; default reads PROTOCOL.md.")
    parser.add_argument("--hold-steps", type=int, default=0, help="Requested hold steps for --algorithm-keyframes; default reads PROTOCOL.md.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_case(results: list[dict[str, str]], args: argparse.Namespace) -> dict[str, str] | None:
    rows = [
        row
        for row in results
        if row.get("gravity") == args.gravity
        and row.get("strategy") == args.strategy
        and row.get("target_name") == args.target
        and int(row.get("success", 0)) == 1
    ]
    if args.trial is not None:
        rows = [row for row in rows if int(row["trial"]) == int(args.trial)]
    rows.sort(key=lambda row: (-float(row.get("stack_height_m", 0.0)), float(row.get("velocity_inf_norm", 0.0))))
    return rows[0] if rows else None


def select_placements(rows: list[dict[str, str]], case: dict[str, str]) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row.get("gravity") == case.get("gravity")
        and row.get("strategy") == case.get("strategy")
        and row.get("target_name") == case.get("target_name")
        and int(row.get("trial", -1)) == int(case.get("trial", -2))
        and int(float(row.get("rock_index", "-1"))) >= 0
        and int(float(row.get("placement_skipped", "0") or 0)) == 0
    ]
    selected.sort(key=lambda row: (int(float(row.get("course", 0))), int(float(row.get("slot_id", 0)))))
    return selected


def render_replay(
    xml_path: Path,
    features: list[dict[str, str]],
    placement_rows: list[dict[str, str]],
    case: dict[str, str],
    frames_dir: Path,
    width: int,
    height: int,
    frame_stride: int,
    settle_steps: int,
    final_hold_steps: int,
    fps: int,
    save_every_frame: bool,
    combined_view: bool,
) -> tuple[list[Path], dict[str, Any]]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path.resolve()))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    hide_unused_rocks(model, data, features)
    mujoco.mj_forward(model, data)

    per_rock_steps = int(settle_steps or float(case.get("effective_steps_per_rock", 720)))
    frame_paths: list[Path] = []
    in_memory_frames: list[Image.Image] = []
    placed: list[int] = []
    frame_index = 0

    def capture(label: str, force_save: bool = False) -> None:
        nonlocal frame_index
        img = render_frame(renderer, data, label, width, height, combined_view=combined_view)
        frame_path = frames_dir / f"frame_{frame_index:04d}.png"
        if save_every_frame or force_save:
            img.save(frame_path)
        else:
            temp_path = frames_dir / f"_gif_frame_{frame_index:04d}.png"
            img.save(temp_path)
            frame_path = temp_path
        frame_paths.append(frame_path)
        in_memory_frames.append(img)
        frame_index += 1

    capture("start | empty target area", force_save=True)
    for placement_id, row in enumerate(placement_rows, start=1):
        rock_index = int(float(row["rock_index"]))
        pos = np.array([float(row["placed_x"]), float(row["placed_y"]), float(row["placed_z"])], dtype=float)
        quat = np.array([float(row["quat_w"]), float(row["quat_x"]), float(row["quat_y"]), float(row["quat_z"])], dtype=float)
        _set_freejoint_pose(model, data, rock_index, pos, quat)
        mujoco.mj_forward(model, data)
        placed.append(rock_index)
        label = (
            f"{case['gravity']} trial {int(case['trial']):02d} | "
            f"stone {placement_id}/{len(placement_rows)} | course {row.get('course')} {row.get('role')}"
        )
        capture(label, force_save=True)
        simulate_with_frames(
            mujoco=mujoco,
            model=model,
            data=data,
            max_steps=per_rock_steps,
            frame_stride=frame_stride,
            capture=lambda step: capture(f"{label} | settle step {step}", force_save=False),
        )
        capture(f"{label} | settled", force_save=True)

    simulate_with_frames(
        mujoco=mujoco,
        model=model,
        data=data,
        max_steps=final_hold_steps,
        frame_stride=max(frame_stride, 100),
        capture=lambda step: capture(f"final hold | step {step}", force_save=False),
    )
    capture("final stable wall", force_save=True)

    # Re-save GIF frame paths from memory to avoid depending on all frame PNGs when --save-frames is off.
    for path, img in zip(frame_paths, in_memory_frames):
        if not path.exists():
            img.save(path)

    renderer.close()
    manifest = {
        "purpose": "Process replay: selected stone poses from placement_log, then MuJoCo free-settle rendering.",
        "not_robot_trajectory": True,
        "xml": str(xml_path),
        "gravity": case.get("gravity"),
        "trial": int(case.get("trial", 0)),
        "target_name": case.get("target_name"),
        "strategy": case.get("strategy"),
        "success": int(case.get("success", 0)),
        "shape_success": int(case.get("shape_success", 0)),
        "rock_count": len(placement_rows),
        "effective_steps_per_rock": per_rock_steps,
        "final_hold_steps": final_hold_steps,
        "fps": fps,
        "view": "front+top" if combined_view else "front",
        "placed_order": [int(float(row["rock_index"])) for row in placement_rows],
    }
    return frame_paths, manifest


def render_algorithm_keyframes(
    output_dir: Path,
    case: dict[str, str],
    frames_dir: Path,
    width: int,
    height: int,
    fps: int,
    save_every_frame: bool,
    combined_view: bool,
    base_seed: int | None,
    target_index: int,
    strategy_index: int,
    steps_per_rock: int,
    hold_steps: int,
) -> tuple[list[Path], dict[str, Any]]:
    protocol = read_protocol_config(output_dir / "PROTOCOL.md")
    if base_seed is None:
        seed_text = protocol.get("seed", "")
        if not seed_text:
            raise SystemExit("--base-seed is required for --algorithm-keyframes because PROTOCOL.md has no seed entry.")
        base_seed = int(float(seed_text))
    steps = int(steps_per_rock or float(protocol.get("steps_per_rock", 0) or 0))
    hold = int(hold_steps or float(protocol.get("hold_steps", 0) or 0))
    if steps <= 0 or hold <= 0:
        raise SystemExit("--steps-per-rock and --hold-steps are required when PROTOCOL.md does not provide them.")

    rows = [coerce_feature_row(row) for row in read_csv(output_dir / "features.csv")]
    target_name = str(case["target_name"])
    strategy = str(case["strategy"])
    gravity = str(case["gravity"])
    trial = int(case["trial"])
    gravity_value = float(case["gravity_m_s2"])
    slots = slots_for_target(target_name)
    order_rng = np.random.default_rng(base_seed + 20011 * target_index + 503 * strategy_index + 37 * trial + int(gravity_value * 10))
    order = structured_order(rows, slots, strategy=strategy, rng=order_rng)
    snapshots: list[dict[str, Any]] = []

    detailed = run_structured_trial_detailed(
        xml_path=Path(case["xml"]),
        rows=[dict(row) for row in rows],
        slots=slots,
        order=order,
        gravity_label=gravity,
        trial_id=trial,
        seed=base_seed + 30011 * target_index + 1009 * (strategy_index + 1) + trial * 37 + int(gravity_value * 10),
        steps_per_rock=steps,
        hold_steps=hold,
        strategy=strategy,
        candidate_count=int(float(case["candidate_count"])),
        target_name=target_name,
        candidate_pose_ranker=load_ranker_from_case(case, "candidate_pose_ranker_dir", load_candidate_pose_ranker),
        candidate_pose_top_k=int(float(case.get("candidate_pose_top_k_requested") or case.get("candidate_pose_top_k") or 0)),
        pose_risk_ranker=load_ranker_from_case(case, "pose_risk_ranker_dir", load_pose_risk_ranker),
        pose_risk_weight=float(case.get("pose_risk_weight_requested") or case.get("pose_risk_weight") or 0.0),
        stone_fit_ranker=load_ranker_from_case(case, "stone_fit_ranker_dir", load_stone_fit_ranker),
        stone_fit_top_k=int(float(case.get("stone_fit_top_k_requested") or case.get("stone_fit_top_k") or 0)),
        state_snapshots=snapshots,
    )

    frame_paths = render_snapshots(
        xml_path=Path(case["xml"]),
        snapshots=detailed.get("state_snapshots", []),
        frames_dir=frames_dir,
        width=width,
        height=height,
        save_every_frame=save_every_frame,
        combined_view=combined_view,
    )
    save_snapshot_npz(frames_dir.parent / "state_snapshots.npz", detailed.get("state_snapshots", []))
    summary = detailed["summary"]
    manifest = {
        "purpose": "Algorithm keyframe replay: rerun placement algorithm and render exact after-placement states.",
        "not_robot_trajectory": True,
        "xml": str(Path(case["xml"])),
        "gravity": gravity,
        "trial": trial,
        "target_name": target_name,
        "strategy": strategy,
        "original_case_success": int(case.get("success", 0)),
        "rerun_success": int(summary.get("success", 0)),
        "rerun_shape_success": int(summary.get("shape_success", 0)),
        "rerun_stable_count": int(summary.get("stable_count", 0)),
        "rerun_failure_count": int(summary.get("failure_count", 0)),
        "rerun_stack_height_m": float(summary.get("stack_height_m", 0.0)),
        "rerun_target_rmse_xy_m": float(summary.get("target_rmse_xy_m", 0.0)),
        "rerun_max_horizontal_drift_m": float(summary.get("max_horizontal_drift_m", 0.0)),
        "rock_count": int(summary.get("rock_count", 0)),
        "steps_per_rock": steps,
        "hold_steps": hold,
        "fps": fps,
        "view": "front+top" if combined_view else "front",
        "placed_order": str(summary.get("order", "")),
    }
    return frame_paths, manifest


def read_protocol_config(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    config: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- ") or ":" not in stripped:
            continue
        key, value = stripped[2:].split(":", 1)
        config[key.strip()] = value.strip()
    return config


def coerce_feature_row(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in {"source_kind", "cluster_label"}:
            out[key] = value
        elif key in {"index", "seed", "cluster_id"}:
            out[key] = int(float(value))
        else:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def load_ranker_from_case(case: dict[str, str], key: str, loader: Any) -> dict[str, Any] | None:
    value = str(case.get(key, "") or "").strip()
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return loader(path)


def render_snapshots(
    xml_path: Path,
    snapshots: list[dict[str, Any]],
    frames_dir: Path,
    width: int,
    height: int,
    save_every_frame: bool,
    combined_view: bool,
) -> list[Path]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path.resolve()))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    frame_paths: list[Path] = []
    for frame_index, snapshot in enumerate(snapshots):
        data.qpos[:] = snapshot["qpos"]
        data.qvel[:] = snapshot["qvel"]
        mujoco.mj_forward(model, data)
        label = snapshot_label(snapshot, frame_index, len(snapshots))
        image = render_frame(renderer, data, label, width, height, combined_view=combined_view)
        frame_path = frames_dir / f"frame_{frame_index:04d}.png"
        if save_every_frame or frame_index in {0, len(snapshots) - 1}:
            image.save(frame_path)
        else:
            temp_path = frames_dir / f"_gif_frame_{frame_index:04d}.png"
            image.save(temp_path)
            frame_path = temp_path
        frame_paths.append(frame_path)
    renderer.close()
    return frame_paths


def snapshot_label(snapshot: dict[str, Any], frame_index: int, frame_count: int) -> str:
    label = str(snapshot.get("label", "state"))
    rock_index = int(snapshot.get("rock_index", -1))
    if label == "placed" and rock_index >= 0:
        return (
            f"keyframe {frame_index + 1}/{frame_count} | rock {rock_index:03d} | "
            f"course {snapshot.get('course')} {snapshot.get('role')}"
        )
    return f"keyframe {frame_index + 1}/{frame_count} | {label}"


def save_snapshot_npz(path: Path, snapshots: list[dict[str, Any]]) -> None:
    if not snapshots:
        return
    labels = np.array([str(item.get("label", "")) for item in snapshots])
    qpos = np.stack([item["qpos"] for item in snapshots], axis=0)
    qvel = np.stack([item["qvel"] for item in snapshots], axis=0)
    rock_index = np.array([int(item.get("rock_index", -1)) for item in snapshots], dtype=np.int32)
    slot_id = np.array([int(item.get("slot_id", -1)) for item in snapshots], dtype=np.int32)
    course = np.array([int(item.get("course", -1)) for item in snapshots], dtype=np.int32)
    np.savez_compressed(path, labels=labels, qpos=qpos, qvel=qvel, rock_index=rock_index, slot_id=slot_id, course=course)


def hide_unused_rocks(model: Any, data: Any, features: list[dict[str, str]]) -> None:
    for offset, row in enumerate(features):
        idx = int(float(row["index"]))
        pos = np.array([20.0 + 0.20 * (offset % 12), 20.0 + 0.20 * (offset // 12), 0.35], dtype=float)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        _set_freejoint_pose(model, data, idx, pos, quat)


def simulate_with_frames(
    mujoco: Any,
    model: Any,
    data: Any,
    max_steps: int,
    frame_stride: int,
    capture: Any,
) -> None:
    min_steps = min(max_steps, 250)
    for step in range(1, max_steps + 1):
        mujoco.mj_step(model, data)
        if step % frame_stride == 0:
            capture(step)
        if step >= min_steps and step % 100 == 0:
            if float(np.linalg.norm(data.qvel, ord=np.inf)) < 0.035:
                break


def render_frame(
    renderer: Any,
    data: Any,
    label: str,
    width: int,
    height: int,
    combined_view: bool,
) -> Image.Image:
    import mujoco

    front = render_camera(renderer, data, azimuth=0.0, elevation=-12.0, distance=1.45, lookat_z=0.17)
    if combined_view:
        top = render_camera(renderer, data, azimuth=0.0, elevation=-88.0, distance=1.05, lookat_z=0.16)
        image = Image.new("RGB", (width * 2, height), "black")
        image.paste(front, (0, 0))
        image.paste(top, (width, 0))
    else:
        image = front
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
    draw.text((8, 7), label, fill=(238, 232, 210), font=font)
    if combined_view:
        draw.text((8, height - 18), "front", fill=(238, 232, 210), font=font)
        draw.text((width + 8, height - 18), "top", fill=(238, 232, 210), font=font)
    return image


def render_camera(renderer: Any, data: Any, azimuth: float, elevation: float, distance: float, lookat_z: float) -> Image.Image:
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array([0.0, 0.0, lookat_z])
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    return Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")


def write_gif(frame_paths: list[Path], gif_path: Path, fps: int) -> None:
    frames = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
    if not frames:
        raise RuntimeError("No frames were rendered.")
    duration_ms = max(1, int(round(1000 / max(fps, 1))))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
