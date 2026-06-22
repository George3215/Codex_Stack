from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moon_rock_stack.run_experiment import write_csv
from moon_rock_stack.simulate import run_trial_detailed


def main() -> int:
    args = parse_args()
    output_dir = args.output.resolve()
    captures_dir = output_dir / args.capture_dir_name
    captures_dir.mkdir(parents=True, exist_ok=True)

    features = read_csv(output_dir / "features.csv")
    rows = [coerce_feature_row(row) for row in features]
    results = read_csv(output_dir / "results.csv")
    selected = select_cases(
        results,
        max_success=args.max_success,
        max_failure=args.max_failure,
        per_target=args.per_target,
    )
    if not selected:
        print(f"No cases selected from {output_dir / 'results.csv'}")
        return 0

    strategy_positions = first_seen_strategy_positions(results)
    capture_rows: list[dict[str, Any]] = []
    for case_id, result in enumerate(selected):
        strategy = result["strategy"]
        gravity = result["gravity"]
        trial = int(result["trial"])
        xml_path = Path(result["xml"])
        state_value = result.get("state_path", "")
        state_path = Path(state_value) if state_value else Path("__missing_state__.npz")
        if not state_path.exists():
            existing_state = find_existing_replayed_state(output_dir, strategy, gravity, trial)
            if existing_state is not None:
                state_path = existing_state
        if state_value and state_path.exists():
            qpos, qvel = load_state(state_path)
        elif not state_value and state_path.exists():
            qpos, qvel = load_state(state_path)
        else:
            order = [int(token) for token in result["order"].split()]
            strategy_position = strategy_positions.get(strategy, 0)
            seed = args.seed + args.strategy_offset * (strategy_position + 1) + trial * 37 + int(float(result["gravity_m_s2"]) * 10)
            detailed = run_trial_detailed(
                xml_path=xml_path,
                rows=[dict(row) for row in rows],
                order=order,
                gravity_label=gravity,
                trial_id=trial,
                seed=seed,
                steps_per_rock=args.steps_per_rock,
                hold_steps=args.hold_steps,
                strategy=strategy,
                candidate_count=int(result["candidate_count"]),
                stack_rocks=int(result["rock_count"]),
            )
            qpos = detailed["state"]["qpos"]
            qvel = detailed["state"]["qvel"]
            state_path = captures_dir / f"replayed_state_{case_id:02d}_{strategy}_{gravity}_trial_{trial:02d}.npz"
            np.savez_compressed(state_path, qpos=qpos, qvel=qvel, strategy=strategy, gravity=gravity, trial=trial)

        label = "success" if int(result["success"]) == 1 else "failure"
        target_name = result.get("target_name", "structured")
        case_dir = captures_dir / f"{case_id:02d}_{target_name}_{label}_{strategy}_{gravity}_trial_{trial:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        rendered = render_case(xml_path, qpos, qvel, case_dir, width=args.width, height=args.height)
        capture_rows.append(
            {
                "case_id": case_id,
                "target_name": target_name,
                "label": label,
                "strategy": strategy,
                "gravity": gravity,
                "trial": trial,
                "success": result["success"],
                "stable_count": result["stable_count"],
                "failure_count": result["failure_count"],
                "stack_height_m": result["stack_height_m"],
                "xml": str(xml_path),
                "state": str(state_path),
                "capture_dir": str(case_dir),
                "images": " ".join(rendered),
            }
        )

    write_csv(captures_dir / "capture_manifest.csv", capture_rows)
    print(captures_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture RGB/depth images for selected MuJoCo stacking cases.")
    parser.add_argument("--output", type=Path, required=True, help="Experiment output directory containing results.csv.")
    parser.add_argument("--max-success", type=int, default=2, help="Number of successful cases to capture.")
    parser.add_argument("--max-failure", type=int, default=2, help="Number of failure cases to capture.")
    parser.add_argument("--seed", type=int, default=7, help="Original experiment seed.")
    parser.add_argument("--strategy-offset", type=int, default=1009, help="Strategy seed offset used by run_experiment.")
    parser.add_argument("--steps-per-rock", type=int, default=2200, help="Replay steps per rock if no state file exists.")
    parser.add_argument("--hold-steps", type=int, default=4200, help="Replay hold steps if no state file exists.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--capture-dir-name", default="captures", help="Subdirectory name for rendered images.")
    parser.add_argument("--per-target", action="store_true", help="Select success/failure examples separately for each target.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def select_cases(
    results: list[dict[str, str]],
    max_success: int,
    max_failure: int,
    per_target: bool = False,
) -> list[dict[str, str]]:
    if per_target:
        selected: list[dict[str, str]] = []
        targets = sorted({row.get("target_name", "structured") for row in results})
        for target_name in targets:
            selected.extend(
                select_cases(
                    [row for row in results if row.get("target_name", "structured") == target_name],
                    max_success=max_success,
                    max_failure=max_failure,
                    per_target=False,
                )
            )
        return selected

    successes = [row for row in results if int(row["success"]) == 1]
    successes = sorted(successes, key=lambda row: (-float(row["stack_height_m"]), float(row["velocity_inf_norm"])))
    failures = [row for row in results if int(row["success"]) == 0]
    failures = sorted(
        failures,
        key=lambda row: (-float(row["failure_count"]), -float(row["max_radial_distance_m"]), -float(row["velocity_inf_norm"])),
    )
    selected: list[dict[str, str]] = []
    selected.extend(successes[:max_success])
    selected.extend(failures[:max_failure])
    return selected


def first_seen_strategy_positions(results: list[dict[str, str]]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for row in results:
        strategy = row["strategy"]
        if strategy not in positions:
            positions[strategy] = len(positions)
    return positions


def load_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["qpos"].copy(), data["qvel"].copy()


def find_existing_replayed_state(output_dir: Path, strategy: str, gravity: str, trial: int) -> Path | None:
    pattern = f"captures*/replayed_state_*_{strategy}_{gravity}_trial_{trial:02d}.npz"
    matches = sorted(output_dir.glob(pattern))
    return matches[0] if matches else None


def render_case(xml_path: Path, qpos: np.ndarray, qvel: np.ndarray, case_dir: Path, width: int, height: int) -> list[str]:
    import matplotlib.pyplot as plt
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    camera_specs = [
        ("front", 0.0, -25.0, 1.15),
        # Wall targets extend laterally in the x direction; azimuth=90 is the
        # more useful face-on view, while azimuth=0 is closer to an end view.
        ("wall_front", 90.0, -12.0, 1.45),
        ("right", 90.0, -25.0, 1.15),
        ("back", 180.0, -25.0, 1.15),
        ("left", 270.0, -25.0, 1.15),
        ("top", 45.0, -75.0, 1.05),
        ("wall_top", 0.0, -88.0, 1.05),
    ]
    written: list[str] = []
    for name, azimuth, elevation, distance in camera_specs:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = np.array([0.0, 0.0, 0.16])
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation

        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        rgb_path = case_dir / f"{name}_rgb.png"
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(rgb_path, rgb)
        written.append(str(rgb_path))

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam)
        depth = renderer.render()
        depth_npy_path = case_dir / f"{name}_depth.npy"
        depth_npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(depth_npy_path, depth)
        depth_png_path = case_dir / f"{name}_depth.png"
        depth_vis = normalize_depth(depth)
        plt.imsave(depth_png_path, depth_vis, cmap="magma")
        written.extend([str(depth_png_path), str(depth_npy_path)])

        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)
        segmentation = renderer.render()
        renderer.disable_segmentation_rendering()
        object_depth = np.where(rock_segmentation_mask(segmentation), depth, np.nan)
        object_depth_npy_path = case_dir / f"{name}_object_depth.npy"
        object_depth_npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(object_depth_npy_path, object_depth)
        object_depth_png_path = case_dir / f"{name}_object_depth.png"
        object_depth_vis = normalize_depth(object_depth)
        plt.imsave(object_depth_png_path, object_depth_vis, cmap="magma")
        written.extend([str(object_depth_png_path), str(object_depth_npy_path)])

    write_wall_camera_aliases(case_dir, written)
    renderer.close()
    return written


def write_wall_camera_aliases(case_dir: Path, written: list[str]) -> None:
    """Write explicit wall-camera filenames expected by wall experiments."""
    alias_specs = [
        ("wall_front_rgb.png", "front_rgb.png"),
        ("wall_front_depth.png", "front_depth.png"),
        ("wall_front_depth.npy", "front_depth.npy"),
        ("wall_front_object_depth.png", "front_object_depth.png"),
        ("wall_front_object_depth.npy", "front_object_depth.npy"),
        ("wall_top_depth.png", "top_depth.png"),
        ("wall_top_depth.npy", "top_depth.npy"),
        ("wall_top_object_depth.png", "top_object_depth.png"),
        ("wall_top_object_depth.npy", "top_object_depth.npy"),
    ]
    for alias_name, fallback_name in alias_specs:
        alias_path = case_dir / alias_name
        if alias_path.exists():
            written.append(str(alias_path))
            continue
        source_path = case_dir / fallback_name
        if not source_path.exists():
            continue
        alias_path.write_bytes(source_path.read_bytes())
        written.append(str(alias_path))


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros_like(depth)
    far_background = _far_background_mask(depth, finite)
    valid = finite & ~far_background
    if not valid.any():
        valid = finite
    values = depth[valid]
    lo = float(np.percentile(values, 2))
    hi = float(np.percentile(values, 98))
    if hi <= lo:
        hi = lo + 1e-6
    clipped = np.clip(depth, lo, hi)
    normalized = 1.0 - (clipped - lo) / (hi - lo)
    normalized[~valid] = 0.0
    return normalized


def rock_segmentation_mask(segmentation: np.ndarray) -> np.ndarray:
    if segmentation.ndim != 3 or segmentation.shape[-1] < 2:
        return np.zeros(segmentation.shape[:2], dtype=bool)
    obj_id = segmentation[:, :, 0]
    obj_type = segmentation[:, :, 1]
    return (obj_type == 5) & (obj_id > 0)


def _far_background_mask(depth: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """Detect MuJoCo far-plane background so depth PNGs use object/floor scale."""
    mask = np.zeros_like(depth, dtype=bool)
    values = depth[finite]
    if values.size == 0:
        return mask
    far_value = float(np.max(values))
    near_value = float(np.min(values))
    if far_value <= near_value:
        return mask
    tolerance = max(1e-3, abs(far_value) * 1e-6)
    far_pixels = finite & (depth >= far_value - tolerance)
    far_fraction = float(np.mean(far_pixels))
    if far_fraction < 0.01:
        return mask
    dynamic_range = far_value - near_value
    near_range = float(np.percentile(values[values < far_value - tolerance], 98) - near_value) if np.any(values < far_value - tolerance) else 0.0
    if dynamic_range > max(10.0 * near_range, 10.0):
        mask = far_pixels
    return mask


if __name__ == "__main__":
    raise SystemExit(main())
