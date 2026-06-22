from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


GRAVITIES = {
    "earth": 9.80665,
    "moon": 1.624,
}


def stacking_order(
    rows: list[dict[str, Any]],
    strategy: str = "paper_baseline",
    rng: np.random.Generator | None = None,
    risk_by_cluster: dict[str, float] | None = None,
    stack_rocks: int | None = None,
) -> list[int]:
    selected = list(rows)
    if strategy == "random":
        if rng is None:
            rng = np.random.default_rng(0)
        rng.shuffle(selected)
    elif strategy == "risk_aware":
        risk_by_cluster = risk_by_cluster or {}
        selected = sorted(selected, key=lambda row: _risk_aware_key(row, risk_by_cluster))
    elif strategy in {"paper_baseline", "physics_filter", "support_first"}:
        selected = sorted(selected, key=_paper_baseline_key)
    elif strategy == "strength":
        selected = sorted(selected, key=_strength_key)
    else:
        raise ValueError(f"Unknown stacking strategy: {strategy}")

    if stack_rocks is not None:
        selected = selected[:stack_rocks]
    return [int(row["index"]) for row in selected]


def _paper_baseline_key(row: dict[str, Any]) -> tuple[int, float, float]:
    priority = {
        "wedge_or_broad_clast": 0,
        "subangular_block": 1,
        "equant_clast": 2,
        "angular_clast": 3,
        "fractured_clast": 4,
        "elongated_clast": 5,
        "spiky_reject": 9,
    }
    label_priority = _cluster_priority(str(row["cluster_label"]), priority)
    return (label_priority, -float(row["stability_score"]), float(row["volume"]))


def _strength_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (-float(row["stability_score"]), float(row["roughness"]), -float(row["volume"]))


def _risk_aware_key(row: dict[str, Any], risk_by_cluster: dict[str, float]) -> tuple[float, int, float, float]:
    label = str(row["cluster_label"])
    intrinsic = _intrinsic_geometry_risk(row)
    if label in risk_by_cluster:
        risk = 0.72 * risk_by_cluster[label] + 0.28 * intrinsic
    else:
        risk = intrinsic
    return (risk, *_paper_baseline_key(row))


def _intrinsic_geometry_risk(row: dict[str, Any]) -> float:
    label = str(row["cluster_label"])
    roughness = float(row["roughness"])
    elongation = float(row["elongation"])
    sphericity = float(row["sphericity"])
    angularity = float(row.get("angularity", 0.0))
    compactness = float(row.get("compactness", 0.0))
    spike_score = float(row.get("spike_score", 0.0))
    risk = 0.35 * min(roughness / 0.25, 1.5)
    risk += 0.18 * max(0.0, elongation - 1.45)
    risk += 0.25 * max(0.0, 0.86 - sphericity)
    risk += 0.22 * min(angularity / 0.22, 1.4)
    risk += 0.18 * max(0.0, 0.32 - compactness) / 0.32
    risk += 0.55 * min(spike_score / 0.18, 1.6)
    if label.startswith("spiky_reject"):
        risk += 1.0
    elif label.startswith("angular_clast"):
        risk += 0.12
    elif label.startswith("fractured_clast"):
        risk += 0.12
    elif label.startswith("elongated_clast"):
        risk += 0.18
    elif label.startswith("wedge_or_broad_clast"):
        risk -= 0.08
    return max(0.0, float(risk))


def _cluster_priority(label: str, priority: dict[str, int]) -> int:
    for prefix, value in priority.items():
        if label.startswith(prefix):
            return value
    return 5


def run_trial(
    xml_path: Path,
    rows: list[dict[str, Any]],
    order: list[int],
    gravity_label: str,
    trial_id: int,
    seed: int,
    steps_per_rock: int,
    hold_steps: int,
    strategy: str = "paper_baseline",
    candidate_count: int = 1,
    stack_rocks: int | None = None,
) -> dict[str, Any]:
    result = run_trial_detailed(
        xml_path=xml_path,
        rows=rows,
        order=order,
        gravity_label=gravity_label,
        trial_id=trial_id,
        seed=seed,
        steps_per_rock=steps_per_rock,
        hold_steps=hold_steps,
        strategy=strategy,
        candidate_count=candidate_count,
        stack_rocks=stack_rocks,
    )
    return result["summary"]


def run_trial_detailed(
    xml_path: Path,
    rows: list[dict[str, Any]],
    order: list[int],
    gravity_label: str,
    trial_id: int,
    seed: int,
    steps_per_rock: int,
    hold_steps: int,
    strategy: str,
    candidate_count: int,
    stack_rocks: int | None,
) -> dict[str, Any]:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo Python is not installed. Activate the Conda env and install dependencies first."
        ) from exc

    rng = np.random.default_rng(seed + trial_id * 9973 + int(GRAVITIES[gravity_label] * 100))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    row_by_index = {int(row["index"]): row for row in rows}
    placed: list[int] = []
    placement_rows: list[dict[str, Any]] = []

    mujoco.mj_forward(model, data)
    if stack_rocks is not None:
        order = order[:stack_rocks]
    for stack_level, rock_index in enumerate(order):
        selected = _place_with_candidate_search(
            mujoco=mujoco,
            model=model,
            data=data,
            row_by_index=row_by_index,
            placed=placed,
            rock_index=rock_index,
            stack_level=stack_level,
            rng=rng,
            steps_per_rock=steps_per_rock,
            candidate_count=max(candidate_count, 1),
            strategy=strategy,
        )
        placed.append(rock_index)
        selected.update(
            {
                "gravity": gravity_label,
                "trial": trial_id,
                "strategy": strategy,
                "rock_index": rock_index,
                "stack_level": stack_level,
                "cluster_label": row_by_index[rock_index]["cluster_label"],
                "source_kind": row_by_index[rock_index]["source_kind"],
            }
        )
        placement_rows.append(selected)

    before = _body_positions(model, data, placed)
    _simulate_until_quiet(mujoco, model, data, hold_steps)
    after = _body_positions(model, data, placed)
    speeds = np.linalg.norm(data.qvel.reshape(-1)[:], ord=np.inf)
    drifts = [float(np.linalg.norm(after[idx][:2] - before[idx][:2])) for idx in placed]
    radial = [float(np.linalg.norm(after[idx][:2])) for idx in placed]
    stable_flags = []
    for idx in placed:
        row = row_by_index[idx]
        z = float(after[idx][2])
        stable_flags.append(z > 0.35 * float(row["bbox_z"]) and float(np.linalg.norm(after[idx][:2])) < 0.65)
    stable_by_index = {idx: bool(flag) for idx, flag in zip(placed, stable_flags)}
    top_z = _current_stack_top(model, data, row_by_index, placed)
    max_drift = max(drifts) if drifts else 0.0
    stable_count = int(sum(stable_flags))
    success = stable_count == len(order) and max_drift < 0.12 and speeds < 0.25

    failure_rows = _failure_rows(
        gravity_label=gravity_label,
        trial_id=trial_id,
        strategy=strategy,
        row_by_index=row_by_index,
        placed=placed,
        before=before,
        after=after,
        stable_by_index=stable_by_index,
    )

    summary = {
        "gravity": gravity_label,
        "gravity_m_s2": GRAVITIES[gravity_label],
        "trial": trial_id,
        "strategy": strategy,
        "candidate_count": candidate_count,
        "rock_count": len(order),
        "stable_count": stable_count,
        "failure_count": len(order) - stable_count,
        "success": int(success),
        "stack_height_m": float(top_z),
        "max_horizontal_drift_m": float(max_drift),
        "max_radial_distance_m": float(max(radial) if radial else 0.0),
        "velocity_inf_norm": float(speeds),
        "order": " ".join(f"{idx:03d}" for idx in order),
        "xml": str(xml_path),
    }
    return {
        "summary": summary,
        "placements": placement_rows,
        "failures": failure_rows,
        "state": {
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
        },
    }


def _place_with_candidate_search(
    mujoco: Any,
    model: Any,
    data: Any,
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    rock_index: int,
    stack_level: int,
    rng: np.random.Generator,
    steps_per_rock: int,
    candidate_count: int,
    strategy: str,
) -> dict[str, Any]:
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()
    best_score = float("inf")
    best_qpos = qpos0.copy()
    best_qvel = qvel0.copy()
    best_metrics: dict[str, Any] = {}

    for candidate_id in range(candidate_count):
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        candidate = _candidate_pose(
            row_by_index[rock_index], row_by_index, placed, stack_level, candidate_id, rng, strategy
        )
        _set_freejoint_pose(model, data, rock_index, candidate["pos"], candidate["quat"])
        mujoco.mj_forward(model, data)
        _simulate_until_quiet(mujoco, model, data, steps_per_rock)
        metrics = _candidate_metrics(model, data, row_by_index, placed, rock_index)
        score = _candidate_score(metrics, stack_level, strategy)
        if score < best_score:
            best_score = score
            best_qpos = data.qpos.copy()
            best_qvel = data.qvel.copy()
            best_metrics = {**candidate, **metrics, "candidate_id": candidate_id, "candidate_score": score}

    data.qpos[:] = best_qpos
    data.qvel[:] = best_qvel
    mujoco.mj_forward(model, data)
    if "settled_z" in best_metrics:
        row_by_index[rock_index]["last_top_z"] = float(best_metrics["settled_z"]) + 0.5 * float(
            row_by_index[rock_index]["bbox_z"]
        )
    return _serializable_candidate(best_metrics)


def _candidate_pose(
    row: dict[str, Any],
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    stack_level: int,
    candidate_id: int,
    rng: np.random.Generator,
    strategy: str,
) -> dict[str, Any]:
    max_z = 0.0
    center = np.zeros(2)
    if placed:
        # Use the geometric center of the placed stack as a simple online correction.
        max_z = max(float(row_by_index[idx]["last_top_z"]) for idx in placed if "last_top_z" in row_by_index[idx])
    half_height = 0.5 * float(row["bbox_z"])
    if strategy == "support_first":
        radius_limit = 0.018 + 0.0025 * stack_level
    else:
        radius_limit = 0.035 + 0.006 * stack_level
    radius = 0.0 if candidate_id == 0 else rng.uniform(0.0, radius_limit)
    angle = rng.uniform(0.0, 2.0 * math.pi)
    offset = np.array([math.cos(angle), math.sin(angle)]) * radius
    drop_clearance = 0.075 if strategy == "support_first" else 0.10
    pos = np.array([center[0] + offset[0], center[1] + offset[1], max_z + half_height + drop_clearance], dtype=float)
    label = str(row["cluster_label"])
    is_base_friendly = label.startswith("wedge_or_broad_clast") or label.startswith("subangular_block")
    if strategy == "support_first":
        tilt = 0.10 if is_base_friendly else 0.20
    else:
        tilt = 0.16 if is_base_friendly else 0.34
    quat = _random_quaternion(rng, max_tilt=tilt)
    return {"pos": pos, "quat": quat}


def _candidate_metrics(
    model: Any,
    data: Any,
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    rock_index: int,
) -> dict[str, float]:
    positions = _body_positions(model, data, placed + [rock_index])
    pos = positions[rock_index]
    row = row_by_index[rock_index]
    top_z = float(pos[2]) + 0.5 * float(row["bbox_z"])
    row_by_index[rock_index]["last_top_z"] = top_z
    radial = float(np.linalg.norm(pos[:2]))
    speed = float(np.linalg.norm(data.qvel, ord=np.inf))
    support_overlap = _support_overlap_score(positions, row_by_index, placed, rock_index)
    height_gain = top_z - _current_stack_top(model, data, row_by_index, placed)
    return {
        "settled_x": float(pos[0]),
        "settled_y": float(pos[1]),
        "settled_z": float(pos[2]),
        "radial_distance_m": radial,
        "velocity_inf_norm_after_place": speed,
        "support_overlap": support_overlap,
        "height_gain_m": float(height_gain),
    }


def _support_overlap_score(
    positions: dict[int, np.ndarray],
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    rock_index: int,
) -> float:
    if not placed:
        return 1.0
    pos = positions[rock_index]
    row = row_by_index[rock_index]
    half_xy = 0.25 * (float(row["bbox_x"]) + float(row["bbox_y"]))
    overlaps = []
    for idx in placed:
        base = positions[idx]
        base_row = row_by_index[idx]
        base_half_xy = 0.25 * (float(base_row["bbox_x"]) + float(base_row["bbox_y"]))
        distance = float(np.linalg.norm(pos[:2] - base[:2]))
        overlaps.append(max(0.0, 1.0 - distance / max(half_xy + base_half_xy, 1e-6)))
    return float(max(overlaps) if overlaps else 0.0)


def _candidate_score(metrics: dict[str, float], stack_level: int, strategy: str) -> float:
    if strategy == "support_first":
        return (
            6.5 * metrics["radial_distance_m"]
            + 3.0 * metrics["velocity_inf_norm_after_place"]
            - 1.8 * metrics["support_overlap"]
            - 0.08 * metrics["height_gain_m"]
            + 0.018 * stack_level
        )
    return (
        3.0 * metrics["radial_distance_m"]
        + 1.8 * metrics["velocity_inf_norm_after_place"]
        - 0.7 * metrics["support_overlap"]
        - 0.4 * metrics["height_gain_m"]
        + 0.01 * stack_level
    )


def _serializable_candidate(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    pos = result.pop("pos", np.zeros(3))
    quat = result.pop("quat", np.array([1.0, 0.0, 0.0, 0.0]))
    result.update(
        {
            "placed_x": float(pos[0]),
            "placed_y": float(pos[1]),
            "placed_z": float(pos[2]),
            "quat_w": float(quat[0]),
            "quat_x": float(quat[1]),
            "quat_y": float(quat[2]),
            "quat_z": float(quat[3]),
        }
    )
    return result


def _failure_rows(
    gravity_label: str,
    trial_id: int,
    strategy: str,
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    before: dict[int, np.ndarray],
    after: dict[int, np.ndarray],
    stable_by_index: dict[int, bool],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stack_level, idx in enumerate(placed):
        if stable_by_index[idx]:
            continue
        row = row_by_index[idx]
        horizontal_drift = float(np.linalg.norm(after[idx][:2] - before[idx][:2]))
        radial = float(np.linalg.norm(after[idx][:2]))
        z = float(after[idx][2])
        reason = []
        if radial >= 0.65:
            reason.append("left_stack_radius")
        if z <= 0.35 * float(row["bbox_z"]):
            reason.append("low_or_fallen")
        if not reason:
            reason.append("unstable_after_hold")
        rows.append(
            {
                "gravity": gravity_label,
                "trial": trial_id,
                "strategy": strategy,
                "rock_index": idx,
                "stack_level": stack_level,
                "cluster_label": row["cluster_label"],
                "source_kind": row["source_kind"],
                "failure_reason": "+".join(reason),
                "final_x": float(after[idx][0]),
                "final_y": float(after[idx][1]),
                "final_z": z,
                "horizontal_drift_m": horizontal_drift,
                "radial_distance_m": radial,
                "volume": float(row["volume"]),
                "roughness": float(row["roughness"]),
                "flatness": float(row["flatness"]),
                "elongation": float(row["elongation"]),
                "stability_score": float(row["stability_score"]),
            }
        )
    return rows


def _set_freejoint_pose(model: Any, data: Any, rock_index: int, pos: np.ndarray, quat: np.ndarray) -> None:
    import mujoco

    joint_name = f"rock_{rock_index:03d}_free"
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"Joint not found: {joint_name}")
    qpos_addr = int(model.jnt_qposadr[joint_id])
    qvel_addr = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_addr : qpos_addr + 3] = pos
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0


def _body_positions(model: Any, data: Any, indices: list[int]) -> dict[int, np.ndarray]:
    import mujoco

    positions: dict[int, np.ndarray] = {}
    for idx in indices:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"rock_{idx:03d}")
        positions[idx] = data.xpos[body_id].copy()
    return positions


def _current_stack_top(model: Any, data: Any, row_by_index: dict[int, dict[str, Any]], indices: list[int]) -> float:
    if not indices:
        return 0.0
    positions = _body_positions(model, data, indices)
    top = 0.0
    for idx in indices:
        top = max(top, float(positions[idx][2]) + 0.5 * float(row_by_index[idx]["bbox_z"]))
    return top


def _simulate_until_quiet(mujoco: Any, model: Any, data: Any, max_steps: int) -> None:
    min_steps = min(max_steps, 250)
    for step in range(max_steps):
        mujoco.mj_step(model, data)
        if step >= min_steps and step % 100 == 0:
            if float(np.linalg.norm(data.qvel, ord=np.inf)) < 0.035:
                break


def _random_quaternion(rng: np.random.Generator, max_tilt: float) -> np.ndarray:
    yaw = rng.uniform(0.0, 2.0 * math.pi)
    tilt_axis = rng.uniform(0.0, 2.0 * math.pi)
    tilt = rng.uniform(0.0, max_tilt)
    q_yaw = _axis_angle(np.array([0.0, 0.0, 1.0]), yaw)
    q_tilt = _axis_angle(np.array([math.cos(tilt_axis), math.sin(tilt_axis), 0.0]), tilt)
    q = _quat_mul(q_yaw, q_tilt)
    return q / np.linalg.norm(q)


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    half = 0.5 * angle
    return np.array([math.cos(half), *(math.sin(half) * axis)], dtype=float)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )
