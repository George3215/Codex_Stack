from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from moon_rock_stack.features import FEATURE_COLUMNS, extract_features
from moon_rock_stack.fractal_rocks import RockMesh


FEATURE_PREFIX = "rock_"
MESH_BACKFILL_COLUMNS = tuple(FEATURE_COLUMNS) + ("mass",)


def main() -> int:
    args = parse_args()
    batch_root = args.batch_root.resolve()
    run_dirs = discover_run_dirs(batch_root, args.run)
    if not run_dirs:
        raise SystemExit(f"No run directories with results.csv found under {batch_root}")

    output_dir = unique_dir(args.output.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)

    run_rows: list[dict[str, Any]] = []
    placement_rows: list[dict[str, Any]] = []
    candidate_pose_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for run_dir in run_dirs:
        run_results = read_csv(run_dir / "results.csv")
        if not run_results:
            continue
        run_id = run_dir.name
        feature_rows = index_by_int(read_csv(run_dir / "features.csv"), "index")
        backfill_feature_rows_from_meshes(run_dir, feature_rows)
        slot_rows_by_target = read_target_slots(run_dir)
        failure_map = read_failure_map(run_dir)

        for row in run_results:
            run_rows.append(flatten_run_row(run_dir, row))

        placements = read_csv(run_dir / "placement_log.csv")
        if not placements:
            warnings.append(f"{run_id}: missing placement_log.csv")
        for row in placements:
            placement_rows.append(
                build_placement_example(
                    run_dir=run_dir,
                    run_id=run_id,
                    row=row,
                    feature_rows=feature_rows,
                    failure_map=failure_map,
                    result_rows=run_results,
                )
            )

        candidate_pose_log = read_csv(run_dir / "candidate_pose_log.csv")
        run_candidate_pose_rows = build_candidate_pose_examples(
            run_dir=run_dir,
            run_id=run_id,
            feature_rows=feature_rows,
            candidate_pose_rows=candidate_pose_log,
            placements=placements,
            failure_map=failure_map,
        )
        candidate_pose_rows.extend(run_candidate_pose_rows)

        assignment_rows.extend(
            build_assignment_examples(
                run_dir=run_dir,
                run_id=run_id,
                feature_rows=feature_rows,
                slot_rows_by_target=slot_rows_by_target,
                placements=placements,
                candidate_pose_examples=run_candidate_pose_rows,
            )
        )

    write_csv(output_dir / "run_examples.csv", run_rows)
    write_csv(output_dir / "placement_examples.csv", placement_rows)
    write_csv(output_dir / "candidate_pose_examples.csv", candidate_pose_rows)
    write_csv(output_dir / "assignment_candidate_examples.csv", assignment_rows)
    write_jsonl(output_dir / "placement_examples.jsonl", placement_rows)
    write_jsonl(output_dir / "candidate_pose_examples.jsonl", candidate_pose_rows)
    write_summary(output_dir, batch_root, run_dirs, run_rows, placement_rows, candidate_pose_rows, assignment_rows, warnings)
    write_readme(output_dir, batch_root, run_dirs, warnings)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MoonStack learning tables from MuJoCo run logs.")
    parser.add_argument("--batch-root", type=Path, default=Path("batch_runs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        type=Path,
        default=[],
        help="Optional run directory. Can be passed multiple times. Defaults to all runs under --batch-root.",
    )
    return parser.parse_args()


def discover_run_dirs(batch_root: Path, requested_runs: list[Path]) -> list[Path]:
    if requested_runs:
        dirs = [(path if path.is_absolute() else batch_root / path).resolve() for path in requested_runs]
        return sorted(path for path in dirs if (path / "results.csv").exists())
    return sorted(path.parent for path in batch_root.glob("**/results.csv"))


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def collect_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames or ["empty"]


def index_by_int(rows: list[dict[str, str]], key: str) -> dict[int, dict[str, str]]:
    indexed: dict[int, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if value in {"", None}:
            continue
        indexed[int(float(value))] = row
    return indexed


def backfill_feature_rows_from_meshes(run_dir: Path, feature_rows: dict[int, dict[str, str]]) -> None:
    mesh_dir = run_dir / "meshes"
    if not mesh_dir.exists():
        return
    for rock_index, row in feature_rows.items():
        missing = [column for column in MESH_BACKFILL_COLUMNS if row.get(column, "") in {"", None}]
        if not missing:
            continue
        mesh_path = mesh_dir / f"rock_{rock_index:03d}.obj"
        if not mesh_path.exists():
            continue
        vertices, faces = read_obj_mesh(mesh_path)
        if len(vertices) == 0 or len(faces) == 0:
            continue
        rock = RockMesh(
            index=rock_index,
            kind=str(row.get("source_kind", "")),
            vertices=vertices,
            faces=faces,
            seed=parse_int(row.get("seed", "")) or -1,
        )
        derived = extract_features(rock)
        for column in missing:
            value = derived.get(column)
            if value is not None:
                row[column] = str(value)


def read_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                parts = line.split()[1:4]
                faces.append([int(part.split("/")[0]) - 1 for part in parts])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def read_target_slots(run_dir: Path) -> dict[str, dict[int, dict[str, str]]]:
    slots: dict[str, dict[int, dict[str, str]]] = {}
    for path in run_dir.glob("target_slots_*.csv"):
        target_name = path.stem.removeprefix("target_slots_")
        slots[target_name] = index_by_int(read_csv(path), "slot_id")
    return slots


def read_failure_map(run_dir: Path) -> dict[tuple[str, str, str, str, str, str], dict[str, str]]:
    failures: dict[tuple[str, str, str, str, str, str], dict[str, str]] = {}
    for row in read_csv(run_dir / "failure_cases.csv"):
        failures[failure_key(row)] = row
    return failures


def failure_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row.get("target_name", "stack"),
        row.get("strategy", ""),
        row.get("gravity", ""),
        row.get("trial", ""),
        row.get("slot_id", ""),
        row.get("rock_index", ""),
    )


def flatten_run_row(run_dir: Path, row: dict[str, str]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "run_name": run_dir.name,
        "run_path": str(run_dir),
    }
    output.update(row)
    return output


def build_placement_example(
    run_dir: Path,
    run_id: str,
    row: dict[str, str],
    feature_rows: dict[int, dict[str, str]],
    failure_map: dict[tuple[str, str, str, str, str, str], dict[str, str]],
    result_rows: list[dict[str, str]],
) -> dict[str, Any]:
    rock_index = parse_int(row.get("rock_index", ""))
    fallback_rock_index = parse_int(row.get("best_rejected_rock_index", ""))
    assignment_rock_index = parse_int(row.get("assignment_candidate_rock_index", ""))
    candidate_rock_index = first_nonnegative(rock_index, fallback_rock_index, assignment_rock_index)
    failed = failure_map.get(failure_key(row))
    skipped = parse_bool(row.get("placement_skipped", ""))
    result = find_result_for_placement(result_rows, row)

    output: dict[str, Any] = {
        "example_id": make_example_id(run_id, row),
        "run_name": run_id,
        "run_path": str(run_dir),
        "target_name": row.get("target_name", "stack"),
        "strategy": row.get("strategy", ""),
        "gravity": row.get("gravity", ""),
        "gravity_m_s2": result.get("gravity_m_s2", ""),
        "trial": row.get("trial", ""),
        "slot_id": row.get("slot_id", ""),
        "course": row.get("course", ""),
        "role": row.get("role", ""),
        "candidate_rock_index": candidate_rock_index if candidate_rock_index is not None else "",
        "committed_rock_index": rock_index if rock_index is not None and rock_index >= 0 else "",
        "is_skipped_slot": int(skipped),
        "is_failure_case": int(failed is not None),
        "label_success": int((not skipped) and failed is None and rock_index is not None and rock_index >= 0),
        "label_run_success": result.get("success", ""),
        "label_run_shape_success": result.get("shape_success", ""),
        "failure_reason": failure_reason(row, failed),
        "target_x": row.get("target_x", ""),
        "target_y": row.get("target_y", ""),
        "placed_x": row.get("placed_x", ""),
        "placed_y": row.get("placed_y", ""),
        "placed_z": row.get("placed_z", ""),
        "quat_w": row.get("quat_w", ""),
        "quat_x": row.get("quat_x", ""),
        "quat_y": row.get("quat_y", ""),
        "quat_z": row.get("quat_z", ""),
        "settled_x": row.get("settled_x", ""),
        "settled_y": row.get("settled_y", ""),
        "settled_z": row.get("settled_z", ""),
        "target_error_xy_m": row.get("target_error_xy_m", ""),
        "target_x_error_m": row.get("target_x_error_m", ""),
        "target_y_error_m": row.get("target_y_error_m", ""),
        "radial_distance_m": row.get("radial_distance_m", ""),
        "support_overlap": row.get("support_overlap", ""),
        "support_contact_count": row.get("support_contact_count", ""),
        "support_balance_error_m": row.get("support_balance_error_m", ""),
        "bearing_pressure_proxy": row.get("bearing_pressure_proxy", ""),
        "placed_disturbance_xy_m": row.get("placed_disturbance_xy_m", ""),
        "velocity_inf_norm_after_place": row.get("velocity_inf_norm_after_place", ""),
        "height_gain_m": row.get("height_gain_m", ""),
        "candidate_id": row.get("candidate_id", ""),
        "candidate_score": row.get("candidate_score", ""),
        "assignment_fallback_attempt": row.get("assignment_fallback_attempt", ""),
        "assignment_candidate_count": row.get("assignment_candidate_count", ""),
        "assignment_selected_primary": row.get("assignment_selected_primary", ""),
        "assignment_tried_count": row.get("assignment_tried_count", ""),
        "state_path": result.get("state_path", ""),
        "xml_path": result.get("xml", ""),
    }
    output.update(prefixed_features(feature_rows.get(feature_lookup_key(candidate_rock_index), {})))
    return output


def find_result_for_placement(result_rows: list[dict[str, str]], row: dict[str, str]) -> dict[str, str]:
    for result in result_rows:
        if (
            result.get("target_name", "stack") == row.get("target_name", "stack")
            and result.get("strategy", "") == row.get("strategy", "")
            and result.get("gravity", "") == row.get("gravity", "")
            and result.get("trial", "") == row.get("trial", "")
        ):
            return result
    return {}


def build_assignment_examples(
    run_dir: Path,
    run_id: str,
    feature_rows: dict[int, dict[str, str]],
    slot_rows_by_target: dict[str, dict[int, dict[str, str]]],
    placements: list[dict[str, str]],
    candidate_pose_examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    selected_counts = count_selected_candidate_rocks(placements)
    paths = sorted(run_dir.glob("assignment_candidates_*.csv"))
    for path in paths:
        target_name = path.stem.removeprefix("assignment_candidates_")
        slots = slot_rows_by_target.get(target_name, {})
        for row in read_csv(path):
            rock_index = parse_int(row.get("rock_index", ""))
            slot_id = parse_int(row.get("slot_id", ""))
            key = (
                target_name,
                str(row.get("slot_id", "")),
                str(row.get("rock_index", "")),
            )
            slot = slots.get(feature_lookup_key(slot_id), {})
            output: dict[str, Any] = {
                "example_id": f"{run_id}:{target_name}:slot-{row.get('slot_id', '')}:cand-{row.get('candidate_rank', '')}",
                "run_name": run_id,
                "run_path": str(run_dir),
                "target_name": target_name,
                "slot_id": row.get("slot_id", ""),
                "course": row.get("course", slot.get("course", "")),
                "role": row.get("role", slot.get("role", "")),
                "target_x": slot.get("x", ""),
                "target_y": slot.get("y", ""),
                "candidate_rank": row.get("candidate_rank", ""),
                "candidate_rock_index": row.get("rock_index", ""),
                "is_primary_assignment": row.get("is_primary_assignment", ""),
                "candidate_count_for_slot": row.get("candidate_count_for_slot", ""),
                "selected_count_in_placement_log": selected_counts.get(key, 0),
            }
            output.update(prefixed_features(feature_rows.get(feature_lookup_key(rock_index), {})))
            outputs.append(output)
    if not paths and candidate_pose_examples:
        outputs.extend(synthesize_assignment_examples_from_candidate_poses(run_id, candidate_pose_examples))
    return outputs


def synthesize_assignment_examples_from_candidate_poses(
    run_id: str,
    candidate_pose_examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    slot_groups: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    for row in candidate_pose_examples:
        rock_index = str(row.get("candidate_rock_index", ""))
        slot_id = str(row.get("slot_id", ""))
        if rock_index in {"", "-1"} or slot_id == "":
            continue
        key = (
            str(row.get("target_name", "")),
            str(row.get("strategy", "")),
            str(row.get("gravity", "")),
            str(row.get("trial", "")),
            slot_id,
            rock_index,
        )
        grouped[key].append(row)
        slot_groups[key[:5]].add(rock_index)

    ranked_by_slot: dict[tuple[str, str, str, str, str], dict[str, int]] = {}
    for slot_key, rocks in slot_groups.items():
        scored = []
        for rock_index in rocks:
            rows = grouped.get((*slot_key, rock_index), [])
            best_score = max((parse_float_any(row.get("candidate_score", "")) for row in rows), default=0.0)
            scored.append((rock_index, best_score))
        ranked_by_slot[slot_key] = {
            rock_index: rank
            for rank, (rock_index, _score) in enumerate(sorted(scored, key=lambda item: -item[1]))
        }

    outputs: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        target_name, strategy, gravity, trial, slot_id, rock_index = key
        representative = rows[0]
        selected = any(int(parse_float_any(row.get("label_committed_success", "")) > 0.0) for row in rows)
        slot_key = key[:5]
        output: dict[str, Any] = {
            "example_id": (
                f"{run_id}:{target_name}:{strategy}:{gravity}:trial-{trial}:"
                f"slot-{slot_id}:synth-rock-{rock_index}"
            ),
            "run_name": run_id,
            "run_path": representative.get("run_path", ""),
            "target_name": target_name,
            "slot_id": slot_id,
            "course": representative.get("course", ""),
            "role": representative.get("role", ""),
            "target_x": representative.get("target_x", ""),
            "target_y": representative.get("target_y", ""),
            "candidate_rank": ranked_by_slot.get(slot_key, {}).get(rock_index, ""),
            "candidate_rock_index": rock_index,
            "is_primary_assignment": int(selected),
            "candidate_count_for_slot": len(slot_groups.get(slot_key, set())),
            "selected_count_in_placement_log": int(selected),
            "synthetic_source": "candidate_pose_log",
        }
        for column, value in representative.items():
            if str(column).startswith(FEATURE_PREFIX):
                output[str(column)] = value
        outputs.append(output)
    return outputs


def build_candidate_pose_examples(
    run_dir: Path,
    run_id: str,
    feature_rows: dict[int, dict[str, str]],
    candidate_pose_rows: list[dict[str, str]],
    placements: list[dict[str, str]],
    failure_map: dict[tuple[str, str, str, str, str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    placement_by_candidate = {
        (
            row.get("target_name", "stack"),
            row.get("strategy", ""),
            row.get("gravity", ""),
            row.get("trial", ""),
            row.get("slot_id", ""),
            row.get("rock_index", ""),
            row.get("candidate_id", ""),
        ): row
        for row in placements
        if row.get("candidate_id", "") not in {"", None}
    }
    outputs: list[dict[str, Any]] = []
    for row in candidate_pose_rows:
        rock_index = parse_int(row.get("rock_index", ""))
        key = (
            row.get("target_name", "stack"),
            row.get("strategy", ""),
            row.get("gravity", ""),
            row.get("trial", ""),
            row.get("slot_id", ""),
            row.get("rock_index", ""),
            row.get("candidate_id", ""),
        )
        placement = placement_by_candidate.get(key)
        failed = failure_map.get(failure_key(row)) if placement is not None else None
        selected = parse_bool(row.get("selected_by_pose_search", ""))
        output: dict[str, Any] = {
            "example_id": (
                f"{run_id}:{row.get('target_name', 'stack')}:{row.get('gravity', '')}:"
                f"{row.get('strategy', '')}:trial-{row.get('trial', '')}:slot-{row.get('slot_id', '')}:"
                f"rock-{row.get('rock_index', '')}:cand-{row.get('candidate_id', '')}"
            ),
            "run_name": run_id,
            "run_path": str(run_dir),
            "target_name": row.get("target_name", "stack"),
            "strategy": row.get("strategy", ""),
            "gravity": row.get("gravity", ""),
            "trial": row.get("trial", ""),
            "slot_id": row.get("slot_id", ""),
            "course": row.get("course", ""),
            "role": row.get("role", ""),
            "target_x": row.get("target_x", ""),
            "target_y": row.get("target_y", ""),
            "candidate_rock_index": row.get("rock_index", ""),
            "source_kind": row.get("source_kind", ""),
            "cluster_label": row.get("cluster_label", ""),
            "candidate_id": row.get("candidate_id", ""),
            "candidate_count": row.get("candidate_count", ""),
            "pose_x": row.get("pose_x", ""),
            "pose_y": row.get("pose_y", ""),
            "pose_z": row.get("pose_z", ""),
            "pose_qw": row.get("pose_qw", ""),
            "pose_qx": row.get("pose_qx", ""),
            "pose_qy": row.get("pose_qy", ""),
            "pose_qz": row.get("pose_qz", ""),
            "candidate_score": row.get("candidate_score", ""),
            "label_selected_by_pose_search": int(selected),
            "label_committed_success": int(selected and placement is not None and failed is None),
            "failure_reason": failure_reason(placement or {}, failed) if selected else "",
        }
        for key_name in (
            "settled_x",
            "settled_y",
            "settled_z",
            "target_error_xy_m",
            "target_x_error_m",
            "target_y_error_m",
            "radial_distance_m",
            "support_overlap",
            "support_contact_count",
            "support_balance_error_m",
            "bearing_pressure_proxy",
            "placed_disturbance_xy_m",
            "velocity_inf_norm_after_place",
            "height_gain_m",
        ):
            output[key_name] = row.get(key_name, "")
        output.update(prefixed_features(feature_rows.get(feature_lookup_key(rock_index), {})))
        outputs.append(output)
    return outputs


def count_selected_candidate_rocks(placements: list[dict[str, str]]) -> Counter[tuple[str, str, str]]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for row in placements:
        rock_index = row.get("rock_index", "")
        if rock_index in {"", "-1"}:
            continue
        counter[(row.get("target_name", "stack"), row.get("slot_id", ""), rock_index)] += 1
    return counter


def prefixed_features(row: dict[str, str]) -> dict[str, str]:
    return {f"{FEATURE_PREFIX}{key}": value for key, value in row.items()}


def make_example_id(run_id: str, row: dict[str, str]) -> str:
    return (
        f"{run_id}:{row.get('target_name', 'stack')}:{row.get('gravity', '')}:"
        f"{row.get('strategy', '')}:trial-{row.get('trial', '')}:slot-{row.get('slot_id', '')}"
    )


def failure_reason(row: dict[str, str], failed: dict[str, str] | None) -> str:
    if row.get("skip_reason", ""):
        return row["skip_reason"]
    if failed:
        return failed.get("failure_reason", "")
    return ""


def first_nonnegative(*values: int | None) -> int | None:
    for value in values:
        if value is not None and value >= 0:
            return value
    return None


def feature_lookup_key(value: int | None) -> int:
    return value if value is not None else -1


def parse_int(value: str | None) -> int | None:
    if value in {"", None}:
        return None
    return int(float(value))


def parse_bool(value: str | None) -> bool:
    if value in {"", None}:
        return False
    return bool(int(float(value)))


def parse_float_any(value: Any) -> float:
    if value in {"", None}:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def write_summary(
    output_dir: Path,
    batch_root: Path,
    run_dirs: list[Path],
    run_rows: list[dict[str, Any]],
    placement_rows: list[dict[str, Any]],
    candidate_pose_rows: list[dict[str, Any]],
    assignment_rows: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    by_gravity = defaultdict(lambda: {"examples": 0, "success": 0, "failure": 0, "skipped": 0})
    by_role = defaultdict(lambda: {"examples": 0, "success": 0, "failure": 0, "skipped": 0})
    by_kind = defaultdict(lambda: {"examples": 0, "success": 0, "failure": 0, "skipped": 0})
    for row in placement_rows:
        for group, key in (
            (by_gravity, row.get("gravity", "")),
            (by_role, row.get("role", "")),
            (by_kind, row.get("rock_source_kind", "")),
        ):
            item = group[key]
            item["examples"] += 1
            item["success"] += int(row.get("label_success", 0))
            item["failure"] += int(row.get("is_failure_case", 0))
            item["skipped"] += int(row.get("is_skipped_slot", 0))

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "batch_root": str(batch_root),
        "output_dir": str(output_dir),
        "run_dir_count": len(run_dirs),
        "run_example_count": len(run_rows),
        "placement_example_count": len(placement_rows),
        "candidate_pose_example_count": len(candidate_pose_rows),
        "assignment_candidate_example_count": len(assignment_rows),
        "placement_by_gravity": dict(sorted(by_gravity.items())),
        "placement_by_role": dict(sorted(by_role.items())),
        "placement_by_source_kind": dict(sorted(by_kind.items())),
        "warnings": warnings,
        "known_limitations": [
            "placement_examples.csv records committed placements, skipped slots, and best rejected slot summaries.",
            "candidate_pose_examples.csv contains sampled candidate poses only for runs that have candidate_pose_log.csv.",
            "Older runs do not have dense candidate-pose labels.",
        ],
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_readme(output_dir: Path, batch_root: Path, run_dirs: list[Path], warnings: list[str]) -> None:
    lines = [
        "# MoonStack Learning Dataset V0",
        "",
        f"Batch root: `{batch_root}`",
        f"Run directories scanned: {len(run_dirs)}",
        "",
        "## Files",
        "",
        "- `run_examples.csv`: one row per structured simulation result.",
        "- `placement_examples.csv`: one row per committed placement, skipped slot, or best-rejected slot summary.",
        "- `placement_examples.jsonl`: JSONL mirror of placement examples.",
        "- `candidate_pose_examples.csv`: one row per sampled candidate pose when `candidate_pose_log.csv` exists.",
        "- `candidate_pose_examples.jsonl`: JSONL mirror of candidate-pose examples.",
        "- `assignment_candidate_examples.csv`: one row per assignment/fallback candidate when available.",
        "- `dataset_summary.json`: counts, groups, and limitations.",
        "",
        "## Labels",
        "",
        "- `label_success=1`: the placement was committed and is not present in `failure_cases.csv`.",
        "- `label_success=0`: the placement was skipped or recorded as a failure.",
        "- `label_run_success`: run-level strict success from `results.csv`.",
        "- `label_run_shape_success`: run-level shape success from `results.csv`.",
        "",
        "## Limitation",
        "",
        "This is mixed V0/V1 data. It is useful for tabular compatibility, failure prediction, and imitation of current search results. Dense GraspNet-style pose labels are available only for new runs that contain `candidate_pose_log.csv`.",
        "",
        "The next data-collection step is to generate larger candidate-pose batches with higher `--candidates` values under both Earth and Moon gravity.",
    ]
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
