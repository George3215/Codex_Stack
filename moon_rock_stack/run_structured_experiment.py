from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .clustering import cluster_features
from .features import extract_features
from .fractal_rocks import generate_rocks, write_all_objs
from .mjcf import write_world_xml
from .run_experiment import write_csv
from .simulate import GRAVITIES
from .structured import run_structured_trial_detailed, slots_for_target, structured_order


def main() -> None:
    args = parse_args()
    output_dir = args.output.resolve()
    mesh_dir = output_dir / "meshes"
    mjcf_dir = output_dir / "mjcf"
    output_dir.mkdir(parents=True, exist_ok=True)
    mjcf_dir.mkdir(parents=True, exist_ok=True)

    rocks = generate_rocks(args.rocks, seed=args.seed, profile=args.rock_profile)
    write_all_objs(mesh_dir, rocks)
    rows = [extract_features(rock) for rock in rocks]
    labels, names = cluster_features(rows, clusters=args.clusters, seed=args.seed)
    for row, label in zip(rows, labels):
        row["cluster_id"] = int(label)
        row["cluster_label"] = names[int(label)]

    write_csv(output_dir / "features.csv", rows)
    write_csv(output_dir / "cluster_summary.csv", cluster_summary(rows))
    write_protocol(output_dir / "PROTOCOL.md", args)

    results: list[dict[str, Any]] = []
    placements: list[dict[str, Any]] = []
    candidate_poses: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    targets = parse_csv_arg(args.targets)
    strategies = parse_csv_arg(args.strategies)
    gravity_labels = parse_csv_arg(args.gravities)
    unknown_gravities = sorted(set(gravity_labels).difference(GRAVITIES))
    if unknown_gravities:
        raise ValueError(f"Unknown gravity labels: {unknown_gravities}. Valid labels: {sorted(GRAVITIES)}")
    assignment_plan = read_assignment_plan(args.assignment_plan)
    role_screening = read_role_screening(args.role_screening)
    candidate_pose_ranker = load_candidate_pose_ranker(args.candidate_pose_ranker_dir)
    pose_risk_ranker = load_pose_risk_ranker(args.pose_risk_ranker_dir)
    stone_fit_ranker = load_stone_fit_ranker(args.stone_fit_ranker_dir)
    if assignment_plan is not None and len(targets) != 1:
        raise ValueError("--assignment-plan currently supports exactly one target. Pass one --targets value.")
    if role_screening is not None and assignment_plan is None:
        raise ValueError("--role-screening requires --assignment-plan.")
    tasks: list[dict[str, Any]] = []
    for target_index, target_name in enumerate(targets):
        slots = slots_for_target(target_name)
        assignment_candidates_by_slot = None
        if assignment_plan is not None:
            validate_assignment_plan(target_name, slots, assignment_plan, rows)
            assignment_candidates_by_slot = build_assignment_candidates(
                slots=slots,
                assignment_plan=assignment_plan,
                role_screening=role_screening,
                fallback_count=args.assignment_fallbacks,
            )
            write_csv(
                output_dir / f"assignment_candidates_{target_name}.csv",
                assignment_candidate_rows(slots, assignment_candidates_by_slot),
            )
        write_csv(output_dir / f"target_slots_{target_name}.csv", [slot.__dict__ for slot in slots])
        for strategy_index, strategy in enumerate(strategies):
            for gravity_label in gravity_labels:
                gravity = GRAVITIES[gravity_label]
                for trial_id in range(args.trials):
                    tasks.append(
                        {
                            "target_name": target_name,
                            "target_index": target_index,
                            "strategy": strategy,
                            "strategy_index": strategy_index,
                            "gravity_label": gravity_label,
                            "gravity": gravity,
                            "trial_id": trial_id,
                            "seed": args.seed,
                            "steps_per_rock": args.steps_per_rock,
                            "hold_steps": args.hold_steps,
                            "candidate_count": args.candidates,
                            "rows": rows,
                            "assignment_plan": assignment_plan,
                            "assignment_plan_path": str(args.assignment_plan.resolve()) if args.assignment_plan else "",
                            "assignment_gate": bool(args.assignment_gate),
                            "assignment_candidates_by_slot": assignment_candidates_by_slot,
                            "assignment_fallbacks": int(args.assignment_fallbacks),
                            "assignment_probe_steps": int(args.assignment_probe_steps),
                            "candidate_probe_steps": int(args.candidate_probe_steps),
                            "candidate_probe_hard_gate": bool(args.candidate_probe_hard_gate),
                            "moon_gate_strict": bool(args.moon_gate_strict),
                            "candidate_pose_ranker": candidate_pose_ranker,
                            "candidate_pose_ranker_dir": str(args.candidate_pose_ranker_dir.resolve()) if args.candidate_pose_ranker_dir else "",
                            "candidate_pose_top_k": int(args.candidate_pose_top_k),
                            "candidate_pose_ranker_max_course": int(args.candidate_pose_ranker_max_course),
                            "pose_risk_ranker": pose_risk_ranker,
                            "pose_risk_ranker_dir": str(args.pose_risk_ranker_dir.resolve()) if args.pose_risk_ranker_dir else "",
                            "pose_risk_weight": float(args.pose_risk_weight),
                            "pose_risk_ranker_max_course": int(args.pose_risk_ranker_max_course),
                            "stone_fit_ranker": stone_fit_ranker,
                            "stone_fit_ranker_dir": str(args.stone_fit_ranker_dir.resolve()) if args.stone_fit_ranker_dir else "",
                            "stone_fit_top_k": int(args.stone_fit_top_k),
                            "stone_fit_ranker_max_course": int(args.stone_fit_ranker_max_course),
                            "commit_best_rejected": bool(args.commit_best_rejected),
                            "role_screening_path": str(args.role_screening.resolve()) if args.role_screening else "",
                            "mjcf_dir": mjcf_dir,
                            "output_dir": output_dir,
                        }
                    )

    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_structured_task, task) for task in tasks]
            for future in as_completed(futures):
                collect_task_result(future.result(), output_dir, results, placements, candidate_poses, failures)
    else:
        for task in tasks:
            collect_task_result(run_structured_task(task), output_dir, results, placements, candidate_poses, failures)

    write_outputs(output_dir, results, placements, candidate_poses, failures)
    summary = summarize(results)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output_dir / "STRUCTURED_REPORT.md", summary, results, failures)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured dry-stacking experiments.")
    parser.add_argument("--rocks", type=int, default=24)
    parser.add_argument("--rock-profile", default="balanced", help="Rock generation profile, e.g. balanced or wall_statics.")
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--targets", default="wall_segment_v1,pillar_v1")
    parser.add_argument("--strategies", default="geometry_bonded,support_first,risk_aware,centered_compact,random_order")
    parser.add_argument("--gravities", default="earth,moon", help="Comma-separated gravity labels to run: earth,moon.")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--steps-per-rock", type=int, default=1200)
    parser.add_argument("--hold-steps", type=int, default=2600)
    parser.add_argument("--workers", type=int, default=2, help="Parallel MuJoCo worker processes. Use 1 for sequential.")
    parser.add_argument(
        "--assignment-plan",
        type=Path,
        help="Optional CSV from screen_target_rocks.py. Applies pre-screened rock_index assignments by slot_id.",
    )
    parser.add_argument(
        "--assignment-gate",
        action="store_true",
        help="When using --assignment-plan, skip candidate placements that fail target/support feasibility gates.",
    )
    parser.add_argument(
        "--role-screening",
        type=Path,
        help="Optional role_screening.csv from screen_target_rocks.py. Enables same-role fallback candidates.",
    )
    parser.add_argument(
        "--assignment-fallbacks",
        type=int,
        default=0,
        help="Number of same-role fallback rocks to try after the primary assignment candidate.",
    )
    parser.add_argument(
        "--assignment-probe-steps",
        type=int,
        default=0,
        help="Extra short hold after an assignment candidate settles, used to reject drift-prone candidates.",
    )
    parser.add_argument(
        "--candidate-probe-steps",
        type=int,
        default=0,
        help="Extra short hold after any candidate settles, used to penalize or reject slow drift before committing.",
    )
    parser.add_argument(
        "--candidate-probe-hard-gate",
        action="store_true",
        help="When --candidate-probe-steps is enabled, reject probe-drifting candidates instead of only scoring them.",
    )
    parser.add_argument(
        "--moon-gate-strict",
        action="store_true",
        help="Use stricter assignment-gate limits under Moon gravity for wall-depth, support balance, and probe drift.",
    )
    parser.add_argument(
        "--candidate-pose-ranker-dir",
        type=Path,
        help="Optional directory containing candidate_pose_rank_net.npz and candidate_pose_rank_net_schema.json.",
    )
    parser.add_argument(
        "--candidate-pose-top-k",
        type=int,
        default=0,
        help="If a candidate pose ranker is provided, simulate only the top-K ranked pose candidates per rock.",
    )
    parser.add_argument(
        "--candidate-pose-ranker-max-course",
        type=int,
        default=-1,
        help="Use the candidate-pose ranker only for slots with course <= this value. -1 keeps it active for all courses.",
    )
    parser.add_argument(
        "--pose-risk-ranker-dir",
        type=Path,
        help="Optional directory containing pose_risk_net.npz and pose_risk_net_schema.json.",
    )
    parser.add_argument(
        "--pose-risk-weight",
        type=float,
        default=0.0,
        help="Penalty multiplier for pre-simulation pose risk probability during candidate ranking and scoring.",
    )
    parser.add_argument(
        "--pose-risk-ranker-max-course",
        type=int,
        default=-1,
        help="Use the pose-risk ranker only for slots with course <= this value. -1 keeps it active for all courses.",
    )
    parser.add_argument(
        "--stone-fit-ranker-dir",
        type=Path,
        help="Optional directory containing stone_fit_net.npz and stone_fit_net_schema.json.",
    )
    parser.add_argument(
        "--stone-fit-top-k",
        type=int,
        default=0,
        help="If a stone-fit ranker is provided, simulate only the top-K ranked stones for each literature/statics slot.",
    )
    parser.add_argument(
        "--stone-fit-ranker-max-course",
        type=int,
        default=-1,
        help="Use the stone-fit ranker only for slots with course <= this value. -1 keeps it active for all courses.",
    )
    parser.add_argument(
        "--commit-best-rejected",
        action="store_true",
        help="Curriculum-data mode: when no feasible literature/statics candidate exists, commit the best rejected candidate instead of skipping the slot.",
    )
    parser.add_argument("--output", type=Path, default=Path("generated_structured"))
    return parser.parse_args()


def run_structured_task(task: dict[str, Any]) -> dict[str, Any]:
    target_name = str(task["target_name"])
    target_index = int(task["target_index"])
    strategy = str(task["strategy"])
    strategy_index = int(task["strategy_index"])
    gravity_label = str(task["gravity_label"])
    gravity = float(task["gravity"])
    trial_id = int(task["trial_id"])
    seed = int(task["seed"])
    rows = [dict(row) for row in task["rows"]]
    slots = slots_for_target(target_name)

    rng = np.random.default_rng(seed + 20011 * target_index + 503 * strategy_index + 37 * trial_id + int(gravity * 10))
    assignment_plan = task.get("assignment_plan")
    if assignment_plan is not None and strategy not in {"literature_column", "literature_wall", "statics_wall", "statics_wall_line_lock"}:
        order = assignment_order(rows, slots, assignment_plan)
    else:
        order = structured_order(rows, slots, strategy=strategy, rng=rng)
    assignment_candidates_by_slot = task.get("assignment_candidates_by_slot")
    xml_path = (
        Path(task["mjcf_dir"])
        / f"{target_name}_{strategy}_{gravity_label}_trial_{trial_id:02d}.xml"
    )
    write_world_xml(xml_path, rows, gravity=gravity, trial_id=trial_id)
    detailed = run_structured_trial_detailed(
        xml_path=xml_path,
        rows=[dict(row) for row in rows],
        slots=slots,
        order=order,
        gravity_label=gravity_label,
        trial_id=trial_id,
        seed=seed + 30011 * target_index + 1009 * (strategy_index + 1) + trial_id * 37 + int(gravity * 10),
        steps_per_rock=int(task["steps_per_rock"]),
        hold_steps=int(task["hold_steps"]),
        strategy=strategy,
        candidate_count=int(task["candidate_count"]),
        target_name=target_name,
        assignment_gate=bool(task.get("assignment_gate", False)),
        assignment_candidates_by_slot=assignment_candidates_by_slot,
        assignment_probe_steps=int(task.get("assignment_probe_steps", 0)),
        candidate_probe_steps=int(task.get("candidate_probe_steps", 0)),
        candidate_probe_hard_gate=bool(task.get("candidate_probe_hard_gate", False)),
        moon_gate_strict=bool(task.get("moon_gate_strict", False)),
        candidate_pose_ranker=task.get("candidate_pose_ranker"),
        candidate_pose_top_k=int(task.get("candidate_pose_top_k", 0)),
        candidate_pose_ranker_max_course=int(task.get("candidate_pose_ranker_max_course", -1)),
        pose_risk_ranker=task.get("pose_risk_ranker"),
        pose_risk_weight=float(task.get("pose_risk_weight", 0.0)),
        pose_risk_ranker_max_course=int(task.get("pose_risk_ranker_max_course", -1)),
        stone_fit_ranker=task.get("stone_fit_ranker"),
        stone_fit_top_k=int(task.get("stone_fit_top_k", 0)),
        stone_fit_ranker_max_course=int(task.get("stone_fit_ranker_max_course", -1)),
        commit_best_rejected=bool(task.get("commit_best_rejected", False)),
        progress_path=Path(task["output_dir"]) / "structured_progress.csv",
    )
    state_path = (
        Path(task["output_dir"])
        / "states"
        / f"{target_name}_{strategy}_{gravity_label}_trial_{trial_id:02d}.npz"
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        state_path,
        qpos=detailed["state"]["qpos"],
        qvel=detailed["state"]["qvel"],
        target_name=target_name,
        strategy=strategy,
        gravity=gravity_label,
        trial=trial_id,
    )
    detailed["summary"]["state_path"] = str(state_path)
    detailed["summary"]["assignment_plan_path"] = str(task.get("assignment_plan_path", ""))
    detailed["summary"]["assignment_plan_used"] = int(assignment_plan is not None)
    detailed["summary"]["assignment_gate_requested"] = int(bool(task.get("assignment_gate", False)))
    detailed["summary"]["assignment_fallbacks_requested"] = int(task.get("assignment_fallbacks", 0))
    detailed["summary"]["assignment_probe_steps_requested"] = int(task.get("assignment_probe_steps", 0))
    detailed["summary"]["candidate_probe_steps_requested"] = int(task.get("candidate_probe_steps", 0))
    detailed["summary"]["candidate_probe_hard_gate_requested"] = int(bool(task.get("candidate_probe_hard_gate", False)))
    detailed["summary"]["moon_gate_strict_requested"] = int(bool(task.get("moon_gate_strict", False)))
    detailed["summary"]["candidate_pose_ranker_dir"] = str(task.get("candidate_pose_ranker_dir", ""))
    detailed["summary"]["candidate_pose_top_k_requested"] = int(task.get("candidate_pose_top_k", 0))
    detailed["summary"]["candidate_pose_ranker_max_course_requested"] = int(task.get("candidate_pose_ranker_max_course", -1))
    detailed["summary"]["pose_risk_ranker_dir"] = str(task.get("pose_risk_ranker_dir", ""))
    detailed["summary"]["pose_risk_weight_requested"] = float(task.get("pose_risk_weight", 0.0))
    detailed["summary"]["pose_risk_ranker_max_course_requested"] = int(task.get("pose_risk_ranker_max_course", -1))
    detailed["summary"]["stone_fit_ranker_dir"] = str(task.get("stone_fit_ranker_dir", ""))
    detailed["summary"]["stone_fit_top_k_requested"] = int(task.get("stone_fit_top_k", 0))
    detailed["summary"]["stone_fit_ranker_max_course_requested"] = int(task.get("stone_fit_ranker_max_course", -1))
    detailed["summary"]["commit_best_rejected_requested"] = int(bool(task.get("commit_best_rejected", False)))
    detailed["summary"]["role_screening_path"] = str(task.get("role_screening_path", ""))
    return {
        "summary": detailed["summary"],
        "placements": detailed["placements"],
        "candidate_poses": detailed.get("candidate_poses", []),
        "failures": detailed["failures"],
    }


def collect_task_result(
    result: dict[str, Any],
    output_dir: Path,
    results: list[dict[str, Any]],
    placements: list[dict[str, Any]],
    candidate_poses: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    results.append(result["summary"])
    placements.extend(result["placements"])
    candidate_poses.extend(result.get("candidate_poses", []))
    failures.extend(result["failures"])
    write_outputs(output_dir, sorted_rows(results), sorted_rows(placements), sorted_rows(candidate_poses), sorted_rows(failures))


def sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("target_name", "")),
            str(row.get("strategy", "")),
            str(row.get("gravity", "")),
            int(row.get("trial", 0)),
            int(row.get("course", 0)),
            int(row.get("slot_id", 0)),
        ),
    )


def parse_csv_arg(value: str) -> list[str]:
    output = [item.strip() for item in value.split(",") if item.strip()]
    if not output:
        raise ValueError("CSV argument cannot be empty.")
    return output


def read_assignment_plan(path: Path | None) -> dict[int, dict[str, Any]] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Assignment plan does not exist: {path}")
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"slot_id", "course", "role", "target_x", "target_y", "rock_index"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Assignment plan is missing required columns: {sorted(missing)}")
        for raw in reader:
            slot_id = int(raw["slot_id"])
            if slot_id in rows:
                raise ValueError(f"Assignment plan has duplicate slot_id: {slot_id}")
            rows[slot_id] = {
                "slot_id": slot_id,
                "course": int(raw["course"]),
                "role": str(raw["role"]),
                "target_x": float(raw["target_x"]),
                "target_y": float(raw["target_y"]),
                "rock_index": int(raw["rock_index"]),
            }
    return rows


def read_role_screening(path: Path | None) -> dict[str, list[dict[str, Any]]] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Role screening file does not exist: {path}")
    by_role: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"index", "role", "role_score", "screen_accept"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Role screening file is missing required columns: {sorted(missing)}")
        for raw in reader:
            if int(float(raw["screen_accept"])) != 1:
                continue
            role = str(raw["role"])
            by_role.setdefault(role, []).append(
                {
                    "index": int(float(raw["index"])),
                    "role": role,
                    "role_score": float(raw["role_score"]),
                    "source_kind": raw.get("source_kind", ""),
                    "cluster_label": raw.get("cluster_label", ""),
                }
            )
    for role_rows in by_role.values():
        role_rows.sort(key=lambda row: float(row["role_score"]))
    return by_role


def load_candidate_pose_ranker(path: Path | None) -> dict[str, Any] | None:
    if path is not None and (path / "support_map_cnn_ranker.pt").exists() and (path / "schema.json").exists():
        metrics_path = path / "metrics.json"
        return {
            "kind": "torch_support_map_cnn",
            "model_path": str((path / "support_map_cnn_ranker.pt").resolve()),
            "schema_path": str((path / "schema.json").resolve()),
            "schema": json.loads((path / "schema.json").read_text(encoding="utf-8")),
            "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {},
        }
    return load_named_binary_model(path, "candidate_pose_rank_net", "Candidate pose ranker")


def load_stone_fit_ranker(path: Path | None) -> dict[str, Any] | None:
    return load_named_binary_model(path, "stone_fit_net", "Stone fit ranker")


def load_pose_risk_ranker(path: Path | None) -> dict[str, Any] | None:
    return load_named_binary_model(path, "pose_risk_net", "Pose risk ranker")


def load_named_binary_model(path: Path | None, name: str, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    model_path = path / f"{name}.npz"
    schema_path = path / f"{name}_schema.json"
    if not model_path.exists() or not schema_path.exists():
        raise FileNotFoundError(f"{label} files missing under: {path}")
    data = np.load(model_path)
    model = {key: data[key] for key in data.files}
    model["schema"] = json.loads(schema_path.read_text(encoding="utf-8"))
    return model


def validate_assignment_plan(
    target_name: str,
    slots: list[Any],
    assignment_plan: dict[int, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    slot_ids = {int(slot.slot_id) for slot in slots}
    plan_ids = set(assignment_plan)
    missing = sorted(slot_ids.difference(plan_ids))
    extra = sorted(plan_ids.difference(slot_ids))
    if missing or extra:
        raise ValueError(
            f"Assignment plan does not match target {target_name}: missing slots={missing}, extra slots={extra}"
        )
    rock_indices = {int(row["index"]) for row in rows}
    used: set[int] = set()
    for slot in slots:
        planned = assignment_plan[int(slot.slot_id)]
        if int(planned["course"]) != int(slot.course):
            raise ValueError(f"Assignment plan course mismatch for slot {slot.slot_id}.")
        if str(planned["role"]) != str(slot.role):
            raise ValueError(f"Assignment plan role mismatch for slot {slot.slot_id}.")
        if abs(float(planned["target_x"]) - float(slot.x)) > 1e-6:
            raise ValueError(f"Assignment plan target_x mismatch for slot {slot.slot_id}.")
        if abs(float(planned["target_y"]) - float(slot.y)) > 1e-6:
            raise ValueError(f"Assignment plan target_y mismatch for slot {slot.slot_id}.")
        rock_index = int(planned["rock_index"])
        if rock_index not in rock_indices:
            raise ValueError(f"Assignment plan uses rock_index {rock_index}, but the generated catalog lacks it.")
        if rock_index in used:
            raise ValueError(f"Assignment plan reuses rock_index {rock_index}.")
        used.add(rock_index)


def build_assignment_candidates(
    slots: list[Any],
    assignment_plan: dict[int, dict[str, Any]],
    role_screening: dict[str, list[dict[str, Any]]] | None,
    fallback_count: int,
) -> dict[int, list[int]]:
    fallback_count = max(0, int(fallback_count))
    planned_primary = {int(row["rock_index"]) for row in assignment_plan.values()}
    output: dict[int, list[int]] = {}
    for slot in slots:
        primary = int(assignment_plan[int(slot.slot_id)]["rock_index"])
        candidates = [primary]
        if role_screening is not None and fallback_count > 0:
            for role_row in role_screening.get(str(slot.role), []):
                idx = int(role_row["index"])
                if idx == primary or idx in planned_primary:
                    continue
                candidates.append(idx)
                if len(candidates) >= fallback_count + 1:
                    break
        output[int(slot.slot_id)] = candidates
    return output


def assignment_candidate_rows(slots: list[Any], candidates_by_slot: dict[int, list[int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot in slots:
        candidates = candidates_by_slot[int(slot.slot_id)]
        for rank, rock_index in enumerate(candidates):
            rows.append(
                {
                    "slot_id": int(slot.slot_id),
                    "course": int(slot.course),
                    "role": str(slot.role),
                    "candidate_rank": rank,
                    "rock_index": int(rock_index),
                    "is_primary_assignment": int(rank == 0),
                    "candidate_count_for_slot": len(candidates),
                }
            )
    return rows


def assignment_order(
    rows: list[dict[str, Any]],
    slots: list[Any],
    assignment_plan: dict[int, dict[str, Any]],
) -> list[int]:
    rock_indices = {int(row["index"]) for row in rows}
    order: list[int] = []
    used: set[int] = set()
    for slot in slots:
        rock_index = int(assignment_plan[int(slot.slot_id)]["rock_index"])
        if rock_index not in rock_indices:
            raise ValueError(f"Assignment plan uses rock_index {rock_index}, but the generated catalog lacks it.")
        if rock_index in used:
            raise ValueError(f"Assignment plan reuses rock_index {rock_index}.")
        order.append(rock_index)
        used.add(rock_index)
    return order


def write_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    placements: list[dict[str, Any]],
    candidate_poses: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    write_csv(output_dir / "results.csv", results)
    write_csv(output_dir / "placement_log.csv", placements)
    write_csv(output_dir / "candidate_pose_log.csv", candidate_poses)
    write_csv(output_dir / "failure_cases.csv", failures)


def cluster_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for label in sorted({str(row["cluster_label"]) for row in rows}):
        subset = [row for row in rows if str(row["cluster_label"]) == label]
        output.append(
            {
                "cluster_label": label,
                "count": len(subset),
                "source_kind_counts": json.dumps(count_by(subset, "source_kind"), sort_keys=True),
                "mean_volume": mean(float(row["volume"]) for row in subset),
                "mean_elongation": mean(float(row["elongation"]) for row in subset),
                "mean_flatness": mean(float(row["flatness"]) for row in subset),
                "mean_angularity": mean(float(row["angularity"]) for row in subset),
                "mean_spike_score": mean(float(row.get("spike_score", 0.0)) for row in subset),
                "mean_compactness": mean(float(row["compactness"]) for row in subset),
                "mean_stability_score": mean(float(row["stability_score"]) for row in subset),
                "mean_major_face_count": mean(float(row.get("major_face_count", 0.0)) for row in subset),
                "mean_largest_face_area_ratio": mean(float(row.get("largest_face_area_ratio", 0.0)) for row in subset),
                "mean_support_face_area_ratio": mean(float(row.get("support_face_area_ratio", 0.0)) for row in subset),
                "mean_opposing_face_pair_count": mean(float(row.get("opposing_face_pair_count", 0.0)) for row in subset),
                "mean_face_planarity": mean(float(row.get("face_planarity", 0.0)) for row in subset),
                "mean_support_plane_quality": mean(float(row.get("support_plane_quality", 0.0)) for row in subset),
            }
        )
    return output


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for target_name in sorted({str(row["target_name"]) for row in results}):
        summary[target_name] = {}
        for strategy in sorted({str(row["strategy"]) for row in results if row["target_name"] == target_name}):
            summary[target_name][strategy] = {}
            for gravity in sorted({str(row["gravity"]) for row in results if row["target_name"] == target_name and row["strategy"] == strategy}):
                subset = [
                    row
                    for row in results
                    if row["target_name"] == target_name and row["strategy"] == strategy and row["gravity"] == gravity
                ]
                summary[target_name][strategy][gravity] = {
                    "trials": len(subset),
                    "success_rate": mean(float(row["success"]) for row in subset),
                    "shape_success_rate": mean(float(row["shape_success"]) for row in subset),
                    "mean_structure_score": mean(float(row["structure_score"]) for row in subset),
                    "mean_stable_count": mean(float(row["stable_count"]) for row in subset),
                    "mean_failure_count": mean(float(row["failure_count"]) for row in subset),
                    "mean_target_rmse_xy_m": mean(float(row["target_rmse_xy_m"]) for row in subset),
                    "mean_target_max_xy_error_m": mean(float(row["target_max_xy_error_m"]) for row in subset),
                    "mean_visible_courses": mean(float(row["visible_courses"]) for row in subset),
                    "mean_stack_height_m": mean(float(row["stack_height_m"]) for row in subset),
                    "mean_max_drift_m": mean(float(row["max_horizontal_drift_m"]) for row in subset),
                    "mean_velocity_inf_norm": mean(float(row["velocity_inf_norm"]) for row in subset),
                }
    return summary


def mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def write_protocol(path: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Structured Dry-Stacking Protocol",
        "",
        "Purpose: compare structured dry-stacking strategies for a short wall and a pillar under Earth and Moon gravity.",
        "",
        "This protocol is derived from local dry-stacking papers in `D:/MoonStack/Asset/Papers`: Furrer et al. 2017, Johns et al. 2020, Liu et al. 2018, Liu et al. 2021, Liu and Napp 2023, and Menezes et al. 2021.",
        "",
        "Key experimental commitments:",
        "",
        "- Build target structures, not loose piles.",
        "- Use geology-prior clast geometry with spike rejection metrics.",
        "- Use same generated rock library for Earth and Moon gravity.",
        "- Compare geometry-bonded ordering, support-first placement, risk-aware ordering, centered-compact placement, and random-order control.",
        "- Evaluate target-shape error, visible courses, residual velocity, drift, and failures.",
        "",
        "Configuration:",
        "",
        f"- rocks: {args.rocks}",
        f"- rock_profile: {args.rock_profile}",
        f"- clusters: {args.clusters}",
        f"- trials: {args.trials}",
        f"- targets: {args.targets}",
        f"- strategies: {args.strategies}",
        f"- gravities: {args.gravities}",
        f"- candidates: {args.candidates}",
        f"- steps_per_rock: {args.steps_per_rock}",
        f"- hold_steps: {args.hold_steps}",
        f"- workers: {args.workers}",
        f"- assignment_plan: {args.assignment_plan or ''}",
        f"- assignment_gate: {int(bool(args.assignment_gate))}",
        f"- role_screening: {args.role_screening or ''}",
        f"- assignment_fallbacks: {args.assignment_fallbacks}",
        f"- assignment_probe_steps: {args.assignment_probe_steps}",
        f"- candidate_probe_steps: {args.candidate_probe_steps}",
        f"- candidate_probe_hard_gate: {int(bool(args.candidate_probe_hard_gate))}",
        f"- moon_gate_strict: {int(bool(args.moon_gate_strict))}",
        f"- candidate_pose_ranker_dir: {args.candidate_pose_ranker_dir or ''}",
        f"- candidate_pose_top_k: {args.candidate_pose_top_k}",
        f"- candidate_pose_ranker_max_course: {args.candidate_pose_ranker_max_course}",
        f"- pose_risk_ranker_dir: {args.pose_risk_ranker_dir or ''}",
        f"- pose_risk_weight: {args.pose_risk_weight}",
        f"- pose_risk_ranker_max_course: {args.pose_risk_ranker_max_course}",
        f"- stone_fit_ranker_dir: {args.stone_fit_ranker_dir or ''}",
        f"- stone_fit_top_k: {args.stone_fit_top_k}",
        f"- stone_fit_ranker_max_course: {args.stone_fit_ranker_max_course}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any], results: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    lines = [
        "# Structured Dry-Stacking Report",
        "",
        "This is a structured experiment: target wall and pillar, not a loose mound.",
        "",
        "## Method Summary",
        "",
    ]
    ranked = sorted(
        results,
        key=lambda row: (
            str(row["target_name"]),
            -float(row["structure_score"]),
            float(row["target_rmse_xy_m"]),
            float(row["velocity_inf_norm"]),
        ),
    )
    for row in ranked[:24]:
        lines.append(
            f"- `{row['target_name']}` `{row['strategy']}` `{row['gravity']}` trial={row['trial']}: "
            f"success={row['success']}, shape={row['shape_success']}, score={float(row['structure_score']):.3f}, "
            f"stable={row['stable_count']}/{row['rock_count']}, rmse={float(row['target_rmse_xy_m']):.3f} m, "
            f"courses={row['visible_courses']}, drift={float(row['max_horizontal_drift_m']):.3f} m, "
            f"velocity={float(row['velocity_inf_norm']):.3f}"
        )

    lines.extend(["", "## Aggregated Summary", "", "```json", json.dumps(summary, indent=2), "```", ""])
    lines.extend(["## Failure Counts", ""])
    if failures:
        counts: dict[str, int] = {}
        for failure in failures:
            key = f"{failure['target_name'] if 'target_name' in failure else 'structured'}:{failure['failure_reason']}"
            counts[key] = counts.get(key, 0) + 1
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{key}`: {count}")
    else:
        lines.append("No structured failures recorded.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
