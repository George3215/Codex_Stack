from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moon_rock_stack.clustering import cluster_features
from moon_rock_stack.features import extract_features
from moon_rock_stack.fractal_rocks import ROCK_PROFILES, generate_rocks, write_all_objs
from moon_rock_stack.run_experiment import write_csv
from moon_rock_stack.structured import slots_for_target


ROLES = ("base", "middle", "tie", "cap", "chock")


def main() -> int:
    args = parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = output_dir / "meshes"

    rocks = generate_rocks(args.rocks, seed=args.seed, profile=args.profile)
    if args.write_meshes:
        write_all_objs(mesh_dir, rocks)

    rows = [extract_features(rock) for rock in rocks]
    labels, names = cluster_features(rows, clusters=args.clusters, seed=args.seed)
    for row, label in zip(rows, labels):
        row["cluster_id"] = int(label)
        row["cluster_label"] = names[int(label)]
        row.update(quality_flags(row))

    role_rows = score_roles(rows)
    slot_rows = assign_to_target(rows, role_rows, args.target)

    write_csv(output_dir / "features.csv", rows)
    write_csv(output_dir / "cluster_summary.csv", cluster_summary(rows))
    write_csv(output_dir / "role_screening.csv", role_rows)
    write_csv(output_dir / f"assignment_plan_{args.target}.csv", slot_rows)
    write_report(output_dir / "README.md", args, rows, role_rows, slot_rows)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and screen target-conditioned angular rock catalogs.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rocks", type=int, default=360)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--profile", default="wall_statics", choices=sorted(ROCK_PROFILES))
    parser.add_argument("--target", default="single_face_wall_4course_v1")
    parser.add_argument("--write-meshes", action="store_true")
    return parser.parse_args()


def quality_flags(row: dict[str, Any]) -> dict[str, Any]:
    spike = float(row["spike_score"])
    flatness = float(row["flatness"])
    elongation = float(row["elongation"])
    compactness = float(row["compactness"])
    bbox = sorted((float(row["bbox_x"]), float(row["bbox_y"]), float(row["bbox_z"])), reverse=True)
    short_to_mid = bbox[2] / max(bbox[1], 1e-9)
    reject_spike = spike > 0.16
    reject_slab = flatness > 1.62 or short_to_mid < 0.62
    reject_stringer = elongation > 1.85
    reject_low_compactness = compactness < 0.22
    acceptable = not (reject_spike or reject_slab or reject_stringer or reject_low_compactness)
    return {
        "short_to_mid": short_to_mid,
        "reject_spike": int(reject_spike),
        "reject_slab": int(reject_slab),
        "reject_stringer": int(reject_stringer),
        "reject_low_compactness": int(reject_low_compactness),
        "screen_accept": int(acceptable),
    }


def score_roles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        for role in ROLES:
            score, reason = role_score(row, role)
            output.append(
                {
                    "index": row["index"],
                    "source_kind": row["source_kind"],
                    "cluster_label": row["cluster_label"],
                    "role": role,
                    "role_score": score,
                    "screen_accept": row["screen_accept"],
                    "reason": reason,
                    "volume": row["volume"],
                    "bbox_x": row["bbox_x"],
                    "bbox_y": row["bbox_y"],
                    "bbox_z": row["bbox_z"],
                    "elongation": row["elongation"],
                    "flatness": row["flatness"],
                    "compactness": row["compactness"],
                    "spike_score": row["spike_score"],
                    "stability_score": row["stability_score"],
                    "short_to_mid": row["short_to_mid"],
                }
            )
    return output


def role_score(row: dict[str, Any], role: str) -> tuple[float, str]:
    source = str(row["source_kind"])
    footprint = 0.5 * (float(row["bbox_x"]) + float(row["bbox_y"]))
    bbox_z = float(row["bbox_z"])
    volume = float(row["volume"])
    compactness = float(row["compactness"])
    elongation = float(row["elongation"])
    flatness = float(row["flatness"])
    spike = float(row["spike_score"])
    stability = float(row["stability_score"])
    screen_penalty = 100.0 if int(row["screen_accept"]) == 0 else 0.0

    if role == "base":
        target_footprint = 0.160
        target_z = 0.118
        score = (
            30.0 * abs(footprint - target_footprint)
            + 6.0 * abs(bbox_z - target_z)
            - 0.75 * compactness
            - 0.10 * stability
            + 5.0 * spike
            + 0.65 * max(0.0, elongation - 1.45)
            + 0.45 * max(0.0, flatness - 1.35)
            + screen_penalty
        )
        if source in {"bearing_block_clast", "buttress_clast", "subangular_block"}:
            score -= 0.34
        if source in {"chock_clast", "elongated_clast", "upright_block_clast", "tie_bridge_clast", "cap_block_clast"}:
            score += 0.28
        return score, "wide compact bearing stone with low spike and non-slab thickness"

    if role == "middle":
        target_footprint = 0.125
        target_z = 0.115
        score = (
            24.0 * abs(footprint - target_footprint)
            + 5.0 * abs(bbox_z - target_z)
            - 0.65 * compactness
            + 0.32 * abs(elongation - 1.18)
            + 0.28 * max(0.0, flatness - 1.32)
            + 5.0 * spike
            + screen_penalty
        )
        if source in {"course_block_clast", "compact_block_clast", "wall_block_clast", "equant_clast"}:
            score -= 0.26
        return score, "course stone with moderate height and broad contact"

    if role == "tie":
        target_elongation = 1.45
        target_footprint = 0.140
        score = (
            18.0 * abs(footprint - target_footprint)
            + 0.85 * abs(elongation - target_elongation)
            + 0.38 * max(0.0, flatness - 1.30)
            - 0.48 * compactness
            + 5.5 * spike
            + screen_penalty
        )
        if source in {"tie_bridge_clast", "wall_block_clast", "course_block_clast"}:
            score -= 0.32
        if elongation < 1.18:
            score += 0.20
        return score, "through/tie candidate with length but still thick enough"

    if role == "cap":
        target_footprint = 0.115
        target_z = 0.105
        score = (
            22.0 * abs(footprint - target_footprint)
            + 4.0 * abs(bbox_z - target_z)
            - 0.50 * compactness
            + 0.28 * max(0.0, elongation - 1.35)
            + 0.32 * max(0.0, flatness - 1.36)
            + 5.0 * spike
            + screen_penalty
        )
        if source in {"cap_block_clast", "compact_block_clast", "equant_clast"}:
            score -= 0.24
        return score, "upper locking stone, not a thin slab"

    target_volume = 0.00045
    score = (
        220.0 * abs(volume - target_volume)
        + 0.22 * abs(elongation - 1.12)
        + 0.24 * max(0.0, flatness - 1.35)
        + 5.0 * spike
        - 0.30 * compactness
        + screen_penalty
    )
    if source in {"chock_clast", "interlock_block_clast", "equant_clast"}:
        score -= 0.22
    return score, "small angular chock/fill candidate, not spike or slab"


def assign_to_target(
    rows: list[dict[str, Any]],
    role_rows: list[dict[str, Any]],
    target: str,
) -> list[dict[str, Any]]:
    slots = slots_for_target(target)
    scores = {
        (int(row["index"]), str(row["role"])): float(row["role_score"])
        for row in role_rows
    }
    row_by_index = {int(row["index"]): row for row in rows}
    used: set[int] = set()
    output: list[dict[str, Any]] = []
    for slot in slots:
        role = slot.role if slot.role in ROLES else "middle"
        ranked = sorted(
            (row for row in rows if int(row["index"]) not in used and int(row["screen_accept"]) == 1),
            key=lambda row: scores[(int(row["index"]), role)],
        )
        if not ranked:
            ranked = sorted(
                (row for row in rows if int(row["index"]) not in used),
                key=lambda row: scores[(int(row["index"]), role)],
            )
        chosen = ranked[0]
        idx = int(chosen["index"])
        used.add(idx)
        output.append(
            {
                "slot_id": slot.slot_id,
                "course": slot.course,
                "role": role,
                "target_x": slot.x,
                "target_y": slot.y,
                "rock_index": idx,
                "source_kind": chosen["source_kind"],
                "cluster_label": chosen["cluster_label"],
                "role_score": scores[(idx, role)],
                "screen_accept": chosen["screen_accept"],
                "bbox_x": chosen["bbox_x"],
                "bbox_y": chosen["bbox_y"],
                "bbox_z": chosen["bbox_z"],
                "elongation": chosen["elongation"],
                "flatness": chosen["flatness"],
                "compactness": chosen["compactness"],
                "spike_score": chosen["spike_score"],
                "short_to_mid": chosen["short_to_mid"],
            }
        )
    return output


def cluster_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    labels = sorted({str(row["cluster_label"]) for row in rows})
    for label in labels:
        subset = [row for row in rows if str(row["cluster_label"]) == label]
        output.append(
            {
                "cluster_label": label,
                "count": len(subset),
                "accepted": sum(int(row["screen_accept"]) for row in subset),
                "source_kind_counts": json.dumps(count_by(subset, "source_kind"), sort_keys=True),
                "mean_bbox_z": mean(float(row["bbox_z"]) for row in subset),
                "mean_elongation": mean(float(row["elongation"]) for row in subset),
                "mean_flatness": mean(float(row["flatness"]) for row in subset),
                "mean_spike_score": mean(float(row["spike_score"]) for row in subset),
                "mean_compactness": mean(float(row["compactness"]) for row in subset),
            }
        )
    return output


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def write_report(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    role_rows: list[dict[str, Any]],
    slot_rows: list[dict[str, Any]],
) -> None:
    accepted = [row for row in rows if int(row["screen_accept"]) == 1]
    rejected = [row for row in rows if int(row["screen_accept"]) == 0]
    source_counts = count_by(rows, "source_kind")
    accepted_counts = count_by(accepted, "source_kind")
    slot_counts = count_by(slot_rows, "source_kind")
    lines = [
        "# Targeted Rock Catalog Screening",
        "",
        f"- target: `{args.target}`",
        f"- profile: `{args.profile}`",
        f"- generated rocks: {len(rows)}",
        f"- accepted by geometry screen: {len(accepted)}",
        f"- rejected by geometry screen: {len(rejected)}",
        "",
        "## Geometry Screen",
        "",
        "Rejects candidates with excessive spike score, slab-like thickness, excessive elongation, or very low compactness. These are conservative filters; MuJoCo still performs final validation.",
        "",
        "## Source Kind Counts",
        "",
        "```json",
        json.dumps(source_counts, indent=2, sort_keys=True),
        "```",
        "",
        "## Accepted Source Kind Counts",
        "",
        "```json",
        json.dumps(accepted_counts, indent=2, sort_keys=True),
        "```",
        "",
        "## Greedy Target Assignment Counts",
        "",
        "```json",
        json.dumps(slot_counts, indent=2, sort_keys=True),
        "```",
        "",
        "## Role Guidance",
        "",
        "- base: wide compact bearing stones, typically `bearing_block_clast`, `buttress_clast`, or `subangular_block`.",
        "- middle: moderate compact course stones, typically `course_block_clast`, `compact_block_clast`, `wall_block_clast`, or `equant_clast`.",
        "- tie: length-biased but non-slab stones, typically `tie_bridge_clast` or thick `wall_block_clast`.",
        "- cap: compact locking stones, not thin plates.",
        "- chock: smaller angular fill stones for repair; do not count as structural wall stones unless explicitly placed as repair.",
        "",
        "## Next Use",
        "",
        "Use this catalog to decide whether a target has enough suitable lower-course stones before running a full MuJoCo experiment. If base or course-1 slots cannot be assigned accepted stones, generate a larger or more targeted catalog instead of proceeding to high-course simulation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
