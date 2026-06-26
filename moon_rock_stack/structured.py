from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .simulate import (
    GRAVITIES,
    _axis_angle,
    _body_positions,
    _quat_mul,
    _random_quaternion,
    _set_freejoint_pose,
    _simulate_until_quiet,
)


@dataclass(frozen=True)
class TargetSlot:
    slot_id: int
    course: int
    x: float
    y: float
    role: str


PROGRESS_FIELDS = [
    "time_s",
    "elapsed_s",
    "event",
    "target_name",
    "strategy",
    "gravity",
    "trial",
    "slot_index",
    "slot_id",
    "course",
    "role",
    "placed_count",
    "rock_index",
    "candidate_count",
    "stone_pool_size",
    "message",
]


def slots_for_target(target_name: str) -> list[TargetSlot]:
    if target_name == "wall_segment_v1":
        return landmark_wall_slots()
    if target_name == "single_face_wall_2course_v1":
        return single_face_wall_2course_slots_v1()
    if target_name == "single_face_wall_3course_v1":
        return single_face_wall_3course_slots_v1()
    if target_name == "tall_wall_v1":
        return tall_wall_slots()
    if target_name == "tall_wall_v2":
        return tall_wall_slots_v2()
    if target_name == "tall_wall_v3":
        return tall_wall_slots_v3()
    if target_name == "tall_wall_thick_v1":
        return tall_wall_thick_slots_v1()
    if target_name == "tall_wall_thick_v2":
        return tall_wall_thick_slots_v2()
    if target_name == "single_wall_strict_v1":
        return single_wall_strict_slots_v1()
    if target_name == "single_face_wall_v1":
        return single_face_wall_slots_v1()
    if target_name == "single_face_wall_high_v1":
        return single_face_wall_high_slots_v1()
    if target_name == "single_face_wall_extra_high_v1":
        return single_face_wall_extra_high_slots_v1()
    if target_name == "tied_high_wall_v1":
        return tied_high_wall_slots_v1()
    if target_name == "tied_high_wall_core_v1":
        return tied_high_wall_core_slots_v1()
    if target_name == "tied_wall_4course_v1":
        return tied_wall_4course_slots_v1()
    if target_name == "single_face_wall_4course_v1":
        return single_face_wall_4course_slots_v1()
    if target_name == "single_face_wall_5course_v1":
        return single_face_wall_5course_slots_v1()
    if target_name == "pillar_v1":
        return pillar_slots()
    if target_name == "single_column_v2":
        return single_column_slots()
    if target_name == "single_column_v3":
        return single_column_slots_v3()
    if target_name == "single_column_v4":
        return single_column_slots_v4()
    if target_name == "tall_pillar_v3":
        return tall_pillar_slots_v3()
    if target_name in {"multi_stone_column_v1", "stone_column_v1"}:
        return multi_stone_column_slots_v1()
    if target_name in {"multi_stone_column_v2", "stone_column_v2"}:
        return multi_stone_column_slots_v2()
    if target_name in {"multi_stone_column_v3", "stone_column_v3"}:
        return multi_stone_column_slots_v3()
    if target_name == "pillar_tripod_v2":
        return tapered_tripod_pillar_slots()
    if target_name == "four_wall_square_v1":
        return four_wall_square_slots()
    raise ValueError(f"Unknown structured target: {target_name}")


def _append_progress(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    payload = {field: row.get(field, "") for field in PROGRESS_FIELDS}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROGRESS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(payload)


def landmark_wall_slots(slot_spacing: float = 0.145) -> list[TargetSlot]:
    slots: list[TargetSlot] = []
    slot_id = 0
    for x in (-1.5 * slot_spacing, -0.5 * slot_spacing, 0.5 * slot_spacing, 1.5 * slot_spacing):
        slots.append(TargetSlot(slot_id=slot_id, course=0, x=x, y=0.0, role="base"))
        slot_id += 1
    for x in (-slot_spacing, 0.0, slot_spacing):
        slots.append(TargetSlot(slot_id=slot_id, course=1, x=x, y=0.0, role="middle"))
        slot_id += 1
    for x in (-0.5 * slot_spacing, 0.5 * slot_spacing):
        slots.append(TargetSlot(slot_id=slot_id, course=2, x=x, y=0.0, role="cap"))
        slot_id += 1
    return slots


def tall_wall_slots(slot_spacing: float = 0.128) -> list[TargetSlot]:
    slots: list[TargetSlot] = []
    slot_id = 0
    courses = [
        ("base", (-2.0, -1.0, 0.0, 1.0, 2.0)),
        ("middle", (-1.5, -0.5, 0.5, 1.5)),
        ("middle", (-1.2, -0.4, 0.4, 1.2)),
        ("middle", (-0.75, 0.0, 0.75)),
        ("cap", (-0.5, 0.5)),
    ]
    for course, (role, offsets) in enumerate(courses):
        for offset in offsets:
            slots.append(
                TargetSlot(
                    slot_id=slot_id,
                    course=course,
                    x=float(offset * slot_spacing),
                    y=0.0,
                    role=role,
                )
            )
            slot_id += 1
    return slots


def tall_wall_slots_v2(slot_spacing: float = 0.145, y_stagger: float = 0.018) -> list[TargetSlot]:
    slots: list[TargetSlot] = []
    slot_id = 0
    courses = [
        ("base", 0.0, (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)),
        ("middle", y_stagger, (-2.0, -1.0, 0.0, 1.0, 2.0)),
        ("middle", -y_stagger, (-1.5, -0.5, 0.5, 1.5)),
        ("middle", 0.5 * y_stagger, (-1.0, 0.0, 1.0)),
        ("cap", 0.0, (-0.5, 0.5)),
    ]
    for course, (role, y, offsets) in enumerate(courses):
        for offset in offsets:
            slots.append(
                TargetSlot(
                    slot_id=slot_id,
                    course=course,
                    x=float(offset * slot_spacing),
                    y=float(y),
                    role=role,
                )
            )
            slot_id += 1
    return slots


def tall_wall_slots_v3(slot_spacing: float = 0.142, y_stagger: float = 0.022) -> list[TargetSlot]:
    """Seven-course high single-wall target with a broad dry-stack base."""
    slots: list[TargetSlot] = []
    slot_id = 0
    courses = [
        ("base", 0.0, (-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5)),
        ("middle", y_stagger, (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)),
        ("middle", -y_stagger, (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)),
        ("middle", 0.65 * y_stagger, (-2.0, -1.0, 0.0, 1.0, 2.0)),
        ("middle", -0.50 * y_stagger, (-1.5, -0.5, 0.5, 1.5)),
        ("middle", 0.35 * y_stagger, (-1.0, 0.0, 1.0)),
        ("cap", 0.0, (-0.5, 0.5)),
    ]
    for course, (role, y, offsets) in enumerate(courses):
        for offset in offsets:
            slots.append(
                TargetSlot(
                    slot_id=slot_id,
                    course=course,
                    x=float(offset * slot_spacing),
                    y=float(y),
                    role=role,
                )
            )
            slot_id += 1
    return slots


def tall_wall_thick_slots_v1(slot_spacing: float = 0.138, half_depth: float = 0.058) -> list[TargetSlot]:
    """Seven-course single wall segment with thickness, batter, and tie stones."""
    slots: list[TargetSlot] = []
    slot_id = 0
    courses = [
        (0, 0.000, (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5), (-1.5, 1.5)),
        (1, 0.010, (-2.0, -1.0, 0.0, 1.0, 2.0), (-1.0, 1.0)),
        (2, 0.018, (-2.0, -1.0, 0.0, 1.0, 2.0), (0.0,)),
        (3, 0.026, (-1.5, -0.5, 0.5, 1.5), (-0.5, 0.5)),
        (4, 0.034, (-1.5, -0.5, 0.5, 1.5), (0.0,)),
        (5, 0.042, (-1.0, 0.0, 1.0), (0.0,)),
        (6, 0.050, (-1.0, 0.0, 1.0), ()),
    ]
    for course, batter, offsets, tie_offsets in courses:
        role = "base" if course == 0 else "cap" if course == len(courses) - 1 else "middle"
        depth = max(0.026, half_depth - batter)
        for face_y in (-depth, depth):
            for offset in offsets:
                slots.append(
                    TargetSlot(
                        slot_id=slot_id,
                        course=course,
                        x=float(offset * slot_spacing),
                        y=float(face_y),
                        role=role,
                    )
                )
                slot_id += 1
        if course < len(courses) - 1:
            for offset in tie_offsets:
                slots.append(
                    TargetSlot(
                        slot_id=slot_id,
                        course=course,
                        x=float(offset * slot_spacing),
                        y=0.0,
                        role="tie",
                    )
                )
                slot_id += 1
    return slots


def tall_wall_thick_slots_v2(slot_spacing: float = 0.132, half_depth: float = 0.050) -> list[TargetSlot]:
    """Narrower seven-course thick wall for height with less footprint scatter."""
    slots: list[TargetSlot] = []
    slot_id = 0
    courses = [
        (0, 0.000, (-2.0, -1.0, 0.0, 1.0, 2.0), (-1.0, 1.0)),
        (1, 0.008, (-1.5, -0.5, 0.5, 1.5), (0.0,)),
        (2, 0.014, (-1.5, -0.5, 0.5, 1.5), (-0.5, 0.5)),
        (3, 0.020, (-1.0, 0.0, 1.0), (0.0,)),
        (4, 0.026, (-1.0, 0.0, 1.0), (0.0,)),
        (5, 0.032, (-0.5, 0.5), (0.0,)),
        (6, 0.038, (-0.5, 0.5), ()),
    ]
    for course, batter, offsets, tie_offsets in courses:
        role = "base" if course == 0 else "cap" if course == len(courses) - 1 else "middle"
        depth = max(0.024, half_depth - batter)
        for face_y in (-depth, depth):
            for offset in offsets:
                slots.append(
                    TargetSlot(
                        slot_id=slot_id,
                        course=course,
                        x=float(offset * slot_spacing),
                        y=float(face_y),
                        role=role,
                    )
                )
                slot_id += 1
        if course < len(courses) - 1:
            for offset in tie_offsets:
                slots.append(
                    TargetSlot(
                        slot_id=slot_id,
                        course=course,
                        x=float(offset * slot_spacing),
                        y=0.0,
                        role="tie",
                    )
                )
                slot_id += 1
    return slots


def single_wall_strict_slots_v1(slot_spacing: float = 0.125, half_depth: float = 0.052) -> list[TargetSlot]:
    """Strict freestanding single wall: long x-axis, narrow y-axis, bonded courses."""
    slots: list[TargetSlot] = []
    slot_id = 0
    courses = [
        (0, 0.000, (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0), (-2.0, 0.0, 2.0)),
        (1, 0.006, (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5), (-1.5, 0.5)),
        (2, 0.012, (-2.0, -1.0, 0.0, 1.0, 2.0), (-1.0, 1.0)),
        (3, 0.018, (-1.5, -0.5, 0.5, 1.5), (-0.5, 0.5)),
        (4, 0.024, (-1.0, 0.0, 1.0), (0.0,)),
    ]
    for course, batter, offsets, tie_offsets in courses:
        role = "base" if course == 0 else "cap" if course == len(courses) - 1 else "middle"
        depth = max(0.030, half_depth - batter)
        for face_y in (-depth, depth):
            for offset in offsets:
                slots.append(
                    TargetSlot(
                        slot_id=slot_id,
                        course=course,
                        x=float(offset * slot_spacing),
                        y=float(face_y),
                        role=role,
                    )
                )
                slot_id += 1
        if course < len(courses) - 1:
            for offset in tie_offsets:
                slots.append(TargetSlot(slot_id=slot_id, course=course, x=float(offset * slot_spacing), y=0.0, role="tie"))
                slot_id += 1
    return slots


def single_face_wall_slots_v1(slot_spacing: float = 0.118, y_stagger: float = 0.008) -> list[TargetSlot]:
    """One-plane wall segment: a long bonded wall face, not a thick rubble pile."""
    slots: list[TargetSlot] = []
    slot_id = 0
    courses = [
        ("base", 0.0, (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)),
        ("middle", y_stagger, (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)),
        ("middle", -y_stagger, (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)),
        ("middle", y_stagger, (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)),
        ("cap", 0.0, (-2.0, -1.0, 0.0, 1.0, 2.0)),
    ]
    for course, (role, y, offsets) in enumerate(courses):
        for offset in offsets:
            slots.append(TargetSlot(slot_id=slot_id, course=course, x=float(offset * slot_spacing), y=float(y), role=role))
            slot_id += 1
    return slots


def single_face_wall_2course_slots_v1(slot_spacing: float = 0.125, y_stagger: float = 0.006) -> list[TargetSlot]:
    """Two-course single-face wall benchmark for foundation and low-wall calibration."""
    slots: list[TargetSlot] = []
    slot_id = 0
    courses = [
        ("base", 0.0, (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)),
        ("middle", y_stagger, (-2.0, -1.0, 0.0, 1.0, 2.0)),
    ]
    for course, (role, y, offsets) in enumerate(courses):
        for offset in offsets:
            slots.append(TargetSlot(slot_id=slot_id, course=course, x=float(offset * slot_spacing), y=float(y), role=role))
            slot_id += 1
    return slots


def single_face_wall_3course_slots_v1(slot_spacing: float = 0.122, y_stagger: float = 0.006) -> list[TargetSlot]:
    """Three-course single-face wall target for neural low-wall curriculum."""
    slots: list[TargetSlot] = []
    slot_id = 0
    courses = [
        ("base", 0.0, (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)),
        ("middle", y_stagger, (-2.0, -1.0, 0.0, 1.0, 2.0)),
        ("cap", 0.0, (-1.5, -0.5, 0.5, 1.5)),
    ]
    for course, (role, y, offsets) in enumerate(courses):
        for offset in offsets:
            slots.append(TargetSlot(slot_id=slot_id, course=course, x=float(offset * slot_spacing), y=float(y), role=role))
            slot_id += 1
    return slots


def single_face_wall_high_slots_v1(slot_spacing: float = 0.116, y_stagger: float = 0.006) -> list[TargetSlot]:
    """High one-plane wall segment: many bonded courses with strict depth control."""
    slots: list[TargetSlot] = []
    slot_id = 0
    courses = [
        ("base", 0.0, (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)),
        ("middle", y_stagger, (-2.0, -1.0, 0.0, 1.0, 2.0)),
        ("middle", -y_stagger, (-2.2, -1.1, 0.0, 1.1, 2.2)),
        ("middle", y_stagger, (-1.7, -0.55, 0.55, 1.7)),
        ("middle", -y_stagger, (-1.45, -0.45, 0.45, 1.45)),
        ("middle", 0.5 * y_stagger, (-1.0, 0.0, 1.0)),
        ("middle", -0.5 * y_stagger, (-0.75, 0.75)),
        ("cap", 0.0, (-0.45, 0.45)),
    ]
    for course, (role, y, offsets) in enumerate(courses):
        for offset in offsets:
            slots.append(TargetSlot(slot_id=slot_id, course=course, x=float(offset * slot_spacing), y=float(y), role=role))
            slot_id += 1
    return slots


def single_face_wall_extra_high_slots_v1(slot_spacing: float = 0.112, y_stagger: float = 0.005) -> list[TargetSlot]:
    """Ten-course one-plane wall target used to push beyond the eight-course high-wall benchmark."""
    slots: list[TargetSlot] = []
    slot_id = 0
    courses = [
        ("base", 0.0, (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)),
        ("middle", y_stagger, (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)),
        ("middle", -y_stagger, (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)),
        ("middle", 0.8 * y_stagger, (-2.0, -1.0, 0.0, 1.0, 2.0)),
        ("middle", -0.8 * y_stagger, (-2.0, -1.0, 0.0, 1.0, 2.0)),
        ("middle", 0.5 * y_stagger, (-1.5, -0.5, 0.5, 1.5)),
        ("middle", -0.5 * y_stagger, (-1.5, -0.5, 0.5, 1.5)),
        ("middle", 0.25 * y_stagger, (-1.0, 0.0, 1.0)),
        ("middle", -0.25 * y_stagger, (-0.75, 0.75)),
        ("cap", 0.0, (-0.45, 0.45)),
    ]
    for course, (role, y, offsets) in enumerate(courses):
        for offset in offsets:
            slots.append(TargetSlot(slot_id=slot_id, course=course, x=float(offset * slot_spacing), y=float(y), role=role))
            slot_id += 1
    return slots


def single_face_wall_4course_slots_v1(slot_spacing: float = 0.118, y_stagger: float = 0.006) -> list[TargetSlot]:
    """Four-course long single-face wall segment for staged height experiments."""
    slots: list[TargetSlot] = []
    slot_id = 0
    courses = [
        ("base", 0.0, (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)),
        ("middle", y_stagger, (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)),
        ("middle", -y_stagger, (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)),
        ("cap", 0.0, (-2.0, -1.0, 0.0, 1.0, 2.0)),
    ]
    for course, (role, y, offsets) in enumerate(courses):
        for offset in offsets:
            slots.append(TargetSlot(slot_id=slot_id, course=course, x=float(offset * slot_spacing), y=float(y), role=role))
            slot_id += 1
    return slots


def single_face_wall_5course_slots_v1(slot_spacing: float = 0.116, y_stagger: float = 0.006) -> list[TargetSlot]:
    """Five-course single-face wall benchmark used after the four-course gate is passed."""
    slots: list[TargetSlot] = []
    slot_id = 0
    courses = [
        ("base", 0.0, (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)),
        ("middle", y_stagger, (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)),
        ("middle", -y_stagger, (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)),
        ("middle", 0.5 * y_stagger, (-2.0, -1.0, 0.0, 1.0, 2.0)),
        ("cap", 0.0, (-1.5, -0.5, 0.5, 1.5)),
    ]
    for course, (role, y, offsets) in enumerate(courses):
        for offset in offsets:
            slots.append(TargetSlot(slot_id=slot_id, course=course, x=float(offset * slot_spacing), y=float(y), role=role))
            slot_id += 1
    return slots


def tied_high_wall_slots_v1(slot_spacing: float = 0.116, half_depth: float = 0.046) -> list[TargetSlot]:
    """High dry-stone wall segment with two faces, tie stones, and a narrow controlled thickness."""
    slots: list[TargetSlot] = []
    slot_id = 0
    courses = [
        (0, 0.000, (-1.5, -0.5, 0.5, 1.5), (0.0,)),
        (1, 0.005, (-1.0, 0.0, 1.0), ()),
        (2, 0.010, (-1.0, 0.0, 1.0), (0.0,)),
        (3, 0.015, (-0.5, 0.5), ()),
        (4, 0.020, (-0.5, 0.5), (0.0,)),
        (5, 0.025, (0.0,), ()),
        (6, 0.030, (0.0,), ()),
    ]
    for course, batter, offsets, tie_offsets in courses:
        role = "base" if course == 0 else "cap" if course == len(courses) - 1 else "middle"
        depth = max(0.024, half_depth - batter)
        for face_y in (-depth, depth):
            for offset in offsets:
                slots.append(TargetSlot(slot_id=slot_id, course=course, x=float(offset * slot_spacing), y=float(face_y), role=role))
                slot_id += 1
        for offset in tie_offsets:
            slots.append(TargetSlot(slot_id=slot_id, course=course, x=float(offset * slot_spacing), y=0.0, role="tie"))
            slot_id += 1
    return slots


def tied_high_wall_core_slots_v1(slot_spacing: float = 0.116, half_depth: float = 0.044) -> list[TargetSlot]:
    """Seven-course dry-stone wall core used before extending to a longer wall segment."""
    slots: list[TargetSlot] = []
    slot_id = 0
    courses = [
        (0, 0.000, (-0.75, 0.75), (0.0,)),
        (1, 0.005, (-0.55, 0.55), ()),
        (2, 0.010, (-0.50, 0.50), (0.0,)),
        (3, 0.015, (-0.38, 0.38), ()),
        (4, 0.020, (0.0,), (0.0,)),
        (5, 0.025, (0.0,), ()),
        (6, 0.030, (0.0,), ()),
    ]
    for course, batter, offsets, tie_offsets in courses:
        role = "base" if course == 0 else "cap" if course == len(courses) - 1 else "middle"
        depth = max(0.024, half_depth - batter)
        for face_y in (-depth, depth):
            for offset in offsets:
                slots.append(TargetSlot(slot_id=slot_id, course=course, x=float(offset * slot_spacing), y=float(face_y), role=role))
                slot_id += 1
        for offset in tie_offsets:
            slots.append(TargetSlot(slot_id=slot_id, course=course, x=float(offset * slot_spacing), y=0.0, role="tie"))
            slot_id += 1
    return slots


def tied_wall_4course_slots_v1(slot_spacing: float = 0.116, half_depth: float = 0.044) -> list[TargetSlot]:
    """Four-course tied wall base used as a staged precursor to high-wall experiments."""
    slots: list[TargetSlot] = []
    slot_id = 0
    courses = [
        (0, 0.000, (-0.75, 0.75), (0.0,)),
        (1, 0.006, (-0.55, 0.55), ()),
        (2, 0.012, (-0.50, 0.50), (0.0,)),
        (3, 0.018, (-0.38, 0.38), ()),
    ]
    for course, batter, offsets, tie_offsets in courses:
        role = "base" if course == 0 else "cap" if course == len(courses) - 1 else "middle"
        depth = max(0.026, half_depth - batter)
        for face_y in (-depth, depth):
            for offset in offsets:
                slots.append(TargetSlot(slot_id=slot_id, course=course, x=float(offset * slot_spacing), y=float(face_y), role=role))
                slot_id += 1
        for offset in tie_offsets:
            slots.append(TargetSlot(slot_id=slot_id, course=course, x=float(offset * slot_spacing), y=0.0, role="tie"))
            slot_id += 1
    return slots


def pillar_slots(course_count: int = 6) -> list[TargetSlot]:
    slots: list[TargetSlot] = []
    for course in range(course_count):
        role = "base" if course == 0 else "cap" if course == course_count - 1 else "middle"
        slots.append(TargetSlot(slot_id=course, course=course, x=0.0, y=0.0, role=role))
    return slots


def single_column_slots(course_count: int = 5, micro_offset: float = 0.012) -> list[TargetSlot]:
    offsets = [
        (0.0, 0.0),
        (micro_offset, 0.0),
        (-micro_offset, 0.0),
        (0.0, micro_offset),
        (0.0, -micro_offset),
    ]
    slots: list[TargetSlot] = []
    for course in range(course_count):
        role = "base" if course == 0 else "cap" if course == course_count - 1 else "middle"
        x, y = offsets[course % len(offsets)]
        slots.append(TargetSlot(slot_id=course, course=course, x=x, y=y, role=role))
    return slots


def single_column_slots_v3(course_count: int = 8, micro_offset: float = 0.010) -> list[TargetSlot]:
    """Eight-course high freestanding column target with small contact-search offsets."""
    offsets = [
        (0.0, 0.0),
        (micro_offset, 0.0),
        (0.0, micro_offset),
        (-micro_offset, 0.0),
        (0.0, -micro_offset),
        (0.65 * micro_offset, 0.65 * micro_offset),
        (-0.65 * micro_offset, 0.65 * micro_offset),
        (0.65 * micro_offset, -0.65 * micro_offset),
    ]
    slots: list[TargetSlot] = []
    for course in range(course_count):
        role = "base" if course == 0 else "cap" if course == course_count - 1 else "middle"
        x, y = offsets[course % len(offsets)]
        slots.append(TargetSlot(slot_id=course, course=course, x=float(x), y=float(y), role=role))
    return slots


def single_column_slots_v4(course_count: int = 10, micro_offset: float = 0.006) -> list[TargetSlot]:
    """Strict one-stone-per-course column target; no heap surrogate is allowed."""
    offsets = [
        (0.0, 0.0),
        (micro_offset, 0.0),
        (0.0, micro_offset),
        (-micro_offset, 0.0),
        (0.0, -micro_offset),
        (0.55 * micro_offset, 0.55 * micro_offset),
        (-0.55 * micro_offset, 0.55 * micro_offset),
        (-0.55 * micro_offset, -0.55 * micro_offset),
        (0.55 * micro_offset, -0.55 * micro_offset),
        (0.0, 0.0),
    ]
    slots: list[TargetSlot] = []
    for course in range(course_count):
        role = "base" if course == 0 else "cap" if course == course_count - 1 else "middle"
        x, y = offsets[course % len(offsets)]
        slots.append(TargetSlot(slot_id=course, course=course, x=float(x), y=float(y), role=role))
    return slots


def tall_pillar_slots_v3(base_radius: float = 0.095, course_gap: float = 0.010) -> list[TargetSlot]:
    """High cairn-like pillar with multi-stone lower support courses."""
    slots: list[TargetSlot] = []
    slot_id = 0
    course_specs = [
        ("base", base_radius, (45.0, 135.0, 225.0, 315.0)),
        ("middle", base_radius * 0.78, (90.0, 210.0, 330.0)),
        ("middle", base_radius * 0.66, (30.0, 150.0, 270.0)),
        ("middle", base_radius * 0.52, (90.0, 270.0)),
        ("middle", base_radius * 0.40, (0.0, 180.0)),
        ("middle", 0.012 + course_gap, (90.0,)),
        ("middle", 0.010, (270.0,)),
        ("cap", 0.0, (0.0,)),
    ]
    for course, (role, radius, angles) in enumerate(course_specs):
        for angle in angles:
            radians = np.deg2rad(angle)
            slots.append(
                TargetSlot(
                    slot_id=slot_id,
                    course=course,
                    x=float(radius * np.cos(radians)),
                    y=float(radius * np.sin(radians)),
                    role=role,
                )
            )
            slot_id += 1
    return slots


def multi_stone_column_slots_v1() -> list[TargetSlot]:
    """Many-stone vertical column: stacked rings plus compact core stones."""
    slots: list[TargetSlot] = []
    slot_id = 0
    course_specs = [
        ("base", 0.125, 6, 0.0, True),
        ("middle", 0.114, 6, 30.0, True),
        ("middle", 0.102, 5, 0.0, True),
        ("middle", 0.090, 5, 36.0, True),
        ("middle", 0.078, 4, 45.0, True),
        ("middle", 0.064, 4, 0.0, False),
        ("middle", 0.050, 3, 60.0, False),
        ("cap", 0.036, 2, 90.0, False),
    ]
    for course, (role, radius, count, phase_deg, include_core) in enumerate(course_specs):
        for item in range(count):
            radians = np.deg2rad(phase_deg + item * 360.0 / count)
            slots.append(
                TargetSlot(
                    slot_id=slot_id,
                    course=course,
                    x=float(radius * np.cos(radians)),
                    y=float(radius * np.sin(radians)),
                    role=role,
                )
            )
            slot_id += 1
        if include_core:
            slots.append(TargetSlot(slot_id=slot_id, course=course, x=0.0, y=0.0, role="core"))
            slot_id += 1
    return slots


def multi_stone_column_slots_v2() -> list[TargetSlot]:
    """Narrow many-stone column: many rocks, tighter radius, still not a heap."""
    slots: list[TargetSlot] = []
    slot_id = 0
    course_specs = [
        ("base", 0.098, 6, 0.0, True),
        ("middle", 0.090, 6, 30.0, True),
        ("middle", 0.080, 5, 0.0, True),
        ("middle", 0.070, 5, 36.0, True),
        ("middle", 0.058, 4, 45.0, True),
        ("middle", 0.048, 4, 0.0, False),
        ("middle", 0.038, 3, 60.0, False),
        ("cap", 0.028, 2, 90.0, False),
    ]
    for course, (role, radius, count, phase_deg, include_core) in enumerate(course_specs):
        for item in range(count):
            radians = np.deg2rad(phase_deg + item * 360.0 / count)
            slots.append(
                TargetSlot(
                    slot_id=slot_id,
                    course=course,
                    x=float(radius * np.cos(radians)),
                    y=float(radius * np.sin(radians)),
                    role=role,
                )
            )
            slot_id += 1
        if include_core:
            slots.append(TargetSlot(slot_id=slot_id, course=course, x=0.0, y=0.0, role="core"))
            slot_id += 1
    return slots


def multi_stone_column_slots_v3() -> list[TargetSlot]:
    """Tapered many-stone column: compact rings with core fill and a modest base."""
    slots: list[TargetSlot] = []
    slot_id = 0
    course_specs = [
        ("base", 0.116, 6, 0.0, True),
        ("middle", 0.106, 6, 30.0, True),
        ("middle", 0.094, 5, 0.0, True),
        ("middle", 0.082, 5, 36.0, True),
        ("middle", 0.070, 4, 45.0, True),
        ("middle", 0.058, 4, 0.0, False),
        ("middle", 0.046, 3, 60.0, False),
        ("cap", 0.034, 2, 90.0, False),
    ]
    for course, (role, radius, count, phase_deg, include_core) in enumerate(course_specs):
        for item in range(count):
            radians = np.deg2rad(phase_deg + item * 360.0 / count)
            slots.append(
                TargetSlot(
                    slot_id=slot_id,
                    course=course,
                    x=float(radius * np.cos(radians)),
                    y=float(radius * np.sin(radians)),
                    role=role,
                )
            )
            slot_id += 1
        if include_core:
            slots.append(TargetSlot(slot_id=slot_id, course=course, x=0.0, y=0.0, role="core"))
            slot_id += 1
    return slots


def tapered_tripod_pillar_slots(base_radius: float = 0.078, middle_radius: float = 0.043) -> list[TargetSlot]:
    slots: list[TargetSlot] = []
    slot_id = 0
    for angle in (90.0, 210.0, 330.0):
        radians = np.deg2rad(angle)
        slots.append(
            TargetSlot(
                slot_id=slot_id,
                course=0,
                x=float(base_radius * np.cos(radians)),
                y=float(base_radius * np.sin(radians)),
                role="base",
            )
        )
        slot_id += 1
    for angle in (0.0, 180.0):
        radians = np.deg2rad(angle)
        slots.append(
            TargetSlot(
                slot_id=slot_id,
                course=1,
                x=float(middle_radius * np.cos(radians)),
                y=float(middle_radius * np.sin(radians)),
                role="middle",
            )
        )
        slot_id += 1
    slots.append(TargetSlot(slot_id=slot_id, course=2, x=0.0, y=0.0, role="cap"))
    return slots


def four_wall_square_slots(half_extent: float = 0.245, slot_spacing: float = 0.126) -> list[TargetSlot]:
    """Small four-wall enclosure: 4 sides, 3 courses, 24 stones total."""
    slots: list[TargetSlot] = []
    side_specs = (
        ("north", np.array([1.0, 0.0]), np.array([0.0, half_extent])),
        ("south", np.array([1.0, 0.0]), np.array([0.0, -half_extent])),
        ("east", np.array([0.0, 1.0]), np.array([half_extent, 0.0])),
        ("west", np.array([0.0, 1.0]), np.array([-half_extent, 0.0])),
    )
    slot_id = 0
    for _side_name, axis, center in side_specs:
        for offset in (-slot_spacing, 0.0, slot_spacing):
            point = center + axis * offset
            slots.append(TargetSlot(slot_id=slot_id, course=0, x=float(point[0]), y=float(point[1]), role="base"))
            slot_id += 1
        for offset in (-0.5 * slot_spacing, 0.5 * slot_spacing):
            point = center + axis * offset
            slots.append(TargetSlot(slot_id=slot_id, course=1, x=float(point[0]), y=float(point[1]), role="middle"))
            slot_id += 1
        slots.append(TargetSlot(slot_id=slot_id, course=2, x=float(center[0]), y=float(center[1]), role="cap"))
        slot_id += 1
    return slots


def structured_order(
    rows: list[dict[str, Any]],
    slots: list[TargetSlot],
    strategy: str = "geometry_bonded",
    rng: np.random.Generator | None = None,
) -> list[int]:
    if strategy == "random_order":
        if rng is None:
            rng = np.random.default_rng(0)
        selected = [int(row["index"]) for row in rows]
        rng.shuffle(selected)
        return selected[: len(slots)]

    remaining = {int(row["index"]): dict(row) for row in rows}
    order: list[int] = []
    for slot in slots:
        chosen = min(remaining.values(), key=lambda row: _stone_role_score(row, slot.role, strategy))
        idx = int(chosen["index"])
        order.append(idx)
        remaining.pop(idx)
    return order


def _ranker_for_course(
    ranker: dict[str, Any] | None,
    course: int,
    max_course: int,
) -> dict[str, Any] | None:
    if ranker is None:
        return None
    if int(max_course) < 0:
        return ranker
    if int(course) <= int(max_course):
        return ranker
    return None


def run_structured_trial_detailed(
    xml_path: Path,
    rows: list[dict[str, Any]],
    slots: list[TargetSlot],
    order: list[int],
    gravity_label: str,
    trial_id: int,
    seed: int,
    steps_per_rock: int,
    hold_steps: int,
    strategy: str,
    candidate_count: int,
    target_name: str = "wall_segment_v1",
    assignment_gate: bool = False,
    assignment_candidates_by_slot: dict[int, list[int]] | None = None,
    assignment_probe_steps: int = 0,
    candidate_probe_steps: int = 0,
    candidate_probe_hard_gate: bool = False,
    moon_gate_strict: bool = False,
    candidate_pose_ranker: dict[str, Any] | None = None,
    candidate_pose_top_k: int = 0,
    candidate_pose_ranker_max_course: int = -1,
    pose_risk_ranker: dict[str, Any] | None = None,
    pose_risk_weight: float = 0.0,
    pose_risk_ranker_max_course: int = -1,
    stone_fit_ranker: dict[str, Any] | None = None,
    stone_fit_top_k: int = 0,
    stone_fit_ranker_max_course: int = -1,
    state_snapshots: list[dict[str, Any]] | None = None,
    commit_best_rejected: bool = False,
    low_release_search: bool = False,
    release_search_step_m: float = 0.004,
    release_extra_clearance_m: float = 0.003,
    base_support_prior: bool = False,
    base_support_prior_weight: float = 1.0,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo Python is not installed in the active environment.") from exc

    start_time = time.monotonic()
    progress_base = {
        "target_name": target_name,
        "strategy": strategy,
        "gravity": gravity_label,
        "trial": trial_id,
    }
    _append_progress(
        progress_path,
        {
            **progress_base,
            "time_s": time.time(),
            "elapsed_s": 0.0,
            "event": "trial_start",
            "slot_index": -1,
            "slot_id": -1,
            "course": -1,
            "placed_count": 0,
            "rock_index": -1,
            "candidate_count": candidate_count,
            "message": str(xml_path),
        },
    )
    rng = np.random.default_rng(seed + trial_id * 7919 + int(GRAVITIES[gravity_label] * 100))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    _append_progress(
        progress_path,
        {
            **progress_base,
            "time_s": time.time(),
            "elapsed_s": time.monotonic() - start_time,
            "event": "model_loaded",
            "slot_index": -1,
            "slot_id": -1,
            "course": -1,
            "placed_count": 0,
            "rock_index": -1,
            "candidate_count": candidate_count,
        },
    )
    row_by_index = {int(row["index"]): row for row in rows}
    placed: list[int] = []
    slot_by_rock: dict[int, TargetSlot] = {}
    placement_rows: list[dict[str, Any]] = []
    candidate_pose_rows: list[dict[str, Any]] = []

    effective_steps_per_rock = _effective_settle_steps(steps_per_rock, gravity_label)
    effective_hold_steps = _effective_settle_steps(hold_steps, gravity_label)
    effective_candidate_probe_steps = _effective_settle_steps(candidate_probe_steps, gravity_label)

    mujoco.mj_forward(model, data)
    if state_snapshots is not None:
        _append_state_snapshot(
            state_snapshots,
            data,
            label="initial",
            slot_index=-1,
            rock_index=-1,
            slot=None,
            placed=placed,
        )
    for slot_index, slot in enumerate(slots):
        _append_progress(
            progress_path,
            {
                **progress_base,
                "time_s": time.time(),
                "elapsed_s": time.monotonic() - start_time,
                "event": "slot_start",
                "slot_index": slot_index,
                "slot_id": slot.slot_id,
                "course": slot.course,
                "role": slot.role,
                "placed_count": len(placed),
                "rock_index": -1,
                "candidate_count": candidate_count,
            },
        )
        slot_candidate_pose_ranker = _ranker_for_course(
            candidate_pose_ranker,
            int(slot.course),
            int(candidate_pose_ranker_max_course),
        )
        slot_candidate_pose_top_k = int(candidate_pose_top_k) if slot_candidate_pose_ranker is not None else 0
        slot_pose_risk_ranker = _ranker_for_course(
            pose_risk_ranker,
            int(slot.course),
            int(pose_risk_ranker_max_course),
        )
        slot_pose_risk_weight = float(pose_risk_weight) if slot_pose_risk_ranker is not None else 0.0
        slot_stone_fit_ranker = _ranker_for_course(
            stone_fit_ranker,
            int(slot.course),
            int(stone_fit_ranker_max_course),
        )
        slot_stone_fit_top_k = int(stone_fit_top_k) if slot_stone_fit_ranker is not None else 0
        if strategy in {"literature_column", "literature_wall", "statics_wall", "statics_wall_line_lock"}:
            rock_index, selected = _place_literature_slot(
                mujoco=mujoco,
                model=model,
                data=data,
                row_by_index=row_by_index,
                placed=placed,
                slot_by_rock=slot_by_rock,
                slot=slot,
                rng=rng,
                steps_per_rock=effective_steps_per_rock,
                candidate_count=max(candidate_count, 1),
                used=set(placed),
                strategy=strategy,
                target_name=target_name,
                candidate_pose_rows=candidate_pose_rows,
                candidate_context={
                    "gravity": gravity_label,
                    "gravity_m_s2": GRAVITIES[gravity_label],
                    "trial": trial_id,
                    "strategy": strategy,
                    "target_name": target_name,
                },
                candidate_pose_ranker=slot_candidate_pose_ranker,
                candidate_pose_top_k=slot_candidate_pose_top_k,
                pose_risk_ranker=slot_pose_risk_ranker,
                pose_risk_weight=slot_pose_risk_weight,
                stone_fit_ranker=slot_stone_fit_ranker,
                stone_fit_top_k=slot_stone_fit_top_k,
                commit_best_rejected=commit_best_rejected,
                candidate_probe_steps=effective_candidate_probe_steps,
                candidate_probe_hard_gate=candidate_probe_hard_gate,
                low_release_search=low_release_search,
                release_search_step_m=release_search_step_m,
                release_extra_clearance_m=release_extra_clearance_m,
                base_support_prior=base_support_prior,
                base_support_prior_weight=base_support_prior_weight,
            )
        else:
            if slot_index >= len(order):
                break
            if assignment_candidates_by_slot is not None:
                candidate_rocks = assignment_candidates_by_slot.get(slot.slot_id, [order[slot_index]])
                rock_index, selected = _place_assignment_slot(
                    mujoco=mujoco,
                    model=model,
                    data=data,
                    row_by_index=row_by_index,
                    placed=placed,
                    slot_by_rock=slot_by_rock,
                    candidate_rocks=candidate_rocks,
                    slot=slot,
                    rng=rng,
                    steps_per_rock=effective_steps_per_rock,
                    candidate_count=max(candidate_count, 1),
                    strategy=strategy,
                    target_name=target_name,
                    assignment_gate=assignment_gate,
                    gravity_label=gravity_label,
                    trial_id=trial_id,
                    assignment_probe_steps=assignment_probe_steps,
                    candidate_probe_steps=effective_candidate_probe_steps,
                    candidate_probe_hard_gate=candidate_probe_hard_gate,
                    moon_gate_strict=moon_gate_strict,
                    candidate_pose_rows=candidate_pose_rows,
                    candidate_pose_ranker=slot_candidate_pose_ranker,
                    candidate_pose_top_k=slot_candidate_pose_top_k,
                    pose_risk_ranker=slot_pose_risk_ranker,
                    pose_risk_weight=slot_pose_risk_weight,
                    low_release_search=low_release_search,
                    release_search_step_m=release_search_step_m,
                    release_extra_clearance_m=release_extra_clearance_m,
                )
            else:
                qpos_before_slot = data.qpos.copy()
                qvel_before_slot = data.qvel.copy()
                rock_index = order[slot_index]
                row_snapshot = dict(row_by_index[rock_index])
                selected = _place_for_target_slot(
                    mujoco=mujoco,
                    model=model,
                    data=data,
                    row_by_index=row_by_index,
                    placed=placed,
                    slot_by_rock=slot_by_rock,
                    rock_index=rock_index,
                    slot=slot,
                    rng=rng,
                    steps_per_rock=effective_steps_per_rock,
                    candidate_count=max(candidate_count, 1),
                    strategy=strategy,
                    candidate_pose_rows=candidate_pose_rows,
                    candidate_context={
                        "gravity": gravity_label,
                        "gravity_m_s2": GRAVITIES[gravity_label],
                        "trial": trial_id,
                        "strategy": strategy,
                        "target_name": target_name,
                        "assignment_fallback_attempt": 0,
                        "assignment_candidate_count": 1,
                        "assignment_candidate_rock_index": rock_index,
                    },
                    candidate_pose_ranker=slot_candidate_pose_ranker,
                    candidate_pose_top_k=slot_candidate_pose_top_k,
                    pose_risk_ranker=slot_pose_risk_ranker,
                    pose_risk_weight=slot_pose_risk_weight,
                    low_release_search=low_release_search,
                    release_search_step_m=release_search_step_m,
                    release_extra_clearance_m=release_extra_clearance_m,
                )
                if assignment_probe_steps > 0:
                    _probe_assignment_candidate(
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        row_by_index=row_by_index,
                        placed=placed,
                        slot_by_rock=slot_by_rock,
                        rock_index=rock_index,
                        slot=slot,
                        selected=selected,
                        probe_steps=assignment_probe_steps,
                    )
                if assignment_gate and not _assignment_candidate_is_feasible(
                    selected,
                    slot,
                    target_name,
                    gravity_label=gravity_label,
                    moon_gate_strict=moon_gate_strict,
                ):
                    rejected_rock_index = rock_index
                    data.qpos[:] = qpos_before_slot
                    data.qvel[:] = qvel_before_slot
                    row_by_index[rejected_rock_index].clear()
                    row_by_index[rejected_rock_index].update(row_snapshot)
                    mujoco.mj_forward(model, data)
                    rock_index = -1
                    selected = {
                        **selected,
                        "placement_skipped": 1,
                        "skip_reason": "assignment_gate_infeasible",
                        "best_rejected_rock_index": rejected_rock_index,
                    }
        if rock_index < 0:
            selected.update(
                {
                    "gravity": gravity_label,
                    "trial": trial_id,
                    "strategy": strategy,
                    "target_name": target_name,
                    "rock_index": rock_index,
                    "slot_id": slot.slot_id,
                    "course": slot.course,
                    "role": slot.role,
                    "target_x": slot.x,
                    "target_y": slot.y,
                    "cluster_label": "skipped",
                    "source_kind": "skipped",
                }
            )
            placement_rows.append(selected)
            _append_progress(
                progress_path,
                {
                    **progress_base,
                    "time_s": time.time(),
                    "elapsed_s": time.monotonic() - start_time,
                    "event": "slot_done",
                    "slot_index": slot_index,
                    "slot_id": slot.slot_id,
                    "course": slot.course,
                    "role": slot.role,
                    "placed_count": len(placed),
                    "rock_index": rock_index,
                    "candidate_count": candidate_count,
                    "stone_pool_size": selected.get("stone_pool_size", ""),
                    "message": selected.get("skip_reason", "skipped"),
                },
            )
            continue
        placed.append(rock_index)
        slot_by_rock[rock_index] = slot
        selected.update(
            {
                "gravity": gravity_label,
                "trial": trial_id,
                "strategy": strategy,
                "target_name": target_name,
                "rock_index": rock_index,
                "slot_id": slot.slot_id,
                "course": slot.course,
                "role": slot.role,
                "target_x": slot.x,
                "target_y": slot.y,
                "cluster_label": row_by_index[rock_index]["cluster_label"],
                "source_kind": row_by_index[rock_index]["source_kind"],
            }
        )
        placement_rows.append(selected)
        _append_progress(
            progress_path,
            {
                **progress_base,
                "time_s": time.time(),
                "elapsed_s": time.monotonic() - start_time,
                "event": "slot_done",
                "slot_index": slot_index,
                "slot_id": slot.slot_id,
                "course": slot.course,
                "role": slot.role,
                "placed_count": len(placed),
                "rock_index": rock_index,
                "candidate_count": candidate_count,
                "stone_pool_size": selected.get("stone_pool_size", ""),
                "message": selected.get("skip_reason", ""),
            },
        )
        if state_snapshots is not None:
            mujoco.mj_forward(model, data)
            _append_state_snapshot(
                state_snapshots,
                data,
                label="placed",
                slot_index=slot_index,
                rock_index=rock_index,
                slot=slot,
                placed=placed,
            )

    _append_progress(
        progress_path,
        {
            **progress_base,
            "time_s": time.time(),
            "elapsed_s": time.monotonic() - start_time,
            "event": "final_hold_start",
            "slot_index": len(slots),
            "slot_id": -1,
            "course": -1,
            "placed_count": len(placed),
            "rock_index": -1,
            "candidate_count": candidate_count,
        },
    )
    before = _body_positions(model, data, placed)
    _simulate_until_quiet(mujoco, model, data, effective_hold_steps)
    after = _body_positions(model, data, placed)
    speed = float(np.linalg.norm(data.qvel, ord=np.inf))
    drifts = [float(np.linalg.norm(after[idx][:2] - before[idx][:2])) for idx in placed]
    target_errors = {
        idx: float(np.linalg.norm(after[idx][:2] - np.array([slot_by_rock[idx].x, slot_by_rock[idx].y])))
        for idx in placed
    }
    stable_target_error_limit = _stable_target_error_limit(target_name)
    radial_limit = _target_radius_limit(target_name)
    wall_y_limit = _target_wall_y_limit(target_name)
    stable_by_index: dict[int, bool] = {}
    for idx in placed:
        row = row_by_index[idx]
        stable_by_index[idx] = (
            float(after[idx][2]) > 0.35 * float(row["bbox_z"])
            and target_errors[idx] < stable_target_error_limit
            and float(np.linalg.norm(after[idx][:2])) < radial_limit
            and (wall_y_limit <= 0.0 or abs(float(after[idx][1])) <= wall_y_limit)
        )
    stable_count = int(sum(stable_by_index.values()))
    rmse = float(np.sqrt(np.mean([error * error for error in target_errors.values()]))) if target_errors else 0.0
    max_error = max(target_errors.values()) if target_errors else 0.0
    top_z = _structured_top(after, row_by_index, placed)
    max_drift = max(drifts) if drifts else 0.0
    max_radial_distance = float(max(float(np.linalg.norm(after[idx][:2])) for idx in placed) if placed else 0.0)
    visible_courses = _visible_course_count(after, row_by_index, slot_by_rock, placed)
    wall_metrics = _wall_shape_metrics(target_name, after, placed)
    stable_fraction = stable_count / max(len(placed), 1)
    structure_score = (
        stable_fraction
        + 0.20 * visible_courses
        - 2.0 * rmse
        - 0.8 * max_drift
        - 0.12 * speed
    )
    if _is_single_wall_target(target_name):
        structure_score += 0.20 * min(wall_metrics["wall_aspect_xy"], 4.0)
        structure_score -= 0.9 * max(0.0, wall_metrics["wall_y_span_m"] - _target_wall_max_y_span(target_name))
        structure_score -= 0.18 * wall_metrics["wall_outlier_count"]
    required_courses = _required_visible_courses(target_name, slots)
    if target_name == "four_wall_square_v1":
        allowed_unstable = 2
    elif target_name == "single_face_wall_3course_v1":
        allowed_unstable = 3
    elif target_name == "tall_wall_thick_v1":
        allowed_unstable = 7
    elif target_name == "tall_wall_thick_v2":
        allowed_unstable = 5
    elif target_name in {"single_wall_strict_v1", "single_face_wall_v1"}:
        allowed_unstable = 8
    elif target_name in {"single_face_wall_high_v1", "single_face_wall_extra_high_v1"}:
        allowed_unstable = 9
    elif target_name == "tied_high_wall_v1":
        allowed_unstable = 10
    elif target_name == "tied_high_wall_core_v1":
        allowed_unstable = 6
    elif target_name == "tied_wall_4course_v1":
        allowed_unstable = 4
    elif target_name == "single_face_wall_4course_v1":
        allowed_unstable = 5
    elif target_name == "single_face_wall_5course_v1":
        allowed_unstable = 6
    elif target_name == "tall_wall_v3":
        allowed_unstable = 4
    elif target_name == "tall_pillar_v3":
        allowed_unstable = 2
    elif target_name in {"multi_stone_column_v1", "stone_column_v1"}:
        allowed_unstable = 8
    elif target_name in {"multi_stone_column_v2", "stone_column_v2"}:
        allowed_unstable = 8
    elif target_name in {"multi_stone_column_v3", "stone_column_v3"}:
        allowed_unstable = 7
    elif target_name == "tall_wall_v2":
        allowed_unstable = 2
    elif target_name == "single_column_v3":
        allowed_unstable = 1
    elif target_name == "single_column_v4":
        allowed_unstable = 1
    elif target_name == "single_column_v2":
        allowed_unstable = 0
    else:
        allowed_unstable = 1
    if target_name == "wall_segment_v1":
        rmse_limit = 0.095
        max_error_limit = 0.18
    elif target_name == "single_face_wall_3course_v1":
        rmse_limit = 0.150
        max_error_limit = 0.280
    elif target_name == "tall_wall_v1":
        rmse_limit = 0.120
        max_error_limit = 0.240
    elif target_name == "tall_wall_v2":
        rmse_limit = 0.150
        max_error_limit = 0.320
    elif target_name == "tall_wall_v3":
        rmse_limit = 0.175
        max_error_limit = 0.380
    elif target_name == "tall_wall_thick_v1":
        rmse_limit = 0.190
        max_error_limit = 0.420
    elif target_name == "tall_wall_thick_v2":
        rmse_limit = 0.165
        max_error_limit = 0.340
    elif target_name in {"single_wall_strict_v1", "single_face_wall_v1"}:
        rmse_limit = 0.155
        max_error_limit = 0.300
    elif target_name == "single_face_wall_extra_high_v1":
        rmse_limit = 0.185
        max_error_limit = 0.380
    elif target_name == "single_face_wall_high_v1":
        rmse_limit = 0.170
        max_error_limit = 0.330
    elif target_name == "tied_high_wall_v1":
        rmse_limit = 0.185
        max_error_limit = 0.350
    elif target_name == "tied_high_wall_core_v1":
        rmse_limit = 0.175
        max_error_limit = 0.320
    elif target_name == "tied_wall_4course_v1":
        rmse_limit = 0.145
        max_error_limit = 0.260
    elif target_name == "single_face_wall_4course_v1":
        rmse_limit = 0.150
        max_error_limit = 0.290
    elif target_name == "single_face_wall_5course_v1":
        rmse_limit = 0.165
        max_error_limit = 0.320
    elif target_name == "tall_pillar_v3":
        rmse_limit = 0.170
        max_error_limit = 0.310
    elif target_name in {"multi_stone_column_v1", "stone_column_v1"}:
        rmse_limit = 0.190
        max_error_limit = 0.380
    elif target_name in {"multi_stone_column_v2", "stone_column_v2"}:
        rmse_limit = 0.165
        max_error_limit = 0.320
    elif target_name in {"multi_stone_column_v3", "stone_column_v3"}:
        rmse_limit = 0.150
        max_error_limit = 0.280
    elif target_name == "single_column_v3":
        rmse_limit = 0.180
        max_error_limit = 0.300
    elif target_name == "single_column_v4":
        rmse_limit = 0.125
        max_error_limit = 0.210
    elif target_name == "single_column_v2":
        rmse_limit = 0.155
        max_error_limit = 0.240
    elif target_name == "pillar_tripod_v2":
        rmse_limit = 0.095
        max_error_limit = 0.18
    elif target_name == "four_wall_square_v1":
        rmse_limit = 0.120
        max_error_limit = 0.240
    else:
        rmse_limit = 0.080
        max_error_limit = 0.14
    shape_success = int(
        len(placed) >= len(slots) - allowed_unstable
        and stable_count >= len(placed) - allowed_unstable
        and rmse < rmse_limit
        and max_error < max_error_limit
        and visible_courses >= required_courses
        and top_z >= _target_min_height(target_name)
        and max_radial_distance <= radial_limit
        and _wall_shape_success(target_name, wall_metrics)
    )
    success = int(shape_success and max_drift < 0.15 and speed < 0.25)

    failures = _structured_failure_rows(
        gravity_label=gravity_label,
        trial_id=trial_id,
        strategy=strategy,
        target_name=target_name,
        row_by_index=row_by_index,
        placed=placed,
        slot_by_rock=slot_by_rock,
        before=before,
        after=after,
        stable_by_index=stable_by_index,
        target_errors=target_errors,
    )
    summary = {
        "gravity": gravity_label,
        "gravity_m_s2": GRAVITIES[gravity_label],
        "trial": trial_id,
        "strategy": strategy,
        "target_name": target_name,
        "candidate_count": candidate_count,
        "effective_steps_per_rock": effective_steps_per_rock,
        "effective_hold_steps": effective_hold_steps,
        "rock_count": len(placed),
        "skipped_slot_count": int(sum(int(row.get("placement_skipped", 0)) for row in placement_rows)),
        "stable_count": stable_count,
        "failure_count": len(placed) - stable_count,
        "success": success,
        "shape_success": shape_success,
        "visible_courses": visible_courses,
        "target_rmse_xy_m": rmse,
        "target_max_xy_error_m": float(max_error),
        "structure_score": float(structure_score),
        "stack_height_m": float(top_z),
        "max_horizontal_drift_m": float(max_drift),
        "max_radial_distance_m": max_radial_distance,
        "wall_x_span_m": wall_metrics["wall_x_span_m"],
        "wall_y_span_m": wall_metrics["wall_y_span_m"],
        "wall_aspect_xy": wall_metrics["wall_aspect_xy"],
        "wall_outlier_count": wall_metrics["wall_outlier_count"],
        "velocity_inf_norm": speed,
        "assignment_gate_used": int(assignment_gate),
        "assignment_probe_steps": int(assignment_probe_steps),
        "candidate_probe_steps": int(effective_candidate_probe_steps),
        "candidate_probe_hard_gate": int(bool(candidate_probe_hard_gate)),
        "moon_gate_strict": int(moon_gate_strict),
        "candidate_pose_ranker_used": int(candidate_pose_ranker is not None),
        "candidate_pose_top_k": int(candidate_pose_top_k),
        "candidate_pose_ranker_max_course": int(candidate_pose_ranker_max_course),
        "pose_risk_ranker_used": int(pose_risk_ranker is not None),
        "pose_risk_weight": float(pose_risk_weight),
        "pose_risk_ranker_max_course": int(pose_risk_ranker_max_course),
        "stone_fit_ranker_used": int(stone_fit_ranker is not None),
        "stone_fit_top_k": int(stone_fit_top_k),
        "stone_fit_ranker_max_course": int(stone_fit_ranker_max_course),
        "commit_best_rejected": int(commit_best_rejected),
        "base_support_prior": int(bool(base_support_prior)),
        "base_support_prior_weight": float(base_support_prior_weight),
        "order": " ".join(f"{idx:03d}" for idx in placed),
        "xml": str(xml_path),
    }
    if state_snapshots is not None:
        _append_state_snapshot(
            state_snapshots,
            data,
            label="final_hold",
            slot_index=len(slots),
            rock_index=-1,
            slot=None,
            placed=placed,
        )
    _append_progress(
        progress_path,
        {
            **progress_base,
            "time_s": time.time(),
            "elapsed_s": time.monotonic() - start_time,
            "event": "trial_done",
            "slot_index": len(slots),
            "slot_id": -1,
            "course": -1,
            "placed_count": len(placed),
            "rock_index": -1,
            "candidate_count": candidate_count,
            "message": f"success={success} shape={shape_success}",
        },
    )
    return {
        "summary": summary,
        "placements": placement_rows,
        "candidate_poses": candidate_pose_rows,
        "failures": failures,
        "state": {"qpos": data.qpos.copy(), "qvel": data.qvel.copy()},
        "state_snapshots": state_snapshots or [],
    }


def _append_state_snapshot(
    snapshots: list[dict[str, Any]],
    data: Any,
    label: str,
    slot_index: int,
    rock_index: int,
    slot: TargetSlot | None,
    placed: list[int],
) -> None:
    snapshots.append(
        {
            "label": label,
            "slot_index": int(slot_index),
            "rock_index": int(rock_index),
            "slot_id": -1 if slot is None else int(slot.slot_id),
            "course": -1 if slot is None else int(slot.course),
            "role": "" if slot is None else str(slot.role),
            "placed": [int(idx) for idx in placed],
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
        }
    )


def _place_for_target_slot(
    mujoco: Any,
    model: Any,
    data: Any,
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    slot_by_rock: dict[int, TargetSlot],
    rock_index: int,
    slot: TargetSlot,
    rng: np.random.Generator,
    steps_per_rock: int,
    candidate_count: int,
    strategy: str,
    candidate_pose_rows: list[dict[str, Any]] | None = None,
    candidate_context: dict[str, Any] | None = None,
    candidate_pose_ranker: dict[str, Any] | None = None,
    candidate_pose_top_k: int = 0,
    pose_risk_ranker: dict[str, Any] | None = None,
    pose_risk_weight: float = 0.0,
    candidate_probe_steps: int = 0,
    candidate_probe_hard_gate: bool = False,
    low_release_search: bool = False,
    release_search_step_m: float = 0.004,
    release_extra_clearance_m: float = 0.003,
) -> dict[str, Any]:
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()
    best_score = float("inf")
    best_qpos = qpos0.copy()
    best_qvel = qvel0.copy()
    best: dict[str, Any] = {}
    target_name = str((candidate_context or {}).get("target_name", ""))
    local_candidate_rows: list[dict[str, Any]] = []
    pre_positions = _body_positions(model, data, placed) if placed else {}
    candidate_items: list[tuple[int, dict[str, Any], float | None, float | None, float, int]] = []

    for candidate_id in range(candidate_count):
        candidate = _target_candidate_pose(
            row_by_index=row_by_index,
            placed=placed,
            slot_by_rock=slot_by_rock,
            rock_index=rock_index,
            slot=slot,
            rng=rng,
            candidate_id=candidate_id,
            strategy=strategy,
            target_name=target_name,
        )
        if low_release_search and slot.course > 0:
            candidate = _lower_candidate_release_height(
                mujoco=mujoco,
                model=model,
                data=data,
                row_by_index=row_by_index,
                rock_index=rock_index,
                candidate=candidate,
                search_step_m=release_search_step_m,
                extra_clearance_m=release_extra_clearance_m,
            )
        ranker_prob = _candidate_ranker_prob(
            candidate_pose_ranker,
            candidate_context or {},
            row_by_index,
            placed,
            slot_by_rock,
            rock_index,
            slot,
            candidate_count,
            candidate_id,
            candidate,
        )
        risk_prob = _pose_risk_prob(
            pose_risk_ranker,
            candidate_context or {},
            row_by_index,
            rock_index,
            slot,
            candidate_count,
            candidate_id,
            candidate,
        )
        rank_score = float(ranker_prob if ranker_prob is not None else 0.0) - float(pose_risk_weight) * float(risk_prob if risk_prob is not None else 0.0)
        candidate_items.append((candidate_id, candidate, ranker_prob, risk_prob, rank_score, 0))

    if candidate_pose_ranker is not None and candidate_pose_top_k > 0 and candidate_pose_top_k < len(candidate_items):
        ranked = sorted(candidate_items, key=lambda item: float(item[4]), reverse=True)
        chosen_rank = {int(item[0]): rank for rank, item in enumerate(ranked[:candidate_pose_top_k])}
        candidate_items = [
            (candidate_id, candidate, ranker_prob, risk_prob, rank_score, chosen_rank[int(candidate_id)])
            for candidate_id, candidate, ranker_prob, risk_prob, rank_score, _rank in candidate_items
            if int(candidate_id) in chosen_rank
        ]
        candidate_items.sort(key=lambda item: item[5])

    for candidate_id, candidate, ranker_prob, risk_prob, rank_score, ranker_rank in candidate_items:
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        _set_freejoint_pose(model, data, rock_index, candidate["pos"], candidate["quat"])
        mujoco.mj_forward(model, data)
        _simulate_until_quiet(mujoco, model, data, steps_per_rock)
        probe_steps = max(0, int(candidate_probe_steps))
        probe_rock_drift = 0.0
        probe_placed_disturbance = 0.0
        if probe_steps > 0:
            before_probe = _body_positions(model, data, placed + [rock_index])
            _simulate_until_quiet(mujoco, model, data, probe_steps)
            after_probe = _body_positions(model, data, placed + [rock_index])
            probe_rock_drift = float(np.linalg.norm(after_probe[rock_index][:2] - before_probe[rock_index][:2]))
            probe_placed_disturbance = (
                max(float(np.linalg.norm(after_probe[idx][:2] - before_probe[idx][:2])) for idx in placed)
                if placed
                else 0.0
            )
        metrics = _target_candidate_metrics(model, data, row_by_index, placed, slot_by_rock, rock_index, slot, pre_positions)
        if probe_steps > 0:
            metrics.update(
                {
                    "candidate_probe_steps": float(probe_steps),
                    "candidate_probe_hard_gate": float(int(bool(candidate_probe_hard_gate))),
                    "candidate_probe_rock_drift_m": probe_rock_drift,
                    "candidate_probe_placed_disturbance_m": probe_placed_disturbance,
                    "candidate_probe_speed": float(np.linalg.norm(data.qvel, ord=np.inf)),
                }
            )
        base_score = _target_candidate_score(metrics, slot, strategy, target_name=target_name)
        risk_penalty = float(pose_risk_weight) * float(risk_prob if risk_prob is not None else 0.0)
        score = base_score + risk_penalty
        local_candidate_rows.append(
            _candidate_pose_row(
                candidate_context=candidate_context or {},
                row_by_index=row_by_index,
                rock_index=rock_index,
                slot=slot,
                candidate_count=candidate_count,
                candidate_id=candidate_id,
                candidate=candidate,
                metrics=metrics,
                score=score,
                ranker_prob=ranker_prob,
                ranker_rank=ranker_rank,
                ranker_top_k=candidate_pose_top_k if candidate_pose_ranker is not None else 0,
                pose_risk_prob=risk_prob,
                pose_risk_weight=pose_risk_weight,
                pose_risk_penalty=risk_penalty,
                pose_rank_score=rank_score,
                base_candidate_score=base_score,
            )
        )
        if score < best_score:
            best_score = score
            best_qpos = data.qpos.copy()
            best_qvel = data.qvel.copy()
            best = {**candidate, **metrics, "candidate_id": candidate_id, "candidate_score": score}

    if candidate_pose_rows is not None:
        selected_candidate_id = int(best.get("candidate_id", -1))
        for row in local_candidate_rows:
            row["selected_by_pose_search"] = int(int(row["candidate_id"]) == selected_candidate_id)
        candidate_pose_rows.extend(local_candidate_rows)

    data.qpos[:] = best_qpos
    data.qvel[:] = best_qvel
    mujoco.mj_forward(model, data)
    if "settled_z" in best:
        row_by_index[rock_index]["last_top_z"] = float(best["settled_z"]) + 0.5 * float(row_by_index[rock_index]["bbox_z"])
        row_by_index[rock_index]["last_center_x"] = float(best.get("settled_x", 0.0))
        row_by_index[rock_index]["last_center_y"] = float(best.get("settled_y", 0.0))
        row_by_index[rock_index]["support_load_path_count"] = float(
            best.get("support_load_path_count", 1.0 if slot.course == 0 else 0.0)
        )
        row_by_index[rock_index]["direct_support_count_course_below"] = float(
            best.get("direct_support_count_course_below", 1.0 if slot.course == 0 else 0.0)
        )
        quat = best.get("quat", np.array([1.0, 0.0, 0.0, 0.0]))
        row_by_index[rock_index]["last_quat_w"] = float(quat[0])
        row_by_index[rock_index]["last_quat_x"] = float(quat[1])
        row_by_index[rock_index]["last_quat_y"] = float(quat[2])
        row_by_index[rock_index]["last_quat_z"] = float(quat[3])
    return _serializable_candidate(best)


def _place_assignment_slot(
    mujoco: Any,
    model: Any,
    data: Any,
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    slot_by_rock: dict[int, TargetSlot],
    candidate_rocks: list[int],
    slot: TargetSlot,
    rng: np.random.Generator,
    steps_per_rock: int,
    candidate_count: int,
    strategy: str,
    target_name: str,
    assignment_gate: bool,
    gravity_label: str,
    trial_id: int,
    assignment_probe_steps: int,
    candidate_probe_steps: int,
    candidate_probe_hard_gate: bool,
    moon_gate_strict: bool,
    candidate_pose_rows: list[dict[str, Any]] | None = None,
    candidate_pose_ranker: dict[str, Any] | None = None,
    candidate_pose_top_k: int = 0,
    pose_risk_ranker: dict[str, Any] | None = None,
    pose_risk_weight: float = 0.0,
    low_release_search: bool = False,
    release_search_step_m: float = 0.004,
    release_extra_clearance_m: float = 0.003,
) -> tuple[int, dict[str, Any]]:
    qpos_before_slot = data.qpos.copy()
    qvel_before_slot = data.qvel.copy()
    best_rejected: dict[str, Any] = {}
    best_rejected_score = float("inf")
    tried = 0
    used = set(placed)
    for fallback_attempt_id, rock_index in enumerate(candidate_rocks):
        rock_index = int(rock_index)
        if rock_index in used or rock_index not in row_by_index:
            continue
        tried += 1
        row_snapshot = dict(row_by_index[rock_index])
        data.qpos[:] = qpos_before_slot
        data.qvel[:] = qvel_before_slot
        mujoco.mj_forward(model, data)
        selected = _place_for_target_slot(
            mujoco=mujoco,
            model=model,
            data=data,
            row_by_index=row_by_index,
            placed=placed,
            slot_by_rock=slot_by_rock,
            rock_index=rock_index,
            slot=slot,
            rng=rng,
            steps_per_rock=steps_per_rock,
            candidate_count=candidate_count,
            strategy=strategy,
            candidate_pose_rows=candidate_pose_rows,
            candidate_context={
                "gravity": gravity_label,
                "gravity_m_s2": GRAVITIES[gravity_label],
                "trial": trial_id,
                "strategy": strategy,
                "target_name": target_name,
                "assignment_fallback_attempt": fallback_attempt_id,
                "assignment_candidate_count": len(candidate_rocks),
                "assignment_candidate_rock_index": rock_index,
            },
            candidate_pose_ranker=candidate_pose_ranker,
            candidate_pose_top_k=candidate_pose_top_k,
            pose_risk_ranker=pose_risk_ranker,
            pose_risk_weight=pose_risk_weight,
            candidate_probe_steps=candidate_probe_steps,
            candidate_probe_hard_gate=candidate_probe_hard_gate,
            low_release_search=low_release_search,
            release_search_step_m=release_search_step_m,
            release_extra_clearance_m=release_extra_clearance_m,
        )
        selected["assignment_fallback_attempt"] = fallback_attempt_id
        selected["assignment_candidate_count"] = len(candidate_rocks)
        selected["assignment_candidate_rock_index"] = rock_index
        selected["assignment_selected_primary"] = int(fallback_attempt_id == 0)
        if assignment_probe_steps > 0:
            _probe_assignment_candidate(
                mujoco=mujoco,
                model=model,
                data=data,
                row_by_index=row_by_index,
                placed=placed,
                slot_by_rock=slot_by_rock,
                rock_index=rock_index,
                slot=slot,
                selected=selected,
                probe_steps=assignment_probe_steps,
            )
        if not assignment_gate or _assignment_candidate_is_feasible(
            selected,
            slot,
            target_name,
            gravity_label=gravity_label,
            moon_gate_strict=moon_gate_strict,
        ):
            return rock_index, selected

        rejected_score = float(selected.get("candidate_score", 1e6))
        if rejected_score < best_rejected_score:
            best_rejected_score = rejected_score
            best_rejected = dict(selected)
            best_rejected["best_rejected_rock_index"] = rock_index
        row_by_index[rock_index].clear()
        row_by_index[rock_index].update(row_snapshot)

    data.qpos[:] = qpos_before_slot
    data.qvel[:] = qvel_before_slot
    mujoco.mj_forward(model, data)
    return -1, {
        **best_rejected,
        "placement_skipped": 1,
        "skip_reason": "assignment_gate_no_feasible_fallback" if tried else "assignment_no_unused_candidates",
        "assignment_candidate_count": len(candidate_rocks),
        "assignment_tried_count": tried,
        "best_rejected_score": best_rejected_score,
    }


def _probe_assignment_candidate(
    mujoco: Any,
    model: Any,
    data: Any,
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    slot_by_rock: dict[int, TargetSlot],
    rock_index: int,
    slot: TargetSlot,
    selected: dict[str, Any],
    probe_steps: int,
) -> None:
    if probe_steps <= 0:
        return
    before_probe = _body_positions(model, data, placed + [rock_index])
    _simulate_until_quiet(mujoco, model, data, probe_steps)
    after_probe = _body_positions(model, data, placed + [rock_index])
    rock_drift = float(np.linalg.norm(after_probe[rock_index][:2] - before_probe[rock_index][:2]))
    placed_disturbance = (
        max(float(np.linalg.norm(after_probe[idx][:2] - before_probe[idx][:2])) for idx in placed)
        if placed
        else 0.0
    )
    metrics = _target_candidate_metrics(
        model=model,
        data=data,
        row_by_index=row_by_index,
        placed=placed,
        slot_by_rock=slot_by_rock,
        rock_index=rock_index,
        slot=slot,
        pre_positions={idx: before_probe[idx] for idx in placed if idx in before_probe},
    )
    selected.update(metrics)
    selected.update(
        {
            "assignment_probe_steps": int(probe_steps),
            "assignment_probe_rock_drift_m": rock_drift,
            "assignment_probe_placed_disturbance_m": placed_disturbance,
            "assignment_probe_speed": float(np.linalg.norm(data.qvel, ord=np.inf)),
        }
    )
    row_by_index[rock_index]["last_top_z"] = float(selected["settled_z"]) + 0.5 * float(row_by_index[rock_index]["bbox_z"])
    row_by_index[rock_index]["last_center_x"] = float(selected.get("settled_x", 0.0))
    row_by_index[rock_index]["last_center_y"] = float(selected.get("settled_y", 0.0))
    row_by_index[rock_index]["support_load_path_count"] = float(
        selected.get("support_load_path_count", 1.0 if slot.course == 0 else 0.0)
    )
    row_by_index[rock_index]["direct_support_count_course_below"] = float(
        selected.get("direct_support_count_course_below", 1.0 if slot.course == 0 else 0.0)
    )


def _place_literature_slot(
    mujoco: Any,
    model: Any,
    data: Any,
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    slot_by_rock: dict[int, TargetSlot],
    slot: TargetSlot,
    rng: np.random.Generator,
    steps_per_rock: int,
    candidate_count: int,
    used: set[int],
    strategy: str,
    target_name: str,
    candidate_pose_rows: list[dict[str, Any]] | None = None,
    candidate_context: dict[str, Any] | None = None,
    candidate_pose_ranker: dict[str, Any] | None = None,
    candidate_pose_top_k: int = 0,
    pose_risk_ranker: dict[str, Any] | None = None,
    pose_risk_weight: float = 0.0,
    stone_fit_ranker: dict[str, Any] | None = None,
    stone_fit_top_k: int = 0,
    commit_best_rejected: bool = False,
    candidate_probe_steps: int = 0,
    candidate_probe_hard_gate: bool = False,
    low_release_search: bool = False,
    release_search_step_m: float = 0.004,
    release_extra_clearance_m: float = 0.003,
    base_support_prior: bool = False,
    base_support_prior_weight: float = 1.0,
) -> tuple[int, dict[str, Any]]:
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()
    slot_candidate_count = _literature_slot_candidate_count(slot, candidate_count, strategy)
    pool, pool_meta = _literature_stone_pool(
        row_by_index=row_by_index,
        slot=slot,
        used=used,
        candidate_count=slot_candidate_count,
        strategy=strategy,
        candidate_context=candidate_context,
        stone_fit_ranker=stone_fit_ranker,
        stone_fit_top_k=stone_fit_top_k,
        target_name=target_name,
        base_support_prior=base_support_prior,
        base_support_prior_weight=base_support_prior_weight,
    )
    if not pool:
        raise RuntimeError("No unused stones are available for literature_column placement.")

    best_score = float("inf")
    best_rock_index = pool[0]
    best_qpos = qpos0.copy()
    best_qvel = qvel0.copy()
    best_row_state = dict(row_by_index[best_rock_index])
    best_selected: dict[str, Any] = {}
    best_feasible_score = float("inf")
    best_feasible_rock_index = -1
    best_feasible_qpos = qpos0.copy()
    best_feasible_qvel = qvel0.copy()
    best_feasible_row_state: dict[str, Any] = {}
    best_feasible_selected: dict[str, Any] = {}

    for rock_index in pool:
        row_snapshot = dict(row_by_index[rock_index])
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        mujoco.mj_forward(model, data)
        selected = _place_for_target_slot(
            mujoco=mujoco,
            model=model,
            data=data,
            row_by_index=row_by_index,
            placed=placed,
            slot_by_rock=slot_by_rock,
            rock_index=rock_index,
            slot=slot,
            rng=rng,
            steps_per_rock=steps_per_rock,
            candidate_count=slot_candidate_count,
            strategy=strategy,
            candidate_pose_rows=candidate_pose_rows,
            candidate_context={
                **(candidate_context or {}),
                "stone_pool_size": len(pool),
                "stone_pool_rock_index": rock_index,
                **pool_meta.get(rock_index, {}),
            },
            candidate_pose_ranker=candidate_pose_ranker,
            candidate_pose_top_k=candidate_pose_top_k,
            pose_risk_ranker=pose_risk_ranker,
            pose_risk_weight=pose_risk_weight,
            candidate_probe_steps=candidate_probe_steps,
            candidate_probe_hard_gate=candidate_probe_hard_gate,
            low_release_search=low_release_search,
            release_search_step_m=release_search_step_m,
            release_extra_clearance_m=release_extra_clearance_m,
        )
        selected["slot_candidate_count"] = slot_candidate_count
        selected.update(pool_meta.get(rock_index, {}))
        score = _literature_stone_candidate_score(selected, slot, strategy, target_name)
        is_feasible = _literature_candidate_is_feasible(selected, slot, strategy, target_name)
        selected["candidate_feasible"] = int(is_feasible)
        if score < best_score:
            best_score = score
            best_rock_index = rock_index
            best_qpos = data.qpos.copy()
            best_qvel = data.qvel.copy()
            best_row_state = dict(row_by_index[rock_index])
            best_selected = dict(selected)
            best_selected["stone_pool_size"] = len(pool)
            best_selected["stone_selection_score"] = score
        if is_feasible and score < best_feasible_score:
            best_feasible_score = score
            best_feasible_rock_index = rock_index
            best_feasible_qpos = data.qpos.copy()
            best_feasible_qvel = data.qvel.copy()
            best_feasible_row_state = dict(row_by_index[rock_index])
            best_feasible_selected = dict(selected)
            best_feasible_selected["stone_pool_size"] = len(pool)
            best_feasible_selected["stone_selection_score"] = score
        row_by_index[rock_index].clear()
        row_by_index[rock_index].update(row_snapshot)

    if best_feasible_rock_index < 0:
        if commit_best_rejected and best_selected:
            data.qpos[:] = best_qpos
            data.qvel[:] = best_qvel
            row_by_index[best_rock_index].update(best_row_state)
            mujoco.mj_forward(model, data)
            selected = dict(best_selected)
            selected["candidate_feasible"] = 0
            selected["commit_best_rejected"] = 1
            selected["skip_reason"] = ""
            selected["stone_pool_size"] = len(pool)
            selected["stone_selection_score"] = best_score
            return best_rock_index, selected
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        mujoco.mj_forward(model, data)
        return -1, {
            "placement_skipped": 1,
            "skip_reason": "no_feasible_pose",
            "commit_best_rejected": 0,
            "best_rejected_score": best_score,
            "best_rejected_rock_index": best_rock_index,
            **best_selected,
        }

    data.qpos[:] = best_feasible_qpos
    data.qvel[:] = best_feasible_qvel
    row_by_index[best_feasible_rock_index].update(best_feasible_row_state)
    mujoco.mj_forward(model, data)
    return best_feasible_rock_index, best_feasible_selected


def _literature_slot_candidate_count(slot: TargetSlot, candidate_count: int, strategy: str) -> int:
    return candidate_count


def _is_wall_online_strategy(strategy: str) -> bool:
    return strategy in {"literature_wall", "statics_wall", "statics_wall_line_lock"}


def _literature_stone_pool(
    row_by_index: dict[int, dict[str, Any]],
    slot: TargetSlot,
    used: set[int],
    candidate_count: int,
    strategy: str,
    candidate_context: dict[str, Any] | None = None,
    stone_fit_ranker: dict[str, Any] | None = None,
    stone_fit_top_k: int = 0,
    target_name: str = "",
    base_support_prior: bool = False,
    base_support_prior_weight: float = 1.0,
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    if _is_wall_online_strategy(strategy):
        if candidate_count >= 5:
            target_pool_size = 8
        elif candidate_count >= 3:
            target_pool_size = 6
        else:
            target_pool_size = 5
        pool_size = min(target_pool_size, max(1, len(row_by_index) - len(used)))
    else:
        pool_size = min(8 if candidate_count >= 6 else 5, max(1, len(row_by_index) - len(used)))
    rows = [row for idx, row in row_by_index.items() if idx not in used]
    if not rows:
        return [], {}
    prior_enabled = bool(base_support_prior and slot.course == 0 and slot.role == "base" and _is_wall_online_strategy(strategy))
    prior_weight = max(0.0, float(base_support_prior_weight))
    hand_ranked = sorted(
        rows,
        key=lambda row: (
            _primary_sort_score(_stone_role_score(row, slot.role, strategy))
            + (prior_weight * _base_support_prior_score(row, slot, target_name) if prior_enabled else 0.0),
            int(row["index"]),
        ),
    )
    if stone_fit_ranker is None:
        pool = [int(row["index"]) for row in hand_ranked[:pool_size]]
        return pool, {
            rock_index: {
                "stone_fit_prob": "",
                "stone_fit_rank": rank,
                "stone_fit_top_k": 0,
                "base_support_prior_enabled": int(prior_enabled),
                "base_support_prior_weight": float(prior_weight if prior_enabled else 0.0),
                "base_support_prior_score": (
                    float(_base_support_prior_score(row_by_index[rock_index], slot, target_name)) if prior_enabled else 0.0
                ),
            }
            for rank, rock_index in enumerate(pool)
        }

    scored: list[tuple[float, float, int, dict[str, Any]]] = []
    for row in rows:
        rock_index = int(row["index"])
        prob = _stone_fit_prob(stone_fit_ranker, candidate_context or {}, row, rock_index, slot)
        role_score = _primary_sort_score(_stone_role_score(row, slot.role, strategy))
        if prob is None:
            prob = 0.0
        prior_score = _base_support_prior_score(row, slot, target_name) if prior_enabled else 0.0
        hybrid_score = (1.0 - float(prob)) + 0.035 * min(float(role_score), 10.0) + prior_weight * float(prior_score)
        scored.append((hybrid_score, -float(prob), rock_index, row))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    top_k = min(max(1, stone_fit_top_k if stone_fit_top_k > 0 else pool_size), len(scored))
    pool = [rock_index for _hybrid, _neg_prob, rock_index, _row in scored[:top_k]]
    meta: dict[int, dict[str, Any]] = {}
    for rank, (hybrid_score, neg_prob, rock_index, _row) in enumerate(scored[:top_k]):
        meta[rock_index] = {
            "stone_fit_prob": float(-neg_prob),
            "stone_fit_rank": rank,
            "stone_fit_hybrid_score": float(hybrid_score),
            "stone_fit_top_k": top_k,
            "base_support_prior_enabled": int(prior_enabled),
            "base_support_prior_weight": float(prior_weight if prior_enabled else 0.0),
            "base_support_prior_score": (
                float(_base_support_prior_score(_row, slot, target_name)) if prior_enabled else 0.0
            ),
        }
    return pool, meta


def _primary_sort_score(value: Any) -> float:
    if isinstance(value, (tuple, list)):
        return float(value[0]) if value else 0.0
    return float(value)


def _base_support_prior_score(row: dict[str, Any], slot: TargetSlot, target_name: str) -> float:
    if slot.course != 0 or slot.role != "base":
        return 0.0

    bbox_x = float(row["bbox_x"])
    bbox_y = float(row["bbox_y"])
    bbox_z = float(row["bbox_z"])
    footprint = 0.5 * (bbox_x + bbox_y)
    footprint_area = bbox_x * bbox_y
    min_span = min(bbox_x, bbox_y)
    volume = float(row["volume"])
    compactness = float(row["compactness"])
    elongation = float(row["elongation"])
    flatness = float(row["flatness"])
    spike = float(row.get("spike_score", 0.0))
    support_count = float(row.get("support_face_count", 0.0))
    support_ratio = float(row.get("support_face_area_ratio", 0.0))
    opposing_ratio = float(row.get("opposing_face_area_ratio", 0.0))
    source = str(row.get("source_kind", ""))
    label = str(row.get("cluster_label", ""))

    if target_name == "single_face_wall_4course_v1":
        min_footprint = 0.132
        min_area = 0.0165
        min_volume = 0.00088
        min_span_target = 0.092
    elif "wall" in target_name:
        min_footprint = 0.128
        min_area = 0.0155
        min_volume = 0.00082
        min_span_target = 0.088
    else:
        min_footprint = 0.118
        min_area = 0.0135
        min_volume = 0.00072
        min_span_target = 0.082

    height_ratio = bbox_z / max(footprint, 1e-6)
    score = 0.0
    score += 8.0 * max(0.0, min_footprint - footprint)
    score += 22.0 * max(0.0, min_area - footprint_area)
    score += 420.0 * max(0.0, min_volume - volume)
    score += 4.5 * max(0.0, min_span_target - min_span)
    score += 0.30 * max(0.0, height_ratio - 1.00)
    score += 0.22 * max(0.0, elongation - 1.45)
    score += 0.18 * max(0.0, flatness - 1.38)
    score += 2.8 * spike

    score -= 2.2 * min(max(0.0, footprint - min_footprint), 0.045)
    score -= 12.0 * min(max(0.0, footprint_area - min_area), 0.010)
    score -= 180.0 * min(max(0.0, volume - min_volume), 0.00065)
    score -= 0.10 * min(max(0.0, compactness - 0.60), 0.40)
    score -= 0.14 * min(max(0.0, support_ratio - 0.18), 0.22)
    score -= 0.04 * min(max(0.0, support_count - 1.0), 3.0)
    score -= 0.08 * min(max(0.0, opposing_ratio - 0.15), 0.30)

    if source in {"bearing_block_clast", "wall_block_clast", "buttress_clast", "subangular_block", "wedge_clast"}:
        score -= 0.07
    if source in {"chock_clast", "cap_block_clast", "tie_bridge_clast"}:
        score += 0.10
    if label.startswith("spiky_reject"):
        score += 1.50
    return float(score)


def _literature_stone_candidate_score(selected: dict[str, Any], slot: TargetSlot, strategy: str, target_name: str) -> float:
    score = float(selected.get("candidate_score", 1e6))
    target_error = float(selected.get("target_error_xy_m", 1.0))
    radial_distance = float(selected.get("radial_distance_m", 1.0))
    target_y_error = float(selected.get("target_y_error_m", 1.0))
    support_count = float(selected.get("support_contact_count", 0.0))
    support_overlap = float(selected.get("support_overlap", 0.0))
    support_balance_error = float(selected.get("support_balance_error_m", 0.0))
    height_gain = float(selected.get("height_gain_m", -1.0))
    placed_disturbance = float(selected.get("placed_disturbance_xy_m", 0.0))
    post_place_speed = float(selected.get("velocity_inf_norm_after_place", 0.0))
    probe_steps = int(float(selected.get("candidate_probe_steps", 0.0)))
    probe_hard_gate = int(float(selected.get("candidate_probe_hard_gate", 0.0)))
    probe_rock_drift = float(selected.get("candidate_probe_rock_drift_m", 0.0))
    probe_placed_disturbance = float(selected.get("candidate_probe_placed_disturbance_m", 0.0))
    probe_speed = float(selected.get("candidate_probe_speed", 0.0))
    support_overlap_min = _literature_min_support_overlap(target_name)
    if slot.course > 0 and support_count < 1.0 and support_overlap < support_overlap_min:
        score += 8.0
    if _is_wall_online_strategy(strategy):
        target_error_soft = _literature_target_error_soft_limit(target_name)
        y_error_soft = _literature_y_error_soft_limit(target_name)
        if target_error > target_error_soft:
            score += 5.0 * (target_error - target_error_soft)
        if target_y_error > y_error_soft:
            score += 9.0 * (target_y_error - y_error_soft)
        if _is_strict_single_face_wall_target(target_name) and slot.course > 0:
            disturbance_soft = 0.040 if slot.course < 3 else 0.030
            score += 18.0 * max(0.0, placed_disturbance - disturbance_soft)
            score += 1.2 * max(0.0, post_place_speed - 0.25)
            score += 6.0 * max(0.0, support_balance_error - 0.105)
            if probe_steps > 0:
                score += 18.0 * max(0.0, probe_rock_drift - (0.032 if slot.course < 3 else 0.024))
                score += 26.0 * max(0.0, probe_placed_disturbance - 0.016)
                score += 1.5 * max(0.0, probe_speed - 0.22)
        if strategy == "statics_wall_line_lock":
            score += 8.0 * max(0.0, target_error - 0.85 * target_error_soft)
            score += 18.0 * max(0.0, target_y_error - 0.70 * y_error_soft)
            score += 9.0 * max(0.0, support_balance_error - 0.085)
    else:
        if target_error > 0.22:
            score += 4.0 * (target_error - 0.22)
        if radial_distance > 0.24:
            score += 10.0 * (radial_distance - 0.24)
    if height_gain < -0.02:
        score += 4.0 * abs(height_gain)
    return score


def _literature_candidate_is_feasible(selected: dict[str, Any], slot: TargetSlot, strategy: str, target_name: str) -> bool:
    if not selected:
        return False
    target_error = float(selected.get("target_error_xy_m", 1.0))
    target_y_error = float(selected.get("target_y_error_m", 1.0))
    settled_y = abs(float(selected.get("settled_y", 999.0)))
    support_count = float(selected.get("support_contact_count", 0.0))
    support_overlap = float(selected.get("support_overlap", 0.0))
    support_balance_error = float(selected.get("support_balance_error_m", 0.0))
    height_gain = float(selected.get("height_gain_m", -1.0))
    placed_disturbance = float(selected.get("placed_disturbance_xy_m", 0.0))
    post_place_speed = float(selected.get("velocity_inf_norm_after_place", 0.0))
    probe_steps = int(float(selected.get("candidate_probe_steps", 0.0)))
    probe_hard_gate = int(float(selected.get("candidate_probe_hard_gate", 0.0)))
    probe_rock_drift = float(selected.get("candidate_probe_rock_drift_m", 0.0))
    probe_placed_disturbance = float(selected.get("candidate_probe_placed_disturbance_m", 0.0))
    probe_speed = float(selected.get("candidate_probe_speed", 0.0))
    if _is_wall_online_strategy(strategy):
        target_error_hard = _literature_target_error_hard_limit(target_name)
        y_error_hard = _literature_y_error_hard_limit(target_name)
        settled_y_hard = _literature_settled_y_hard_limit(target_name)
        if strategy == "statics_wall_line_lock":
            target_error_hard *= 0.86
            y_error_hard *= 0.78
            settled_y_hard *= 0.82
        if (
            target_error > target_error_hard
            or target_y_error > y_error_hard
            or settled_y > settled_y_hard
        ):
            return False
        if slot.course > 0 and support_count < 1.0 and support_overlap < _literature_min_support_overlap(target_name):
            return False
        if height_gain < _literature_min_height_gain(target_name):
            return False
        if _is_strict_single_face_wall_target(target_name) and slot.course > 0:
            disturbance_limit = 0.090 if slot.course < 3 else 0.070
            if placed_disturbance > disturbance_limit:
                return False
            if post_place_speed > 0.55:
                return False
            if probe_steps > 0 and probe_hard_gate:
                if probe_rock_drift > (0.055 if slot.course < 3 else 0.040):
                    return False
                if probe_placed_disturbance > 0.030:
                    return False
                if probe_speed > 0.45:
                    return False
            if support_balance_error > 0.135:
                return False
            if strategy == "statics_wall_line_lock" and support_balance_error > 0.115:
                return False
    return True


def _assignment_candidate_is_feasible(
    selected: dict[str, Any],
    slot: TargetSlot,
    target_name: str,
    gravity_label: str,
    moon_gate_strict: bool = False,
) -> bool:
    if not selected:
        return False
    target_error = float(selected.get("target_error_xy_m", 1.0))
    target_y_error = float(selected.get("target_y_error_m", 1.0))
    settled_y = abs(float(selected.get("settled_y", 999.0)))
    support_count = float(selected.get("support_contact_count", 0.0))
    support_overlap = float(selected.get("support_overlap", 0.0))
    support_balance_error = float(selected.get("support_balance_error_m", 0.0))
    height_gain = float(selected.get("height_gain_m", -1.0))
    target_error_limit = _literature_target_error_hard_limit(target_name)
    y_error_limit = _literature_y_error_hard_limit(target_name)
    settled_y_limit = _literature_settled_y_hard_limit(target_name)
    min_support_overlap = _literature_min_support_overlap(target_name)
    rock_probe_limit = 0.090
    placed_probe_limit = 0.060
    if gravity_label == "moon" and moon_gate_strict:
        target_error_limit = min(target_error_limit, 0.240)
        y_error_limit = min(y_error_limit, 0.070)
        settled_y_limit = min(settled_y_limit, 0.120)
        min_support_overlap = max(min_support_overlap, 0.240 if slot.course > 1 else 0.200)
        rock_probe_limit = 0.026 if slot.course > 0 else 0.040
        placed_probe_limit = 0.018
    if target_error > target_error_limit:
        return False
    if target_y_error > y_error_limit:
        return False
    if settled_y > settled_y_limit:
        return False
    if slot.course > 0 and support_count < 1.0 and support_overlap < min_support_overlap:
        return False
    if gravity_label == "moon" and moon_gate_strict and slot.course > 0:
        if support_overlap < min_support_overlap:
            return False
        if support_balance_error > 0.115:
            return False
    if height_gain < _literature_min_height_gain(target_name):
        return False
    if "assignment_probe_rock_drift_m" in selected:
        if float(selected["assignment_probe_rock_drift_m"]) > rock_probe_limit:
            return False
    if "assignment_probe_placed_disturbance_m" in selected:
        if float(selected["assignment_probe_placed_disturbance_m"]) > placed_probe_limit:
            return False
    return True


def _literature_target_error_soft_limit(target_name: str) -> float:
    if _is_strict_single_face_wall_target(target_name):
        return 0.135 if target_name == "single_face_wall_4course_v1" else 0.145
    if target_name in {"tied_high_wall_v1", "tied_high_wall_core_v1", "tied_wall_4course_v1"}:
        return 0.22
    return 0.17


def _literature_y_error_soft_limit(target_name: str) -> float:
    if _is_strict_single_face_wall_target(target_name):
        return 0.045 if target_name == "single_face_wall_4course_v1" else 0.055
    if target_name in {"tied_high_wall_v1", "tied_high_wall_core_v1", "tied_wall_4course_v1"}:
        return 0.12
    return 0.080


def _literature_target_error_hard_limit(target_name: str) -> float:
    if _is_strict_single_face_wall_target(target_name):
        return 0.235 if target_name == "single_face_wall_4course_v1" else 0.255
    if target_name in {"tied_high_wall_v1", "tied_high_wall_core_v1", "tied_wall_4course_v1"}:
        return 0.42
    return 0.30


def _literature_y_error_hard_limit(target_name: str) -> float:
    if _is_strict_single_face_wall_target(target_name):
        return 0.080 if target_name == "single_face_wall_4course_v1" else 0.095
    if target_name in {"tied_high_wall_v1", "tied_high_wall_core_v1", "tied_wall_4course_v1"}:
        return 0.22
    return 0.12


def _literature_settled_y_hard_limit(target_name: str) -> float:
    if _is_strict_single_face_wall_target(target_name):
        return 0.120 if target_name == "single_face_wall_4course_v1" else 0.145
    if target_name in {"tied_high_wall_v1", "tied_high_wall_core_v1", "tied_wall_4course_v1"}:
        return 0.30
    return 0.22


def _literature_min_support_overlap(target_name: str) -> float:
    if _is_strict_single_face_wall_target(target_name):
        return 0.24 if target_name == "single_face_wall_4course_v1" else 0.22
    if target_name in {"tied_high_wall_v1", "tied_high_wall_core_v1", "tied_wall_4course_v1"}:
        return 0.12
    return 0.20


def _literature_min_height_gain(target_name: str) -> float:
    if target_name in {"tied_high_wall_v1", "tied_high_wall_core_v1", "tied_wall_4course_v1"}:
        return -0.10
    return -0.08


def _required_visible_courses(target_name: str, slots: list[TargetSlot]) -> int:
    target_courses = max((slot.course for slot in slots), default=0) + 1
    if target_name in {
        "single_face_wall_2course_v1",
        "single_face_wall_3course_v1",
        "tall_wall_v3",
        "single_wall_strict_v1",
        "single_face_wall_v1",
        "single_face_wall_high_v1",
        "single_face_wall_extra_high_v1",
        "tied_high_wall_v1",
        "tied_high_wall_core_v1",
        "tied_wall_4course_v1",
        "single_face_wall_4course_v1",
        "single_face_wall_5course_v1",
        "single_column_v3",
        "single_column_v4",
        "tall_wall_thick_v1",
        "tall_wall_thick_v2",
        "tall_pillar_v3",
        "multi_stone_column_v1",
        "stone_column_v1",
        "multi_stone_column_v2",
        "stone_column_v2",
        "multi_stone_column_v3",
        "stone_column_v3",
    }:
        return target_courses
    return min(5, target_courses)


def _stable_target_error_limit(target_name: str) -> float:
    if target_name == "single_face_wall_2course_v1":
        return 0.160
    if target_name == "single_face_wall_3course_v1":
        return 0.170
    if target_name in {"single_wall_strict_v1", "single_face_wall_v1"}:
        return 0.175
    if target_name == "single_face_wall_extra_high_v1":
        return 0.205
    if target_name == "single_face_wall_high_v1":
        return 0.190
    if target_name == "single_face_wall_4course_v1":
        return 0.160
    if target_name == "single_face_wall_5course_v1":
        return 0.175
    if target_name in {"tied_high_wall_v1", "tied_high_wall_core_v1", "tied_wall_4course_v1"}:
        return 0.220
    if target_name == "single_column_v4":
        return 0.145
    if target_name in {"single_column_v2", "single_column_v3"}:
        return 0.170
    if target_name in {"multi_stone_column_v1", "stone_column_v1"}:
        return 0.240
    if target_name in {"multi_stone_column_v2", "stone_column_v2"}:
        return 0.210
    if target_name in {"multi_stone_column_v3", "stone_column_v3"}:
        return 0.190
    return 0.20


def _target_radius_limit(target_name: str) -> float:
    if target_name in {
        "single_face_wall_2course_v1",
        "single_face_wall_3course_v1",
        "single_wall_strict_v1",
        "single_face_wall_v1",
        "single_face_wall_high_v1",
        "single_face_wall_extra_high_v1",
        "single_face_wall_4course_v1",
        "single_face_wall_5course_v1",
        "tied_high_wall_v1",
        "tied_high_wall_core_v1",
        "tied_wall_4course_v1",
    }:
        return 0.95
    if target_name == "single_column_v4":
        return 0.220
    if target_name in {"single_column_v2", "single_column_v3"}:
        return 0.300
    if target_name in {"multi_stone_column_v1", "stone_column_v1"}:
        return 0.360
    if target_name in {"multi_stone_column_v2", "stone_column_v2"}:
        return 0.300
    if target_name in {"multi_stone_column_v3", "stone_column_v3"}:
        return 0.285
    return 0.75


def _target_min_height(target_name: str) -> float:
    if target_name == "single_face_wall_2course_v1":
        return 0.13
    if target_name == "single_face_wall_3course_v1":
        return 0.19
    if target_name == "single_wall_strict_v1":
        return 0.30
    if target_name == "single_face_wall_v1":
        return 0.28
    if target_name == "single_face_wall_extra_high_v1":
        return 0.52
    if target_name == "single_face_wall_high_v1":
        return 0.42
    if target_name == "single_face_wall_4course_v1":
        return 0.24
    if target_name == "single_face_wall_5course_v1":
        return 0.30
    if target_name == "tied_high_wall_v1":
        return 0.42
    if target_name == "tied_high_wall_core_v1":
        return 0.38
    if target_name == "tied_wall_4course_v1":
        return 0.27
    if target_name == "tall_wall_v3":
        return 0.34
    if target_name == "tall_wall_thick_v1":
        return 0.36
    if target_name == "tall_wall_thick_v2":
        return 0.34
    if target_name == "tall_pillar_v3":
        return 0.32
    if target_name in {"multi_stone_column_v1", "stone_column_v1"}:
        return 0.38
    if target_name in {"multi_stone_column_v2", "stone_column_v2"}:
        return 0.36
    if target_name in {"multi_stone_column_v3", "stone_column_v3"}:
        return 0.36
    if target_name == "single_column_v3":
        return 0.32
    if target_name == "single_column_v4":
        return 0.36
    return 0.0


def _is_single_wall_target(target_name: str) -> bool:
    return target_name in {
        "single_face_wall_2course_v1",
        "single_face_wall_3course_v1",
        "single_wall_strict_v1",
        "single_face_wall_v1",
        "single_face_wall_high_v1",
        "single_face_wall_extra_high_v1",
        "single_face_wall_4course_v1",
        "single_face_wall_5course_v1",
        "tied_high_wall_v1",
        "tied_high_wall_core_v1",
        "tied_wall_4course_v1",
    }


def _is_strict_single_face_wall_target(target_name: str) -> bool:
    return target_name in {
        "single_face_wall_2course_v1",
        "single_face_wall_3course_v1",
        "single_wall_strict_v1",
        "single_face_wall_v1",
        "single_face_wall_high_v1",
        "single_face_wall_extra_high_v1",
        "single_face_wall_4course_v1",
        "single_face_wall_5course_v1",
    }


def _target_wall_y_limit(target_name: str) -> float:
    if target_name == "single_face_wall_2course_v1":
        return 0.16
    if target_name == "single_face_wall_3course_v1":
        return 0.17
    if target_name == "single_wall_strict_v1":
        return 0.24
    if target_name == "single_face_wall_v1":
        return 0.18
    if target_name == "single_face_wall_extra_high_v1":
        return 0.19
    if target_name == "single_face_wall_high_v1":
        return 0.18
    if target_name == "single_face_wall_4course_v1":
        return 0.115
    if target_name == "single_face_wall_5course_v1":
        return 0.130
    if target_name in {"tied_high_wall_v1", "tied_high_wall_core_v1", "tied_wall_4course_v1"}:
        return 0.22
    return 0.0


def _target_wall_min_x_span(target_name: str) -> float:
    if target_name == "single_face_wall_2course_v1":
        return 0.48
    if target_name == "single_face_wall_3course_v1":
        return 0.48
    if target_name == "single_wall_strict_v1":
        return 0.56
    if target_name == "single_face_wall_v1":
        return 0.58
    if target_name == "single_face_wall_extra_high_v1":
        return 0.50
    if target_name == "single_face_wall_high_v1":
        return 0.48
    if target_name == "single_face_wall_4course_v1":
        return 0.56
    if target_name == "single_face_wall_5course_v1":
        return 0.52
    if target_name == "tied_high_wall_v1":
        return 0.34
    if target_name == "tied_high_wall_core_v1":
        return 0.22
    if target_name == "tied_wall_4course_v1":
        return 0.20
    return 0.0


def _target_wall_max_y_span(target_name: str) -> float:
    if target_name == "single_face_wall_2course_v1":
        return 0.22
    if target_name == "single_face_wall_3course_v1":
        return 0.24
    if target_name == "single_wall_strict_v1":
        return 0.38
    if target_name == "single_face_wall_v1":
        return 0.28
    if target_name == "single_face_wall_extra_high_v1":
        return 0.30
    if target_name == "single_face_wall_high_v1":
        return 0.27
    if target_name == "single_face_wall_4course_v1":
        return 0.190
    if target_name == "single_face_wall_5course_v1":
        return 0.220
    if target_name in {"tied_high_wall_v1", "tied_high_wall_core_v1", "tied_wall_4course_v1"}:
        return 0.34
    return 999.0


def _target_wall_min_aspect(target_name: str) -> float:
    if target_name == "single_face_wall_2course_v1":
        return 2.20
    if target_name == "single_face_wall_3course_v1":
        return 2.10
    if target_name == "single_wall_strict_v1":
        return 1.85
    if target_name == "single_face_wall_v1":
        return 2.45
    if target_name == "single_face_wall_extra_high_v1":
        return 2.05
    if target_name == "single_face_wall_high_v1":
        return 2.20
    if target_name == "single_face_wall_4course_v1":
        return 2.80
    if target_name == "single_face_wall_5course_v1":
        return 2.35
    if target_name == "tied_high_wall_v1":
        return 1.35
    if target_name == "tied_high_wall_core_v1":
        return 1.05
    if target_name == "tied_wall_4course_v1":
        return 1.00
    return 0.0


def _wall_shape_metrics(target_name: str, positions: dict[int, np.ndarray], placed: list[int]) -> dict[str, float]:
    if not _is_single_wall_target(target_name) or not placed:
        return {
            "wall_x_span_m": 0.0,
            "wall_y_span_m": 0.0,
            "wall_aspect_xy": 0.0,
            "wall_outlier_count": 0.0,
        }
    xy = np.array([positions[idx][:2] for idx in placed], dtype=float)
    x_span = float(np.max(xy[:, 0]) - np.min(xy[:, 0]))
    y_span = float(np.max(xy[:, 1]) - np.min(xy[:, 1]))
    aspect = float(x_span / max(y_span, 1e-6))
    y_limit = _target_wall_y_limit(target_name)
    outliers = float(np.sum(np.abs(xy[:, 1]) > y_limit))
    return {
        "wall_x_span_m": x_span,
        "wall_y_span_m": y_span,
        "wall_aspect_xy": aspect,
        "wall_outlier_count": outliers,
    }


def _wall_shape_success(target_name: str, metrics: dict[str, float]) -> bool:
    if not _is_single_wall_target(target_name):
        return True
    max_outliers = 2 if _is_strict_single_face_wall_target(target_name) else 4
    return (
        metrics["wall_x_span_m"] >= _target_wall_min_x_span(target_name)
        and metrics["wall_y_span_m"] <= _target_wall_max_y_span(target_name)
        and metrics["wall_aspect_xy"] >= _target_wall_min_aspect(target_name)
        and metrics["wall_outlier_count"] <= max_outliers
    )


def _target_candidate_pose(
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    slot_by_rock: dict[int, TargetSlot],
    rock_index: int,
    slot: TargetSlot,
    rng: np.random.Generator,
    candidate_id: int,
    strategy: str,
    target_name: str = "",
) -> dict[str, np.ndarray]:
    row = row_by_index[rock_index]
    half_height = 0.5 * float(row["bbox_z"])
    support_top = _support_top_for_slot(row_by_index, placed, slot_by_rock, slot)
    jitter_scale = {
        "column_shell": 0.005,
        "literature_column": 0.004,
        "literature_wall": 0.004,
        "statics_wall_line_lock": 0.003,
        "single_column_contact": 0.003,
        "single_column_strict": 0.002,
        "dry_wall": 0.004,
        "pillar_bonded": 0.007,
        "height_bonded": 0.008,
        "height_column": 0.006,
        "wall_bonded": 0.006,
        "column_centered": 0.002,
        "centered_compact": 0.004,
        "geometry_bonded": 0.010,
        "support_first": 0.012,
        "risk_aware": 0.014,
        "random_order": 0.018,
    }.get(strategy, 0.014)
    if candidate_id == 0:
        jitter = np.zeros(2)
    else:
        jitter = rng.normal(0.0, jitter_scale, size=2)
        jitter[1] *= 0.45
    if strategy == "column_shell":
        drop_clearance = 0.040 if slot.course == 0 else 0.052
    elif strategy == "literature_column":
        drop_clearance = 0.008 if slot.course == 0 else 0.034
    elif _is_wall_online_strategy(strategy):
        drop_clearance = 0.004 if slot.course == 0 else 0.014
    elif strategy == "single_column_contact":
        drop_clearance = 0.036 if slot.course == 0 else 0.044
    elif strategy == "single_column_strict":
        drop_clearance = 0.036 if slot.course == 0 else 0.046
    elif strategy == "dry_wall":
        drop_clearance = 0.048 if slot.course == 0 else 0.058
    elif strategy == "pillar_bonded":
        drop_clearance = 0.042 if slot.course == 0 else 0.052
    elif strategy == "height_bonded":
        drop_clearance = 0.052 if slot.course == 0 else 0.058
    elif strategy == "height_column":
        drop_clearance = 0.040 if slot.course == 0 else 0.050
    elif strategy == "wall_bonded":
        drop_clearance = 0.052 if slot.course == 0 else 0.064
    elif strategy == "column_centered":
        drop_clearance = 0.038 if slot.course == 0 else 0.048
    elif strategy == "centered_compact":
        drop_clearance = 0.045 if slot.course == 0 else 0.055
    else:
        drop_clearance = 0.060 if slot.course == 0 else 0.075
    target_xy = np.array([slot.x, slot.y], dtype=float)
    if strategy == "single_column_contact" and slot.course > 0:
        support_xy = _support_xy_for_slot(row_by_index, placed, slot_by_rock, slot)
        target_xy = 0.55 * target_xy + 0.45 * support_xy
    elif strategy == "literature_column" and slot.course > 0:
        support_xy = _support_xy_for_slot(row_by_index, placed, slot_by_rock, slot)
        support_fraction = 0.25 if slot.role == "core" else 0.35
        target_xy = (1.0 - support_fraction) * target_xy + support_fraction * support_xy
    elif _is_wall_online_strategy(strategy) and slot.course > 0:
        support_xy = _support_xy_for_slot(row_by_index, placed, slot_by_rock, slot)
        if slot.role == "tie":
            support_fraction = 0.40 if strategy in {"statics_wall", "statics_wall_line_lock"} else 0.34
        elif slot.course >= 4:
            support_fraction = 0.70 if strategy in {"statics_wall", "statics_wall_line_lock"} else 0.62
        elif slot.course >= 2:
            support_fraction = 0.58 if strategy in {"statics_wall", "statics_wall_line_lock"} else 0.48
        else:
            support_fraction = 0.44 if strategy in {"statics_wall", "statics_wall_line_lock"} else 0.36
        target_xy = (1.0 - support_fraction) * target_xy + support_fraction * support_xy
        if _is_strict_single_face_wall_target(target_name):
            if strategy == "statics_wall_line_lock":
                target_xy[1] = 0.98 * float(slot.y) + 0.02 * target_xy[1]
                y_window = 0.33 * _literature_y_error_soft_limit(target_name)
            else:
                target_xy[1] = 0.90 * float(slot.y) + 0.10 * target_xy[1]
                y_window = 0.5 * _literature_y_error_soft_limit(target_name)
            target_xy[1] = float(np.clip(target_xy[1], slot.y - y_window, slot.y + y_window))
        else:
            target_xy[1] = 0.72 * float(slot.y) + 0.28 * target_xy[1]
    pos = np.array([target_xy[0] + jitter[0], target_xy[1] + jitter[1], support_top + half_height + drop_clearance], dtype=float)
    if strategy == "column_shell":
        max_tilt = 0.045 if slot.role in {"base", "core"} else 0.075
    elif strategy == "literature_column":
        max_tilt = 0.040 if slot.role in {"base", "core"} else 0.065
    elif _is_wall_online_strategy(strategy):
        max_tilt = 0.045 if slot.role in {"base", "tie"} else 0.075
    elif strategy == "single_column_contact":
        max_tilt = 0.030 if slot.role == "base" else 0.050
    elif strategy == "single_column_strict":
        max_tilt = 0.035 if slot.role == "base" else 0.055
    elif strategy == "dry_wall":
        max_tilt = 0.065 if slot.role == "base" else 0.100
    elif strategy == "pillar_bonded":
        max_tilt = 0.052 if slot.role == "base" else 0.080
    elif strategy == "height_bonded":
        max_tilt = 0.070 if slot.role == "base" else 0.105
    elif strategy == "height_column":
        max_tilt = 0.040 if slot.role == "base" else 0.065
    elif strategy == "wall_bonded":
        max_tilt = 0.080 if slot.role == "base" else 0.120
    elif strategy == "column_centered":
        max_tilt = 0.045 if slot.role == "base" else 0.075
    elif strategy == "centered_compact":
        max_tilt = 0.06 if slot.role == "base" else 0.10
    else:
        max_tilt = 0.10 if slot.role == "base" else 0.18
    if _is_wall_online_strategy(strategy):
        quat = _literature_wall_quaternion(rng, candidate_id, max_tilt)
    else:
        quat = _random_quaternion(rng, max_tilt=max_tilt)
    return {"pos": pos, "quat": quat}


def _lower_candidate_release_height(
    mujoco: Any,
    model: Any,
    data: Any,
    row_by_index: dict[int, dict[str, Any]],
    rock_index: int,
    candidate: dict[str, Any],
    search_step_m: float,
    extra_clearance_m: float,
) -> dict[str, Any]:
    """Move a candidate down to the lowest pre-contact release height.

    The original pose is a bbox/support-top estimate. For angular rocks this
    can leave unnecessary vertical fall distance. We scan downward with MuJoCo
    contact detection and release just above the first collision.
    """
    original_qpos = data.qpos.copy()
    original_qvel = data.qvel.copy()
    adjusted = dict(candidate)
    pos = np.asarray(candidate["pos"], dtype=float).copy()
    quat = np.asarray(candidate["quat"], dtype=float).copy()
    original_z = float(pos[2])
    search_step = max(0.001, float(search_step_m))
    extra_clearance = max(0.0, float(extra_clearance_m))
    max_raise = 0.10
    row = row_by_index[rock_index]
    min_z = max(0.001, 0.10 * float(row["bbox_z"]))

    checks = 0
    high_z = original_z
    while _candidate_pose_has_contact(mujoco, model, data, rock_index, pos, quat):
        checks += 1
        high_z += search_step
        pos[2] = high_z
        if high_z - original_z > max_raise:
            data.qpos[:] = original_qpos
            data.qvel[:] = original_qvel
            mujoco.mj_forward(model, data)
            adjusted.update(
                {
                    "low_release_search": 1,
                    "low_release_failed": 1,
                    "release_original_z": original_z,
                    "release_z": float(pos[2]),
                    "release_drop_reduction_m": 0.0,
                    "release_search_checks": checks,
                    "release_contact_z": "",
                    "release_contact_clearance_m": "",
                }
            )
            return adjusted

    best_free_z = high_z
    contact_z: float | None = None
    test_z = high_z - search_step
    while test_z >= min_z:
        checks += 1
        pos[2] = float(test_z)
        if _candidate_pose_has_contact(mujoco, model, data, rock_index, pos, quat):
            contact_z = float(test_z)
            break
        best_free_z = float(test_z)
        test_z -= search_step

    release_z = best_free_z + extra_clearance
    pos[2] = float(release_z)
    data.qpos[:] = original_qpos
    data.qvel[:] = original_qvel
    mujoco.mj_forward(model, data)

    adjusted["pos"] = pos
    adjusted.update(
        {
            "low_release_search": 1,
            "low_release_failed": 0,
            "release_original_z": original_z,
            "release_z": float(release_z),
            "release_drop_reduction_m": max(0.0, original_z - float(release_z)),
            "release_search_checks": checks,
            "release_contact_z": "" if contact_z is None else float(contact_z),
            "release_contact_clearance_m": "" if contact_z is None else float(release_z - contact_z),
        }
    )
    return adjusted


def _candidate_pose_has_contact(
    mujoco: Any,
    model: Any,
    data: Any,
    rock_index: int,
    pos: np.ndarray,
    quat: np.ndarray,
) -> bool:
    _set_freejoint_pose(model, data, rock_index, pos, quat)
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"rock_{rock_index:03d}")
    if body_id < 0:
        return False
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        body1 = int(model.geom_bodyid[int(contact.geom1)])
        body2 = int(model.geom_bodyid[int(contact.geom2)])
        if body_id not in {body1, body2}:
            continue
        other_body = body2 if body1 == body_id else body1
        if other_body != body_id:
            return True
    return False


def _literature_wall_quaternion(rng: np.random.Generator, candidate_id: int, max_tilt: float) -> np.ndarray:
    yaw_sequence = (
        0.0,
        math.pi / 2.0,
        -math.pi / 2.0,
        math.pi,
        0.08,
        -0.08,
        math.pi / 4.0,
        -math.pi / 4.0,
    )
    yaw = yaw_sequence[candidate_id % len(yaw_sequence)]
    if candidate_id >= len(yaw_sequence):
        yaw += rng.normal(0.0, 0.10)
    tilt_axis = 0.0 if candidate_id % 2 == 0 else math.pi / 2.0
    tilt = min(max_tilt, 0.012 * (candidate_id // 2))
    if candidate_id >= len(yaw_sequence):
        tilt = rng.uniform(0.0, max_tilt)
        tilt_axis = rng.uniform(0.0, 2.0 * math.pi)
    q_yaw = _axis_angle(np.array([0.0, 0.0, 1.0]), yaw)
    q_tilt = _axis_angle(np.array([math.cos(tilt_axis), math.sin(tilt_axis), 0.0]), tilt)
    q = _quat_mul(q_yaw, q_tilt)
    return q / np.linalg.norm(q)


def _target_candidate_metrics(
    model: Any,
    data: Any,
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    slot_by_rock: dict[int, TargetSlot],
    rock_index: int,
    slot: TargetSlot,
    pre_positions: dict[int, np.ndarray],
) -> dict[str, float]:
    positions = _body_positions(model, data, placed + [rock_index])
    pos = positions[rock_index]
    row = row_by_index[rock_index]
    target_error = float(np.linalg.norm(pos[:2] - np.array([slot.x, slot.y])))
    target_x_error = float(abs(pos[0] - slot.x))
    target_y_error = float(abs(pos[1] - slot.y))
    support = _target_support_score(positions, row_by_index, placed, slot_by_rock, rock_index, slot)
    support_count = _target_support_contact_count(positions, row_by_index, placed, slot_by_rock, rock_index, slot)
    support_centroid = _support_centroid_for_slot(positions, placed, slot_by_rock, slot)
    support_balance_error = float(np.linalg.norm(pos[:2] - support_centroid)) if slot.course > 0 else 0.0
    placed_disturbance = (
        max(float(np.linalg.norm(positions[idx][:2] - pre_positions[idx][:2])) for idx in placed if idx in pre_positions)
        if placed
        else 0.0
    )
    speed = float(np.linalg.norm(data.qvel, ord=np.inf))
    top_z = float(pos[2]) + 0.5 * float(row["bbox_z"])
    continuity = _support_continuity_metrics(
        positions=positions,
        row_by_index=row_by_index,
        placed=placed,
        slot_by_rock=slot_by_rock,
        rock_index=rock_index,
        slot=slot,
    )
    return {
        "settled_x": float(pos[0]),
        "settled_y": float(pos[1]),
        "settled_z": float(pos[2]),
        "target_error_xy_m": target_error,
        "target_x_error_m": target_x_error,
        "target_y_error_m": target_y_error,
        "radial_distance_m": float(np.linalg.norm(pos[:2])),
        "support_overlap": support,
        "support_contact_count": float(support_count),
        "support_balance_error_m": support_balance_error,
        "bearing_pressure_proxy": float(row["mass"]) / max(support + 0.08 * support_count, 0.08),
        "placed_disturbance_xy_m": placed_disturbance,
        "velocity_inf_norm_after_place": speed,
        "height_gain_m": top_z - _support_top_for_slot(row_by_index, placed, slot_by_rock, slot),
        **continuity,
    }


def _support_continuity_metrics(
    positions: dict[int, np.ndarray],
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    slot_by_rock: dict[int, TargetSlot],
    rock_index: int,
    slot: TargetSlot,
) -> dict[str, float]:
    pos = positions[rock_index]
    row = row_by_index[rock_index]
    half_x = 0.5 * float(row["bbox_x"])
    top_z = float(pos[2]) + 0.5 * float(row["bbox_z"])

    same_course = [idx for idx in placed if slot_by_rock[idx].course == slot.course and idx in positions]
    same_left = [idx for idx in same_course if float(positions[idx][0]) < float(pos[0])]
    same_right = [idx for idx in same_course if float(positions[idx][0]) > float(pos[0])]

    def neighbor_gap(idx: int, side: str) -> float:
        npos = positions[idx]
        nrow = row_by_index[idx]
        n_half_x = 0.5 * float(nrow["bbox_x"])
        if side == "left":
            return float((pos[0] - half_x) - (npos[0] + n_half_x))
        return float((npos[0] - n_half_x) - (pos[0] + half_x))

    left_gap = 0.0
    right_gap = 0.0
    if same_left:
        left_idx = max(same_left, key=lambda idx: float(positions[idx][0]))
        left_gap = neighbor_gap(left_idx, "left")
    if same_right:
        right_idx = min(same_right, key=lambda idx: float(positions[idx][0]))
        right_gap = neighbor_gap(right_idx, "right")

    course_tops = [top_z]
    course_ys = [float(pos[1])]
    for idx in same_course:
        course_tops.append(float(positions[idx][2]) + 0.5 * float(row_by_index[idx]["bbox_z"]))
        course_ys.append(float(positions[idx][1]))

    direct_supports: list[int] = []
    if slot.course > 0:
        for idx in placed:
            below_slot = slot_by_rock[idx]
            if below_slot.course != slot.course - 1 or idx not in positions:
                continue
            below_pos = positions[idx]
            below_row = row_by_index[idx]
            below_half_x = 0.5 * float(below_row["bbox_x"])
            candidate_half_xy = 0.25 * (float(row["bbox_x"]) + float(row["bbox_y"]))
            below_half_xy = 0.25 * (float(below_row["bbox_x"]) + float(below_row["bbox_y"]))
            distance = float(np.linalg.norm(pos[:2] - below_pos[:2]))
            if distance <= 1.02 * max(candidate_half_xy + below_half_xy, 1e-6):
                direct_supports.append(idx)

    load_paths = 1.0 if slot.course == 0 else 0.0
    support_span = 0.0
    underhang_left = 0.0
    underhang_right = 0.0
    if direct_supports:
        support_edges = []
        for idx in direct_supports:
            below_pos = positions[idx]
            below_row = row_by_index[idx]
            below_half_x = 0.5 * float(below_row["bbox_x"])
            support_edges.append((float(below_pos[0] - below_half_x), float(below_pos[0] + below_half_x)))
        support_min = min(edge[0] for edge in support_edges)
        support_max = max(edge[1] for edge in support_edges)
        support_span = float(max(0.0, support_max - support_min))
        cand_min = float(pos[0] - half_x)
        cand_max = float(pos[0] + half_x)
        underhang_left = float(max(0.0, support_min - cand_min))
        underhang_right = float(max(0.0, cand_max - support_max))
        load_paths = float(
            sum(float(row_by_index[idx].get("support_load_path_count", 1.0 if slot_by_rock[idx].course == 0 else 0.0)) for idx in direct_supports)
        )

    positive_gaps = [max(0.0, left_gap), max(0.0, right_gap)]
    return {
        "same_course_placed_count": float(len(same_course)),
        "left_neighbor_present": float(1 if same_left else 0),
        "right_neighbor_present": float(1 if same_right else 0),
        "neighbor_gap_left_m": float(left_gap),
        "neighbor_gap_right_m": float(right_gap),
        "neighbor_gap_max_positive_m": float(max(positive_gaps) if positive_gaps else 0.0),
        "course_height_std_after_m": float(np.std(np.asarray(course_tops, dtype=float))) if course_tops else 0.0,
        "course_y_std_after_m": float(np.std(np.asarray(course_ys, dtype=float))) if course_ys else 0.0,
        "course_y_abs_max_after_m": float(max(abs(y) for y in course_ys)) if course_ys else 0.0,
        "direct_support_count_course_below": float(len(direct_supports)),
        "support_load_path_count": float(load_paths),
        "support_span_x_m": float(support_span),
        "support_span_cover_ratio": float(support_span / max(float(row["bbox_x"]), 1e-6)),
        "support_underhang_left_m": float(underhang_left),
        "support_underhang_right_m": float(underhang_right),
        "support_underhang_max_m": float(max(underhang_left, underhang_right)),
    }


def _target_candidate_score(metrics: dict[str, float], slot: TargetSlot, strategy: str, target_name: str = "") -> float:
    radial_weight = 0.0
    radial_soft_limit = 0.22
    disturbance_weight = 0.0
    contact_weight = 0.0
    balance_weight = 0.0
    bearing_weight = 0.0
    y_line_weight = 0.0
    if strategy == "column_shell":
        target_weight = 9.2
        support_weight = 4.4 if slot.course > 0 else 1.1
        velocity_weight = 4.0
        height_reward = 0.65
        radial_weight = 4.0
        radial_soft_limit = 0.18
        if slot.role == "core":
            support_weight += 0.8
            target_weight -= 0.5
            radial_soft_limit = 0.12
    elif strategy == "literature_column":
        target_weight = 12.4
        support_weight = 5.0 if slot.course > 0 else 1.0
        velocity_weight = 5.2
        height_reward = 0.42
        radial_weight = 5.6
        radial_soft_limit = 0.19
        disturbance_weight = 7.5
        contact_weight = 0.16 if slot.course > 0 else 0.0
        if slot.role == "core":
            support_weight += 0.9
            radial_soft_limit = 0.13
            contact_weight += 0.08
        elif slot.role == "cap":
            height_reward = 0.22
    elif strategy == "literature_wall":
        target_weight = max(8.8, 11.8 - 0.55 * slot.course)
        support_weight = (5.0 + 0.45 * slot.course) if slot.course > 0 else 1.0
        velocity_weight = 5.0
        height_reward = 0.38 if slot.course < 3 else 0.56
        disturbance_weight = 7.0
        contact_weight = 0.18 if slot.course > 0 else 0.0
        if _is_strict_single_face_wall_target(target_name):
            y_line_weight = 10.0 if slot.course > 0 else 7.0
        if slot.role == "tie":
            support_weight += 0.8
            target_weight -= 0.4
            contact_weight += 0.08
        elif slot.role == "cap":
            height_reward = 0.34
    elif strategy in {"statics_wall", "statics_wall_line_lock"}:
        target_weight = max(7.4, 10.4 - 0.55 * slot.course)
        support_weight = (6.4 + 0.70 * slot.course) if slot.course > 0 else 1.3
        velocity_weight = 5.4
        height_reward = 0.34 if slot.course < 3 else 0.48
        disturbance_weight = 8.0
        contact_weight = 0.28 if slot.course > 0 else 0.0
        balance_weight = 5.8 if slot.course > 0 else 0.0
        bearing_weight = 0.035 if slot.course > 0 else 0.0
        if _is_strict_single_face_wall_target(target_name):
            y_line_weight = 14.0 if slot.course > 0 else 9.0
            balance_weight += 1.0 if slot.course > 0 else 0.0
            disturbance_weight += 2.0
        if strategy == "statics_wall_line_lock":
            target_weight += 1.2 if slot.course > 0 else 0.8
            y_line_weight += 9.0 if slot.course > 0 else 6.0
            disturbance_weight += 2.5
            balance_weight += 2.2 if slot.course > 0 else 0.0
            bearing_weight += 0.012 if slot.course > 0 else 0.0
            height_reward = max(0.20, height_reward - 0.08)
        if slot.role == "tie":
            support_weight += 1.2
            target_weight -= 0.8
            contact_weight += 0.14
            balance_weight += 1.5
        elif slot.role == "cap":
            height_reward = 0.30
            balance_weight += 1.2
    elif strategy == "single_column_contact":
        target_weight = 13.0
        support_weight = 6.2 if slot.course > 0 else 1.2
        velocity_weight = 5.4
        height_reward = 0.45
    elif strategy == "single_column_strict":
        target_weight = 14.0
        support_weight = 5.4 if slot.course > 0 else 1.2
        velocity_weight = 5.0
        height_reward = 0.55
    elif strategy == "dry_wall":
        target_weight = 10.4
        support_weight = 2.0 if slot.course > 0 else 0.8
        velocity_weight = 3.7
        height_reward = 0.55
        if slot.role == "tie":
            support_weight += 0.4
            target_weight -= 0.3
    elif strategy == "pillar_bonded":
        target_weight = 9.8
        support_weight = 4.0 if slot.course > 0 else 1.1
        velocity_weight = 4.1
        height_reward = 0.8
        radial_weight = 2.4
        radial_soft_limit = 0.20
    elif strategy == "height_bonded":
        target_weight = 8.2
        support_weight = 3.1 if slot.course > 0 else 0.9
        velocity_weight = 3.4
        height_reward = 1.4
    elif strategy == "height_column":
        target_weight = 10.2
        support_weight = 3.6 if slot.course > 0 else 1.0
        velocity_weight = 3.9
        height_reward = 1.0
    elif strategy == "wall_bonded":
        target_weight = 8.6
        support_weight = 2.8 if slot.course > 0 else 0.8
        velocity_weight = 3.0
        height_reward = 0.0
    elif strategy == "column_centered":
        target_weight = 11.0
        support_weight = 3.2 if slot.course > 0 else 0.9
        velocity_weight = 3.6
        height_reward = 0.0
    elif strategy == "centered_compact":
        target_weight = 9.8
        support_weight = 2.6 if slot.course > 0 else 0.8
        velocity_weight = 3.2
        height_reward = 0.0
    elif strategy == "support_first":
        target_weight = 5.6
        support_weight = 2.4 if slot.course > 0 else 0.6
        velocity_weight = 2.6
        height_reward = 0.0
    elif strategy == "risk_aware":
        target_weight = 6.8
        support_weight = 2.0 if slot.course > 0 else 0.5
        velocity_weight = 2.4
        height_reward = 0.0
    elif strategy == "random_order":
        target_weight = 6.2
        support_weight = 1.6 if slot.course > 0 else 0.4
        velocity_weight = 2.0
        height_reward = 0.0
        radial_weight = 1.5
        radial_soft_limit = 0.23
    else:
        target_weight = 7.4
        support_weight = 1.8 if slot.course > 0 else 0.4
        velocity_weight = 2.2
        height_reward = 0.0
    height_penalty = 0.25 * max(0.0, 0.035 - metrics["height_gain_m"])
    radial_penalty = radial_weight * max(0.0, metrics["radial_distance_m"] - radial_soft_limit)
    hard_penalty = 0.0
    probe_steps = int(float(metrics.get("candidate_probe_steps", 0.0)))
    probe_penalty = 0.0
    if probe_steps > 0:
        probe_rock_drift = float(metrics.get("candidate_probe_rock_drift_m", 0.0))
        probe_placed_disturbance = float(metrics.get("candidate_probe_placed_disturbance_m", 0.0))
        probe_speed = float(metrics.get("candidate_probe_speed", 0.0))
        probe_drift_soft = 0.032 if slot.course < 3 else 0.024
        probe_penalty = (
            20.0 * max(0.0, probe_rock_drift - probe_drift_soft)
            + 28.0 * max(0.0, probe_placed_disturbance - 0.016)
            + 1.8 * max(0.0, probe_speed - 0.22)
        )
    if strategy == "literature_column":
        if slot.course > 0 and metrics["support_contact_count"] < 1.0:
            hard_penalty += 8.0
        hard_penalty += 4.5 * max(0.0, metrics["target_error_xy_m"] - 0.22)
        hard_penalty += 10.0 * max(0.0, metrics["radial_distance_m"] - 0.24)
        hard_penalty += 3.0 * max(0.0, -0.02 - metrics["height_gain_m"])
    elif strategy in {"literature_wall", "statics_wall", "statics_wall_line_lock"}:
        support_overlap_min = _literature_min_support_overlap(target_name)
        target_error_soft = _literature_target_error_soft_limit(target_name)
        y_error_soft = _literature_y_error_soft_limit(target_name)
        settled_y_limit = _target_wall_y_limit(target_name)
        if slot.course > 0 and metrics["support_contact_count"] < 1.0 and metrics["support_overlap"] < support_overlap_min:
            hard_penalty += 8.0
        hard_penalty += 5.5 * max(0.0, metrics["target_error_xy_m"] - target_error_soft)
        hard_penalty += 9.0 * max(0.0, metrics["target_y_error_m"] - y_error_soft)
        hard_penalty += 3.0 * max(0.0, -0.02 - metrics["height_gain_m"])
        if settled_y_limit > 0.0:
            hard_penalty += 7.0 * max(0.0, abs(metrics["settled_y"]) - settled_y_limit)
        if strategy in {"statics_wall", "statics_wall_line_lock"}:
            hard_penalty += 9.0 * max(0.0, metrics["support_balance_error_m"] - 0.070)
            hard_penalty += 0.055 * max(0.0, metrics["bearing_pressure_proxy"] - 0.035)
        if strategy == "statics_wall_line_lock":
            hard_penalty += 7.5 * max(0.0, metrics["target_error_xy_m"] - 0.82 * target_error_soft)
            hard_penalty += 16.0 * max(0.0, metrics["target_y_error_m"] - 0.70 * y_error_soft)
            hard_penalty += 12.0 * max(0.0, abs(metrics["settled_y"]) - 0.82 * settled_y_limit) if settled_y_limit > 0.0 else 0.0
            hard_penalty += 14.0 * max(0.0, metrics["support_balance_error_m"] - 0.055)
    return (
        target_weight * metrics["target_error_xy_m"]
        + velocity_weight * metrics["velocity_inf_norm_after_place"]
        + disturbance_weight * metrics["placed_disturbance_xy_m"]
        + balance_weight * metrics.get("support_balance_error_m", 0.0)
        + bearing_weight * metrics.get("bearing_pressure_proxy", 0.0)
        + y_line_weight * metrics.get("target_y_error_m", 0.0)
        - support_weight * metrics["support_overlap"]
        - contact_weight * metrics["support_contact_count"]
        - height_reward * metrics["height_gain_m"]
        + height_penalty
        + radial_penalty
        + hard_penalty
        + probe_penalty
        + 0.04 * slot.course
    )


def _support_top_for_slot(
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    slot_by_rock: dict[int, TargetSlot],
    slot: TargetSlot,
) -> float:
    if slot.course == 0 or not placed:
        return 0.0
    candidates = [
        idx
        for idx in placed
        if slot_by_rock[idx].course == slot.course - 1
        and ((slot_by_rock[idx].x - slot.x) ** 2 + (slot_by_rock[idx].y - slot.y) ** 2) ** 0.5 <= 0.18
    ]
    if not candidates:
        candidates = placed
    target_xy = np.array([slot.x, slot.y], dtype=float)
    nearest = sorted(
        candidates,
        key=lambda idx: float(
            np.linalg.norm(
                np.array([slot_by_rock[idx].x, slot_by_rock[idx].y], dtype=float)
                - target_xy
            )
        ),
    )[:2]
    return max(float(row_by_index[idx].get("last_top_z", 0.0)) for idx in nearest)


def _support_xy_for_slot(
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    slot_by_rock: dict[int, TargetSlot],
    slot: TargetSlot,
) -> np.ndarray:
    if slot.course == 0 or not placed:
        return np.array([slot.x, slot.y], dtype=float)
    candidates = [
        idx
        for idx in placed
        if slot_by_rock[idx].course == slot.course - 1
        and "last_center_x" in row_by_index[idx]
        and "last_center_y" in row_by_index[idx]
    ]
    if not candidates:
        candidates = [idx for idx in placed if "last_center_x" in row_by_index[idx] and "last_center_y" in row_by_index[idx]]
    if not candidates:
        return np.array([slot.x, slot.y], dtype=float)
    target_xy = np.array([slot.x, slot.y], dtype=float)
    ranked = sorted(
        candidates,
        key=lambda item: float(
            np.linalg.norm(
                np.array([float(row_by_index[item]["last_center_x"]), float(row_by_index[item]["last_center_y"])])
                - target_xy
            )
        ),
    )[:3]
    centers = [
        np.array([float(row_by_index[idx]["last_center_x"]), float(row_by_index[idx]["last_center_y"])], dtype=float)
        for idx in ranked
    ]
    distances = np.array([max(float(np.linalg.norm(center - target_xy)), 1e-4) for center in centers], dtype=float)
    weights = 1.0 / distances
    weights /= float(np.sum(weights))
    return np.sum([weight * center for weight, center in zip(weights, centers)], axis=0)


def _support_centroid_for_slot(
    positions: dict[int, np.ndarray],
    placed: list[int],
    slot_by_rock: dict[int, TargetSlot],
    slot: TargetSlot,
) -> np.ndarray:
    target_xy = np.array([slot.x, slot.y], dtype=float)
    if slot.course == 0 or not placed:
        return target_xy
    candidates = [
        idx
        for idx in placed
        if slot_by_rock[idx].course < slot.course
    ]
    if not candidates:
        return target_xy
    ranked = sorted(candidates, key=lambda idx: float(np.linalg.norm(positions[idx][:2] - target_xy)))[:3]
    centers = [positions[idx][:2] for idx in ranked]
    distances = np.array([max(float(np.linalg.norm(center - target_xy)), 1e-4) for center in centers], dtype=float)
    weights = 1.0 / distances
    weights /= float(np.sum(weights))
    return np.sum([weight * center for weight, center in zip(weights, centers)], axis=0)


def _target_support_score(
    positions: dict[int, np.ndarray],
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    slot_by_rock: dict[int, TargetSlot],
    rock_index: int,
    slot: TargetSlot,
) -> float:
    if slot.course == 0 or not placed:
        return 1.0
    pos = positions[rock_index]
    row = row_by_index[rock_index]
    half_xy = 0.25 * (float(row["bbox_x"]) + float(row["bbox_y"]))
    overlaps: list[float] = []
    for idx in placed:
        below_slot = slot_by_rock[idx]
        if below_slot.course >= slot.course:
            continue
        below_pos = positions[idx]
        below_row = row_by_index[idx]
        below_half_xy = 0.25 * (float(below_row["bbox_x"]) + float(below_row["bbox_y"]))
        distance = float(np.linalg.norm(pos[:2] - below_pos[:2]))
        overlaps.append(max(0.0, 1.0 - distance / max(half_xy + below_half_xy, 1e-6)))
    return float(max(overlaps) if overlaps else 0.0)


def _target_support_contact_count(
    positions: dict[int, np.ndarray],
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    slot_by_rock: dict[int, TargetSlot],
    rock_index: int,
    slot: TargetSlot,
) -> int:
    if slot.course == 0 or not placed:
        return 1
    pos = positions[rock_index]
    row = row_by_index[rock_index]
    half_xy = 0.25 * (float(row["bbox_x"]) + float(row["bbox_y"]))
    count = 0
    for idx in placed:
        below_slot = slot_by_rock[idx]
        if below_slot.course >= slot.course:
            continue
        below_pos = positions[idx]
        below_row = row_by_index[idx]
        below_half_xy = 0.25 * (float(below_row["bbox_x"]) + float(below_row["bbox_y"]))
        distance = float(np.linalg.norm(pos[:2] - below_pos[:2]))
        if distance <= 0.92 * max(half_xy + below_half_xy, 1e-6):
            count += 1
    return count


def _stone_role_score(row: dict[str, Any], role: str, strategy: str) -> tuple[float, float]:
    label = str(row.get("cluster_label", ""))
    source = str(row.get("source_kind", ""))
    volume = float(row["volume"])
    footprint = 0.5 * (float(row["bbox_x"]) + float(row["bbox_y"]))
    bbox_z = float(row["bbox_z"])
    compactness = float(row["compactness"])
    elongation = float(row["elongation"])
    flatness = float(row["flatness"])
    spike = float(row.get("spike_score", 0.0))
    stability = float(row["stability_score"])
    base_bonus = 0.0
    if label.startswith("wedge_or_broad_clast") or source == "wedge_clast":
        base_bonus -= 0.18
    if label.startswith("subangular_block") or source == "subangular_block":
        base_bonus -= 0.12
    if label.startswith("compact_block_clast") or source == "compact_block_clast":
        base_bonus -= 0.14
    if label.startswith("upright_block_clast") or source == "upright_block_clast":
        base_bonus -= 0.10
    if source == "buttress_clast":
        base_bonus -= 0.16
    if source == "bearing_block_clast":
        base_bonus -= 0.18
    if source == "course_block_clast":
        base_bonus -= 0.10
    if source == "chock_clast":
        base_bonus += 0.12
    if source in {"tie_bridge_clast", "cap_block_clast"}:
        base_bonus += 0.16
    if label.startswith("spiky_reject"):
        base_bonus += 2.0

    if role == "base":
        score = (
            -1.8 * volume
            -0.9 * compactness
            -0.10 * stability
            +0.45 * max(0.0, elongation - 1.35)
            +0.25 * max(0.0, flatness - 1.45)
            +3.0 * spike
            +base_bonus
        )
    elif role == "middle":
        score = (
            -0.8 * compactness
            -0.08 * stability
            +0.22 * abs(elongation - 1.18)
            +2.5 * spike
            +0.5 * base_bonus
        )
    elif role == "core":
        score = (
            -0.95 * compactness
            -0.08 * stability
            +0.35 * max(0.0, elongation - 1.24)
            +0.32 * max(0.0, flatness - 1.20)
            +3.0 * spike
            +0.35 * base_bonus
        )
    elif role == "tie":
        score = (
            -0.6 * compactness
            -0.08 * stability
            -0.20 * min(elongation, 1.45)
            +0.42 * max(0.0, elongation - 1.55)
            +0.34 * max(0.0, flatness - 1.35)
            +3.2 * spike
            +0.30 * base_bonus
        )
    else:
        score = (
            -0.05 * stability
            +0.18 * abs(elongation - 1.25)
            +2.5 * spike
            +0.15 * max(0.0, volume - 0.0015)
        )
    if strategy == "risk_aware":
        score += 0.45 * max(0.0, elongation - 1.35)
        score += 3.5 * spike
        if label.startswith("fractured_clast") or source == "fractured_clast":
            score += 0.08
    elif strategy == "literature_column":
        target_footprint = {
            "base": 0.135,
            "middle": 0.118,
            "core": 0.108,
            "cap": 0.098,
        }.get(role, 0.115)
        target_volume = {
            "base": 0.00095,
            "middle": 0.00070,
            "core": 0.00058,
            "cap": 0.00045,
        }.get(role, 0.00070)
        score += 7.5 * abs(footprint - target_footprint)
        score += 180.0 * abs(volume - target_volume)
        score -= 0.42 * compactness
        score -= 0.10 * stability
        score += 0.75 * max(0.0, elongation - 1.22)
        score += 0.85 * max(0.0, flatness - 1.20)
        score += 5.8 * spike
        if source in {"elongated_clast", "fractured_clast"} or label.startswith("elongated_clast"):
            score += 0.22
        if source in {"compact_block_clast", "equant_clast", "subangular_block", "keystone_clast"}:
            score -= 0.16
        if role == "base":
            if source == "equant_clast":
                score += 0.18
            if source in {"buttress_clast", "subangular_block", "wedge_clast", "compact_block_clast"}:
                score -= 0.20
            score += 0.60 * max(0.0, footprint - 0.165)
            score += 0.35 * max(0.0, bbox_z / max(footprint, 1e-6) - 0.95)
            score += 0.18 * max(0.0, 0.108 - footprint)
            score -= 0.10 if source in {"buttress_clast", "subangular_block"} else 0.0
        elif role == "middle":
            score += 0.34 * max(0.0, bbox_z - 0.145)
            score += 0.25 * max(0.0, volume - 0.00110)
        elif role == "core":
            score += 0.42 * max(0.0, footprint - 0.135)
            score -= 0.10 if source in {"compact_block_clast", "equant_clast"} else 0.0
        elif role == "cap":
            score += 0.70 * max(0.0, footprint - 0.125)
            score += 0.35 * max(0.0, bbox_z - 0.125)
    elif _is_wall_online_strategy(strategy):
        target_footprint = {
            "base": 0.145,
            "middle": 0.120,
            "tie": 0.140,
            "cap": 0.110,
        }.get(role, 0.125)
        target_volume = {
            "base": 0.00105,
            "middle": 0.00086,
            "tie": 0.00088,
            "cap": 0.00066,
        }.get(role, 0.00078)
        score += 6.8 * abs(footprint - target_footprint)
        score += 130.0 * abs(volume - target_volume)
        score -= 0.38 * compactness
        score -= 0.08 * stability
        score += 0.62 * max(0.0, flatness - 1.25)
        score += 0.48 * max(0.0, elongation - 1.42)
        score += 5.6 * spike
        if source in {"fractured_clast", "elongated_clast"} or label.startswith("elongated_clast"):
            score += 0.18
        if source in {
            "compact_block_clast",
            "wall_block_clast",
            "subangular_block",
            "buttress_clast",
            "keystone_clast",
            "bearing_block_clast",
            "course_block_clast",
            "tie_bridge_clast",
            "interlock_block_clast",
            "cap_block_clast",
        }:
            score -= 0.14
        if strategy in {"statics_wall", "statics_wall_line_lock"}:
            score -= 0.18 * stability
            score -= 0.22 * compactness
            score += 0.55 * max(0.0, spike - 0.06)
            score += 0.32 * max(0.0, flatness - 1.38)
        if strategy == "statics_wall_line_lock":
            score -= 0.10 * compactness
            score -= 0.08 * stability
            score += 0.70 * max(0.0, spike - 0.045)
            score += 0.30 * max(0.0, flatness - 1.32)
            score += 0.20 * max(0.0, bbox_z - 0.165)
        if role == "base":
            if source in {"bearing_block_clast", "wall_block_clast", "buttress_clast", "subangular_block", "wedge_clast"}:
                score -= 0.22
            if strategy in {"statics_wall", "statics_wall_line_lock"} and source == "bearing_block_clast":
                score -= 0.16
            if strategy in {"statics_wall", "statics_wall_line_lock"} and source in {"tie_bridge_clast", "cap_block_clast", "chock_clast"}:
                score += 0.34
            if source == "equant_clast":
                score += 0.12
            score += 0.24 * max(0.0, bbox_z / max(footprint, 1e-6) - 1.02)
            score += 0.25 * max(0.0, 0.118 - footprint)
        elif role == "tie":
            if source in {"tie_bridge_clast", "wall_block_clast"}:
                score -= 0.12
            if strategy in {"statics_wall", "statics_wall_line_lock"} and source == "tie_bridge_clast":
                score -= 0.18
            score -= 0.12 * min(elongation, 1.35)
            score += 0.35 * max(0.0, elongation - 1.55)
            score += 0.35 * max(0.0, flatness - 1.28)
        elif role == "middle":
            if source in {"course_block_clast", "wall_block_clast"}:
                score -= 0.16
            if strategy in {"statics_wall", "statics_wall_line_lock"} and source in {"course_block_clast", "compact_block_clast", "equant_clast"}:
                score -= 0.12
            score += 0.25 * max(0.0, footprint - 0.155)
            score -= 0.28 * min(max(0.0, bbox_z - 0.105), 0.045)
            score += 0.72 * max(0.0, bbox_z - 0.165)
            score += 0.22 * max(0.0, 0.098 - bbox_z)
            if strategy in {"statics_wall", "statics_wall_line_lock"}:
                score -= 0.18 * min(max(0.0, bbox_z - 0.105), 0.055)
                score += 0.52 * max(0.0, bbox_z - 0.175)
            if strategy == "statics_wall_line_lock":
                score += 0.42 * max(0.0, bbox_z - 0.150)
                score += 0.18 * max(0.0, flatness - 1.22)
        elif role == "cap":
            score += 0.35 * max(0.0, footprint - 0.135)
            score -= 0.18 * min(max(0.0, bbox_z - 0.100), 0.040)
            score += 0.70 * max(0.0, bbox_z - 0.155)
            if source in {"cap_block_clast", "compact_block_clast", "equant_clast"}:
                score -= 0.12
    elif strategy == "column_shell":
        score -= 0.50 * compactness
        score -= 0.14 * stability
        score -= 0.55 * bbox_z
        score += 0.72 * max(0.0, elongation - 1.30)
        score += 0.72 * max(0.0, flatness - 1.24)
        score += 5.4 * spike
        if source == "compact_block_clast" or label.startswith("compact_block_clast"):
            score -= 0.20
        if source == "upright_block_clast" or label.startswith("upright_block_clast"):
            score -= 0.10
        if role == "base":
            score -= 0.44 * volume
            if source == "buttress_clast":
                score -= 0.18
        elif role == "core":
            score -= 0.28 * compactness
            score += 0.20 * max(0.0, volume - 0.0018)
            score += 0.22 * max(0.0, bbox_z - 0.150)
        elif role == "cap":
            score += 0.24 * volume
            score += 0.14 * max(0.0, elongation - 1.18)
    elif strategy == "single_column_contact":
        score -= 0.64 * compactness
        score -= 0.15 * stability
        score -= 0.95 * bbox_z
        score += 1.00 * max(0.0, elongation - 1.25)
        score += 0.90 * max(0.0, flatness - 1.18)
        score += 6.0 * spike
        if source == "upright_block_clast" or label.startswith("upright_block_clast"):
            score -= 0.14
        if source == "compact_block_clast" or label.startswith("compact_block_clast"):
            score -= 0.20
        if role == "base":
            score -= 0.48 * volume
            score += 0.30 * max(0.0, bbox_z - 0.145)
            if source == "buttress_clast":
                score -= 0.18
        elif role == "cap":
            score += 0.25 * volume
            score += 0.10 * max(0.0, bbox_z - 0.140)
    elif strategy == "single_column_strict":
        score -= 0.62 * compactness
        score -= 0.14 * stability
        score -= 1.15 * bbox_z
        score += 1.10 * max(0.0, elongation - 1.22)
        score += 0.95 * max(0.0, flatness - 1.18)
        score += 6.2 * spike
        if source == "upright_block_clast" or label.startswith("upright_block_clast"):
            score -= 0.22
        if source == "compact_block_clast" or label.startswith("compact_block_clast"):
            score -= 0.18
        if role == "base":
            score -= 0.46 * volume
            score += 0.32 * max(0.0, bbox_z - 0.145)
            if source == "buttress_clast":
                score -= 0.18
        elif role == "cap":
            score += 0.28 * volume
            score += 0.12 * max(0.0, bbox_z - 0.140)
    elif strategy == "dry_wall":
        score -= 0.34 * compactness
        score -= 0.10 * stability
        score += 0.32 * max(0.0, flatness - 1.28)
        score += 0.34 * max(0.0, elongation - 1.35)
        score += 4.2 * spike
        if role == "base":
            score -= 0.46 * volume
            score += 0.12 * max(0.0, bbox_z - 0.135)
        elif role == "tie":
            score -= 0.18 * min(elongation, 1.45)
            score += 0.30 * max(0.0, elongation - 1.65)
        elif role == "cap":
            score += 0.18 * volume
            score += 0.20 * max(0.0, flatness - 1.22)
    elif strategy == "pillar_bonded":
        score -= 0.42 * compactness
        score -= 0.10 * stability
        score -= 1.70 * bbox_z
        score += 0.62 * max(0.0, elongation - 1.18)
        score += 0.54 * max(0.0, flatness - 1.18)
        score += 5.2 * spike
        if role == "base":
            score -= 0.42 * volume
            score += 0.16 * max(0.0, bbox_z - 0.135)
        elif role == "cap":
            score += 0.24 * volume
    elif strategy == "support_first":
        score -= 0.25 * compactness
        score += 0.20 * max(0.0, flatness - 1.35)
    elif strategy == "height_bonded":
        score -= 0.32 * compactness
        score -= 0.10 * stability
        score -= 2.20 * bbox_z
        score += 0.36 * max(0.0, flatness - 1.24)
        score += 0.42 * max(0.0, elongation - 1.28)
        score += 4.0 * spike
        if role == "base":
            score -= 0.25 * volume
            score += 0.18 * max(0.0, bbox_z - 0.130)
    elif strategy == "height_column":
        score -= 0.38 * compactness
        score -= 0.10 * stability
        score -= 2.60 * bbox_z
        score += 0.58 * max(0.0, elongation - 1.18)
        score += 0.50 * max(0.0, flatness - 1.18)
        score += 5.0 * spike
        if role == "base":
            score -= 0.25 * volume
            score += 0.24 * max(0.0, bbox_z - 0.135)
    elif strategy == "wall_bonded":
        score -= 0.30 * compactness
        score -= 0.10 * stability
        score += 0.28 * max(0.0, flatness - 1.30)
        score += 0.35 * max(0.0, elongation - 1.30)
        score += 3.5 * spike
    elif strategy == "centered_compact":
        score -= 0.35 * compactness
        score -= 0.08 * stability
        score += 0.45 * max(0.0, elongation - 1.20)
        score += 0.30 * max(0.0, flatness - 1.25)
        score += 4.5 * spike
        if role == "base":
            score -= 0.35 * volume
    elif strategy == "column_centered":
        score -= 0.45 * compactness
        score -= 0.08 * stability
        score += 0.70 * max(0.0, elongation - 1.16)
        score += 0.45 * max(0.0, flatness - 1.20)
        score += 5.0 * spike
        if role == "base":
            score -= 0.45 * volume
    return (score, volume)


def _effective_settle_steps(requested_steps: int, gravity_label: str) -> int:
    if gravity_label == "moon":
        return max(requested_steps, int(round(requested_steps * 2.4)))
    return requested_steps


def _structured_top(after: dict[int, np.ndarray], row_by_index: dict[int, dict[str, Any]], placed: list[int]) -> float:
    top = 0.0
    for idx in placed:
        top = max(top, float(after[idx][2]) + 0.5 * float(row_by_index[idx]["bbox_z"]))
    return top


def _visible_course_count(
    after: dict[int, np.ndarray],
    row_by_index: dict[int, dict[str, Any]],
    slot_by_rock: dict[int, TargetSlot],
    placed: list[int],
) -> int:
    visible = set()
    for idx in placed:
        if float(after[idx][2]) > 0.35 * float(row_by_index[idx]["bbox_z"]):
            visible.add(slot_by_rock[idx].course)
    return len(visible)


def _structured_failure_rows(
    gravity_label: str,
    trial_id: int,
    strategy: str,
    target_name: str,
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    slot_by_rock: dict[int, TargetSlot],
    before: dict[int, np.ndarray],
    after: dict[int, np.ndarray],
    stable_by_index: dict[int, bool],
    target_errors: dict[int, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in placed:
        if stable_by_index[idx]:
            continue
        row = row_by_index[idx]
        slot = slot_by_rock[idx]
        reasons = []
        if target_errors[idx] >= 0.20:
            reasons.append("missed_target")
        if float(after[idx][2]) <= 0.35 * float(row["bbox_z"]):
            reasons.append("low_or_fallen")
        drift = float(np.linalg.norm(after[idx][:2] - before[idx][:2]))
        if drift >= 0.15:
            reasons.append("post_hold_drift")
        if not reasons:
            reasons.append("unstable_structure")
        rows.append(
            {
                "gravity": gravity_label,
                "trial": trial_id,
                "strategy": strategy,
                "target_name": target_name,
                "rock_index": idx,
                "slot_id": slot.slot_id,
                "course": slot.course,
                "role": slot.role,
                "cluster_label": row["cluster_label"],
                "source_kind": row["source_kind"],
                "failure_reason": "+".join(reasons),
                "target_error_xy_m": target_errors[idx],
                "horizontal_drift_m": drift,
                "final_x": float(after[idx][0]),
                "final_y": float(after[idx][1]),
                "final_z": float(after[idx][2]),
                "volume": float(row["volume"]),
                "roughness": float(row["roughness"]),
                "angularity": float(row["angularity"]),
                "spike_score": float(row.get("spike_score", 0.0)),
                "flatness": float(row["flatness"]),
                "elongation": float(row["elongation"]),
                "stability_score": float(row["stability_score"]),
            }
        )
    return rows


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


def _candidate_pose_row(
    candidate_context: dict[str, Any],
    row_by_index: dict[int, dict[str, Any]],
    rock_index: int,
    slot: TargetSlot,
    candidate_count: int,
    candidate_id: int,
    candidate: dict[str, Any],
    metrics: dict[str, float],
    score: float,
    ranker_prob: float | None = None,
    ranker_rank: int = 0,
    ranker_top_k: int = 0,
    pose_risk_prob: float | None = None,
    pose_risk_weight: float = 0.0,
    pose_risk_penalty: float = 0.0,
    pose_rank_score: float = 0.0,
    base_candidate_score: float | None = None,
) -> dict[str, Any]:
    row = row_by_index[rock_index]
    pos = candidate["pos"]
    quat = candidate["quat"]
    output: dict[str, Any] = {
        "target_name": candidate_context.get("target_name", ""),
        "strategy": candidate_context.get("strategy", ""),
        "gravity": candidate_context.get("gravity", ""),
        "trial": candidate_context.get("trial", ""),
        "slot_id": slot.slot_id,
        "course": slot.course,
        "role": slot.role,
        "target_x": slot.x,
        "target_y": slot.y,
        "rock_index": rock_index,
        "source_kind": row.get("source_kind", ""),
        "cluster_label": row.get("cluster_label", ""),
        "candidate_id": candidate_id,
        "candidate_count": candidate_count,
        "pose_x": float(pos[0]),
        "pose_y": float(pos[1]),
        "pose_z": float(pos[2]),
        "pose_qw": float(quat[0]),
        "pose_qx": float(quat[1]),
        "pose_qy": float(quat[2]),
        "pose_qz": float(quat[3]),
        "candidate_score": float(score),
        "base_candidate_score": "" if base_candidate_score is None else float(base_candidate_score),
        "selected_by_pose_search": 0,
        "ranker_prob": "" if ranker_prob is None else float(ranker_prob),
        "ranker_rank": int(ranker_rank),
        "ranker_top_k": int(ranker_top_k),
        "pose_risk_prob": "" if pose_risk_prob is None else float(pose_risk_prob),
        "pose_risk_weight": float(pose_risk_weight),
        "pose_risk_penalty": float(pose_risk_penalty),
        "pose_rank_score": float(pose_rank_score),
        "low_release_search": int(candidate.get("low_release_search", 0)),
        "low_release_failed": int(candidate.get("low_release_failed", 0)),
        "release_original_z": candidate.get("release_original_z", ""),
        "release_z": candidate.get("release_z", ""),
        "release_drop_reduction_m": candidate.get("release_drop_reduction_m", ""),
        "release_search_checks": candidate.get("release_search_checks", ""),
        "release_contact_z": candidate.get("release_contact_z", ""),
        "release_contact_clearance_m": candidate.get("release_contact_clearance_m", ""),
    }
    for key, value in candidate_context.items():
        output.setdefault(key, value)
    for key, value in metrics.items():
        output[key] = float(value)
    return output


def _candidate_ranker_prob(
    ranker: dict[str, Any] | None,
    candidate_context: dict[str, Any],
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    slot_by_rock: dict[int, TargetSlot],
    rock_index: int,
    slot: TargetSlot,
    candidate_count: int,
    candidate_id: int,
    candidate: dict[str, Any],
) -> float | None:
    if ranker is None:
        return None
    if ranker.get("kind") == "torch_support_map_cnn":
        return _torch_support_map_ranker_score(
            ranker=ranker,
            candidate_context=candidate_context,
            row_by_index=row_by_index,
            placed=placed,
            slot_by_rock=slot_by_rock,
            rock_index=rock_index,
            slot=slot,
            candidate_count=candidate_count,
            candidate_id=candidate_id,
            candidate=candidate,
        )
    row = _candidate_ranker_feature_row(
        candidate_context=candidate_context,
        rock_row=row_by_index[rock_index],
        rock_index=rock_index,
        slot=slot,
        candidate_count=candidate_count,
        candidate_id=candidate_id,
        candidate=candidate,
    )
    return _mlp_binary_prob(ranker, row)


def _stone_fit_prob(
    ranker: dict[str, Any] | None,
    candidate_context: dict[str, Any],
    rock_row: dict[str, Any],
    rock_index: int,
    slot: TargetSlot,
) -> float | None:
    if ranker is None:
        return None
    row = _stone_fit_feature_row(candidate_context, rock_row, rock_index, slot)
    return _mlp_binary_prob(ranker, row)


def _pose_risk_prob(
    ranker: dict[str, Any] | None,
    candidate_context: dict[str, Any],
    row_by_index: dict[int, dict[str, Any]],
    rock_index: int,
    slot: TargetSlot,
    candidate_count: int,
    candidate_id: int,
    candidate: dict[str, Any],
) -> float | None:
    if ranker is None:
        return None
    row = _candidate_ranker_feature_row(
        candidate_context=candidate_context,
        rock_row=row_by_index[rock_index],
        rock_index=rock_index,
        slot=slot,
        candidate_count=candidate_count,
        candidate_id=candidate_id,
        candidate=candidate,
    )
    return _mlp_binary_prob(ranker, row)


def _mlp_binary_prob(model: dict[str, Any], row: dict[str, Any]) -> float | None:
    schema = model.get("schema", {})
    values: list[float] = []
    for column in schema.get("numeric_columns", []):
        raw = row.get(column, "")
        if _is_number(raw):
            values.append(float(raw))
            values.append(0.0)
        else:
            values.append(0.0)
            values.append(1.0)
    for column in schema.get("categorical_columns", []):
        raw = str(row.get(column, ""))
        for category in schema.get("categories", {}).get(column, []):
            values.append(1.0 if raw == str(category) else 0.0)
    x = np.asarray(values, dtype=np.float64).reshape(1, -1)
    mean = np.asarray(schema.get("feature_mean", [0.0] * x.shape[1]), dtype=np.float64).reshape(1, -1)
    std = np.asarray(schema.get("feature_std", [1.0] * x.shape[1]), dtype=np.float64).reshape(1, -1)
    if mean.shape[1] != x.shape[1] or std.shape[1] != x.shape[1]:
        return None
    if not {"w1", "b1", "w2", "b2"}.issubset(model):
        return None
    x = (x - mean) / np.where(std < 1e-6, 1.0, std)
    hidden = np.maximum(np.einsum("ij,jk->ik", x, model["w1"], optimize=False) + model["b1"], 0.0)
    logits = np.einsum("ij,jk->ik", hidden, model["w2"], optimize=False) + model["b2"]
    return float(1.0 / (1.0 + np.exp(-np.clip(logits[0, 0], -40.0, 40.0))))


def _torch_support_map_ranker_score(
    ranker: dict[str, Any],
    candidate_context: dict[str, Any],
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    slot_by_rock: dict[int, TargetSlot],
    rock_index: int,
    slot: TargetSlot,
    candidate_count: int,
    candidate_id: int,
    candidate: dict[str, Any],
) -> float | None:
    runtime = _torch_support_map_runtime(ranker)
    if runtime is None:
        return None
    schema = ranker.get("schema", {})
    map_tensor = _online_support_map_tensor(
        schema=schema,
        candidate_context=candidate_context,
        row_by_index=row_by_index,
        placed=placed,
        slot_by_rock=slot_by_rock,
        rock_index=rock_index,
        slot=slot,
        candidate=candidate,
        grid_size=int(runtime["grid_size"]),
        window_m=float(runtime["window_m"]),
    )
    numeric = _online_support_map_numeric(
        schema=schema,
        candidate_context=candidate_context,
        rock_row=row_by_index[rock_index],
        rock_index=rock_index,
        slot=slot,
        candidate_count=candidate_count,
        candidate_id=candidate_id,
        candidate=candidate,
    )
    if map_tensor is None or numeric is None:
        return None
    torch = runtime["torch"]
    device = runtime["device"]
    with torch.no_grad():
        maps = torch.from_numpy(map_tensor[None, :, :, :].astype(np.float32)).to(device)
        nums = torch.from_numpy(numeric[None, :].astype(np.float32)).to(device)
        score = runtime["model"](maps, nums)
    return float(score.detach().cpu().numpy().reshape(-1)[0])


def _torch_support_map_runtime(ranker: dict[str, Any]) -> dict[str, Any] | None:
    runtime = ranker.get("_runtime")
    if runtime is not None:
        return runtime
    try:
        import torch
        from torch import nn
    except Exception:
        return None

    schema = ranker.get("schema", {})
    metrics = ranker.get("metrics", {})
    model_path = ranker.get("model_path", "")
    if not model_path:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class _SupportMapCNNRanker(nn.Module):
        def __init__(self, map_channels: int, numeric_dim: int, hidden: int, dropout: float) -> None:
            super().__init__()
            self.map_encoder = nn.Sequential(
                nn.Conv2d(map_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.numeric_encoder = nn.Sequential(
                nn.Linear(numeric_dim, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Sequential(
                nn.Linear(128 + 64, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        def forward(self, maps: Any, numeric: Any) -> Any:
            map_features = self.map_encoder(maps)
            numeric_features = self.numeric_encoder(numeric)
            return self.head(torch.cat([map_features, numeric_features], dim=1)).squeeze(-1)

    try:
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(model_path), map_location="cpu")
    hidden = int(metrics.get("hidden", checkpoint.get("metrics", {}).get("hidden", 128)))
    dropout = float(metrics.get("dropout", checkpoint.get("metrics", {}).get("dropout", 0.0)))
    map_channels = len(schema.get("channels", [])) or int(metrics.get("map_shape", [8])[0])
    numeric_dim = len(schema.get("numeric_mean", []))
    if numeric_dim <= 0:
        return None
    model = _SupportMapCNNRanker(map_channels=map_channels, numeric_dim=numeric_dim, hidden=hidden, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    map_shape = checkpoint.get("metrics", {}).get("map_shape", metrics.get("map_shape", [map_channels, 64, 64]))
    runtime = {
        "torch": torch,
        "model": model,
        "device": device,
        "grid_size": int(map_shape[1]) if len(map_shape) >= 3 else 64,
        "window_m": float(schema.get("window_m", metrics.get("window_m", 0.9))),
    }
    ranker["_runtime"] = runtime
    return runtime


def _online_support_map_numeric(
    schema: dict[str, Any],
    candidate_context: dict[str, Any],
    rock_row: dict[str, Any],
    rock_index: int,
    slot: TargetSlot,
    candidate_count: int,
    candidate_id: int,
    candidate: dict[str, Any],
) -> np.ndarray | None:
    feature_row = _candidate_ranker_feature_row(
        candidate_context=candidate_context,
        rock_row=rock_row,
        rock_index=rock_index,
        slot=slot,
        candidate_count=candidate_count,
        candidate_id=candidate_id,
        candidate=candidate,
    )
    numeric_features = list(schema.get("numeric_features", []))
    values: list[float] = []
    present: list[float] = []
    for column in numeric_features:
        raw = feature_row.get(str(column), "")
        if _is_number(raw):
            values.append(float(raw))
            present.append(1.0)
        else:
            values.append(0.0)
            present.append(0.0)
    x = np.asarray(values + present, dtype=np.float32)
    mean = np.asarray(schema.get("numeric_mean", []), dtype=np.float32)
    std = np.asarray(schema.get("numeric_std", []), dtype=np.float32)
    if mean.shape[0] != x.shape[0] or std.shape[0] != x.shape[0]:
        return None
    return (x - mean) / np.where(std < 1e-6, 1.0, std)


def _online_support_map_tensor(
    schema: dict[str, Any],
    candidate_context: dict[str, Any],
    row_by_index: dict[int, dict[str, Any]],
    placed: list[int],
    slot_by_rock: dict[int, TargetSlot],
    rock_index: int,
    slot: TargetSlot,
    candidate: dict[str, Any],
    grid_size: int,
    window_m: float,
) -> np.ndarray | None:
    cell = window_m / max(grid_size, 1)
    axis = np.linspace(-0.5 * window_m + 0.5 * cell, 0.5 * window_m - 0.5 * cell, grid_size, dtype=np.float32)
    local_x, local_y = np.meshgrid(axis, axis)
    world_x = local_x + float(slot.x)
    world_y = local_y + float(slot.y)
    front_height_m = float(schema.get("front_height_m", 0.55))
    front_z_axis = np.linspace(
        0.5 * front_height_m / max(grid_size, 1),
        front_height_m - 0.5 * front_height_m / max(grid_size, 1),
        grid_size,
        dtype=np.float32,
    )
    front_x, front_z = np.meshgrid(axis + float(slot.x), front_z_axis)
    height = np.zeros_like(local_x, dtype=np.float32)
    support_count = np.zeros_like(local_x, dtype=np.float32)
    front_silhouette = np.zeros_like(local_x, dtype=np.float32)
    front_depth = np.zeros_like(local_x, dtype=np.float32)
    support_top = 0.0
    for support_idx in placed:
        support_row = row_by_index.get(support_idx, {})
        if "last_center_x" not in support_row or "last_center_y" not in support_row:
            continue
        sx = _float_or_default(support_row.get("last_center_x"), 0.0)
        sy = _float_or_default(support_row.get("last_center_y"), 0.0)
        bbox_x = _float_or_default(support_row.get("bbox_x"), 0.12)
        bbox_y = _float_or_default(support_row.get("bbox_y"), 0.10)
        bbox_z = _float_or_default(support_row.get("bbox_z"), 0.08)
        yaw = _quat_yaw(
            _float_or_default(support_row.get("last_quat_w"), 1.0),
            _float_or_default(support_row.get("last_quat_x"), 0.0),
            _float_or_default(support_row.get("last_quat_y"), 0.0),
            _float_or_default(support_row.get("last_quat_z"), 0.0),
        )
        mask = _ellipse_mask(world_x, world_y, sx, sy, bbox_x, bbox_y, yaw)
        top_z = _float_or_default(support_row.get("last_top_z"), bbox_z)
        support_top = max(support_top, top_z)
        support_count[mask] += 1.0
        height[mask] = np.maximum(height[mask], top_z)
        center_z = top_z - 0.5 * bbox_z
        front_mask = _front_rect_mask(front_x, front_z, sx, center_z, bbox_x, bbox_z)
        front_silhouette[front_mask] = 1.0
        front_depth[front_mask] = np.maximum(front_depth[front_mask], _front_depth_value(sy, bbox_y, float(slot.y), window_m))

    candidate_row = row_by_index[rock_index]
    pos = candidate["pos"]
    quat = candidate["quat"]
    bbox_x = _float_or_default(candidate_row.get("bbox_x"), 0.12)
    bbox_y = _float_or_default(candidate_row.get("bbox_y"), 0.10)
    bbox_z = _float_or_default(candidate_row.get("bbox_z"), 0.08)
    yaw = _quat_yaw(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    candidate_mask = _ellipse_mask(world_x, world_y, float(pos[0]), float(pos[1]), bbox_x, bbox_y, yaw)
    front_candidate = _front_rect_mask(front_x, front_z, float(pos[0]), float(pos[2]), bbox_x, bbox_z)
    front_candidate_depth = np.zeros_like(local_x, dtype=np.float32)
    front_candidate_depth[front_candidate] = _front_depth_value(float(pos[1]), bbox_y, float(slot.y), window_m)
    render_front_depth_proxy = np.maximum(front_depth, front_candidate_depth).astype(np.float32)
    render_front_valid_proxy = ((front_silhouette > 0.0) | front_candidate).astype(np.float32)
    candidate_top_z = float(pos[2]) + 0.5 * bbox_z
    top_height_with_candidate = np.maximum(height, candidate_mask.astype(np.float32) * candidate_top_z)
    render_top_depth_proxy = np.clip(0.42 + 0.24 * top_height_with_candidate / max(front_height_m, 1e-6), 0.0, 1.0).astype(np.float32)
    sigma = max(window_m / max(grid_size, 1) * 1.5, 0.035)
    target_gaussian = np.exp(-0.5 * ((world_x - float(slot.x)) ** 2 + (world_y - float(slot.y)) ** 2) / (sigma * sigma))
    target_z = support_top + 0.5 * bbox_z
    sigma_front_z = max(front_height_m / max(grid_size, 1) * 2.0, 0.025)
    front_target_gaussian = np.exp(
        -0.5
        * (
            ((front_x - float(slot.x)) / sigma) ** 2
            + ((front_z - target_z) / sigma_front_z) ** 2
        )
    )
    gravity = _float_or_default(candidate_context.get("gravity_m_s2"), GRAVITIES.get(str(candidate_context.get("gravity", "earth")), 9.80665))
    channel_map = {
        "height_before_m": height,
        "support_occupancy": (support_count > 0.0).astype(np.float32),
        "support_count_clipped": np.clip(support_count, 0.0, 4.0).astype(np.float32) / 4.0,
        "target_gaussian": target_gaussian.astype(np.float32),
        "candidate_footprint": candidate_mask.astype(np.float32),
        "candidate_height_m": candidate_mask.astype(np.float32) * bbox_z,
        "gravity_ratio": np.full_like(local_x, gravity / 9.80665, dtype=np.float32),
        "course_ratio": np.full_like(local_x, float(slot.course) / 6.0, dtype=np.float32),
        "top_height_before_m": height,
        "top_support_occupancy": (support_count > 0.0).astype(np.float32),
        "top_support_count_clipped": np.clip(support_count, 0.0, 4.0).astype(np.float32) / 4.0,
        "top_target_gaussian": target_gaussian.astype(np.float32),
        "top_candidate_footprint": candidate_mask.astype(np.float32),
        "top_candidate_height_m": candidate_mask.astype(np.float32) * bbox_z,
        "front_support_silhouette": front_silhouette,
        "front_support_depth_proxy": front_depth,
        "front_target_gaussian": front_target_gaussian.astype(np.float32),
        "front_candidate_silhouette": front_candidate.astype(np.float32),
        "front_candidate_depth_proxy": front_candidate_depth,
        "render_front_depth_norm": render_front_depth_proxy,
        "render_front_valid": render_front_valid_proxy,
        "render_top_depth_norm": render_top_depth_proxy,
        "render_top_valid": np.ones_like(local_x, dtype=np.float32),
    }
    channels = list(schema.get("channels", []))
    if not channels:
        channels = list(channel_map)
    tensor = np.stack([channel_map.get(name, np.zeros_like(local_x, dtype=np.float32)) for name in channels], axis=0)
    mean = np.asarray(schema.get("map_mean", []), dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(schema.get("map_std", []), dtype=np.float32).reshape(-1, 1, 1)
    if mean.shape[0] != tensor.shape[0] or std.shape[0] != tensor.shape[0]:
        return None
    return (tensor - mean) / np.where(std < 1e-6, 1.0, std)


def _ellipse_mask(
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


def _front_rect_mask(
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


def _front_depth_value(center_y: float, bbox_y: float, target_y: float, window_m: float) -> float:
    camera_y = target_y - 0.5 * window_m
    front_surface_y = center_y - 0.5 * bbox_y
    return float(np.clip((front_surface_y - camera_y) / max(window_m, 1e-6), 0.0, 1.0))


def _quat_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _float_or_default(value: Any, default: float) -> float:
    if _is_number(value):
        return float(value)
    return default


def _candidate_ranker_feature_row(
    candidate_context: dict[str, Any],
    rock_row: dict[str, Any],
    rock_index: int,
    slot: TargetSlot,
    candidate_count: int,
    candidate_id: int,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    pos = candidate["pos"]
    quat = candidate["quat"]
    output: dict[str, Any] = {
        "target_name": candidate_context.get("target_name", ""),
        "strategy": candidate_context.get("strategy", ""),
        "gravity": candidate_context.get("gravity", ""),
        "gravity_m_s2": candidate_context.get("gravity_m_s2", ""),
        "trial": candidate_context.get("trial", ""),
        "slot_id": slot.slot_id,
        "course": slot.course,
        "role": slot.role,
        "target_x": slot.x,
        "target_y": slot.y,
        "candidate_rock_index": rock_index,
        "source_kind": rock_row.get("source_kind", ""),
        "cluster_label": rock_row.get("cluster_label", ""),
        "candidate_id": candidate_id,
        "candidate_count": candidate_count,
        "pose_x": float(pos[0]),
        "pose_y": float(pos[1]),
        "pose_z": float(pos[2]),
        "pose_qw": float(quat[0]),
        "pose_qx": float(quat[1]),
        "pose_qy": float(quat[2]),
        "pose_qz": float(quat[3]),
    }
    for key, value in rock_row.items():
        output[f"rock_{key}"] = value
    return output


def _stone_fit_feature_row(
    candidate_context: dict[str, Any],
    rock_row: dict[str, Any],
    rock_index: int,
    slot: TargetSlot,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "target_name": candidate_context.get("target_name", ""),
        "strategy": candidate_context.get("strategy", ""),
        "gravity": candidate_context.get("gravity", ""),
        "gravity_m_s2": candidate_context.get("gravity_m_s2", ""),
        "trial": candidate_context.get("trial", ""),
        "slot_id": slot.slot_id,
        "course": slot.course,
        "role": slot.role,
        "target_x": slot.x,
        "target_y": slot.y,
        "candidate_rock_index": rock_index,
    }
    for key, value in rock_row.items():
        output[f"rock_{key}"] = value
    return output


def _is_number(value: Any) -> bool:
    if value in {"", None}:
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
