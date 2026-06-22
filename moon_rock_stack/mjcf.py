from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


CLUSTER_COLORS = {
    "equant_clast": "0.68 0.66 0.58 1",
    "subangular_block": "0.56 0.54 0.49 1",
    "wedge_or_broad_clast": "0.62 0.58 0.48 1",
    "fractured_clast": "0.46 0.44 0.40 1",
    "angular_clast": "0.50 0.48 0.43 1",
    "elongated_clast": "0.54 0.49 0.42 1",
    "upright_block_clast": "0.62 0.64 0.54 1",
    "compact_block_clast": "0.67 0.63 0.53 1",
    "wall_block_clast": "0.60 0.57 0.49 1",
    "buttress_clast": "0.58 0.57 0.48 1",
    "keystone_clast": "0.64 0.58 0.46 1",
    "angular_boulder_clast": "0.45 0.44 0.39 1",
    "notched_block_clast": "0.50 0.47 0.41 1",
    "bearing_block_clast": "0.61 0.59 0.50 1",
    "course_block_clast": "0.66 0.62 0.52 1",
    "tie_bridge_clast": "0.57 0.55 0.48 1",
    "chock_clast": "0.53 0.50 0.43 1",
    "interlock_block_clast": "0.49 0.47 0.42 1",
    "cap_block_clast": "0.69 0.64 0.54 1",
    "spiky_reject": "0.78 0.20 0.16 1",
}


CONTACT_FRICTION = (1.15, 0.025, 0.002)
CONTACT_SOLREF = (0.006, 1.0)
CONTACT_SOLIMP = (0.92, 0.98, 0.001)


def write_world_xml(
    xml_path: Path,
    rows: list[dict[str, Any]],
    gravity: float,
    trial_id: int,
    mesh_dir_relative: str = "../meshes",
) -> None:
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f'<mujoco model="moon_rock_stack_trial_{trial_id:02d}">',
        f'  <compiler angle="radian" meshdir="{escape(mesh_dir_relative)}" inertiafromgeom="true" discardvisual="false"/>',
        f'  <option timestep="0.002" gravity="0 0 {-abs(gravity):.8f}" cone="elliptic" iterations="80" tolerance="1e-10"/>',
        '  <size njmax="2000" nconmax="800"/>',
        '  <default>',
        (
            '    <geom condim="4" '
            f'friction="{CONTACT_FRICTION[0]:.3f} {CONTACT_FRICTION[1]:.3f} {CONTACT_FRICTION[2]:.3f}" '
            f'solref="{CONTACT_SOLREF[0]:.3f} {CONTACT_SOLREF[1]:.3f}" '
            f'solimp="{CONTACT_SOLIMP[0]:.3f} {CONTACT_SOLIMP[1]:.3f} {CONTACT_SOLIMP[2]:.3f}"/>'
        ),
        '  </default>',
        '  <asset>',
        '    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.18 0.17 0.15" rgb2="0.26 0.25 0.22"/>',
        '    <material name="regolith" texture="grid" texrepeat="8 8" reflectance="0.08"/>',
    ]
    for row in rows:
        idx = int(row["index"])
        lines.append(f'    <mesh name="rock_{idx:03d}_mesh" file="rock_{idx:03d}.obj"/>')
    lines.extend(
        [
            "  </asset>",
            "  <worldbody>",
            '    <light name="sun" pos="-1.5 -1.5 3.0" dir="0.4 0.4 -1" diffuse="0.9 0.86 0.78"/>',
            '    <geom name="ground" type="plane" size="5 5 0.05" material="regolith" rgba="0.31 0.30 0.27 1"/>',
        ]
    )
    for row in rows:
        idx = int(row["index"])
        cluster = str(row["cluster_label"])
        color = color_for_cluster(cluster)
        mass = max(float(row["mass"]), 0.002)
        x = 2.0 + 0.18 * (idx % 8)
        y = -1.2 + 0.18 * (idx // 8)
        z = 0.40 + 0.03 * idx
        lines.extend(
            [
                f'    <body name="rock_{idx:03d}" pos="{x:.4f} {y:.4f} {z:.4f}" quat="1 0 0 0">',
                f'      <freejoint name="rock_{idx:03d}_free"/>',
                f'      <geom name="rock_{idx:03d}_geom" type="mesh" mesh="rock_{idx:03d}_mesh" mass="{mass:.8f}" rgba="{color}"/>',
                "    </body>",
            ]
        )
    lines.extend(["  </worldbody>", "</mujoco>"])
    xml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def color_for_cluster(cluster_label: str) -> str:
    for prefix, color in CLUSTER_COLORS.items():
        if cluster_label.startswith(prefix):
            return color
    return "0.58 0.56 0.50 1"
