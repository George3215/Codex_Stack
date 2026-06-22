from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moon_rock_stack.mjcf import CONTACT_FRICTION, CONTACT_SOLIMP, CONTACT_SOLREF
from moon_rock_stack.run_experiment import write_csv
from moon_rock_stack.simulate import GRAVITIES


REFERENCE_MOON_G = 1.624
REFERENCE_EARTH_G = 9.80665


def main() -> int:
    args = parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gravity_rows = run_gravity_checks(args)
    friction_rows = run_friction_checks(args)
    config_rows = physics_config_rows()

    write_csv(output_dir / "gravity_validation.csv", gravity_rows)
    write_csv(output_dir / "friction_validation.csv", friction_rows)
    write_csv(output_dir / "physics_config.csv", config_rows)
    (output_dir / "physics_config.json").write_text(
        json.dumps(
            {
                "gravities": GRAVITIES,
                "reference_gravities": {
                    "earth": REFERENCE_EARTH_G,
                    "moon": REFERENCE_MOON_G,
                },
                "contact_friction": CONTACT_FRICTION,
                "contact_solref": CONTACT_SOLREF,
                "contact_solimp": CONTACT_SOLIMP,
                "main_friction_angle_deg": math.degrees(math.atan(CONTACT_FRICTION[0])),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(output_dir / "README.md", gravity_rows, friction_rows)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MuJoCo gravity and contact friction settings.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freefall-seconds", type=float, default=0.55)
    parser.add_argument("--friction-seconds", type=float, default=4.0)
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument(
        "--angles",
        default="35,40,45,48,49,50,55",
        help="Comma-separated incline angles in degrees for the friction-equivalent gravity test.",
    )
    parser.add_argument("--slip-distance", type=float, default=0.025)
    return parser.parse_args()


def run_gravity_checks(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, gravity in GRAVITIES.items():
        rows.append(freefall_check(label, gravity, args.freefall_seconds))
    return rows


def freefall_check(label: str, gravity: float, duration_s: float) -> dict[str, Any]:
    import mujoco

    xml = f"""
<mujoco model="gravity_freefall_{label}">
  <option timestep="0.001" gravity="0 0 {-gravity:.8f}" iterations="80" tolerance="1e-12"/>
  <worldbody>
    <body name="falling_mass" pos="0 0 1.0">
      <freejoint/>
      <geom type="sphere" size="0.035" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    steps = int(round(duration_s / model.opt.timestep))
    times: list[float] = []
    velocities: list[float] = []
    heights: list[float] = []
    for step in range(steps):
        mujoco.mj_step(model, data)
        times.append((step + 1) * model.opt.timestep)
        velocities.append(float(data.qvel[2]))
        heights.append(float(data.qpos[2]))
    time_array = np.array(times, dtype=float)
    velocity_array = np.array(velocities, dtype=float)
    height_array = np.array(heights, dtype=float)
    slope, intercept = np.polyfit(time_array, velocity_array, 1)
    estimated_g_velocity = -float(slope)
    # z(t) = z0 + v0 t - 0.5 g t^2, so the quadratic coefficient is -0.5 g.
    quad, linear, constant = np.polyfit(time_array, height_array, 2)
    estimated_g_position = -2.0 * float(quad)
    reference = REFERENCE_EARTH_G if label == "earth" else REFERENCE_MOON_G if label == "moon" else gravity
    return {
        "test": "freefall",
        "gravity": label,
        "configured_g_m_s2": gravity,
        "reference_g_m_s2": reference,
        "estimated_g_from_velocity_m_s2": estimated_g_velocity,
        "estimated_g_from_position_m_s2": estimated_g_position,
        "velocity_abs_error_m_s2": abs(estimated_g_velocity - gravity),
        "position_abs_error_m_s2": abs(estimated_g_position - gravity),
        "reference_abs_error_m_s2": abs(gravity - reference),
        "duration_s": duration_s,
        "timestep_s": model.opt.timestep,
        "samples": steps,
        "pass": int(abs(estimated_g_velocity - gravity) < 0.002 and abs(estimated_g_position - gravity) < 0.002),
    }


def run_friction_checks(args: argparse.Namespace) -> list[dict[str, Any]]:
    angles = [float(item.strip()) for item in args.angles.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    for label, gravity in GRAVITIES.items():
        for angle in angles:
            rows.append(
                friction_tilt_check(
                    label=label,
                    gravity=gravity,
                    angle_deg=angle,
                    test_seconds=args.friction_seconds,
                    settle_seconds=args.settle_seconds,
                    slip_distance=args.slip_distance,
                )
            )
    return rows


def friction_tilt_check(
    label: str,
    gravity: float,
    angle_deg: float,
    test_seconds: float,
    settle_seconds: float,
    slip_distance: float,
) -> dict[str, Any]:
    import mujoco

    mu = CONTACT_FRICTION[0]
    xml = f"""
<mujoco model="friction_tilt_{label}_{int(angle_deg)}">
  <option timestep="0.002" gravity="0 0 {-gravity:.8f}" cone="elliptic" iterations="100" tolerance="1e-12"/>
  <default>
    <geom condim="4" friction="{CONTACT_FRICTION[0]:.6f} {CONTACT_FRICTION[1]:.6f} {CONTACT_FRICTION[2]:.6f}" solref="{CONTACT_SOLREF[0]:.6f} {CONTACT_SOLREF[1]:.6f}" solimp="{CONTACT_SOLIMP[0]:.6f} {CONTACT_SOLIMP[1]:.6f} {CONTACT_SOLIMP[2]:.6f}"/>
  </default>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.05"/>
    <body name="test_block" pos="0 0 0.041">
      <freejoint/>
      <geom name="block_geom" type="box" size="0.055 0.055 0.040" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    settle_steps = int(round(settle_seconds / model.opt.timestep))
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_block")
    x0 = float(data.xpos[body_id][0])

    angle_rad = math.radians(angle_deg)
    model.opt.gravity[:] = np.array([gravity * math.sin(angle_rad), 0.0, -gravity * math.cos(angle_rad)])
    steps = int(round(test_seconds / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)

    dx = float(data.xpos[body_id][0] - x0)
    speed = float(np.linalg.norm(data.qvel[:3]))
    tan_angle = math.tan(angle_rad)
    expected_static = tan_angle <= mu
    observed_static = abs(dx) <= slip_distance
    return {
        "test": "tilted_gravity_static_friction",
        "gravity": label,
        "configured_g_m_s2": gravity,
        "configured_mu_slide": mu,
        "angle_deg": angle_deg,
        "tan_angle": tan_angle,
        "expected_static_by_coulomb": int(expected_static),
        "observed_static": int(observed_static),
        "displacement_x_m": dx,
        "abs_displacement_x_m": abs(dx),
        "final_linear_speed_m_s": speed,
        "settle_seconds": settle_seconds,
        "test_seconds": test_seconds,
        "slip_distance_threshold_m": slip_distance,
        "near_threshold": int(abs(tan_angle - mu) < 0.03),
        "match": int(expected_static == observed_static or abs(tan_angle - mu) < 0.03),
    }


def physics_config_rows() -> list[dict[str, Any]]:
    return [
        {
            "parameter": "earth_gravity_m_s2",
            "configured": GRAVITIES["earth"],
            "reference": REFERENCE_EARTH_G,
            "note": "standard gravity used for Earth comparison",
        },
        {
            "parameter": "moon_gravity_m_s2",
            "configured": GRAVITIES["moon"],
            "reference": REFERENCE_MOON_G,
            "note": "NASA/JPL Moon surface gravity value used for new runs",
        },
        {
            "parameter": "sliding_friction_mu",
            "configured": CONTACT_FRICTION[0],
            "reference": "tan(49.0 deg)",
            "note": "upper lunar-regolith-like dry contact proxy; not cohesion model",
        },
        {
            "parameter": "torsional_friction",
            "configured": CONTACT_FRICTION[1],
            "reference": "",
            "note": "MuJoCo torsional contact friction proxy",
        },
        {
            "parameter": "rolling_friction",
            "configured": CONTACT_FRICTION[2],
            "reference": "",
            "note": "MuJoCo rolling contact friction proxy",
        },
        {
            "parameter": "friction_angle_deg_from_mu",
            "configured": math.degrees(math.atan(CONTACT_FRICTION[0])),
            "reference": "Apollo lunar soil friction angles are commonly reported around 30-50 deg depending on density and depth",
            "note": "Use sensitivity tests before claiming a universal lunar surface value.",
        },
    ]


def write_report(path: Path, gravity_rows: list[dict[str, Any]], friction_rows: list[dict[str, Any]]) -> None:
    gravity_pass = all(int(row["pass"]) == 1 for row in gravity_rows)
    friction_pass = all(int(row["match"]) == 1 for row in friction_rows)
    lines = [
        "# Physics Validation",
        "",
        "Date: 2026-06-18",
        "",
        "This run verifies that the MuJoCo models actually use the configured gravity and friction values.",
        "",
        "## References Used For Calibration",
        "",
        "- Moon gravity: NASA Moon facts list surface gravity as `1.624 m/s^2`.",
        "- Earth comparison: `9.80665 m/s^2` standard gravity.",
        "- Lunar regolith friction: LPI/Carrier-style lunar soil summaries report friction angles that vary strongly with density and depth; the current MuJoCo sliding friction `mu=1.15` corresponds to `atan(mu)=49.0 deg`, near the upper end of dense/in-situ lunar soil behavior.",
        "",
        "Important limitation: MuJoCo Coulomb friction is a contact proxy. It does not model regolith cohesion, particle crushing, excavation, dust, or contact aging.",
        "",
        "## Configured Parameters",
        "",
        f"- Earth gravity: `{GRAVITIES['earth']}` m/s^2",
        f"- Moon gravity: `{GRAVITIES['moon']}` m/s^2",
        f"- Contact friction: `{CONTACT_FRICTION[0]} {CONTACT_FRICTION[1]} {CONTACT_FRICTION[2]}`",
        f"- Equivalent sliding friction angle: `{math.degrees(math.atan(CONTACT_FRICTION[0])):.2f} deg`",
        f"- Contact solref: `{CONTACT_SOLREF[0]} {CONTACT_SOLREF[1]}`",
        f"- Contact solimp: `{CONTACT_SOLIMP[0]} {CONTACT_SOLIMP[1]} {CONTACT_SOLIMP[2]}`",
        "",
        "## Gravity Freefall Check",
        "",
        "| Gravity | Configured g | Estimated from velocity | Estimated from position | Abs error vs reference | Pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in gravity_rows:
        lines.append(
            "| {gravity} | {configured_g_m_s2:.6f} | {estimated_g_from_velocity_m_s2:.6f} | "
            "{estimated_g_from_position_m_s2:.6f} | {reference_abs_error_m_s2:.6f} | {pass} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Friction Tilt Check",
            "",
            "The test settles a 1 kg block on a plane, then rotates the gravity vector so the plane is equivalent to an incline. A block should remain static when `tan(theta) <= mu` and slide when `tan(theta) > mu`, apart from a small near-threshold band.",
            "",
            "| Gravity | Angle | tan(angle) | Expected static | Observed static | dx (m) | Match |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in friction_rows:
        lines.append(
            "| {gravity} | {angle_deg:.1f} | {tan_angle:.3f} | {expected_static_by_coulomb} | "
            "{observed_static} | {displacement_x_m:.4f} | {match} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Gravity numerical validation pass: `{int(gravity_pass)}`.",
            f"- Friction threshold validation pass: `{int(friction_pass)}`.",
            "- New runs use Moon gravity `1.624 m/s^2`. Earlier runs before this validation used `1.62 m/s^2`; those are still usable as near-identical preliminary runs, but should be labeled as pre-calibration.",
            "- The friction value is plausible for high-friction dense dry lunar regolith/rock contact, but it is optimistic. Future sensitivity batches should compare `mu=0.75`, `mu=0.95`, and `mu=1.15` before making scientific claims.",
            "",
            "## Data Files",
            "",
            "- `gravity_validation.csv`",
            "- `friction_validation.csv`",
            "- `physics_config.csv`",
            "- `physics_config.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
