# Dry-Stacking Structure Plan

This project should not optimize for a loose pile. The experimental target is a usable dry-stacked structure: first a lunar landmark cairn/wall segment, then larger dry-stacked wall modules.

## Literature-Derived Method Points

The local papers in `D:/MoonStack/Asset/Papers` support these implementation choices:

- Furrer et al. 2017: found stones require online next-best object/pose planning. A candidate pose is not accepted only because it is geometrically close; it must pass a physics/stability check, including support contacts and low residual kinetic energy.
- Johns et al. 2020: autonomous dry-stone construction is target-geometry driven. The robot builds a wall, monitors the as-built state, refines stone poses, and replans after each placement.
- Liu et al. 2018: dry stacking should optimize target filling and whole-structure stability. Their reward combines target-shape fit with stability under disturbance, not just local height gain.
- Liu et al. 2021: dry stacking with irregular stones needs geometry-aware placement search and physics validation. Candidate positions/orientations are sampled, settled, and evaluated.
- Liu and Napp 2023: sequence planning matters because locally good placements can weaken the whole structure. Stone ordering should be stability/risk aware.
- Menezes et al. 2021: target volume and height/elevation maps are useful representations for ISRU-style stacking where perfect CAD models are not available.

## Immediate Target

Build a lunar landmark short-wall/cairn segment, not a mound:

- 3 courses.
- Base course has multiple stones along a line.
- Upper courses are offset to form dry-stone-style bonding.
- The structure should have visible length and courses, while remaining small enough for fast MuJoCo iteration.

The first implemented target is `landmark_wall_v1`:

- Course 0: 4 target slots.
- Course 1: 3 target slots, shifted by half a slot.
- Course 2: 2 target slots, shifted again and centered.
- Total: 9 stones.

## Stone Selection Policy

The generator now emits geology-prior clasts without point spikes. Selection uses measured features:

- Base layer: prefer `wedge_or_broad_clast`, `subangular_block`, and compact equant stones; penalize high elongation and `spike_score`.
- Middle layer: prefer subangular/equant stones with good compactness and moderate volume.
- Cap layer: allow smaller elongated or fractured clasts only if they have low `spike_score` and candidate pose validation succeeds.
- Reject or heavily penalize `spiky_reject`.

## Candidate Pose Search

For each target slot:

1. Pick the next stone using role-aware geometry scoring.
2. Generate multiple candidate poses around the slot center.
3. Use MuJoCo settling for each candidate.
4. Score candidates with:
   - target XY error,
   - support overlap with previous course,
   - low residual velocity,
   - low drift from the target slot,
   - useful but bounded height gain.
5. Commit the best candidate, then re-evaluate the as-built state before placing the next stone.

## Evaluation Metrics

The result is not judged by height alone. It records:

- `stable_count`
- `shape_success`
- `target_rmse_xy_m`
- `target_max_xy_error_m`
- `course_count`
- `placed_count`
- `stack_height_m`
- `max_horizontal_drift_m`
- `velocity_inf_norm`
- `structure_score`

Success requires:

- all or nearly all target stones remain stable,
- target XY error stays under a tolerance,
- residual velocity is low,
- drift is controlled,
- the intended number of courses remains visible.

## Next Targets

After `landmark_wall_v1`:

1. `landmark_arrow_v1`: an arrow/waypoint marker using a line plus triangular head.
2. `landmark_ring_v1`: a small ring or boundary marker.
3. `wall_segment_v2`: thicker two-row wall with interlocking courses.
4. `wall_corner_v1`: L-shaped wall corner.
5. `habitat_wall_module_v1`: longer dry-stacked structural module.
