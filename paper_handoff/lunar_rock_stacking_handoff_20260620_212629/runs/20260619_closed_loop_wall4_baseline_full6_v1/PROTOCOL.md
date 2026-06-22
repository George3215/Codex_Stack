# Structured Dry-Stacking Protocol

Purpose: compare structured dry-stacking strategies for a short wall and a pillar under Earth and Moon gravity.

This protocol is derived from local dry-stacking papers in `D:/MoonStack/Asset/Papers`: Furrer et al. 2017, Johns et al. 2020, Liu et al. 2018, Liu et al. 2021, Liu and Napp 2023, and Menezes et al. 2021.

Key experimental commitments:

- Build target structures, not loose piles.
- Use geology-prior clast geometry with spike rejection metrics.
- Use same generated rock library for Earth and Moon gravity.
- Compare geometry-bonded ordering, support-first placement, risk-aware ordering, centered-compact placement, and random-order control.
- Evaluate target-shape error, visible courses, residual velocity, drift, and failures.

Configuration:

- rocks: 480
- rock_profile: wall_statics
- clusters: 12
- trials: 1
- targets: single_face_wall_4course_v1
- strategies: wall_bonded
- gravities: earth,moon
- candidates: 6
- steps_per_rock: 520
- hold_steps: 1600
- workers: 2
- assignment_plan: batch_runs\20260618_targeted_rock_catalog_wall_statics_v1\assignment_plan_single_face_wall_4course_v1.csv
- assignment_gate: 1
- role_screening: batch_runs\20260618_targeted_rock_catalog_wall_statics_v1\role_screening.csv
- assignment_fallbacks: 5
- assignment_probe_steps: 0
- moon_gate_strict: 1
- candidate_pose_ranker_dir: 
- candidate_pose_top_k: 0
- stone_fit_ranker_dir: 
- stone_fit_top_k: 0
