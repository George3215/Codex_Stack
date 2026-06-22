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

- rocks: 140
- rock_profile: high_wall
- clusters: 10
- trials: 2
- targets: single_face_wall_4course_v1
- strategies: statics_wall
- gravities: earth,moon
- candidates: 8
- steps_per_rock: 420
- hold_steps: 1800
- workers: 4
- assignment_plan: 
- assignment_gate: 0
- role_screening: 
- assignment_fallbacks: 0
- assignment_probe_steps: 0
- moon_gate_strict: 0
- candidate_pose_ranker_dir: batch_runs\20260620_torch_support_map_cnn_4course_curriculum_score_v1
- candidate_pose_top_k: 3
- stone_fit_ranker_dir: 
- stone_fit_top_k: 0
