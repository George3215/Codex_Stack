# Lunar Rock Stacking Paper Handoff

Generated on DESKTOP-ML5VHFK for DESKTOP-DRR635P.

Mission: use current lunar rock stacking simulation results to mine literature, design ICRA-style paper story, and decide missing experiments.

Machine roles:
- Windows stone stacking: DESKTOP-ML5VHFK, produces simulation/training/closed-loop validation artifacts.
- Windows paper search: DESKTOP-DRR635P, mines ICRA/arXiv literature and converts artifacts into paper framing, ablations, and writing plan.

Current technical state:
- Task: lunar analogue irregular rock / dry-stone stacking, currently focused on single_face_wall_4course_v1.
- Main evidence so far: heuristic/statics wall baseline, neural/CNN candidate pose rankers, 4-course curriculum ranker, closed-loop simulation validation.
- Recent finding: 4-course-specific CNN has very high offline top-3 candidate ranker metrics, but closed-loop validation remains hard; success is not solved by candidate ranking alone.
- Most common failures: middle/cap missed target, post-hold drift, unstable structure.

Paper-search objectives:
1. Find how ICRA/RAL/T-RO/RSS papers frame irregular object stacking, dry-stone stacking, construction robotics, contact-rich manipulation, failure recovery, and sim-to-real.
2. Extract which contribution framing is strongest: closed-loop failure-aware repair, structure stability metric, curriculum ranker, or sim-to-real gap analysis.
3. Build a paper skeleton: title, problem, method modules, baselines, ablation table, metrics, figure list, related work buckets.
4. Identify missing experiments that are feasible before September submission.

Do not treat offline ranker top-3 as final success. The key paper metric should be closed-loop structure success, shape success, stability after hold, RMSE, drift, failure mode histogram, and repair success.
