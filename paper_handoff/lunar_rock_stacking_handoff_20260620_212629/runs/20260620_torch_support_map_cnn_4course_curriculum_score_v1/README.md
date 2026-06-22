# PyTorch Support-Map CNN Ranker

This model ranks candidate stone poses within each slot/candidate-rock group.

Architecture:

- CNN encoder over local support maps.
- MLP encoder over candidate pose, stone geometry, support metrics, and missing-value mask.
- Score head trained with groupwise softmax loss; target mode: score.

- tensor dir: `D:\MoonStack\experiments\moon_rock_stack\batch_runs\20260620_local_support_maps_4course_curriculum_v1`
- device: cuda
- torch: 2.11.0+cu128
- GPU: NVIDIA GeForce RTX 2080 Ti
- rows: 12484
- rankable groups: 3992
- test top-1 hit: 0.436
- test top-3 hit: 0.989
- train top-1 hit: 0.437
- train top-3 hit: 0.980
- test mean top-1 quality regret: 9.931
- test mean top-3 quality regret: 0.033

In `selected` mode, labels imitate the current heuristic search. In `score` mode, labels use post-simulation candidate quality as supervision while keeping post-simulation fields out of the input when `--exclude-postsim-features` is set.
