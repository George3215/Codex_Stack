# MoonStack Learning Dataset V0

Batch root: `D:\MoonStack\experiments\moon_rock_stack\batch_runs`
Run directories scanned: 7

## Files

- `run_examples.csv`: one row per structured simulation result.
- `placement_examples.csv`: one row per committed placement, skipped slot, or best-rejected slot summary.
- `placement_examples.jsonl`: JSONL mirror of placement examples.
- `candidate_pose_examples.csv`: one row per sampled candidate pose when `candidate_pose_log.csv` exists.
- `candidate_pose_examples.jsonl`: JSONL mirror of candidate-pose examples.
- `assignment_candidate_examples.csv`: one row per assignment/fallback candidate when available.
- `dataset_summary.json`: counts, groups, and limitations.

## Labels

- `label_success=1`: the placement was committed and is not present in `failure_cases.csv`.
- `label_success=0`: the placement was skipped or recorded as a failure.
- `label_run_success`: run-level strict success from `results.csv`.
- `label_run_shape_success`: run-level shape success from `results.csv`.

## Limitation

This is mixed V0/V1 data. It is useful for tabular compatibility, failure prediction, and imitation of current search results. Dense GraspNet-style pose labels are available only for new runs that contain `candidate_pose_log.csv`.

The next data-collection step is to generate larger candidate-pose batches with higher `--candidates` values under both Earth and Moon gravity.
