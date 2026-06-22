# Data Management Policy

This project is intentionally experiment-heavy. Raw MuJoCo states, meshes, RGB/depth captures, and batch CSV files can grow quickly.

## Local Preservation

Do not delete raw experiment outputs during normal iteration. Failed runs are useful scientific data because they identify bad geometry classes, poor placement strategies, missing support, excessive drift, and unrealistic parameter settings.

Primary local output directories:

- `batch_runs/`
- `generated*/`
- `meshes/`
- `mjcf/`
- `states/`
- `captures*/`

These are ignored by Git by default. Ignoring them does not delete them; it only prevents accidental GitHub pushes of large files.

## What To Commit

Commit source code, scripts, environment files, documentation, and small curated reports.

Recommended tracked files:

- `moon_rock_stack/*.py`
- `scripts/*.py`
- `environment.yml`
- `pyproject.toml`
- `README.md`
- `docs/*.md`

For a scientifically important run, commit a small curated summary rather than the whole run directory:

- `README.md` for the run;
- compact CSV summaries;
- selected PNG figures only when needed;
- enough command lines and seeds to reproduce the run.

If a normally ignored artifact must be committed, use an explicit force-add and document why:

```powershell
git add -f batch_runs/<run_name>/README.md
git add -f batch_runs/<run_name>/summary.json
```

## Push Strategy

Before pushing to GitHub:

1. Check `git status --ignored`.
2. Confirm large raw outputs remain ignored.
3. Commit source and curated records.
4. Use Git LFS only for intentionally shared binary assets.

Do not push local copies of papers, third-party repositories, or bulk simulation output unless there is a clear licensing and storage decision.
