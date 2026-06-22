from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


@dataclass
class PhaseResult:
    name: str
    command: list[str]
    returncode: int
    stdout_path: Path
    stderr_path: Path
    parsed_output: Path | None
    started_at: str
    finished_at: str


def main() -> int:
    args = parse_args()
    session = args.session or datetime.now().strftime("%Y%m%d_pointnet_supportmap_%H%M%S")
    output_root = args.output_root.resolve()
    session_dir = output_root / session
    log_dir = session_dir / "logs"
    session_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_paths = resolve_run_paths(args.dataset.resolve(), args.run)
    if not run_paths:
        raise SystemExit("No run directories were resolved for point-cloud export.")

    manifest: dict[str, Any] = {
        "created_at": now(),
        "session": session,
        "session_dir": str(session_dir),
        "dataset": str(args.dataset.resolve()),
        "tensor_dir": str(args.tensor_dir.resolve()),
        "run_paths": [str(path) for path in run_paths],
        "purpose": (
            "Ablate whether a PointNet rock-shape embedding improves the current "
            "support-map candidate-pose ranker on the same 3/4-course curriculum dataset."
        ),
        "phases": [],
    }
    manifest_path = session_dir / "manifest.json"
    write_json(manifest_path, manifest)

    pointcloud_base = output_root / f"{session}_pointclouds"
    pointnet_base = output_root / f"{session}_pointnet_encoder"
    ranker_base = output_root / f"{session}_support_map_pointnet_{args.target_mode}"

    pointcloud_cmd = [
        str(PYTHON),
        "-m",
        "scripts.export_rock_pointclouds",
        "--output",
        str(pointcloud_base),
        "--points",
        str(args.points),
        "--seed",
        str(args.seed),
    ]
    for run_path in run_paths:
        pointcloud_cmd.extend(["--run", str(run_path)])

    pointcloud_result = run_phase("export_pointclouds", pointcloud_cmd, log_dir)
    record_phase(manifest, manifest_path, pointcloud_result)
    if pointcloud_result.returncode != 0 or pointcloud_result.parsed_output is None:
        return pointcloud_result.returncode or 1

    pointnet_cmd = [
        str(PYTHON),
        "-m",
        "scripts.train_torch_pointnet_rock_encoder",
        "--pointcloud-dir",
        str(pointcloud_result.parsed_output),
        "--output",
        str(pointnet_base),
        "--epochs",
        str(args.pointnet_epochs),
        "--batch-size",
        str(args.pointnet_batch_size),
        "--embedding-dim",
        str(args.embedding_dim),
        "--dropout",
        str(args.pointnet_dropout),
        "--lr",
        str(args.pointnet_lr),
        "--weight-decay",
        str(args.pointnet_weight_decay),
        "--test-fraction",
        str(args.test_fraction),
        "--seed",
        str(args.seed + 1),
        "--device",
        args.device,
        "--amp",
        "--augment",
        "--use-normals",
        "--cluster-family",
    ]
    pointnet_result = run_phase("train_pointnet", pointnet_cmd, log_dir)
    record_phase(manifest, manifest_path, pointnet_result)
    if pointnet_result.returncode != 0 or pointnet_result.parsed_output is None:
        return pointnet_result.returncode or 1

    embedding_path = pointnet_result.parsed_output / "rock_pointnet_embeddings.npz"
    if not embedding_path.exists():
        raise SystemExit(f"Expected embedding file was not created: {embedding_path}")

    ranker_cmd = [
        str(PYTHON),
        "-m",
        "scripts.train_torch_support_map_ranker",
        "--tensor-dir",
        str(args.tensor_dir.resolve()),
        "--output",
        str(ranker_base),
        "--epochs",
        str(args.ranker_epochs),
        "--batch-size",
        str(args.ranker_batch_size),
        "--hidden",
        str(args.ranker_hidden),
        "--dropout",
        str(args.ranker_dropout),
        "--lr",
        str(args.ranker_lr),
        "--weight-decay",
        str(args.ranker_weight_decay),
        "--test-fraction",
        str(args.test_fraction),
        "--split-by-run",
        "--seed",
        str(args.seed + 2),
        "--device",
        args.device,
        "--amp",
        "--target-mode",
        args.target_mode,
        "--exclude-postsim-features",
        "--rock-embedding-npz",
        str(embedding_path),
    ]
    ranker_result = run_phase("train_support_map_pointnet_ranker", ranker_cmd, log_dir)
    record_phase(manifest, manifest_path, ranker_result)
    if ranker_result.returncode != 0:
        return ranker_result.returncode or 1

    write_readme(session_dir, manifest)
    print(session_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PointNet + support-map CNN ablation without deleting existing data.")
    parser.add_argument("--session", default="")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("batch_runs"))
    parser.add_argument("--run", action="append", type=Path, default=[], help="Optional run directory; repeatable.")
    parser.add_argument("--points", type=int, default=768)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--pointnet-epochs", type=int, default=70)
    parser.add_argument("--pointnet-batch-size", type=int, default=128)
    parser.add_argument("--pointnet-dropout", type=float, default=0.22)
    parser.add_argument("--pointnet-lr", type=float, default=8e-4)
    parser.add_argument("--pointnet-weight-decay", type=float, default=2e-4)
    parser.add_argument("--ranker-epochs", type=int, default=80)
    parser.add_argument("--ranker-batch-size", type=int, default=128)
    parser.add_argument("--ranker-hidden", type=int, default=192)
    parser.add_argument("--ranker-dropout", type=float, default=0.18)
    parser.add_argument("--ranker-lr", type=float, default=8e-4)
    parser.add_argument("--ranker-weight-decay", type=float, default=4e-4)
    parser.add_argument("--target-mode", choices=("selected", "score", "risk_adjusted", "structure_aware"), default="structure_aware")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=206212010)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_run_paths(dataset: Path, requested: list[Path]) -> list[Path]:
    if requested:
        return sorted({(path if path.is_absolute() else REPO / path).resolve() for path in requested})
    run_examples = dataset / "run_examples.csv"
    if not run_examples.exists():
        raise SystemExit(f"Missing dataset run_examples.csv: {run_examples}")
    paths: set[Path] = set()
    with run_examples.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("run_path", "")
            if raw:
                paths.add(Path(raw).resolve())
    return sorted(path for path in paths if (path / "features.csv").exists() and (path / "meshes").exists())


def run_phase(name: str, command: list[str], log_dir: Path) -> PhaseResult:
    started_at = now()
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    print(f"[{started_at}] START {name}")
    print(" ".join(command))
    result = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True, check=False)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    finished_at = now()
    parsed_output = parse_output_path(result.stdout)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    print(f"[{finished_at}] END {name} returncode={result.returncode} parsed_output={parsed_output or ''}")
    return PhaseResult(
        name=name,
        command=command,
        returncode=int(result.returncode),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        parsed_output=parsed_output,
        started_at=started_at,
        finished_at=finished_at,
    )


def parse_output_path(stdout: str) -> Path | None:
    for line in reversed([item.strip() for item in stdout.splitlines() if item.strip()]):
        candidate = Path(line)
        if candidate.exists():
            return candidate.resolve()
    return None


def record_phase(manifest: dict[str, Any], manifest_path: Path, result: PhaseResult) -> None:
    manifest["phases"].append(
        {
            "name": result.name,
            "command": result.command,
            "returncode": result.returncode,
            "stdout": str(result.stdout_path),
            "stderr": str(result.stderr_path),
            "parsed_output": str(result.parsed_output) if result.parsed_output else "",
            "started_at": result.started_at,
            "finished_at": result.finished_at,
        }
    )
    write_json(manifest_path, manifest)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_readme(session_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# PointNet Support-Map Ablation",
        "",
        "Purpose: test whether a learned rock point-cloud embedding improves the current support-map pose ranker on the same curriculum split.",
        "",
        "This is an ablation, not the default closed-loop policy. The current closed-loop policy should only switch to this ranker if its holdout top-k metrics and MuJoCo strict-wall results improve.",
        "",
        f"- dataset: `{manifest['dataset']}`",
        f"- tensor dir: `{manifest['tensor_dir']}`",
        f"- run count: {len(manifest['run_paths'])}",
        "",
        "Phases:",
    ]
    for phase in manifest.get("phases", []):
        lines.append(f"- {phase['name']}: returncode={phase['returncode']}, output=`{phase.get('parsed_output', '')}`")
    (session_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
