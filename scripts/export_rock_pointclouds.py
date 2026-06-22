from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    args = parse_args()
    output_dir = unique_dir(args.output.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)
    run_dirs = discover_run_dirs(args.batch_root.resolve(), args.run)
    if not run_dirs:
        raise SystemExit("No run directories with features.csv and meshes/*.obj found.")

    rows_out: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    total_rocks = 0
    for run_dir in run_dirs:
        mesh_dir = run_dir / "meshes"
        feature_rows = read_csv(run_dir / "features.csv")
        if not mesh_dir.exists() or not feature_rows:
            continue
        run_output = output_dir / run_dir.name
        run_output.mkdir(parents=True, exist_ok=True)

        clouds: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        indices: list[int] = []
        seeds: list[int] = []
        kinds: list[str] = []
        cluster_ids: list[int] = []
        cluster_labels: list[str] = []
        feature_by_index = {parse_int(row.get("index", "")): row for row in feature_rows if parse_int(row.get("index", "")) is not None}

        for rock_index in sorted(feature_by_index):
            obj_path = mesh_dir / f"rock_{rock_index:03d}.obj"
            if not obj_path.exists():
                continue
            vertices, faces = read_obj(obj_path)
            if len(vertices) == 0 or len(faces) == 0:
                continue
            rng = np.random.default_rng(args.seed + stable_hash(run_dir.name) + rock_index * 7919)
            points, point_normals = sample_mesh(vertices, faces, args.points, rng)
            clouds.append(points.astype(np.float32))
            normals.append(point_normals.astype(np.float32))
            indices.append(rock_index)
            row = feature_by_index[rock_index]
            seeds.append(parse_int(row.get("seed", "")) or 0)
            kinds.append(row.get("source_kind", ""))
            cluster_ids.append(parse_int(row.get("cluster_id", "")) or -1)
            cluster_labels.append(row.get("cluster_label", ""))
            rows_out.append(
                {
                    "run_name": run_dir.name,
                    "run_path": str(run_dir),
                    "tensor_path": str(run_output / "rock_pointclouds.npz"),
                    "rock_index": rock_index,
                    "point_count": args.points,
                    "source_kind": row.get("source_kind", ""),
                    "cluster_id": row.get("cluster_id", ""),
                    "cluster_label": row.get("cluster_label", ""),
                    "seed": row.get("seed", ""),
                }
            )

        if clouds:
            np.savez_compressed(
                run_output / "rock_pointclouds.npz",
                points=np.stack(clouds, axis=0),
                normals=np.stack(normals, axis=0),
                rock_index=np.asarray(indices, dtype=np.int32),
                seed=np.asarray(seeds, dtype=np.int64),
                source_kind=np.asarray(kinds),
                cluster_id=np.asarray(cluster_ids, dtype=np.int32),
                cluster_label=np.asarray(cluster_labels),
            )
            total_rocks += len(clouds)
            summary_rows.append(
                {
                    "run_name": run_dir.name,
                    "rock_count": len(clouds),
                    "point_count": args.points,
                    "tensor_path": str(run_output / "rock_pointclouds.npz"),
                }
            )

    write_csv(output_dir / "rock_pointcloud_index.csv", rows_out)
    write_csv(output_dir / "run_summary.csv", summary_rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "batch_root": str(args.batch_root.resolve()),
        "output_dir": str(output_dir),
        "run_count": len(summary_rows),
        "rock_count": total_rocks,
        "points_per_rock": args.points,
        "format": "Per-run NPZ with arrays: points[N,P,3], normals[N,P,3], rock_index[N].",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_readme(output_dir, summary)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-rock point-cloud tensors from OBJ meshes.")
    parser.add_argument("--batch-root", type=Path, default=Path("batch_runs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="append", type=Path, default=[], help="Optional run directory; repeatable.")
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--seed", type=int, default=620)
    return parser.parse_args()


def discover_run_dirs(batch_root: Path, requested: list[Path]) -> list[Path]:
    if requested:
        dirs = [(path if path.is_absolute() else batch_root / path).resolve() for path in requested]
    else:
        dirs = [path.parent for path in batch_root.glob("**/features.csv")]
    return sorted(path for path in dirs if (path / "features.csv").exists() and (path / "meshes").exists())


def unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.name}_{stamp}")
    suffix = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{stamp}_{suffix:02d}")
        suffix += 1
    return candidate


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                raw = line.split()[1:]
                ids = [int(token.split("/")[0]) - 1 for token in raw]
                if len(ids) == 3:
                    faces.append(ids)
                elif len(ids) > 3:
                    for i in range(1, len(ids) - 1):
                        faces.append([ids[0], ids[i], ids[i + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def sample_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    tris = vertices[faces]
    vec_a = tris[:, 1] - tris[:, 0]
    vec_b = tris[:, 2] - tris[:, 0]
    cross = np.cross(vec_a, vec_b)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = areas > 1e-12
    tris = tris[valid]
    cross = cross[valid]
    areas = areas[valid]
    if len(tris) == 0:
        points = np.repeat(vertices.mean(axis=0, keepdims=True), count, axis=0)
        normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (count, 1))
        return normalize_points(points), normals
    probs = areas / areas.sum()
    chosen = rng.choice(len(tris), size=count, replace=True, p=probs)
    tri = tris[chosen]
    u = rng.random(count)
    v = rng.random(count)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    points = tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])
    normals = cross[chosen]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    return normalize_points(points), normals


def normalize_points(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale < 1e-9:
        return centered
    return centered / scale


def parse_int(value: str | None) -> int | None:
    if value in {"", None}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def stable_hash(text: str) -> int:
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % 2_147_483_647
    return value


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Rock Point-Cloud Tensor Export",
        "",
        "Each run has a `rock_pointclouds.npz` file with fixed-size point samples from generated OBJ meshes.",
        "",
        f"- run count: {summary['run_count']}",
        f"- rock count: {summary['rock_count']}",
        f"- points per rock: {summary['points_per_rock']}",
        "",
        "Arrays:",
        "",
        "- `points`: normalized sampled xyz, shape `[N, P, 3]`.",
        "- `normals`: sampled face normals, shape `[N, P, 3]`.",
        "- `rock_index`: rock id matching `features.csv`, `placement_examples.csv`, and candidate logs.",
        "",
        "This is the input format for a future PointNet/PointNet++ rock encoder.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
