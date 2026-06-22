from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an OBJ rock contact sheet.")
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    paths = sorted(args.mesh_dir.glob("rock_*.obj"))[: args.limit]
    if not paths:
        raise SystemExit(f"No rock_*.obj files found under {args.mesh_dir}")

    cols = min(4, len(paths))
    rows = int(np.ceil(len(paths) / cols))
    fig = plt.figure(figsize=(3.0 * cols, 3.0 * rows), facecolor="white")
    for i, path in enumerate(paths):
        vertices, faces = read_obj(path)
        tris = vertices[faces]
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        mesh = Poly3DCollection(
            tris,
            facecolor=(0.62, 0.60, 0.52, 1.0),
            edgecolor=(0.08, 0.08, 0.08, 0.50),
            linewidths=0.35,
        )
        ax.add_collection3d(mesh)
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        center = 0.5 * (mins + maxs)
        span = max(float((maxs - mins).max()), 1e-9) * 0.62
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.view_init(24, -42)
        ax.set_axis_off()
        ax.set_title(path.stem, fontsize=9)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(args.output.resolve())
    return 0


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(value.split("/")[0]) - 1 for value in line.split()[1:4]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int32)


if __name__ == "__main__":
    raise SystemExit(main())
