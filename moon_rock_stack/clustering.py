from __future__ import annotations

from typing import Any

import numpy as np

from .features import FEATURE_COLUMNS


def cluster_features(
    rows: list[dict[str, Any]],
    clusters: int,
    seed: int,
    iterations: int = 80,
) -> tuple[np.ndarray, dict[int, str]]:
    if clusters < 1:
        raise ValueError("clusters must be >= 1")
    if clusters > len(rows):
        raise ValueError("clusters cannot exceed rock count")

    x = np.asarray([[float(row[col]) for col in FEATURE_COLUMNS] for row in rows], dtype=float)
    x[:, 0] = np.log10(np.maximum(x[:, 0], 1e-12))
    x[:, 1] = np.log10(np.maximum(x[:, 1], 1e-12))
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-9)
    z = (x - mean) / std

    rng = np.random.default_rng(seed)
    centers = z[rng.choice(len(z), size=clusters, replace=False)].copy()
    labels = np.zeros(len(z), dtype=np.int32)

    for _ in range(iterations):
        distances = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(clusters):
            members = z[labels == cluster_id]
            if len(members) == 0:
                centers[cluster_id] = z[rng.integers(0, len(z))]
            else:
                centers[cluster_id] = members.mean(axis=0)

    names = name_clusters(rows, labels, clusters)
    return labels, names


def name_clusters(rows: list[dict[str, Any]], labels: np.ndarray, clusters: int) -> dict[int, str]:
    names: dict[int, str] = {}
    used: dict[str, int] = {}
    for cluster_id in range(clusters):
        members = [row for row, label in zip(rows, labels) if int(label) == cluster_id]
        if not members:
            base = "empty"
        else:
            mean = {key: float(np.mean([float(row[key]) for row in members])) for key in FEATURE_COLUMNS}
            if mean["spike_score"] >= 0.18:
                base = "spiky_reject"
            elif mean.get("support_face_area_ratio", 0.0) >= 0.24 and mean.get("opposing_face_pair_count", 0.0) >= 1.0:
                base = "bearing_block_clast"
            elif mean.get("major_face_count", 0.0) >= 7.0 and mean.get("face_area_entropy", 0.0) >= 0.72:
                base = "multi_facet_clast"
            elif mean["bbox_z"] >= 1.12 * max(mean["bbox_x"], mean["bbox_y"]) and mean["flatness"] < 1.28:
                base = "upright_block_clast"
            elif mean["compactness"] >= 0.34 and mean["elongation"] < 1.20 and mean["flatness"] < 1.22:
                base = "compact_block_clast"
            elif mean["elongation"] >= 1.32:
                base = "elongated_clast"
            elif mean["flatness"] >= 1.32 and mean["compactness"] >= 0.30:
                base = "wedge_or_broad_clast"
            elif mean["compactness"] < 0.29:
                base = "fractured_clast"
            elif mean["angularity"] >= 0.195 or mean["roughness"] >= 0.275:
                base = "angular_clast"
            elif mean["sphericity"] >= 0.82 and mean["roughness"] < 0.10:
                base = "equant_clast"
            else:
                base = "subangular_block"

        used[base] = used.get(base, 0) + 1
        names[cluster_id] = base if used[base] == 1 else f"{base}_{used[base]}"
    return names
