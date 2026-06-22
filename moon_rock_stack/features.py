from __future__ import annotations

import math
from typing import Any

import numpy as np

from .fractal_rocks import RockMesh


FEATURE_COLUMNS = (
    "volume",
    "surface_area",
    "bbox_x",
    "bbox_y",
    "bbox_z",
    "elongation",
    "flatness",
    "sphericity",
    "roughness",
    "angularity",
    "spike_score",
    "compactness",
    "stability_score",
    "major_face_count",
    "largest_face_area_ratio",
    "top3_face_area_ratio",
    "face_area_entropy",
    "normal_concentration",
    "support_face_count",
    "support_face_area_ratio",
    "opposing_face_pair_count",
    "opposing_face_area_ratio",
    "face_planarity",
    "support_plane_quality",
)


def extract_features(rock: RockMesh) -> dict[str, Any]:
    vertices = rock.vertices
    faces = rock.faces
    bbox = vertices.max(axis=0) - vertices.min(axis=0)
    sorted_bbox = np.sort(bbox)[::-1]
    longest, middle, shortest = np.maximum(sorted_bbox, 1e-9)
    surface_area = mesh_surface_area(vertices, faces)
    volume = mesh_volume(vertices, faces)
    equivalent_area = math.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0)
    sphericity = equivalent_area / max(surface_area, 1e-9)
    centroid = vertices.mean(axis=0)
    radii = np.linalg.norm(vertices - centroid, axis=1)
    roughness = float(radii.std() / max(radii.mean(), 1e-9))
    angularity = mesh_angularity(vertices, faces)
    spike_score = mesh_spike_score(vertices, faces)
    compactness = float(volume / max(float(np.prod(bbox)), 1e-9))
    elongation = float(longest / middle)
    flatness = float(middle / shortest)
    stability_score = float((longest * middle) / max(shortest * shortest, 1e-9) * max(sphericity, 0.1))
    face_priors = mesh_face_priors(vertices, faces)

    output = {
        "index": rock.index,
        "source_kind": rock.kind,
        "seed": rock.seed,
        "volume": float(volume),
        "surface_area": float(surface_area),
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_z": float(bbox[2]),
        "elongation": elongation,
        "flatness": flatness,
        "sphericity": float(sphericity),
        "roughness": roughness,
        "angularity": angularity,
        "spike_score": spike_score,
        "compactness": compactness,
        "stability_score": stability_score,
        "mass": float(volume * 2600.0),
    }
    output.update(face_priors)
    return output


def mesh_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    tris = vertices[faces]
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    tris = vertices[faces]
    signed = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])) / 6.0
    return float(abs(signed.sum()))


def mesh_angularity(vertices: np.ndarray, faces: np.ndarray) -> float:
    tris = vertices[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(faces):
        for a, b in ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))):
            key = (a, b) if a < b else (b, a)
            edge_to_faces.setdefault(key, []).append(face_id)

    angles: list[float] = []
    for adjacent in edge_to_faces.values():
        if len(adjacent) == 2:
            dot = float(np.clip(np.dot(normals[adjacent[0]], normals[adjacent[1]]), -1.0, 1.0))
            angles.append(math.acos(dot) / math.pi)
    return float(np.mean(angles)) if angles else 0.0


def mesh_spike_score(vertices: np.ndarray, faces: np.ndarray) -> float:
    centroid = vertices.mean(axis=0)
    radii = np.linalg.norm(vertices - centroid, axis=1)
    adjacency = [set() for _ in range(len(vertices))]
    for face in faces:
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))

    excesses: list[float] = []
    for idx, neighbors in enumerate(adjacency):
        if not neighbors:
            continue
        local_ref = float(np.median(radii[list(neighbors)]))
        excesses.append(max(0.0, radii[idx] / max(local_ref, 1e-9) - 1.0))
    return float(max(excesses) if excesses else 0.0)


def mesh_face_priors(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    tris = vertices[faces]
    raw_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    double_areas = np.linalg.norm(raw_normals, axis=1)
    valid = double_areas > 1e-12
    if not np.any(valid):
        return empty_face_priors()

    areas = 0.5 * double_areas[valid]
    normals = raw_normals[valid] / double_areas[valid, None]
    total_area = float(areas.sum())
    if total_area <= 1e-12:
        return empty_face_priors()

    clusters = cluster_face_normals(normals, areas)
    if not clusters:
        return empty_face_priors()

    clusters.sort(key=lambda item: float(item["area"]), reverse=True)
    ratios = np.asarray([float(item["area"]) / total_area for item in clusters], dtype=np.float64)
    major_mask = ratios >= 0.050
    support_mask = ratios >= 0.080
    entropy = float(-(ratios * np.log(np.maximum(ratios, 1e-12))).sum() / max(math.log(len(ratios)), 1e-9))
    concentration = float(np.square(ratios).sum())
    face_planarity = float(
        sum(float(item["area"]) * float(item["alignment"]) for item in clusters) / max(total_area, 1e-12)
    )

    opposing_count = 0
    opposing_area_ratio = 0.0
    major_indices = [index for index, is_major in enumerate(major_mask) if bool(is_major)]
    for left_pos, left_index in enumerate(major_indices):
        left_normal = np.asarray(clusters[left_index]["normal"], dtype=np.float64)
        for right_index in major_indices[left_pos + 1 :]:
            right_normal = np.asarray(clusters[right_index]["normal"], dtype=np.float64)
            if float(np.dot(left_normal, right_normal)) <= -math.cos(math.radians(20.0)):
                opposing_count += 1
                opposing_area_ratio = max(opposing_area_ratio, min(float(ratios[left_index]), float(ratios[right_index])))

    largest = float(ratios[0])
    support_area = float(ratios[support_mask].sum()) if np.any(support_mask) else largest
    support_count = int(support_mask.sum()) if np.any(support_mask) else 1
    support_plane_quality = float(support_area * (1.0 + min(opposing_count, 3) / 3.0) * face_planarity)
    return {
        "major_face_count": float(int(major_mask.sum())),
        "largest_face_area_ratio": largest,
        "top3_face_area_ratio": float(ratios[:3].sum()),
        "face_area_entropy": entropy,
        "normal_concentration": concentration,
        "support_face_count": float(support_count),
        "support_face_area_ratio": support_area,
        "opposing_face_pair_count": float(opposing_count),
        "opposing_face_area_ratio": float(opposing_area_ratio),
        "face_planarity": face_planarity,
        "support_plane_quality": support_plane_quality,
    }


def cluster_face_normals(normals: np.ndarray, areas: np.ndarray, angle_degrees: float = 18.0) -> list[dict[str, Any]]:
    threshold = math.cos(math.radians(angle_degrees))
    clusters: list[dict[str, Any]] = []
    for triangle_index in np.argsort(-areas):
        normal = normals[int(triangle_index)]
        area = float(areas[int(triangle_index)])
        best_index = -1
        best_dot = threshold
        for cluster_index, cluster in enumerate(clusters):
            dot = float(np.dot(normal, np.asarray(cluster["normal"], dtype=np.float64)))
            if dot > best_dot:
                best_dot = dot
                best_index = cluster_index
        if best_index < 0:
            clusters.append({"normal": normal.copy(), "area": area, "members": [(normal.copy(), area)]})
            continue

        cluster = clusters[best_index]
        weighted = np.asarray(cluster["normal"], dtype=np.float64) * float(cluster["area"]) + normal * area
        norm = float(np.linalg.norm(weighted))
        if norm > 1e-12:
            cluster["normal"] = weighted / norm
        cluster["area"] = float(cluster["area"]) + area
        cluster["members"].append((normal.copy(), area))

    for cluster in clusters:
        cluster_normal = np.asarray(cluster["normal"], dtype=np.float64)
        member_weight = sum(float(area) for _, area in cluster["members"])
        if member_weight <= 1e-12:
            cluster["alignment"] = 0.0
            continue
        alignment = sum(float(area) * max(0.0, float(np.dot(normal, cluster_normal))) for normal, area in cluster["members"])
        cluster["alignment"] = float(alignment / member_weight)
    return clusters


def empty_face_priors() -> dict[str, float]:
    return {
        "major_face_count": 0.0,
        "largest_face_area_ratio": 0.0,
        "top3_face_area_ratio": 0.0,
        "face_area_entropy": 0.0,
        "normal_concentration": 0.0,
        "support_face_count": 0.0,
        "support_face_area_ratio": 0.0,
        "opposing_face_pair_count": 0.0,
        "opposing_face_area_ratio": 0.0,
        "face_planarity": 0.0,
        "support_plane_quality": 0.0,
    }
