from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class RockMesh:
    index: int
    kind: str
    vertices: np.ndarray
    faces: np.ndarray
    seed: int


ROCK_KINDS = (
    "equant_clast",
    "subangular_block",
    "wedge_clast",
    "fractured_clast",
    "elongated_clast",
    "upright_block_clast",
    "compact_block_clast",
    "wall_block_clast",
    "buttress_clast",
    "keystone_clast",
    "angular_boulder_clast",
    "notched_block_clast",
    "bearing_block_clast",
    "course_block_clast",
    "tie_bridge_clast",
    "chock_clast",
    "interlock_block_clast",
    "cap_block_clast",
)


ROCK_PROFILES: dict[str, tuple[str, ...]] = {
    "balanced": ROCK_KINDS,
    "wall_statics": (
        "bearing_block_clast",
        "buttress_clast",
        "subangular_block",
        "course_block_clast",
        "compact_block_clast",
        "wall_block_clast",
        "tie_bridge_clast",
        "equant_clast",
        "course_block_clast",
        "chock_clast",
        "interlock_block_clast",
        "cap_block_clast",
    ),
    "single_face_wall": (
        "course_block_clast",
        "compact_block_clast",
        "wall_block_clast",
        "equant_clast",
        "subangular_block",
        "course_block_clast",
        "compact_block_clast",
        "cap_block_clast",
        "chock_clast",
        "wall_block_clast",
        "course_block_clast",
        "tie_bridge_clast",
    ),
    "high_wall": (
        "bearing_block_clast",
        "buttress_clast",
        "bearing_block_clast",
        "course_block_clast",
        "compact_block_clast",
        "wall_block_clast",
        "tie_bridge_clast",
        "course_block_clast",
        "interlock_block_clast",
        "cap_block_clast",
        "equant_clast",
        "chock_clast",
    ),
    "screening_stress": ROCK_KINDS
    + (
        "bearing_block_clast",
        "course_block_clast",
        "tie_bridge_clast",
        "chock_clast",
        "interlock_block_clast",
        "cap_block_clast",
    ),
}


def generate_rocks(
    count: int,
    seed: int = 7,
    lat_steps: int = 14,
    lon_steps: int = 28,
    profile: str = "balanced",
) -> list[RockMesh]:
    rng = np.random.default_rng(seed)
    kinds = ROCK_PROFILES.get(profile)
    if kinds is None:
        raise ValueError(f"Unknown rock profile: {profile}")
    rocks: list[RockMesh] = []
    for index in range(count):
        kind = kinds[index % len(kinds)]
        rock_seed = int(rng.integers(0, 2**31 - 1))
        vertices, faces = generate_fractal_rock(kind, rock_seed, lat_steps, lon_steps)
        rocks.append(RockMesh(index=index, kind=kind, vertices=vertices, faces=faces, seed=rock_seed))
    return rocks


def generate_fractal_rock(
    kind: str,
    seed: int,
    lat_steps: int = 14,
    lon_steps: int = 28,
) -> tuple[np.ndarray, np.ndarray]:
    _ = lat_steps, lon_steps
    rng = np.random.default_rng(seed)
    params = _kind_params(kind, rng)
    base_radius = rng.uniform(0.055, 0.095)
    effective_radius = base_radius * float(params.get("radius_scale", 1.0))

    vertices, faces = _subdivided_icosahedron(subdivisions=0)
    directions = vertices / np.maximum(np.linalg.norm(vertices, axis=1, keepdims=True), 1e-9)
    radial = _broad_lobed_radius(
        directions=directions,
        faces=faces,
        rng=rng,
        roughness=params["roughness"],
        lobe_count=int(params["lobe_count"]),
    )
    vertices = directions * radial[:, None] * effective_radius * params["scale"]
    vertices = _apply_fracture_planes(
        vertices=vertices,
        rng=rng,
        base_radius=effective_radius,
        count=int(params["chip_count"]),
        depth_range=params["chip_depth"],
    )

    if kind == "wedge_clast":
        vertices = _apply_wedge(vertices, rng)
    elif kind == "fractured_clast":
        vertices = _apply_shear(vertices, rng, amount=0.16)
    elif kind == "elongated_clast":
        vertices = _apply_shear(vertices, rng, amount=0.10)
    elif kind == "buttress_clast":
        vertices = _apply_buttress(vertices)
    elif kind == "keystone_clast":
        vertices = _apply_wedge(vertices, rng)
        vertices = _apply_shear(vertices, rng, amount=0.06)
    elif kind == "notched_block_clast":
        vertices = _apply_shear(vertices, rng, amount=0.12)
    elif kind == "bearing_block_clast":
        vertices = _apply_buttress(vertices)
    elif kind == "chock_clast":
        vertices = _apply_wedge(vertices, rng)
    elif kind == "interlock_block_clast":
        vertices = _apply_shear(vertices, rng, amount=0.08)
    elif kind == "cap_block_clast":
        vertices = _apply_shear(vertices, rng, amount=0.04)

    vertices = _limit_local_protrusions(
        vertices=vertices,
        faces=faces,
        max_excess=float(params["max_local_excess"]),
        iterations=5,
    )
    vertices -= vertices.mean(axis=0)
    vertices = _enforce_non_slab_thickness(vertices, min_short_to_mid=0.62)
    vertices = _limit_local_protrusions(vertices, faces, max_excess=0.16, iterations=3)
    vertices -= vertices.mean(axis=0)
    return vertices, faces


def _kind_params(kind: str, rng: np.random.Generator) -> dict[str, object]:
    if kind == "equant_clast":
        scale = _scale_from_ratios(rng.uniform(1.00, 1.16), rng.uniform(1.00, 1.16), rng)
        return {
            "scale": scale,
            "roughness": rng.uniform(0.075, 0.120),
            "lobe_count": 7,
            "chip_count": 3,
            "chip_depth": (0.080, 0.150),
            "max_local_excess": 0.15,
        }
    if kind == "subangular_block":
        scale = _scale_from_ratios(rng.uniform(1.06, 1.30), rng.uniform(1.00, 1.22), rng)
        return {
            "scale": scale,
            "roughness": rng.uniform(0.100, 0.165),
            "lobe_count": 8,
            "chip_count": 4,
            "chip_depth": (0.090, 0.180),
            "max_local_excess": 0.16,
        }
    if kind == "wedge_clast":
        scale = _scale_from_ratios(rng.uniform(1.10, 1.42), rng.uniform(1.05, 1.30), rng)
        return {
            "scale": scale,
            "roughness": rng.uniform(0.090, 0.150),
            "lobe_count": 8,
            "chip_count": 4,
            "chip_depth": (0.085, 0.170),
            "max_local_excess": 0.16,
        }
    if kind == "fractured_clast":
        scale = _scale_from_ratios(rng.uniform(1.08, 1.42), rng.uniform(1.05, 1.32), rng)
        return {
            "scale": scale,
            "roughness": rng.uniform(0.120, 0.205),
            "lobe_count": 9,
            "chip_count": 6,
            "chip_depth": (0.100, 0.230),
            "max_local_excess": 0.17,
        }
    if kind == "elongated_clast":
        scale = _scale_from_ratios(rng.uniform(1.34, 1.74), rng.uniform(1.00, 1.22), rng)
        return {
            "scale": scale,
            "roughness": rng.uniform(0.085, 0.150),
            "lobe_count": 8,
            "chip_count": 4,
            "chip_depth": (0.080, 0.165),
            "max_local_excess": 0.16,
        }
    if kind == "upright_block_clast":
        scale = _scale_xyz(rng.uniform(0.94, 1.08), rng.uniform(0.94, 1.10), rng.uniform(1.30, 1.68))
        return {
            "scale": scale,
            "radius_scale": rng.uniform(1.02, 1.16),
            "roughness": rng.uniform(0.075, 0.130),
            "lobe_count": 7,
            "chip_count": 4,
            "chip_depth": (0.075, 0.155),
            "max_local_excess": 0.14,
        }
    if kind == "compact_block_clast":
        scale = _scale_xyz(rng.uniform(0.96, 1.12), rng.uniform(0.96, 1.12), rng.uniform(0.96, 1.16))
        return {
            "scale": scale,
            "radius_scale": rng.uniform(0.90, 1.08),
            "roughness": rng.uniform(0.070, 0.125),
            "lobe_count": 7,
            "chip_count": 3,
            "chip_depth": (0.070, 0.145),
            "max_local_excess": 0.14,
        }
    if kind == "wall_block_clast":
        scale = _scale_xyz(rng.uniform(1.18, 1.42), rng.uniform(0.82, 0.98), rng.uniform(0.86, 1.04))
        return {
            "scale": scale,
            "radius_scale": rng.uniform(0.92, 1.08),
            "roughness": rng.uniform(0.065, 0.115),
            "lobe_count": 6,
            "chip_count": 4,
            "chip_depth": (0.070, 0.150),
            "max_local_excess": 0.13,
        }
    if kind == "buttress_clast":
        scale = _scale_xyz(rng.uniform(1.16, 1.36), rng.uniform(1.08, 1.28), rng.uniform(0.86, 1.03))
        return {
            "scale": scale,
            "radius_scale": rng.uniform(1.04, 1.18),
            "roughness": rng.uniform(0.085, 0.145),
            "lobe_count": 8,
            "chip_count": 4,
            "chip_depth": (0.080, 0.165),
            "max_local_excess": 0.15,
        }
    if kind == "keystone_clast":
        scale = _scale_from_ratios(rng.uniform(1.12, 1.42), rng.uniform(1.02, 1.22), rng)
        return {
            "scale": scale,
            "radius_scale": rng.uniform(0.94, 1.10),
            "roughness": rng.uniform(0.095, 0.160),
            "lobe_count": 8,
            "chip_count": 5,
            "chip_depth": (0.085, 0.180),
            "max_local_excess": 0.15,
        }
    if kind == "angular_boulder_clast":
        scale = _scale_from_ratios(rng.uniform(1.00, 1.24), rng.uniform(1.00, 1.18), rng)
        return {
            "scale": scale,
            "radius_scale": rng.uniform(1.08, 1.24),
            "roughness": rng.uniform(0.120, 0.190),
            "lobe_count": 10,
            "chip_count": 6,
            "chip_depth": (0.095, 0.210),
            "max_local_excess": 0.16,
        }
    if kind == "notched_block_clast":
        scale = _scale_from_ratios(rng.uniform(1.10, 1.34), rng.uniform(1.00, 1.20), rng)
        return {
            "scale": scale,
            "radius_scale": rng.uniform(0.92, 1.08),
            "roughness": rng.uniform(0.110, 0.185),
            "lobe_count": 9,
            "chip_count": 7,
            "chip_depth": (0.090, 0.205),
            "max_local_excess": 0.16,
        }
    if kind == "bearing_block_clast":
        scale = _scale_xyz(rng.uniform(1.18, 1.38), rng.uniform(1.04, 1.26), rng.uniform(0.88, 1.04))
        return {
            "scale": scale,
            "radius_scale": rng.uniform(1.06, 1.22),
            "roughness": rng.uniform(0.075, 0.130),
            "lobe_count": 7,
            "chip_count": 4,
            "chip_depth": (0.075, 0.155),
            "max_local_excess": 0.14,
        }
    if kind == "course_block_clast":
        scale = _scale_xyz(rng.uniform(1.08, 1.28), rng.uniform(0.94, 1.10), rng.uniform(0.94, 1.12))
        return {
            "scale": scale,
            "radius_scale": rng.uniform(0.94, 1.10),
            "roughness": rng.uniform(0.070, 0.125),
            "lobe_count": 7,
            "chip_count": 4,
            "chip_depth": (0.070, 0.150),
            "max_local_excess": 0.13,
        }
    if kind == "tie_bridge_clast":
        scale = _scale_xyz(rng.uniform(1.36, 1.62), rng.uniform(0.84, 0.99), rng.uniform(0.84, 1.00))
        return {
            "scale": scale,
            "radius_scale": rng.uniform(0.96, 1.10),
            "roughness": rng.uniform(0.075, 0.130),
            "lobe_count": 7,
            "chip_count": 4,
            "chip_depth": (0.075, 0.155),
            "max_local_excess": 0.13,
        }
    if kind == "chock_clast":
        scale = _scale_from_ratios(rng.uniform(1.02, 1.22), rng.uniform(1.00, 1.16), rng)
        return {
            "scale": scale,
            "radius_scale": rng.uniform(0.74, 0.92),
            "roughness": rng.uniform(0.095, 0.155),
            "lobe_count": 8,
            "chip_count": 5,
            "chip_depth": (0.080, 0.170),
            "max_local_excess": 0.14,
        }
    if kind == "interlock_block_clast":
        scale = _scale_from_ratios(rng.uniform(1.12, 1.34), rng.uniform(1.02, 1.22), rng)
        return {
            "scale": scale,
            "radius_scale": rng.uniform(0.92, 1.08),
            "roughness": rng.uniform(0.105, 0.170),
            "lobe_count": 9,
            "chip_count": 6,
            "chip_depth": (0.085, 0.185),
            "max_local_excess": 0.15,
        }
    if kind == "cap_block_clast":
        scale = _scale_xyz(rng.uniform(1.14, 1.34), rng.uniform(0.92, 1.08), rng.uniform(0.90, 1.06))
        return {
            "scale": scale,
            "radius_scale": rng.uniform(0.90, 1.04),
            "roughness": rng.uniform(0.070, 0.120),
            "lobe_count": 6,
            "chip_count": 4,
            "chip_depth": (0.070, 0.145),
            "max_local_excess": 0.13,
        }
    raise ValueError(f"Unknown rock kind: {kind}")


def _scale_from_ratios(a_over_b: float, b_over_c: float, rng: np.random.Generator) -> np.ndarray:
    c = 1.0
    b = b_over_c * c
    a = a_over_b * b
    scale = np.array([a, b, c], dtype=float)
    scale /= np.cbrt(float(np.prod(scale)))
    if rng.random() < 0.35:
        scale = scale[rng.permutation(3)]
    return scale


def _scale_xyz(x: float, y: float, z: float) -> np.ndarray:
    scale = np.array([x, y, z], dtype=float)
    scale /= np.cbrt(float(np.prod(scale)))
    return scale


def _subdivided_icosahedron(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=float,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    for _ in range(subdivisions):
        vertices_list = [vertex.copy() for vertex in vertices]
        midpoint_cache: dict[tuple[int, int], int] = {}
        new_faces: list[list[int]] = []

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            cached = midpoint_cache.get(key)
            if cached is not None:
                return cached
            point = vertices[a] + vertices[b]
            point /= max(float(np.linalg.norm(point)), 1e-9)
            vertices_list.append(point)
            idx = len(vertices_list) - 1
            midpoint_cache[key] = idx
            return idx

        for a, b, c in faces:
            ab = midpoint(int(a), int(b))
            bc = midpoint(int(b), int(c))
            ca = midpoint(int(c), int(a))
            new_faces.extend([[int(a), ab, ca], [int(b), bc, ab], [int(c), ca, bc], [ab, bc, ca]])
        vertices = np.asarray(vertices_list, dtype=float)
        faces = np.asarray(new_faces, dtype=np.int32)

    return vertices, faces


def _broad_lobed_radius(
    directions: np.ndarray,
    faces: np.ndarray,
    rng: np.random.Generator,
    roughness: float,
    lobe_count: int,
) -> np.ndarray:
    anchors = rng.normal(size=(lobe_count, 3))
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    amplitudes = rng.normal(0.0, roughness, size=lobe_count)
    radial = np.ones(len(directions), dtype=float)
    sigma = rng.uniform(0.22, 0.34)
    for anchor, amplitude in zip(anchors, amplitudes):
        weights = np.exp(((directions @ anchor) - 1.0) / sigma)
        radial += amplitude * weights
    radial = _smooth_vertex_values(radial, faces, iterations=1, strength=0.28)
    radial /= max(float(np.mean(radial)), 1e-9)
    return np.clip(radial, 0.72, 1.34)


def _smooth_vertex_values(values: np.ndarray, faces: np.ndarray, iterations: int, strength: float) -> np.ndarray:
    adjacency = _vertex_adjacency(len(values), faces)
    values = values.copy()
    for _ in range(iterations):
        updated = values.copy()
        for idx, neighbors in enumerate(adjacency):
            if neighbors:
                updated[idx] = (1.0 - strength) * values[idx] + strength * float(np.mean(values[neighbors]))
        values = updated
    return values


def _apply_fracture_planes(
    vertices: np.ndarray,
    rng: np.random.Generator,
    base_radius: float,
    count: int,
    depth_range: tuple[float, float],
) -> np.ndarray:
    vertices = vertices.copy()
    for _ in range(count):
        normal = rng.normal(size=3)
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        projection = vertices @ normal
        cutoff = np.quantile(projection, rng.uniform(0.70, 0.84))
        span = max(float(projection.max() - cutoff), 1e-9)
        mask = projection > cutoff
        t = ((projection[mask] - cutoff) / span)[:, None]
        depth = base_radius * rng.uniform(*depth_range)
        vertices[mask] -= normal * depth * np.clip(t, 0.0, 1.0)
    return vertices


def _apply_wedge(vertices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    vertices = vertices.copy()
    axis = int(rng.integers(0, 2))
    coord = vertices[:, axis]
    normalized = (coord - coord.min()) / max(float(coord.max() - coord.min()), 1e-9)
    vertices[:, 2] *= 0.84 + 0.24 * normalized
    return vertices


def _apply_shear(vertices: np.ndarray, rng: np.random.Generator, amount: float) -> np.ndarray:
    vertices = vertices.copy()
    vertices[:, 0] += amount * vertices[:, 2] * rng.uniform(-1.0, 1.0)
    vertices[:, 1] += 0.55 * amount * vertices[:, 0] * rng.uniform(-1.0, 1.0)
    return vertices


def _apply_buttress(vertices: np.ndarray) -> np.ndarray:
    vertices = vertices.copy()
    z = vertices[:, 2]
    normalized = (z - z.min()) / max(float(z.max() - z.min()), 1e-9)
    widen = 1.0 + 0.14 * (1.0 - normalized)
    vertices[:, 0] *= widen
    vertices[:, 1] *= widen
    return vertices


def _limit_local_protrusions(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_excess: float,
    iterations: int,
) -> np.ndarray:
    vertices = vertices.copy()
    adjacency = _vertex_adjacency(len(vertices), faces)
    for _ in range(iterations):
        center = vertices.mean(axis=0)
        vectors = vertices - center
        radii = np.linalg.norm(vectors, axis=1)
        for idx, neighbors in enumerate(adjacency):
            if not neighbors or radii[idx] <= 1e-9:
                continue
            neighbor_ref = float(np.median(radii[neighbors]))
            limit = neighbor_ref * (1.0 + max_excess)
            if radii[idx] > limit:
                vertices[idx] = center + vectors[idx] * (limit / radii[idx])
    return vertices


def _vertex_adjacency(vertex_count: int, faces: np.ndarray) -> list[list[int]]:
    adjacency = [set() for _ in range(vertex_count)]
    for a, b, c in faces:
        a_i, b_i, c_i = int(a), int(b), int(c)
        adjacency[a_i].update((b_i, c_i))
        adjacency[b_i].update((a_i, c_i))
        adjacency[c_i].update((a_i, b_i))
    return [sorted(neighbors) for neighbors in adjacency]


def _enforce_non_slab_thickness(vertices: np.ndarray, min_short_to_mid: float) -> np.ndarray:
    bbox = vertices.max(axis=0) - vertices.min(axis=0)
    order = np.argsort(bbox)
    shortest_axis = int(order[0])
    middle = float(bbox[order[1]])
    shortest = float(bbox[shortest_axis])
    target = middle * min_short_to_mid
    if shortest < target:
        vertices = vertices.copy()
        vertices[:, shortest_axis] *= target / max(shortest, 1e-9)
    return vertices


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as obj:
        obj.write("# Generated by moon_rock_stack\n")
        for vertex in vertices:
            obj.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for face in faces:
            a, b, c = face + 1
            obj.write(f"f {a} {b} {c}\n")


def write_all_objs(mesh_dir: Path, rocks: Iterable[RockMesh]) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for rock in rocks:
        path = mesh_dir / f"rock_{rock.index:03d}.obj"
        write_obj(path, rock.vertices, rock.faces)
        paths[rock.index] = path
    return paths
