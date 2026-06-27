from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moon_rock_stack.clustering import cluster_features
from moon_rock_stack.features import FEATURE_COLUMNS, extract_features
from moon_rock_stack.fractal_rocks import RockMesh, generate_fractal_rock, write_obj


STRUCTURAL_KINDS = (
    "bearing_block_clast",
    "course_block_clast",
    "compact_block_clast",
    "subangular_block",
    "wall_block_clast",
    "interlock_block_clast",
    "angular_boulder_clast",
    "buttress_clast",
    "cap_block_clast",
    "equant_clast",
)


@dataclass
class NasaObj:
    sample_id: str
    outer_zip: Path
    inner_zip_name: str
    obj_name: str
    vertices: np.ndarray
    faces: np.ndarray


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    original_dir = output / "nasa_original_test_only"
    generated_dir = output / "nasa_like_generated_training"
    original_mesh_dir = original_dir / "meshes"
    generated_mesh_dir = generated_dir / "meshes"
    for path in (original_mesh_dir, generated_mesh_dir):
        path.mkdir(parents=True, exist_ok=True)

    nasa_objects = read_nasa_objects(args.nasa_root)
    if not nasa_objects:
        raise RuntimeError(f"No nested NASA OBJ files found under {args.nasa_root}")

    original_rows, manifest_rows = build_original_test_only(
        nasa_objects=nasa_objects,
        mesh_dir=original_mesh_dir,
        target_max_extent_m=args.test_max_extent_m,
    )
    add_clusters(original_rows, clusters=min(args.clusters, len(original_rows)), seed=args.seed)
    write_csv(original_dir / "nasa_test_features.csv", original_rows)
    write_csv(original_dir / "nasa_test_manifest.csv", manifest_rows)
    (original_dir / "nasa_test_manifest.json").write_text(
        json.dumps(
            {
                "source_root": str(args.nasa_root.resolve()),
                "policy": "TEST_ONLY_FOREVER_DO_NOT_TRAIN_ON_ORIGINAL_NASA_GEOMETRY",
                "sample_count": len(original_rows),
                "samples": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    generated_rows, audit_rows = generate_nasa_like_training_catalog(
        reference_rows=original_rows,
        generated_count=args.generated_count,
        candidate_multiplier=args.candidate_multiplier,
        seed=args.seed,
        mesh_dir=generated_mesh_dir,
        max_spike=args.max_spike,
        max_flatness=args.max_flatness,
        min_short_to_mid=args.min_short_to_mid,
    )
    add_clusters(generated_rows, clusters=min(args.clusters, len(generated_rows)), seed=args.seed + 31)
    write_csv(generated_dir / "features.csv", generated_rows)
    write_csv(generated_dir / "similarity_audit.csv", audit_rows)
    write_csv(generated_dir / "cluster_summary.csv", cluster_summary(generated_rows))

    feature_stats = feature_distribution_summary(original_rows, generated_rows)
    (output / "nasa_reference_feature_stats.json").write_text(json.dumps(feature_stats, indent=2), encoding="utf-8")
    write_readme(output / "README.md", args, original_rows, generated_rows, audit_rows, feature_stats)
    print(output)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a NASA meteorite test-only feature set and a similar-but-not-identical synthetic training catalog."
    )
    parser.add_argument("--nasa-root", type=Path, default=Path(r"D:\MoonStack\NASAstone"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20627001)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--generated-count", type=int, default=320)
    parser.add_argument("--candidate-multiplier", type=int, default=14)
    parser.add_argument("--test-max-extent-m", type=float, default=0.165)
    parser.add_argument("--max-spike", type=float, default=0.160)
    parser.add_argument("--max-flatness", type=float, default=1.620)
    parser.add_argument("--min-short-to-mid", type=float, default=0.620)
    return parser.parse_args()


def read_nasa_objects(root: Path) -> list[NasaObj]:
    objects: list[NasaObj] = []
    for outer_zip in sorted(root.glob("*.zip")):
        with zipfile.ZipFile(outer_zip) as outer:
            nested_entries = [item for item in outer.infolist() if item.filename.lower().endswith(".zip")]
            if not nested_entries:
                continue
            nested_entry = nested_entries[0]
            with zipfile.ZipFile(io.BytesIO(outer.read(nested_entry))) as inner:
                obj_entries = [item for item in inner.infolist() if item.filename.lower().endswith(".obj")]
                if not obj_entries:
                    continue
                obj_entry = obj_entries[0]
                text = inner.read(obj_entry).decode("utf-8", errors="ignore")
        vertices, faces = parse_obj(text)
        if len(vertices) < 4 or len(faces) < 4:
            continue
        objects.append(
            NasaObj(
                sample_id=sample_id_from_name(outer_zip.name),
                outer_zip=outer_zip,
                inner_zip_name=nested_entry.filename,
                obj_name=obj_entry.filename,
                vertices=vertices,
                faces=faces,
            )
        )
    return objects


def parse_obj(text: str) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            try:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
        elif parts[0] == "f" and len(parts) >= 4:
            idx: list[int] = []
            for token in parts[1:]:
                value = token.split("/")[0]
                try:
                    parsed = int(value)
                except ValueError:
                    continue
                if parsed < 0:
                    parsed = len(vertices) + parsed + 1
                idx.append(parsed - 1)
            if len(idx) >= 3:
                for offset in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[offset], idx[offset + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def sample_id_from_name(name: str) -> str:
    stem = Path(name).stem
    return stem.replace("antarctic-meteorite-sample-", "").replace("-", "_")


def build_original_test_only(
    nasa_objects: list[NasaObj],
    mesh_dir: Path,
    target_max_extent_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, item in enumerate(nasa_objects):
        raw_bbox = item.vertices.max(axis=0) - item.vertices.min(axis=0)
        vertices = normalize_to_extent(item.vertices, target_max_extent_m)
        mesh_path = mesh_dir / f"nasa_test_{index:03d}_{item.sample_id}.obj"
        write_obj(mesh_path, vertices, item.faces)
        rock = RockMesh(index=index, kind="nasa_original_test_only", vertices=vertices, faces=item.faces, seed=0)
        row = extract_features(rock)
        row.update(
            {
                "sample_id": item.sample_id,
                "policy": "TEST_ONLY_FOREVER_DO_NOT_TRAIN",
                "outer_zip": str(item.outer_zip),
                "inner_zip": item.inner_zip_name,
                "obj_name": item.obj_name,
                "mesh_path": str(mesh_path),
                "raw_bbox_x": float(raw_bbox[0]),
                "raw_bbox_y": float(raw_bbox[1]),
                "raw_bbox_z": float(raw_bbox[2]),
                "raw_vertex_count": int(len(item.vertices)),
                "raw_face_count": int(len(item.faces)),
            }
        )
        rows.append(row)
        manifest_rows.append(
            {
                "index": index,
                "sample_id": item.sample_id,
                "policy": "TEST_ONLY_FOREVER_DO_NOT_TRAIN",
                "outer_zip": str(item.outer_zip),
                "inner_zip": item.inner_zip_name,
                "obj_name": item.obj_name,
                "mesh_path": str(mesh_path),
                "raw_vertex_count": int(len(item.vertices)),
                "raw_face_count": int(len(item.faces)),
                "raw_bbox_x": float(raw_bbox[0]),
                "raw_bbox_y": float(raw_bbox[1]),
                "raw_bbox_z": float(raw_bbox[2]),
            }
        )
    return rows, manifest_rows


def normalize_to_extent(vertices: np.ndarray, max_extent: float) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0)
    bbox = centered.max(axis=0) - centered.min(axis=0)
    scale = max_extent / max(float(bbox.max()), 1e-9)
    return centered * scale


def generate_nasa_like_training_catalog(
    reference_rows: list[dict[str, Any]],
    generated_count: int,
    candidate_multiplier: int,
    seed: int,
    mesh_dir: Path,
    max_spike: float,
    max_flatness: float,
    min_short_to_mid: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    reference_matrix = feature_matrix(reference_rows)
    mean = reference_matrix.mean(axis=0)
    std = np.maximum(reference_matrix.std(axis=0), 1e-9)
    target_count = max(generated_count, 1)
    candidate_count = max(target_count * max(candidate_multiplier, 1), target_count)
    candidates: list[tuple[float, dict[str, Any], np.ndarray, np.ndarray]] = []

    for candidate_id in range(candidate_count):
        ref = reference_rows[int(rng.integers(0, len(reference_rows)))]
        kind = STRUCTURAL_KINDS[candidate_id % len(STRUCTURAL_KINDS)]
        rock_seed = int(rng.integers(0, 2**31 - 1))
        vertices, faces = generate_fractal_rock(kind, rock_seed)
        vertices = scale_like_reference(vertices, ref, rng)
        rock = RockMesh(index=candidate_id, kind=kind, vertices=vertices, faces=faces, seed=rock_seed)
        row = extract_features(rock)
        row["reference_sample_id"] = ref["sample_id"]
        row["candidate_id"] = candidate_id
        row["source_policy"] = "TRAINING_ALLOWED_SYNTHETIC_NASA_LIKE_NO_ORIGINAL_GEOMETRY"
        row["generator_profile"] = "nasa_like_reference_feature_matching_v1"
        quality = quality_flags(row, max_spike=max_spike, max_flatness=max_flatness, min_short_to_mid=min_short_to_mid)
        row.update(quality)
        if not quality["screen_accept"]:
            continue
        dist = nearest_reference_distance(row, reference_matrix, mean, std)
        row["nearest_reference_feature_distance"] = float(dist)
        candidates.append((dist, row, vertices, faces))

    candidates.sort(key=lambda item: (item[0], int(item[1]["candidate_id"])))
    selected = candidates[:target_count]
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for output_index, (dist, row, vertices, faces) in enumerate(selected):
        mesh_path = mesh_dir / f"rock_{output_index:03d}.obj"
        write_obj(mesh_path, vertices, faces)
        row = dict(row)
        row["index"] = output_index
        row["mesh_path"] = str(mesh_path)
        row["source_kind"] = str(row["source_kind"])
        rows.append(row)
        audit_rows.append(
            {
                "index": output_index,
                "source_kind": row["source_kind"],
                "reference_sample_id": row["reference_sample_id"],
                "nearest_reference_feature_distance": float(dist),
                "screen_accept": int(row["screen_accept"]),
                "short_to_mid": row["short_to_mid"],
                "spike_score": row["spike_score"],
                "flatness": row["flatness"],
                "elongation": row["elongation"],
                "compactness": row["compactness"],
                "support_face_area_ratio": row.get("support_face_area_ratio", 0.0),
                "mesh_path": str(mesh_path),
            }
        )
    return rows, audit_rows


def scale_like_reference(vertices: np.ndarray, reference_row: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0)
    current_bbox = centered.max(axis=0) - centered.min(axis=0)
    ref_dims = np.asarray(
        [float(reference_row["bbox_x"]), float(reference_row["bbox_y"]), float(reference_row["bbox_z"])],
        dtype=np.float64,
    )
    desired_sorted = np.sort(ref_dims)[::-1] * rng.uniform(0.82, 1.12, size=3)
    desired_sorted[2] = max(desired_sorted[2], min(desired_sorted[1] * 0.64, desired_sorted[0] * 0.92))
    desired_dims = desired_sorted[rng.permutation(3)]
    scale = desired_dims / np.maximum(current_bbox, 1e-9)
    scaled = centered * scale
    return scaled - scaled.mean(axis=0)


def feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    matrix = np.asarray([[float(row[col]) for col in FEATURE_COLUMNS] for row in rows], dtype=np.float64)
    matrix[:, 0] = np.log10(np.maximum(matrix[:, 0], 1e-12))
    matrix[:, 1] = np.log10(np.maximum(matrix[:, 1], 1e-12))
    return matrix


def nearest_reference_distance(
    row: dict[str, Any],
    reference_matrix: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> float:
    values = feature_matrix([row])[0]
    z = (values - mean) / std
    ref_z = (reference_matrix - mean) / std
    return float(np.sqrt(np.square(ref_z - z[None, :]).mean(axis=1)).min())


def quality_flags(row: dict[str, Any], max_spike: float, max_flatness: float, min_short_to_mid: float) -> dict[str, Any]:
    bbox = sorted((float(row["bbox_x"]), float(row["bbox_y"]), float(row["bbox_z"])), reverse=True)
    short_to_mid = bbox[2] / max(bbox[1], 1e-9)
    reject_spike = float(row["spike_score"]) > max_spike
    reject_slab = float(row["flatness"]) > max_flatness or short_to_mid < min_short_to_mid
    reject_stringer = float(row["elongation"]) > 1.95
    reject_low_compactness = float(row["compactness"]) < 0.18
    screen_accept = not (reject_spike or reject_slab or reject_stringer or reject_low_compactness)
    return {
        "short_to_mid": float(short_to_mid),
        "reject_spike": int(reject_spike),
        "reject_slab": int(reject_slab),
        "reject_stringer": int(reject_stringer),
        "reject_low_compactness": int(reject_low_compactness),
        "screen_accept": int(screen_accept),
    }


def add_clusters(rows: list[dict[str, Any]], clusters: int, seed: int) -> None:
    labels, names = cluster_features(rows, clusters=clusters, seed=seed)
    for row, label in zip(rows, labels):
        row["cluster_id"] = int(label)
        row["cluster_label"] = names[int(label)]


def cluster_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    labels = sorted({str(row["cluster_label"]) for row in rows})
    for label in labels:
        subset = [row for row in rows if str(row["cluster_label"]) == label]
        output.append(
            {
                "cluster_label": label,
                "count": len(subset),
                "source_kind_counts": json.dumps(count_by(subset, "source_kind"), sort_keys=True),
                "mean_volume": mean(float(row["volume"]) for row in subset),
                "mean_bbox_x": mean(float(row["bbox_x"]) for row in subset),
                "mean_bbox_y": mean(float(row["bbox_y"]) for row in subset),
                "mean_bbox_z": mean(float(row["bbox_z"]) for row in subset),
                "mean_elongation": mean(float(row["elongation"]) for row in subset),
                "mean_flatness": mean(float(row["flatness"]) for row in subset),
                "mean_spike_score": mean(float(row["spike_score"]) for row in subset),
                "mean_compactness": mean(float(row["compactness"]) for row in subset),
                "mean_support_face_area_ratio": mean(float(row["support_face_area_ratio"]) for row in subset),
            }
        )
    return output


def feature_distribution_summary(
    reference_rows: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    keys = (
        "volume",
        "bbox_x",
        "bbox_y",
        "bbox_z",
        "elongation",
        "flatness",
        "roughness",
        "angularity",
        "spike_score",
        "compactness",
        "major_face_count",
        "support_face_area_ratio",
        "support_plane_quality",
    )
    return {
        "nasa_original_policy": "TEST_ONLY_FOREVER_DO_NOT_TRAIN",
        "generated_policy": "TRAINING_ALLOWED_SYNTHETIC_NASA_LIKE",
        "reference_count": len(reference_rows),
        "generated_count": len(generated_rows),
        "reference": {key: describe([float(row[key]) for row in reference_rows]) for key in keys},
        "generated": {key: describe([float(row[key]) for row in generated_rows]) for key in keys},
    }


def describe(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"min": math.nan, "mean": math.nan, "max": math.nan, "std": math.nan}
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(
    path: Path,
    args: argparse.Namespace,
    original_rows: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    feature_stats: dict[str, Any],
) -> None:
    source_counts = count_by(generated_rows, "source_kind")
    accepted_distances = [float(row["nearest_reference_feature_distance"]) for row in audit_rows]
    lines = [
        "# NASA 石头参考集与 NASA-like 训练石头目录",
        "",
        "## 数据隔离规则",
        "",
        "- `nasa_original_test_only/`：来自 `D:\\MoonStack\\NASAstone` 的原始 NASA OBJ，只能作为测试集和审计集，永远不能参与训练。",
        "- `nasa_like_generated_training/`：只使用原始 NASA 石头的几何统计分布进行筛选，mesh 由本项目多面体生成器重新生成，可以作为训练来源之一。",
        "- 当前没有把 NASA 原始顶点、面片或纹理复制到训练目录；训练目录只有重新生成的角状多面体。",
        "",
        "## 本次生成",
        "",
        f"- NASA 原始 test-only 样本数：{len(original_rows)}",
        f"- NASA-like synthetic training 样本数：{len(generated_rows)}",
        f"- seed：{args.seed}",
        f"- test-only 归一化最大外接尺寸：{args.test_max_extent_m:.3f} m",
        f"- synthetic 筛选阈值：spike <= {args.max_spike:.3f}，flatness <= {args.max_flatness:.3f}，short/mid >= {args.min_short_to_mid:.3f}",
        "",
        "## 几何先验",
        "",
        "- 保留角状、多面、块状石头，不生成尖刺状局部突起。",
        "- 不允许特别扁的 slab；短轴/中轴比例低于阈值会被拒绝。",
        "- 训练石头按 NASA test-only 的 bbox 比例、粗糙度、角度、支撑面比例做近似筛选，但不复用原始网格。",
        "- 保留 `bearing_block_clast`、`course_block_clast`、`compact_block_clast`、`wall_block_clast` 等结构语义，便于接入现有墙体堆叠策略。",
        "",
        "## Synthetic source_kind 分布",
        "",
        "```json",
        json.dumps(source_counts, indent=2, sort_keys=True),
        "```",
        "",
        "## 相似度审计",
        "",
        f"- 最近参考特征距离 mean：{mean(accepted_distances):.4f}",
        f"- 最近参考特征距离 min/max：{min(accepted_distances):.4f} / {max(accepted_distances):.4f}",
        "- 这里的距离是标准化几何特征距离，只用于筛选“近似外形分布”；它不是复制检测，也不能说明语义同源。",
        "",
        "## 输出文件",
        "",
        "- `nasa_original_test_only/nasa_test_features.csv`：原始 NASA test-only 几何特征。",
        "- `nasa_original_test_only/nasa_test_manifest.json`：原始样本来源、zip、obj、test-only 策略。",
        "- `nasa_like_generated_training/features.csv`：可训练 synthetic NASA-like 几何特征。",
        "- `nasa_like_generated_training/similarity_audit.csv`：每个 synthetic 石头最接近的 NASA 参考和筛选距离。",
        "- `nasa_reference_feature_stats.json`：原始 test-only 与 synthetic 训练集的几何分布对比。",
        "",
        "## 下一步使用方式",
        "",
        "- 结构仿真侧可先使用 `--rock-profile nasa_like_wall` 运行 3/4 层单面墙小批量评估。",
        "- 学习侧可以把 `nasa_like_generated_training/features.csv` 作为几何预训练/类别扩充数据，但禁止读取 `nasa_original_test_only` 进入训练。",
        "- NASA 原始 test-only 应只在模型冻结或策略冻结后做泛化测试，避免调参时泄漏测试集。",
        "",
        "## 分布摘要",
        "",
        "```json",
        json.dumps(feature_stats, indent=2, sort_keys=True),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
