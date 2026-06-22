from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


COMMON_NUMERIC = [
    "gravity_m_s2",
    "course",
    "target_x",
    "target_y",
    "rock_volume",
    "rock_surface_area",
    "rock_bbox_x",
    "rock_bbox_y",
    "rock_bbox_z",
    "rock_elongation",
    "rock_flatness",
    "rock_sphericity",
    "rock_roughness",
    "rock_compactness",
    "rock_stability_score",
    "rock_mass",
    "rock_angularity",
    "rock_spike_score",
]

CANDIDATE_POSE_NUMERIC = [
    "pose_x",
    "pose_y",
    "pose_z",
    "pose_qw",
    "pose_qx",
    "pose_qy",
    "pose_qz",
    "candidate_id",
    "candidate_count",
]

CATEGORICAL = [
    "target_name",
    "strategy",
    "gravity",
    "role",
    "source_kind",
    "cluster_label",
]

GROUP_COLUMNS = [
    "run_name",
    "target_name",
    "strategy",
    "gravity",
    "trial",
    "slot_id",
    "candidate_rock_index",
]


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    output_dir = unique_dir(args.output.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)

    rows = read_csv(dataset_dir / "candidate_pose_examples.csv")
    if not rows:
        raise SystemExit(f"No candidate_pose_examples.csv found under {dataset_dir}")

    matrix = build_matrix(rows)
    groups = build_groups(rows)
    groups = [group for group in groups if len(group) >= 2 and any(matrix["labels"][idx] > 0.5 for idx in group)]
    if not groups:
        raise SystemExit("No rankable candidate-pose groups found.")

    rng = np.random.default_rng(args.seed)
    split = split_groups(groups, args.test_fraction, rng)
    model = init_model(matrix["x"].shape[1], args.hidden, rng)
    history = train(model, matrix["x"], matrix["labels"], split["train"], args, rng)
    summary, eval_rows = evaluate(model, matrix["x"], matrix["labels"], rows, split["test"])
    train_summary, _ = evaluate(model, matrix["x"], matrix["labels"], rows, split["train"])
    summary.update(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "row_count": len(rows),
            "rankable_group_count": len(groups),
            "train_groups": len(split["train"]),
            "test_groups": len(split["test"]),
            "input_dim": int(matrix["x"].shape[1]),
            "hidden": int(args.hidden),
            "epochs": int(args.epochs),
            "final_train_loss": float(history[-1]) if history else None,
            "train_top1_hit_rate": train_summary["top1_hit_rate"],
            "train_top3_hit_rate": train_summary["top3_hit_rate"],
        }
    )

    save_model(output_dir, model, matrix["schema"], summary)
    write_csv(output_dir / "group_eval.csv", eval_rows)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_readme(output_dir, summary)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a groupwise candidate-pose ranker.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=620)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def build_matrix(rows: list[dict[str, str]]) -> dict[str, Any]:
    numeric_columns = COMMON_NUMERIC + CANDIDATE_POSE_NUMERIC
    categories = {column: sorted({str(row.get(column, "")) for row in rows}) for column in CATEGORICAL}
    x_rows: list[list[float]] = []
    labels: list[float] = []
    for row in rows:
        features: list[float] = []
        for column in numeric_columns:
            raw = row.get(column, "")
            features.append(parse_float(raw))
            features.append(0.0 if is_float(raw) else 1.0)
        for column in CATEGORICAL:
            value = str(row.get(column, ""))
            features.extend(1.0 if value == category else 0.0 for category in categories[column])
        x_rows.append(features)
        labels.append(1.0 if parse_float(row.get("label_selected_by_pose_search", "0")) > 0.5 else 0.0)
    x = np.asarray(x_rows, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    schema = {
        "numeric_columns": numeric_columns,
        "categorical_columns": CATEGORICAL,
        "categories": categories,
        "feature_mean": mean.reshape(-1).tolist(),
        "feature_std": std.reshape(-1).tolist(),
        "group_columns": GROUP_COLUMNS,
        "label_column": "label_selected_by_pose_search",
        "task": "groupwise_candidate_pose_ranking",
    }
    return {"x": (x - mean) / std, "labels": np.asarray(labels, dtype=np.float64), "schema": schema}


def build_groups(rows: list[dict[str, str]]) -> list[np.ndarray]:
    grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        key = tuple(str(row.get(column, "")) for column in GROUP_COLUMNS)
        grouped[key].append(idx)
    return [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]


def split_groups(groups: list[np.ndarray], test_fraction: float, rng: np.random.Generator) -> dict[str, list[np.ndarray]]:
    indices = np.arange(len(groups))
    rng.shuffle(indices)
    test_count = max(1, int(round(len(indices) * test_fraction)))
    test_ids = set(int(idx) for idx in indices[:test_count])
    return {
        "train": [group for i, group in enumerate(groups) if i not in test_ids],
        "test": [group for i, group in enumerate(groups) if i in test_ids],
    }


def init_model(input_dim: int, hidden: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    scale1 = np.sqrt(2.0 / max(input_dim, 1))
    scale2 = np.sqrt(2.0 / max(hidden, 1))
    return {
        "w1": rng.normal(0.0, scale1, size=(input_dim, hidden)),
        "b1": np.zeros(hidden, dtype=np.float64),
        "w2": rng.normal(0.0, scale2, size=(hidden, 1)),
        "b2": np.zeros(1, dtype=np.float64),
    }


def train(
    model: dict[str, np.ndarray],
    x: np.ndarray,
    labels: np.ndarray,
    groups: list[np.ndarray],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[float]:
    history: list[float] = []
    for _epoch in range(args.epochs):
        order = np.arange(len(groups))
        rng.shuffle(order)
        losses: list[float] = []
        for group_idx in order:
            group = groups[int(group_idx)]
            y = labels[group]
            pos_count = float(np.sum(y > 0.5))
            if pos_count <= 0:
                continue
            target = np.where(y > 0.5, 1.0 / pos_count, 0.0)
            loss = train_group(model, x[group], target, args.lr, args.l2)
            losses.append(loss)
        history.append(float(np.mean(losses)) if losses else 0.0)
    return history


def train_group(
    model: dict[str, np.ndarray],
    x: np.ndarray,
    target: np.ndarray,
    lr: float,
    l2: float,
) -> float:
    z1 = x @ model["w1"] + model["b1"]
    h1 = np.maximum(z1, 0.0)
    scores = (h1 @ model["w2"] + model["b2"]).reshape(-1)
    probs = softmax(scores)
    loss = float(-np.sum(target * np.log(np.maximum(probs, 1e-12))))

    grad_scores = (probs - target).reshape(-1, 1)
    grad_w2 = h1.T @ grad_scores + l2 * model["w2"]
    grad_b2 = grad_scores.sum(axis=0)
    grad_h1 = grad_scores @ model["w2"].T
    grad_z1 = grad_h1 * (z1 > 0.0)
    grad_w1 = x.T @ grad_z1 + l2 * model["w1"]
    grad_b1 = grad_z1.sum(axis=0)

    model["w2"] -= lr * grad_w2
    model["b2"] -= lr * grad_b2
    model["w1"] -= lr * grad_w1
    model["b1"] -= lr * grad_b1
    return loss


def score(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    h1 = np.maximum(x @ model["w1"] + model["b1"], 0.0)
    return (h1 @ model["w2"] + model["b2"]).reshape(-1)


def evaluate(
    model: dict[str, np.ndarray],
    x: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, str]],
    groups: list[np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores = score(model, x)
    eval_rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    top1_hits = 0
    top3_hits = 0
    for group in groups:
        selected = set(int(idx) for idx in group if labels[idx] > 0.5)
        if not selected:
            continue
        ranked = sorted((int(idx) for idx in group), key=lambda idx: -float(scores[idx]))
        top1 = set(ranked[:1])
        top3 = set(ranked[: min(3, len(ranked))])
        hit1 = int(bool(selected.intersection(top1)))
        hit3 = int(bool(selected.intersection(top3)))
        top1_hits += hit1
        top3_hits += hit3
        first = rows[int(group[0])]
        item = {
            "run_name": first.get("run_name", ""),
            "target_name": first.get("target_name", ""),
            "strategy": first.get("strategy", ""),
            "gravity": first.get("gravity", ""),
            "trial": first.get("trial", ""),
            "slot_id": first.get("slot_id", ""),
            "candidate_rock_index": first.get("candidate_rock_index", ""),
            "candidate_count": len(group),
            "selected_candidate_ids": " ".join(str(rows[idx].get("candidate_id", "")) for idx in sorted(selected)),
            "net_top1_hit": hit1,
            "net_top3_hit": hit3,
            "net_top1_candidate_id": rows[ranked[0]].get("candidate_id", "") if ranked else "",
            "net_top1_score": float(scores[ranked[0]]) if ranked else 0.0,
        }
        by_key[(item["target_name"], item["gravity"])].append(item)
        eval_rows.append(item)
    group_count = max(len(eval_rows), 1)
    by_target_gravity = {}
    for (target, gravity), items in sorted(by_key.items()):
        count = max(len(items), 1)
        by_target_gravity[f"{target}|{gravity}"] = {
            "groups": len(items),
            "top1_hit_rate": sum(int(item["net_top1_hit"]) for item in items) / count,
            "top3_hit_rate": sum(int(item["net_top3_hit"]) for item in items) / count,
        }
    summary = {
        "groups": len(eval_rows),
        "top1_hit_rate": top1_hits / group_count,
        "top3_hit_rate": top3_hits / group_count,
        "by_target_gravity": by_target_gravity,
    }
    return summary, eval_rows


def save_model(output_dir: Path, model: dict[str, np.ndarray], schema: dict[str, Any], metrics: dict[str, Any]) -> None:
    np.savez_compressed(output_dir / "candidate_pose_group_ranker.npz", **model)
    (output_dir / "candidate_pose_group_ranker_schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "candidate_pose_group_ranker_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    # Runtime-compatible aliases for moon_rock_stack.run_structured_experiment.
    np.savez_compressed(output_dir / "candidate_pose_rank_net.npz", **model)
    (output_dir / "candidate_pose_rank_net_schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_readme(output_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Candidate Pose Groupwise Ranker",
        "",
        "This model is trained with a groupwise softmax loss over candidate poses that share a run, target, strategy, gravity, trial, slot, and candidate rock.",
        "",
        f"- dataset: `{metrics['dataset_dir']}`",
        f"- rows: {metrics['row_count']}",
        f"- rankable groups: {metrics['rankable_group_count']}",
        f"- test top-1 hit: {metrics['top1_hit_rate']:.3f}",
        f"- test top-3 hit: {metrics['top3_hit_rate']:.3f}",
        f"- train top-1 hit: {metrics['train_top1_hit_rate']:.3f}",
        f"- train top-3 hit: {metrics['train_top3_hit_rate']:.3f}",
        "",
        "The labels are still imitation labels from the current hand-coded search, so this is a ranking baseline, not a final physical policy.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(np.clip(shifted, -40.0, 40.0))
    return exp / np.maximum(np.sum(exp), 1e-12)


def parse_float(value: str | None) -> float:
    return float(value) if is_float(value) else 0.0


def is_float(value: str | None) -> bool:
    if value in {"", None}:
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
