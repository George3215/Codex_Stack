from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
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
    "rock_major_face_count",
    "rock_largest_face_area_ratio",
    "rock_top3_face_area_ratio",
    "rock_face_area_entropy",
    "rock_normal_concentration",
    "rock_support_face_count",
    "rock_support_face_area_ratio",
    "rock_opposing_face_pair_count",
    "rock_opposing_face_area_ratio",
    "rock_face_planarity",
    "rock_support_plane_quality",
]

POSE_NUMERIC = [
    "placed_x",
    "placed_y",
    "placed_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "candidate_id",
    "assignment_fallback_attempt",
    "assignment_candidate_count",
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

COMMON_CATEGORICAL = [
    "target_name",
    "strategy",
    "gravity",
    "role",
    "rock_source_kind",
    "rock_cluster_label",
]

CANDIDATE_CATEGORICAL = [
    "target_name",
    "strategy",
    "gravity",
    "role",
    "source_kind",
    "cluster_label",
]

WORLD_TARGETS = [
    "target_error_xy_m",
    "target_y_error_m",
    "velocity_inf_norm_after_place",
    "height_gain_m",
]


@dataclass
class MatrixBundle:
    x: np.ndarray
    y: np.ndarray
    schema: dict[str, Any]
    row_count: int


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    output_dir = unique_dir(args.output.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)

    placement_rows = read_csv(dataset_dir / "placement_examples.csv")
    candidate_pose_rows = read_csv(dataset_dir / "candidate_pose_examples.csv")
    metrics: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "models": {},
    }

    train_binary_model(
        "stone_fit_net",
        placement_rows,
        COMMON_NUMERIC,
        COMMON_CATEGORICAL,
        "label_success",
        output_dir,
        metrics,
        args,
        min_rows=100,
    )
    train_binary_model(
        "pose_accept_net",
        placement_rows,
        COMMON_NUMERIC + POSE_NUMERIC,
        COMMON_CATEGORICAL,
        "label_success",
        output_dir,
        metrics,
        args,
        min_rows=100,
    )
    train_binary_model(
        "moon_drift_risk_net",
        add_moon_drift_risk_label(placement_rows),
        COMMON_NUMERIC + POSE_NUMERIC,
        COMMON_CATEGORICAL,
        "label_moon_drift_risk",
        output_dir,
        metrics,
        args,
        min_rows=100,
    )
    train_regression_model(
        "world_delta_net",
        placement_rows,
        COMMON_NUMERIC + POSE_NUMERIC,
        COMMON_CATEGORICAL,
        WORLD_TARGETS,
        output_dir,
        metrics,
        args,
        min_rows=100,
    )
    train_binary_model(
        "candidate_pose_rank_net",
        candidate_pose_rows,
        COMMON_NUMERIC + CANDIDATE_POSE_NUMERIC,
        CANDIDATE_CATEGORICAL,
        "label_selected_by_pose_search",
        output_dir,
        metrics,
        args,
        min_rows=12,
    )

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_readme(output_dir, metrics)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train modular small NumPy networks for MoonStack placement.")
    parser.add_argument("--dataset", type=Path, required=True, help="Directory from build_learning_dataset.py.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--balanced-binary-loss",
        action="store_true",
        help="Use inverse-frequency weights for binary targets so rare failures influence the classifier.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def train_binary_model(
    name: str,
    rows: list[dict[str, str]],
    numeric_columns: list[str],
    categorical_columns: list[str],
    label_column: str,
    output_dir: Path,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    min_rows: int,
) -> None:
    bundle = build_matrix(rows, numeric_columns, categorical_columns, [label_column], task="binary")
    if bundle.row_count < min_rows:
        metrics["models"][name] = {"status": "skipped", "reason": f"not enough rows: {bundle.row_count} < {min_rows}"}
        return
    rng = np.random.default_rng(args.seed + stable_name_seed(name))
    split = split_indices(bundle.row_count, args.test_fraction, rng)
    model = init_model(bundle.x.shape[1], args.hidden, 1, rng)
    history = train_binary(model, bundle.x[split["train"]], bundle.y[split["train"]], args, rng)
    probs = predict_binary(model, bundle.x[split["test"]]).reshape(-1)
    y_true = bundle.y[split["test"]].reshape(-1)
    model_metrics = binary_metrics(y_true, probs)
    model_metrics.update(
        {
            "status": "trained",
            "row_count": bundle.row_count,
            "train_rows": int(len(split["train"])),
            "test_rows": int(len(split["test"])),
            "input_dim": int(bundle.x.shape[1]),
            "hidden": int(args.hidden),
            "epochs": int(args.epochs),
            "final_train_loss": float(history[-1]) if history else None,
        }
    )
    if bundle.row_count < 100:
        model_metrics["warning"] = "low_sample_smoke_only"
    save_model(output_dir, name, model, bundle.schema, model_metrics)
    metrics["models"][name] = model_metrics


def train_regression_model(
    name: str,
    rows: list[dict[str, str]],
    numeric_columns: list[str],
    categorical_columns: list[str],
    target_columns: list[str],
    output_dir: Path,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    min_rows: int,
) -> None:
    bundle = build_matrix(rows, numeric_columns, categorical_columns, target_columns, task="regression")
    if bundle.row_count < min_rows:
        metrics["models"][name] = {"status": "skipped", "reason": f"not enough rows: {bundle.row_count} < {min_rows}"}
        return
    rng = np.random.default_rng(args.seed + stable_name_seed(name))
    split = split_indices(bundle.row_count, args.test_fraction, rng)
    y_mean = bundle.y[split["train"]].mean(axis=0, keepdims=True)
    y_std = bundle.y[split["train"]].std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    model = init_model(bundle.x.shape[1], args.hidden, bundle.y.shape[1], rng)
    history = train_regression(model, bundle.x[split["train"]], (bundle.y[split["train"]] - y_mean) / y_std, args, rng)
    pred = predict_regression(model, bundle.x[split["test"]]) * y_std + y_mean
    model_metrics = regression_metrics(bundle.y[split["test"]], pred, target_columns)
    model_metrics.update(
        {
            "status": "trained",
            "row_count": bundle.row_count,
            "train_rows": int(len(split["train"])),
            "test_rows": int(len(split["test"])),
            "input_dim": int(bundle.x.shape[1]),
            "hidden": int(args.hidden),
            "epochs": int(args.epochs),
            "final_train_loss": float(history[-1]) if history else None,
        }
    )
    if bundle.row_count < 100:
        model_metrics["warning"] = "low_sample_smoke_only"
    bundle.schema["target_mean"] = y_mean.reshape(-1).tolist()
    bundle.schema["target_std"] = y_std.reshape(-1).tolist()
    save_model(output_dir, name, model, bundle.schema, model_metrics)
    metrics["models"][name] = model_metrics


def build_matrix(
    rows: list[dict[str, str]],
    numeric_columns: list[str],
    categorical_columns: list[str],
    target_columns: list[str],
    task: str,
) -> MatrixBundle:
    filtered = [row for row in rows if targets_available(row, target_columns, task)]
    categories = {column: sorted({str(row.get(column, "")) for row in filtered}) for column in categorical_columns}
    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    for row in filtered:
        features: list[float] = []
        for column in numeric_columns:
            raw = row.get(column, "")
            features.append(parse_float(raw))
            features.append(0.0 if is_float(raw) else 1.0)
        for column in categorical_columns:
            value = str(row.get(column, ""))
            features.extend(1.0 if value == category else 0.0 for category in categories[column])
        x_rows.append(features)
        y_rows.append([parse_float(row[column]) for column in target_columns])
    if not x_rows:
        return MatrixBundle(np.zeros((0, 0), dtype=np.float64), np.zeros((0, len(target_columns))), {}, 0)
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    schema = {
        "task": task,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "categories": categories,
        "target_columns": target_columns,
        "feature_mean": mean.reshape(-1).tolist(),
        "feature_std": std.reshape(-1).tolist(),
    }
    return MatrixBundle(x=(x - mean) / std, y=y, schema=schema, row_count=x.shape[0])


def targets_available(row: dict[str, str], target_columns: list[str], task: str) -> bool:
    if task == "binary":
        return row.get(target_columns[0], "") not in {"", None}
    return all(is_float(row.get(column, "")) for column in target_columns)


def add_moon_drift_risk_label(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        y_error = abs(parse_float(item.get("target_y_error_m", "")))
        target_error = parse_float(item.get("target_error_xy_m", ""))
        failed = parse_float(item.get("is_failure_case", "0")) > 0.5
        skipped = parse_float(item.get("is_skipped_slot", "0")) > 0.5
        moon = str(item.get("gravity", "")) == "moon"
        risk = failed or skipped or (moon and (y_error > 0.10 or target_error > 0.24))
        item["label_moon_drift_risk"] = "1" if risk else "0"
        output.append(item)
    return output


def split_indices(n: int, test_fraction: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    indices = rng.permutation(n)
    test_count = max(1, int(round(n * test_fraction)))
    return {"test": indices[:test_count], "train": indices[test_count:]}


def init_model(input_dim: int, hidden: int, output_dim: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "w1": rng.normal(0.0, math.sqrt(2.0 / max(input_dim, 1)), size=(input_dim, hidden)),
        "b1": np.zeros((1, hidden)),
        "w2": rng.normal(0.0, math.sqrt(2.0 / max(hidden, 1)), size=(hidden, output_dim)),
        "b2": np.zeros((1, output_dim)),
    }


def train_binary(model: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, args: argparse.Namespace, rng: np.random.Generator) -> list[float]:
    y = y.reshape(-1, 1)
    weights = binary_sample_weights(y) if args.balanced_binary_loss else np.ones_like(y)
    history: list[float] = []
    for _epoch in range(args.epochs):
        losses: list[float] = []
        for xb, yb, wb in minibatches(x, y, args.batch_size, rng, weights):
            hidden, logits = forward(model, xb)
            probs = sigmoid(logits)
            losses.append(weighted_binary_cross_entropy(yb, probs, wb) + l2_loss(model, args.l2))
            grad = wb * (probs - yb) / max(float(np.sum(wb)), 1e-9)
            backward(model, xb, hidden, grad, args.lr, args.l2)
        history.append(float(np.mean(losses)))
    return history


def train_regression(model: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, args: argparse.Namespace, rng: np.random.Generator) -> list[float]:
    history: list[float] = []
    for _epoch in range(args.epochs):
        losses: list[float] = []
        for xb, yb, _wb in minibatches(x, y, args.batch_size, rng):
            hidden, pred = forward(model, xb)
            diff = pred - yb
            losses.append(float(np.mean(diff * diff)) + l2_loss(model, args.l2))
            backward(model, xb, hidden, (2.0 * diff) / max(len(xb), 1), args.lr, args.l2)
        history.append(float(np.mean(losses)))
    return history


def minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    weights: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    indices = rng.permutation(len(x))
    if weights is None:
        weights = np.ones((len(x), 1), dtype=np.float64)
    return [
        (x[idx], y[idx], weights[idx])
        for idx in (indices[start : start + batch_size] for start in range(0, len(x), batch_size))
    ]


def forward(model: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.maximum(x @ model["w1"] + model["b1"], 0.0)
    return hidden, hidden @ model["w2"] + model["b2"]


def backward(model: dict[str, np.ndarray], x: np.ndarray, hidden: np.ndarray, grad_output: np.ndarray, lr: float, l2: float) -> None:
    grad_w2 = hidden.T @ grad_output + l2 * model["w2"]
    grad_b2 = grad_output.sum(axis=0, keepdims=True)
    grad_hidden = grad_output @ model["w2"].T
    grad_hidden[hidden <= 0.0] = 0.0
    grad_w1 = x.T @ grad_hidden + l2 * model["w1"]
    grad_b1 = grad_hidden.sum(axis=0, keepdims=True)
    model["w2"] -= lr * grad_w2
    model["b2"] -= lr * grad_b2
    model["w1"] -= lr * grad_w1
    model["b1"] -= lr * grad_b1


def predict_binary(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    _hidden, logits = forward(model, x)
    return sigmoid(logits)


def predict_regression(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    _hidden, output = forward(model, x)
    return output


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def binary_sample_weights(y_true: np.ndarray) -> np.ndarray:
    positive = y_true >= 0.5
    negative = ~positive
    pos_count = max(float(np.sum(positive)), 1.0)
    neg_count = max(float(np.sum(negative)), 1.0)
    total = max(float(len(y_true)), 1.0)
    weights = np.ones_like(y_true, dtype=np.float64)
    weights[positive] = 0.5 * total / pos_count
    weights[negative] = 0.5 * total / neg_count
    return weights


def weighted_binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, weights: np.ndarray) -> float:
    eps = 1e-7
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    loss = -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(np.sum(weights * loss) / max(float(np.sum(weights)), 1e-9))


def l2_loss(model: dict[str, np.ndarray], l2: float) -> float:
    return float(0.5 * l2 * (np.sum(model["w1"] ** 2) + np.sum(model["w2"] ** 2)))


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    pred = (y_prob >= 0.5).astype(np.float64)
    tp = float(np.sum((pred == 1.0) & (y_true == 1.0)))
    tn = float(np.sum((pred == 0.0) & (y_true == 0.0)))
    fp = float(np.sum((pred == 1.0) & (y_true == 0.0)))
    fn = float(np.sum((pred == 0.0) & (y_true == 1.0)))
    total = max(float(len(y_true)), 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(pred)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_columns: list[str]) -> dict[str, Any]:
    diff = y_pred - y_true
    mae = np.mean(np.abs(diff), axis=0)
    rmse = np.sqrt(np.mean(diff * diff, axis=0))
    return {
        "targets": {
            column: {"mae": float(mae[index]), "rmse": float(rmse[index]), "mean_true": float(np.mean(y_true[:, index]))}
            for index, column in enumerate(target_columns)
        }
    }


def save_model(output_dir: Path, name: str, model: dict[str, np.ndarray], schema: dict[str, Any], metrics: dict[str, Any]) -> None:
    np.savez_compressed(output_dir / f"{name}.npz", **model)
    (output_dir / f"{name}_schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


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


def stable_name_seed(name: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(name))


def write_readme(output_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Modular Small-Network Baseline",
        "",
        f"Created at: `{metrics['created_at']}`",
        f"Dataset: `{metrics['dataset_dir']}`",
        "",
        "## Models",
        "",
    ]
    for name, item in metrics["models"].items():
        if item.get("status") != "trained":
            lines.append(f"- `{name}`: skipped, {item.get('reason', '')}")
            continue
        warning = f", warning={item['warning']}" if "warning" in item else ""
        if "accuracy" in item:
            lines.append(
                f"- `{name}`: rows={item['row_count']}, accuracy={item['accuracy']:.3f}, "
                f"precision={item['precision']:.3f}, recall={item['recall']:.3f}, f1={item['f1']:.3f}{warning}"
            )
        else:
            target_bits = ", ".join(
                f"{target}:mae={values['mae']:.4f}" for target, values in item.get("targets", {}).items()
            )
            lines.append(f"- `{name}`: rows={item['row_count']}, {target_bits}{warning}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These are tabular NumPy MLP baselines. They prove the modular data path and give a first ranking signal, but they are not yet point-cloud policies.",
            "",
            "Use these models to reduce candidate trials, then replace tabular rock features with learned mesh or point-cloud encoders.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
