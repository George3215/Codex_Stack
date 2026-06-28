from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


NUMERIC_COLUMNS = [
    "gravity_m_s2",
    "course",
    "target_x",
    "target_y",
    "pose_x",
    "pose_y",
    "pose_z",
    "pose_qw",
    "pose_qx",
    "pose_qy",
    "pose_qz",
    "candidate_id",
    "candidate_count",
    "rock_volume",
    "rock_surface_area",
    "rock_face_count",
    "rock_bbox_x",
    "rock_bbox_y",
    "rock_bbox_z",
    "rock_elongation",
    "rock_flatness",
    "rock_sphericity",
    "rock_roughness",
    "rock_angularity",
    "rock_spike_score",
    "rock_compactness",
    "rock_rectangularity",
    "rock_roundness_proxy",
    "rock_concavity_proxy",
    "rock_stability_score",
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
    "rock_mass",
]

CATEGORICAL_COLUMNS = [
    "target_name",
    "strategy",
    "gravity",
    "role",
    "source_kind",
    "cluster_label",
]

GROUP_COLUMNS = ["run_name", "target_name", "strategy", "gravity", "trial", "slot_id", "candidate_rock_index"]


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    dataset_dir = args.dataset.resolve()
    rows = read_csv(dataset_dir / "candidate_pose_examples.csv")
    rows = filter_rows(rows, args)
    if not rows:
        raise SystemExit("No candidate pose rows remained after filtering.")

    output_dir = unique_dir(args.output.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)
    train_rows, test_rows, split_info = split_rows(rows, args)
    if not train_rows or not test_rows:
        raise SystemExit("Train/test split produced an empty side.")

    x_train_raw, y_train, schema = build_matrix(train_rows, args)
    x_test_raw, y_test, _ = build_matrix(test_rows, args, schema)
    mean = x_train_raw.mean(axis=0, keepdims=True)
    std = np.where(x_train_raw.std(axis=0, keepdims=True) < 1e-6, 1.0, x_train_raw.std(axis=0, keepdims=True))
    x_train = (x_train_raw - mean) / std
    x_test = (x_test_raw - mean) / std
    schema["feature_mean"] = mean.reshape(-1).tolist()
    schema["feature_std"] = std.reshape(-1).tolist()
    schema["target_columns"] = ["label_pose_risk"]
    schema["risk_label_policy"] = risk_policy(args)

    model, history = train_model(x_train, y_train, args)
    train_prob = predict(model, x_train)
    test_prob = predict(model, x_test)
    train_metrics = binary_metrics(y_train, train_prob)
    test_metrics = binary_metrics(y_test, test_prob)
    group_eval_rows, group_metrics = evaluate_groups(test_rows, test_prob, args)

    export_runtime_npz(output_dir / "pose_risk_net.npz", model)
    write_json(output_dir / "pose_risk_net_schema.json", schema)
    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "row_count": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "split": split_info,
        "positive_count": int(sum(risk_label(row, args) for row in rows)),
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "train": train_metrics,
        "test": test_metrics,
        "test_group": group_metrics,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "history": history,
        "purpose": "PoseRiskNet V1: pre-simulation candidate pose risk penalty for wall stacking.",
        "input_policy": "Inputs use geometry, target, gravity, and candidate pose only; post-simulation fields are labels only.",
    }
    write_json(output_dir / "pose_risk_net_metrics.json", metrics)
    write_csv(output_dir / "group_eval.csv", group_eval_rows)
    write_readme(output_dir, metrics)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight pose-risk net from candidate-pose examples.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-contains", action="append", default=[])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.14)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--split-by-run", action="store_true")
    parser.add_argument("--test-run-name", action="append", default=[])
    parser.add_argument("--seed", type=int, default=643)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-error-limit", type=float, default=0.16)
    parser.add_argument("--target-y-error-limit", type=float, default=0.075)
    parser.add_argument("--disturbance-limit", type=float, default=0.080)
    parser.add_argument("--velocity-limit", type=float, default=0.22)
    parser.add_argument(
        "--candidate-metric-labels",
        action="store_true",
        help=(
            "Label risk from each candidate pose's own post-simulation metrics only. "
            "This ignores committed-success and failure_reason fields, which are slot/run outcomes."
        ),
    )
    return parser.parse_args()


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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


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


def filter_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        if args.target_contains and not any(token in row.get("target_name", "") for token in args.target_contains):
            continue
        if row.get("candidate_rock_index", "") in {"", None}:
            continue
        output.append(row)
    return output


def split_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    if args.test_run_name:
        test_runs = set(args.test_run_name)
    elif args.split_by_run:
        runs = sorted({row.get("run_name", "") for row in rows})
        rng.shuffle(runs)
        test_count = max(1, int(round(len(runs) * args.test_fraction)))
        test_runs = set(runs[:test_count])
    else:
        grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[group_key(row)].append(row)
        keys = list(grouped)
        rng.shuffle(keys)
        test_count = max(1, int(round(len(keys) * args.test_fraction)))
        test_keys = set(keys[:test_count])
        train = [row for key, group in grouped.items() for row in group if key not in test_keys]
        test = [row for key, group in grouped.items() for row in group if key in test_keys]
        return train, test, {"mode": "group", "test_fraction": args.test_fraction, "test_group_count": len(test_keys)}
    train_rows = [row for row in rows if row.get("run_name", "") not in test_runs]
    test_rows = [row for row in rows if row.get("run_name", "") in test_runs]
    return train_rows, test_rows, {
        "mode": "run",
        "test_runs": sorted(test_runs),
        "train_runs": sorted({row.get("run_name", "") for row in train_rows}),
    }


def group_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in GROUP_COLUMNS)


def build_matrix(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    schema: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if schema is None:
        categories = {column: sorted({str(row.get(column, "")) for row in rows}) for column in CATEGORICAL_COLUMNS}
        schema = {
            "task": "binary_pose_risk",
            "numeric_columns": NUMERIC_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "categories": categories,
        }
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    for row in rows:
        features: list[float] = []
        for column in NUMERIC_COLUMNS:
            raw = row.get(column, "")
            features.append(parse_float(raw))
            features.append(0.0 if is_float(raw) else 1.0)
        for column in CATEGORICAL_COLUMNS:
            value = str(row.get(column, ""))
            for category in schema.get("categories", {}).get(column, []):
                features.append(1.0 if value == str(category) else 0.0)
        x_rows.append(features)
        y_rows.append(float(risk_label(row, args)))
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32).reshape(-1, 1), schema


def risk_label(row: dict[str, str], args: argparse.Namespace) -> int:
    committed_success = parse_float(row.get("label_committed_success", "")) > 0.5
    target_error = parse_float(row.get("target_error_xy_m", ""))
    target_y_error = abs(parse_float(row.get("target_y_error_m", "")))
    disturbance = parse_float(row.get("placed_disturbance_xy_m", ""))
    velocity = parse_float(row.get("velocity_inf_norm_after_place", ""))
    failed = str(row.get("failure_reason", "")).strip() != ""
    outcome_risk = False if args.candidate_metric_labels else ((not committed_success) or failed)
    return int(
        outcome_risk
        or target_error > args.target_error_limit
        or target_y_error > args.target_y_error_limit
        or disturbance > args.disturbance_limit
        or velocity > args.velocity_limit
    )


def risk_policy(args: argparse.Namespace) -> dict[str, float]:
    return {
        "target_error_limit": float(args.target_error_limit),
        "target_y_error_limit": float(args.target_y_error_limit),
        "disturbance_limit": float(args.disturbance_limit),
        "velocity_limit": float(args.velocity_limit),
        "candidate_metric_labels": int(bool(args.candidate_metric_labels)),
    }


def train_model(x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    import torch
    from torch import nn

    device = choose_device(torch, args.device)
    model = nn.Sequential(
        nn.Linear(x.shape[1], args.hidden),
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(args.hidden, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    positives = float(np.sum(y >= 0.5))
    negatives = float(np.sum(y < 0.5))
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    tx = torch.from_numpy(x).to(device)
    ty = torch.from_numpy(y).to(device)
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for idx in minibatch_indices(len(x), args.batch_size, rng):
            logits = model(tx[idx])
            loss = criterion(logits, ty[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 10) == 0:
            history.append({"epoch": epoch, "loss": float(np.mean(losses))})
    first: nn.Linear = model[0]  # type: ignore[assignment]
    second: nn.Linear = model[3]  # type: ignore[assignment]
    exported = {
        "w1": first.weight.detach().cpu().numpy().T.astype(np.float64),
        "b1": first.bias.detach().cpu().numpy().reshape(1, -1).astype(np.float64),
        "w2": second.weight.detach().cpu().numpy().T.astype(np.float64),
        "b2": second.bias.detach().cpu().numpy().reshape(1, -1).astype(np.float64),
    }
    return exported, history


def choose_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def minibatch_indices(count: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(count)
    return [order[start : start + batch_size] for start in range(0, count, batch_size)]


def predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    try:
        return predict_torch_cpu(model, x)
    except Exception:
        if len(x) <= 1024:
            return predict_chunk(model, x)
        chunks = []
        for start in range(0, len(x), 1024):
            chunks.append(predict_chunk(model, x[start : start + 1024]))
        return np.vstack(chunks)


def predict_torch_cpu(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    import torch

    device = torch.device("cpu")
    w1 = torch.from_numpy(model["w1"].astype(np.float32, copy=False)).to(device)
    b1 = torch.from_numpy(model["b1"].astype(np.float32, copy=False)).to(device)
    w2 = torch.from_numpy(model["w2"].astype(np.float32, copy=False)).to(device)
    b2 = torch.from_numpy(model["b2"].astype(np.float32, copy=False)).to(device)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            batch = torch.from_numpy(x[start : start + 1024].astype(np.float32, copy=False)).to(device)
            hidden = torch.relu(batch @ w1 + b1)
            logits = hidden @ w2 + b2
            outputs.append(torch.sigmoid(logits).cpu().numpy().astype(np.float64))
    return np.vstack(outputs)


def predict_chunk(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    hidden = np.maximum(np.einsum("ij,jk->ik", x, model["w1"], optimize=False) + model["b1"], 0.0)
    logits = np.einsum("ij,jk->ik", hidden, model["w2"], optimize=False) + model["b2"]
    return sigmoid(logits)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    pred = (y_prob >= 0.5).astype(np.float32)
    tp = float(np.sum((pred == 1.0) & (y_true == 1.0)))
    tn = float(np.sum((pred == 0.0) & (y_true == 0.0)))
    fp = float(np.sum((pred == 1.0) & (y_true == 0.0)))
    fn = float(np.sum((pred == 0.0) & (y_true == 1.0)))
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
        "f1": 2.0 * tp / max(2.0 * tp + fp + fn, 1.0),
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(pred)),
    }


def evaluate_groups(rows: list[dict[str, str]], risk_prob: np.ndarray, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, float]]:
    grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[group_key(row)].append(index)
    eval_rows: list[dict[str, Any]] = []
    top1_safe = top3_safe = total = 0
    for key, indices in grouped.items():
        safe = [idx for idx in indices if risk_label(rows[idx], args) == 0]
        if not safe:
            continue
        total += 1
        ranked = sorted(indices, key=lambda idx: float(risk_prob[idx, 0]))
        top1_hit = int(ranked[0] in safe)
        top3_hit = int(bool(set(ranked[: min(3, len(ranked))]).intersection(safe)))
        top1_safe += top1_hit
        top3_safe += top3_hit
        best = ranked[0]
        eval_rows.append(
            {
                "run_name": key[0],
                "target_name": key[1],
                "strategy": key[2],
                "gravity": key[3],
                "trial": key[4],
                "slot_id": key[5],
                "candidate_rock_index": key[6],
                "candidate_count": len(indices),
                "safe_count": len(safe),
                "top1_safe": top1_hit,
                "top3_safe": top3_hit,
                "top1_candidate_id": rows[best].get("candidate_id", ""),
                "top1_risk_prob": float(risk_prob[best, 0]),
            }
        )
    return eval_rows, {
        "rankable_group_count": total,
        "top1_safe_rate": top1_safe / max(total, 1),
        "top3_safe_rate": top3_safe / max(total, 1),
    }


def export_runtime_npz(path: Path, model: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **model)


def parse_float(value: str | None) -> float:
    return float(value) if is_float(value) else 0.0


def is_float(value: str | None) -> bool:
    if value in {"", None}:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def write_readme(output_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# PoseRiskNet V1",
        "",
        "Purpose: penalize risky candidate poses before or during closed-loop wall placement.",
        "",
        f"- rows: {metrics['row_count']}",
        f"- positives/risky: {metrics['positive_count']}",
        f"- test accuracy: {metrics['test']['accuracy']:.3f}",
        f"- test precision: {metrics['test']['precision']:.3f}",
        f"- test recall: {metrics['test']['recall']:.3f}",
        f"- test f1: {metrics['test']['f1']:.3f}",
        f"- test top1 safe: {metrics['test_group']['top1_safe_rate']:.3f}",
        f"- test top3 safe: {metrics['test_group']['top3_safe_rate']:.3f}",
        "",
        "Input excludes post-simulation fields; risk labels are made from committed success, target error, drift, and velocity.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
