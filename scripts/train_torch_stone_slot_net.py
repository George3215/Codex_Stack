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
    "rock_angularity",
    "rock_spike_score",
    "rock_compactness",
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
    "role",
    "rock_source_kind",
    "rock_cluster_label",
]

GROUP_COLUMNS = ["run_name", "target_name", "slot_id"]


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    dataset_dir = args.dataset.resolve()
    rows = read_csv(dataset_dir / "assignment_candidate_examples.csv")
    rows = filter_rows(rows, args)
    if not rows:
        raise SystemExit("No assignment candidate rows remained after filtering.")

    output_dir = unique_dir(args.output.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)

    train_rows, test_rows, split_info = split_rows(rows, args)
    if not train_rows or not test_rows:
        raise SystemExit("Train/test split produced an empty side.")

    x_train_raw, y_train, schema = build_matrix(train_rows)
    x_test_raw, y_test, _ = build_matrix(test_rows, schema)
    mean = x_train_raw.mean(axis=0, keepdims=True)
    std = np.where(x_train_raw.std(axis=0, keepdims=True) < 1e-6, 1.0, x_train_raw.std(axis=0, keepdims=True))
    x_train = (x_train_raw - mean) / std
    x_test = (x_test_raw - mean) / std
    schema["feature_mean"] = mean.reshape(-1).tolist()
    schema["feature_std"] = std.reshape(-1).tolist()
    schema["target_columns"] = ["label_selected_in_placement_log"]

    model, history = train_model(x_train, y_train, train_rows, args)
    train_prob = predict(model, x_train)
    test_prob = predict(model, x_test)
    train_metrics = binary_metrics(y_train, train_prob)
    test_metrics = binary_metrics(y_test, test_prob)
    group_eval_rows, group_metrics = evaluate_groups(test_rows, test_prob)

    export_runtime_npz(output_dir / "stone_fit_net.npz", model)
    write_json(output_dir / "stone_fit_net_schema.json", schema)
    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "row_count": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "split": split_info,
        "positive_count": int(sum(label_for_row(row) for row in rows)),
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
        "role_balance": bool(args.role_balance),
        "role_weights": parse_role_weights(args.role_weight),
        "history": history,
        "purpose": "StoneSlotNet V1: score whether a stone geometry fits a target wall slot before pose search.",
        "input_policy": "No candidate_rank or primary-assignment fields are used as input.",
    }
    write_json(output_dir / "stone_fit_net_metrics.json", metrics)
    write_csv(output_dir / "group_eval.csv", group_eval_rows)
    write_readme(output_dir, metrics)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch stone-slot fit net and export runtime-compatible MLP weights.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-contains", action="append", default=[])
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--split-by-run", action="store_true")
    parser.add_argument("--test-run-name", action="append", default=[])
    parser.add_argument("--seed", type=int, default=641)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--role-balance", action="store_true", help="Apply inverse role-frequency sample weights.")
    parser.add_argument("--role-weight", action="append", default=[], help="Optional role=weight multiplier, for example middle=1.4.")
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
    output = []
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
        return train, test, {
            "mode": "group",
            "test_fraction": args.test_fraction,
            "test_group_count": len(test_keys),
        }
    train_rows = [row for row in rows if row.get("run_name", "") not in test_runs]
    test_rows = [row for row in rows if row.get("run_name", "") in test_runs]
    return train_rows, test_rows, {
        "mode": "run",
        "test_runs": sorted(test_runs),
        "train_runs": sorted({row.get("run_name", "") for row in train_rows}),
    }


def group_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in GROUP_COLUMNS)


def build_matrix(rows: list[dict[str, str]], schema: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if schema is None:
        categories = {
            column: sorted({str(row.get(column, "")) for row in rows})
            for column in CATEGORICAL_COLUMNS
        }
        schema = {
            "task": "binary",
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
        y_rows.append(float(label_for_row(row)))
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32).reshape(-1, 1), schema


def label_for_row(row: dict[str, str]) -> int:
    return int(parse_float(row.get("selected_count_in_placement_log", "")) > 0.0)


def train_model(x: np.ndarray, y: np.ndarray, rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
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
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    sample_weight = torch.from_numpy(build_sample_weights(rows, args).astype(np.float32)).to(device).reshape(-1, 1)
    tx = torch.from_numpy(x).to(device)
    ty = torch.from_numpy(y).to(device)
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for idx in minibatch_indices(len(x), args.batch_size, rng):
            logits = model(tx[idx])
            raw_loss = criterion(logits, ty[idx])
            loss = (raw_loss * sample_weight[idx]).mean()
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


def build_sample_weights(rows: list[dict[str, str]], args: argparse.Namespace) -> np.ndarray:
    weights = np.ones(len(rows), dtype=np.float64)
    role_multipliers = parse_role_weights(args.role_weight)
    if args.role_balance:
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row.get("role", "")] += 1
        role_count = max(len(counts), 1)
        total = max(len(rows), 1)
        for index, row in enumerate(rows):
            role = row.get("role", "")
            weights[index] *= total / max(role_count * counts.get(role, 1), 1)
    for index, row in enumerate(rows):
        weights[index] *= role_multipliers.get(row.get("role", ""), 1.0)
    mean = float(np.mean(weights)) if len(weights) else 1.0
    if mean > 1e-9:
        weights /= mean
    return weights


def parse_role_weights(entries: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            continue
        key, raw_value = entry.split("=", 1)
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if value > 0.0:
            weights[key.strip()] = value
    return weights


def choose_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def minibatch_indices(count: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(count)
    return [order[start : start + batch_size] for start in range(0, count, batch_size)]


def predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    hidden = np.maximum(x @ model["w1"] + model["b1"], 0.0)
    logits = hidden @ model["w2"] + model["b2"]
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


def evaluate_groups(rows: list[dict[str, str]], prob: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, float]]:
    grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[group_key(row)].append(index)
    eval_rows: list[dict[str, Any]] = []
    top1 = top3 = total = 0
    for key, indices in grouped.items():
        positives = [idx for idx in indices if label_for_row(rows[idx]) == 1]
        if not positives:
            continue
        total += 1
        ranked = sorted(indices, key=lambda idx: -float(prob[idx, 0]))
        top1_hit = int(ranked[0] in positives)
        top3_hit = int(bool(set(ranked[: min(3, len(ranked))]).intersection(positives)))
        top1 += top1_hit
        top3 += top3_hit
        best = ranked[0]
        eval_rows.append(
            {
                "run_name": key[0],
                "target_name": key[1],
                "slot_id": key[2],
                "candidate_count": len(indices),
                "positive_count": len(positives),
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
                "top1_rock": rows[best].get("candidate_rock_index", ""),
                "top1_prob": float(prob[best, 0]),
            }
        )
    return eval_rows, {
        "rankable_group_count": total,
        "top1_hit_rate": top1 / max(total, 1),
        "top3_hit_rate": top3 / max(total, 1),
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
        "# StoneSlotNet V1",
        "",
        "Purpose: choose stones for target wall slots before pose search.",
        "",
        f"- rows: {metrics['row_count']}",
        f"- positives: {metrics['positive_count']}",
        f"- test accuracy: {metrics['test']['accuracy']:.3f}",
        f"- test precision: {metrics['test']['precision']:.3f}",
        f"- test recall: {metrics['test']['recall']:.3f}",
        f"- test f1: {metrics['test']['f1']:.3f}",
        f"- test group top1: {metrics['test_group']['top1_hit_rate']:.3f}",
        f"- test group top3: {metrics['test_group']['top3_hit_rate']:.3f}",
        "",
        "Input excludes candidate assignment rank and primary-assignment flags.",
        "Export is runtime-compatible with `--stone-fit-ranker-dir`.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
