from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


GROUP_COLUMNS = [
    "run_name",
    "target_name",
    "strategy",
    "gravity",
    "trial",
    "slot_id",
    "candidate_rock_index",
]

REGRESSION_TARGETS = [
    "abs_target_x_error_m",
    "abs_target_y_error_m",
    "placed_disturbance_xy_m",
    "log_velocity_inf_norm_after_place",
    "height_gain_m",
]

POSTSIM_NUMERIC_FEATURES = {
    "candidate_score",
    "support_overlap",
    "support_contact_count",
    "support_balance_error_m",
    "bearing_pressure_proxy",
}

SUPERVISION_ONLY_FEATURES = {
    "label_success",
    "label_run_success",
    "label_run_shape_success",
    "label_selected_by_pose_search",
    "label_committed_success",
    "target_error_xy_m",
    "target_x_error_m",
    "target_y_error_m",
    "radial_distance_m",
    "placed_disturbance_xy_m",
    "velocity_inf_norm_after_place",
    "height_gain_m",
    "success_rate",
    "failure_rate",
}


@dataclass
class Bundle:
    maps: np.ndarray
    numeric: np.ndarray
    reg_targets: np.ndarray
    fail_targets: np.ndarray
    rows: list[dict[str, str]]
    schema: dict[str, Any]


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    tensor_dir = args.tensor_dir.resolve()
    output_dir = unique_dir(args.output.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)

    bundle = load_bundle(tensor_dir, exclude_postsim=args.exclude_postsim_features)
    train_idx, test_idx, split_info = split_indices(bundle.rows, args)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise SystemExit("Train/test split produced an empty side.")

    train_bundle, test_bundle, norm_schema = normalize_bundle(bundle, train_idx, test_idx)
    bundle.schema.update(norm_schema)
    model, history = train_model(train_bundle, test_bundle, args)
    metrics, eval_rows = evaluate_model(model, test_bundle, bundle.rows, test_idx, args)
    train_metrics, _ = evaluate_model(model, train_bundle, bundle.rows, train_idx, args)

    save_model(output_dir, model, bundle.schema, metrics, train_metrics, history, split_info, args, tensor_dir)
    write_csv(output_dir / "group_eval.csv", eval_rows)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN wall-state critic from front/top depth tensors.")
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-by-run", action="store_true")
    parser.add_argument("--test-run-name", action="append", default=[])
    parser.add_argument("--seed", type=int, default=642)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--exclude-postsim-features", action="store_true")
    return parser.parse_args()


def load_bundle(tensor_dir: Path, exclude_postsim: bool) -> Bundle:
    rows = read_csv(tensor_dir / "examples_index.csv")
    if not rows:
        raise SystemExit(f"No examples_index.csv found in {tensor_dir}")
    summary_path = tensor_dir / "summary.json"
    tensor_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    numeric_names = [str(name) for name in tensor_summary.get("numeric_features", [])]
    keep_indices, kept_numeric = numeric_feature_selection(numeric_names, exclude_postsim)
    maps_parts: list[np.ndarray] = []
    numeric_parts: list[np.ndarray] = []
    fail_parts: list[np.ndarray] = []
    shard_files = sorted({row.get("shard_file", "") for row in rows if row.get("shard_file", "")})
    for shard_file in shard_files:
        data = np.load(tensor_dir / shard_file)
        maps_parts.append(np.asarray(data["maps"], dtype=np.float32))
        numeric = np.asarray(data["numeric_features"], dtype=np.float32)[:, keep_indices]
        present = np.asarray(data["numeric_feature_present"], dtype=np.float32)[:, keep_indices]
        numeric_parts.append(np.concatenate([numeric, present], axis=1))
        success = np.asarray(data["label_committed_success"], dtype=np.float32).reshape(-1, 1)
        fail_parts.append(1.0 - success)
    maps = np.concatenate(maps_parts, axis=0)
    numeric = np.concatenate(numeric_parts, axis=0)
    fail_targets = np.concatenate(fail_parts, axis=0)
    reg_targets = np.asarray([regression_targets(row) for row in rows], dtype=np.float32)
    if len(rows) != maps.shape[0]:
        raise SystemExit(f"Tensor/index mismatch: {maps.shape[0]} rows vs {len(rows)} index rows.")
    schema = {
        "source_tensor_dir": str(tensor_dir),
        "channels": tensor_summary.get("channels", []),
        "window_m": tensor_summary.get("window_m", 0.9),
        "front_height_m": tensor_summary.get("front_height_m", 0.55),
        "numeric_features": kept_numeric,
        "numeric_layout": "numeric features concatenated with missing-value mask",
        "regression_targets": REGRESSION_TARGETS,
        "classification_target": "placement_failure_probability",
        "group_columns": GROUP_COLUMNS,
    }
    return Bundle(maps=maps, numeric=numeric, reg_targets=reg_targets, fail_targets=fail_targets, rows=rows, schema=schema)


def numeric_feature_selection(names: list[str], exclude_postsim: bool) -> tuple[list[int], list[str]]:
    if not exclude_postsim:
        return list(range(len(names))), names
    forbidden = POSTSIM_NUMERIC_FEATURES.union(SUPERVISION_ONLY_FEATURES)
    keep = [idx for idx, name in enumerate(names) if name not in forbidden]
    return keep, [names[idx] for idx in keep]


def regression_targets(row: dict[str, str]) -> list[float]:
    return [
        abs(parse_float(row.get("target_x_error_m", ""), 0.20)),
        abs(parse_float(row.get("target_y_error_m", ""), 0.20)),
        parse_float(row.get("placed_disturbance_xy_m", ""), 0.05),
        float(np.log1p(max(0.0, parse_float(row.get("velocity_inf_norm_after_place", ""), 0.0)))),
        parse_float(row.get("height_gain_m", ""), 0.0),
    ]


def split_indices(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    if args.test_run_name:
        test_runs = set(args.test_run_name)
    elif args.split_by_run:
        runs = sorted({row.get("run_name", "") for row in rows})
        rng.shuffle(runs)
        test_count = max(1, int(round(len(runs) * args.test_fraction)))
        test_runs = set(runs[:test_count])
    else:
        indices = rng.permutation(len(rows))
        test_count = max(1, int(round(len(rows) * args.test_fraction)))
        return indices[test_count:], indices[:test_count], {"mode": "row", "test_fraction": args.test_fraction}
    train = [idx for idx, row in enumerate(rows) if row.get("run_name", "") not in test_runs]
    test = [idx for idx, row in enumerate(rows) if row.get("run_name", "") in test_runs]
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64), {
        "mode": "run",
        "test_runs": sorted(test_runs),
        "train_runs": sorted({rows[idx].get("run_name", "") for idx in train}),
    }


def normalize_bundle(bundle: Bundle, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[Bundle, Bundle, dict[str, Any]]:
    map_mean = bundle.maps[train_idx].mean(axis=(0, 2, 3), keepdims=True)
    map_std = np.where(bundle.maps[train_idx].std(axis=(0, 2, 3), keepdims=True) < 1e-6, 1.0, bundle.maps[train_idx].std(axis=(0, 2, 3), keepdims=True))
    num_mean = bundle.numeric[train_idx].mean(axis=0, keepdims=True)
    num_std = np.where(bundle.numeric[train_idx].std(axis=0, keepdims=True) < 1e-6, 1.0, bundle.numeric[train_idx].std(axis=0, keepdims=True))
    target_mean = bundle.reg_targets[train_idx].mean(axis=0, keepdims=True)
    target_std = np.where(bundle.reg_targets[train_idx].std(axis=0, keepdims=True) < 1e-6, 1.0, bundle.reg_targets[train_idx].std(axis=0, keepdims=True))
    norm_maps = (bundle.maps - map_mean) / map_std
    norm_numeric = (bundle.numeric - num_mean) / num_std
    norm_reg = (bundle.reg_targets - target_mean) / target_std
    schema = {
        "map_mean": map_mean.reshape(-1).tolist(),
        "map_std": map_std.reshape(-1).tolist(),
        "numeric_mean": num_mean.reshape(-1).tolist(),
        "numeric_std": num_std.reshape(-1).tolist(),
        "target_mean": target_mean.reshape(-1).tolist(),
        "target_std": target_std.reshape(-1).tolist(),
    }
    train = Bundle(norm_maps[train_idx], norm_numeric[train_idx], norm_reg[train_idx], bundle.fail_targets[train_idx], [bundle.rows[int(i)] for i in train_idx], bundle.schema)
    test = Bundle(norm_maps[test_idx], norm_numeric[test_idx], norm_reg[test_idx], bundle.fail_targets[test_idx], [bundle.rows[int(i)] for i in test_idx], bundle.schema)
    return train, test, schema


def train_model(train: Bundle, test: Bundle, args: argparse.Namespace) -> tuple[Any, list[dict[str, float]]]:
    import torch
    from torch import nn

    device = choose_device(torch, args.device)

    class WallStateCritic(nn.Module):
        def __init__(self, channels: int, numeric_dim: int, hidden: int, dropout: float) -> None:
            super().__init__()
            self.map_encoder = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.numeric_encoder = nn.Sequential(nn.Linear(numeric_dim, 64), nn.LayerNorm(64), nn.SiLU())
            self.trunk = nn.Sequential(nn.Linear(192, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout))
            self.reg_head = nn.Linear(hidden, len(REGRESSION_TARGETS))
            self.fail_head = nn.Linear(hidden, 1)

        def forward(self, maps: Any, numeric: Any) -> tuple[Any, Any]:
            features = torch.cat([self.map_encoder(maps), self.numeric_encoder(numeric)], dim=1)
            hidden = self.trunk(features)
            return self.reg_head(hidden), self.fail_head(hidden)

    model = WallStateCritic(train.maps.shape[1], train.numeric.shape[1], args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    positives = float(np.sum(train.fail_targets >= 0.5))
    negatives = float(np.sum(train.fail_targets < 0.5))
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float]] = []
    train_tensors = bundle_tensors(torch, train, device)
    test_tensors = bundle_tensors(torch, test, device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for idx in minibatch_indices(train.maps.shape[0], args.batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                reg_pred, fail_logit = model(train_tensors["maps"][idx], train_tensors["numeric"][idx])
                loss = mse(reg_pred, train_tensors["reg"][idx]) + 0.65 * bce(fail_logit, train_tensors["fail"][idx])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 10) == 0:
            test_loss = validation_loss(model, test_tensors, mse, bce, args.batch_size, device)
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "test_loss": test_loss})
    return model, history


def validation_loss(model: Any, tensors: dict[str, Any], mse: Any, bce: Any, batch_size: int, device: Any) -> float:
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, tensors["maps"].shape[0], batch_size):
            sl = slice(start, start + batch_size)
            reg_pred, fail_logit = model(tensors["maps"][sl], tensors["numeric"][sl])
            loss = mse(reg_pred, tensors["reg"][sl]) + 0.65 * bce(fail_logit, tensors["fail"][sl])
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def bundle_tensors(torch: Any, bundle: Bundle, device: Any) -> dict[str, Any]:
    return {
        "maps": torch.from_numpy(bundle.maps.astype(np.float32)).to(device),
        "numeric": torch.from_numpy(bundle.numeric.astype(np.float32)).to(device),
        "reg": torch.from_numpy(bundle.reg_targets.astype(np.float32)).to(device),
        "fail": torch.from_numpy(bundle.fail_targets.astype(np.float32)).to(device),
    }


def evaluate_model(model: Any, bundle: Bundle, all_rows: list[dict[str, str]], source_indices: np.ndarray, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    device = next(model.parameters()).device
    tensors = bundle_tensors(torch, bundle, device)
    model.eval()
    reg_parts = []
    fail_parts = []
    with torch.no_grad():
        for start in range(0, bundle.maps.shape[0], args.batch_size):
            sl = slice(start, start + args.batch_size)
            reg_pred, fail_logit = model(tensors["maps"][sl], tensors["numeric"][sl])
            reg_parts.append(reg_pred.detach().cpu().numpy())
            fail_parts.append(torch.sigmoid(fail_logit).detach().cpu().numpy())
    reg_pred_norm = np.concatenate(reg_parts, axis=0)
    fail_prob = np.concatenate(fail_parts, axis=0)
    schema = bundle.schema
    target_mean = np.asarray(schema["target_mean"], dtype=np.float32).reshape(1, -1)
    target_std = np.asarray(schema["target_std"], dtype=np.float32).reshape(1, -1)
    reg_pred = reg_pred_norm * target_std + target_mean
    reg_true = bundle.reg_targets * target_std + target_mean
    reg_metrics = regression_metrics(reg_true, reg_pred)
    cls_metrics = binary_metrics(bundle.fail_targets, fail_prob)
    group_rows, group_metrics = evaluate_groups(bundle.rows, fail_prob, reg_pred, reg_true)
    return {
        "row_count": int(bundle.maps.shape[0]),
        "regression": reg_metrics,
        "classification": cls_metrics,
        "group_ranking": group_metrics,
    }, group_rows


def evaluate_groups(rows: list[dict[str, str]], fail_prob: np.ndarray, reg_pred: np.ndarray, reg_true: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, float]]:
    grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[group_key(row)].append(idx)
    eval_rows: list[dict[str, Any]] = []
    top1 = top3 = total = 0
    for key, indices in grouped.items():
        if len(indices) < 2:
            continue
        total += 1
        true_risk = {
            idx: float(reg_true[idx, 1] * 4.0 + reg_true[idx, 2] * 2.0 + max(0.0, -reg_true[idx, 4]) * 2.5)
            for idx in indices
        }
        pred_risk = {
            idx: float(fail_prob[idx, 0] + reg_pred[idx, 1] * 4.0 + reg_pred[idx, 2] * 2.0 + max(0.0, -reg_pred[idx, 4]) * 2.5)
            for idx in indices
        }
        true_best = min(indices, key=lambda idx: true_risk[idx])
        ranked = sorted(indices, key=lambda idx: pred_risk[idx])
        hit1 = int(ranked[0] == true_best)
        hit3 = int(true_best in ranked[: min(3, len(ranked))])
        top1 += hit1
        top3 += hit3
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
                "top1_hit": hit1,
                "top3_hit": hit3,
                "pred_top1_candidate_id": rows[ranked[0]].get("candidate_id", ""),
                "true_best_candidate_id": rows[true_best].get("candidate_id", ""),
                "pred_top1_risk": pred_risk[ranked[0]],
                "true_best_risk": true_risk[true_best],
            }
        )
    return eval_rows, {
        "rankable_group_count": total,
        "top1_hit_rate": top1 / max(total, 1),
        "top3_hit_rate": top3 / max(total, 1),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    diff = y_pred - y_true
    mae = np.mean(np.abs(diff), axis=0)
    rmse = np.sqrt(np.mean(diff * diff, axis=0))
    return {
        target: {"mae": float(mae[idx]), "rmse": float(rmse[idx]), "mean_true": float(np.mean(y_true[:, idx]))}
        for idx, target in enumerate(REGRESSION_TARGETS)
    }


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


def save_model(
    output_dir: Path,
    model: Any,
    schema: dict[str, Any],
    test_metrics: dict[str, Any],
    train_metrics: dict[str, Any],
    history: list[dict[str, float]],
    split_info: dict[str, Any],
    args: argparse.Namespace,
    tensor_dir: Path,
) -> None:
    import torch

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "schema": schema,
            "metrics": {
                "hidden": args.hidden,
                "dropout": args.dropout,
                "map_shape": list(next(iter(model.state_dict().values())).shape) if model.state_dict() else [],
            },
        },
        output_dir / "wall_state_critic.pt",
    )
    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tensor_dir": str(tensor_dir),
        "output_dir": str(output_dir),
        "split": split_info,
        "train": train_metrics,
        "test": test_metrics,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "amp": bool(args.amp),
        "history": history,
        "purpose": "WallStateCritic V1 predicts post-placement local structure risk from current wall depth and stone geometry.",
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "schema.json", schema)
    write_readme(output_dir, metrics)


def group_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")) for column in GROUP_COLUMNS)


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


def minibatch_indices(count: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(count)
    return [order[start : start + batch_size] for start in range(0, count, batch_size)]


def choose_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value in {"", None}:
        return default
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if np.isfinite(output) else default


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
    train = metrics["train"]
    test = metrics["test"]
    lines = [
        "# WallStateCritic V1",
        "",
        "Purpose: predict local wall-state risk from rendered/proxy front and top depth tensors.",
        "",
        f"- train rows: {train.get('row_count', '')}",
        f"- test rows: {test.get('row_count', '')}",
        f"- test failure accuracy: {test['classification']['accuracy']:.3f}",
        f"- test failure f1: {test['classification']['f1']:.3f}",
        f"- test group top1: {test['group_ranking']['top1_hit_rate']:.3f}",
        f"- test group top3: {test['group_ranking']['top3_hit_rate']:.3f}",
        "",
        "This model is not yet wired into closed-loop placement; it is the first wall-state critic for rejecting or repairing risky placements.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
