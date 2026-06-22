from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


GROUP_COLUMNS = [
    "run_name",
    "target_name",
    "strategy",
    "gravity",
    "trial",
    "slot_id",
    "candidate_rock_index",
]
QUALITY_TARGET_MODES = {"score", "risk_adjusted", "structure_aware", "interlock_aware", "drift_guarded"}


@dataclass
class TensorBundle:
    maps: np.ndarray
    numeric: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    rows: list[dict[str, str]]
    schema: dict[str, Any]


class GroupIndexDataset(Dataset):
    def __init__(self, groups: list[np.ndarray]) -> None:
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.groups[index]


class SupportMapCNNRanker(nn.Module):
    def __init__(self, map_channels: int, numeric_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=5, padding=2),
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
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 64, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, maps: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        map_features = self.map_encoder(maps)
        numeric_features = self.numeric_encoder(numeric)
        return self.head(torch.cat([map_features, numeric_features], dim=1)).squeeze(-1)


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


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    tensor_dir = args.tensor_dir.resolve()
    output_dir = unique_dir(args.output.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)

    bundle = load_tensor_bundle(tensor_dir, args.rock_embedding_npz, args)
    groups = build_groups(bundle.rows, bundle.labels, target_mode=args.target_mode)
    if not groups:
        raise SystemExit("No rankable groups with positive labels or quality targets found.")

    rng = np.random.default_rng(args.seed)
    split = split_groups(
        groups,
        bundle.rows,
        args.test_fraction,
        rng,
        split_by_run=args.split_by_run,
        requested_test_runs=args.test_run_name,
    )
    train_loader = make_loader(bundle, split["train"], args, shuffle=True)
    test_loader = make_loader(bundle, split["test"], args, shuffle=False)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SupportMapCNNRanker(
        map_channels=bundle.maps.shape[1],
        numeric_dim=bundle.numeric.shape[1],
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_metric = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args)
        scheduler.step()
        test_metrics, _ = evaluate(model, test_loader, bundle.rows, device, target_mode=args.target_mode)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_top1": test_metrics["top1_hit_rate"],
                "test_top3": test_metrics["top3_hit_rate"],
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )
        score = test_metrics["top3_hit_rate"] + 0.25 * test_metrics["top1_hit_rate"]
        if score > best_metric:
            best_metric = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, eval_rows = evaluate(model, test_loader, bundle.rows, device, target_mode=args.target_mode)
    train_metrics, _ = evaluate(
        model,
        make_loader(bundle, split["train"], args, shuffle=False),
        bundle.rows,
        device,
        target_mode=args.target_mode,
    )
    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tensor_dir": str(tensor_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "row_count": len(bundle.rows),
        "rankable_group_count": len(groups),
        "train_groups": len(split["train"]),
        "test_groups": len(split["test"]),
        "split_by_run": bool(args.split_by_run),
        "train_runs": split.get("train_runs", []),
        "test_runs": split.get("test_runs", []),
        "map_shape": list(bundle.maps.shape[1:]),
        "numeric_dim": int(bundle.numeric.shape[1]),
        "rock_embedding_npz": str(args.rock_embedding_npz.resolve()) if args.rock_embedding_npz else "",
        "exclude_postsim_features": bool(args.exclude_postsim_features),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "target_mode": args.target_mode,
        "quality_temperature": args.quality_temperature,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "group_role_weights": parse_weight_entries(args.group_role_weight),
        "group_course_weights": parse_weight_entries(args.group_course_weight),
        "amp": bool(args.amp and device.type == "cuda"),
        "test_top1_hit_rate": test_metrics["top1_hit_rate"],
        "test_top3_hit_rate": test_metrics["top3_hit_rate"],
        "train_top1_hit_rate": train_metrics["top1_hit_rate"],
        "train_top3_hit_rate": train_metrics["top3_hit_rate"],
        "by_target_gravity": test_metrics["by_target_gravity"],
        "architecture": "CNN map encoder + numeric MLP + groupwise softmax ranking loss",
    }
    if "mean_top1_quality_regret" in test_metrics:
        metrics["test_mean_top1_quality_regret"] = test_metrics["mean_top1_quality_regret"]
        metrics["test_mean_top3_quality_regret"] = test_metrics["mean_top3_quality_regret"]
        metrics["train_mean_top1_quality_regret"] = train_metrics["mean_top1_quality_regret"]
        metrics["train_mean_top3_quality_regret"] = train_metrics["mean_top3_quality_regret"]

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "schema": bundle.schema,
            "args": vars(args),
        },
        output_dir / "support_map_cnn_ranker.pt",
    )
    write_csv(output_dir / "group_eval.csv", eval_rows)
    write_csv(output_dir / "history.csv", history)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_dir / "schema.json").write_text(json.dumps(bundle.schema, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_readme(output_dir, metrics)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch CNN groupwise ranker on local support maps.")
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-by-run", action="store_true", help="Hold out whole run_name/catalogs instead of random groups.")
    parser.add_argument("--test-run-name", action="append", default=[], help="Explicit run_name to hold out; repeatable.")
    parser.add_argument("--seed", type=int, default=620)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA AMP mixed precision when available.")
    parser.add_argument("--rock-embedding-npz", type=Path, default=None, help="Optional PointNet embedding NPZ to append by run_name and candidate_rock_index.")
    parser.add_argument(
        "--target-mode",
        choices=("selected", "score", "risk_adjusted", "structure_aware", "interlock_aware", "drift_guarded"),
        default="selected",
        help=(
            "Train against selected labels, candidate_score quality targets, risk-adjusted, "
            "wall-structure-aware, interlock-aware, or drift-guarded targets."
        ),
    )
    parser.add_argument(
        "--quality-temperature",
        type=float,
        default=35.0,
        help="Softmax temperature for --target-mode score. Larger values make quality targets less peaky.",
    )
    parser.add_argument(
        "--exclude-postsim-features",
        action="store_true",
        help="Drop candidate_score/support metrics that are only known after MuJoCo candidate simulation.",
    )
    parser.add_argument(
        "--group-role-weight",
        action="append",
        default=[],
        help="Optional role=weight multiplier for whole candidate groups, for example middle=1.5.",
    )
    parser.add_argument(
        "--group-course-weight",
        action="append",
        default=[],
        help="Optional course=weight multiplier for whole candidate groups, for example 3=1.8.",
    )
    return parser.parse_args()


def load_tensor_bundle(tensor_dir: Path, rock_embedding_npz: Path | None, args: argparse.Namespace) -> TensorBundle:
    rows = read_csv(tensor_dir / "examples_index.csv")
    if not rows:
        raise SystemExit(f"No examples_index.csv found in {tensor_dir}")
    summary_path = tensor_dir / "summary.json"
    tensor_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    numeric_feature_names = list(tensor_summary.get("numeric_features", []))
    keep_indices, kept_numeric_feature_names = numeric_feature_selection(
        numeric_feature_names,
        exclude_postsim_features=bool(args.exclude_postsim_features),
    )
    maps_parts: list[np.ndarray] = []
    numeric_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    shard_files = sorted({row.get("shard_file", "") for row in rows if row.get("shard_file", "")})
    for shard_file in shard_files:
        data = np.load(tensor_dir / shard_file)
        maps_parts.append(np.asarray(data["maps"], dtype=np.float32))
        numeric = np.asarray(data["numeric_features"], dtype=np.float32)
        present = np.asarray(data["numeric_feature_present"], dtype=np.float32)
        numeric = numeric[:, keep_indices]
        present = present[:, keep_indices]
        numeric_parts.append(np.concatenate([numeric, present], axis=1))
        selected = np.asarray(data["label_selected_by_pose_search"], dtype=np.float32)
        quality = np.asarray(data["candidate_score"], dtype=np.float32)
        if args.target_mode == "risk_adjusted":
            start = sum(part.shape[0] for part in score_parts)
            shard_rows = rows[start : start + quality.shape[0]]
            label_parts.append(risk_adjusted_quality_targets(shard_rows, quality))
        elif args.target_mode == "structure_aware":
            start = sum(part.shape[0] for part in score_parts)
            shard_rows = rows[start : start + quality.shape[0]]
            label_parts.append(structure_aware_quality_targets(shard_rows, quality))
        elif args.target_mode == "interlock_aware":
            start = sum(part.shape[0] for part in score_parts)
            shard_rows = rows[start : start + quality.shape[0]]
            label_parts.append(interlock_aware_quality_targets(shard_rows, quality))
        elif args.target_mode == "drift_guarded":
            start = sum(part.shape[0] for part in score_parts)
            shard_rows = rows[start : start + quality.shape[0]]
            label_parts.append(drift_guarded_quality_targets(shard_rows, quality))
        else:
            label_parts.append(quality if args.target_mode == "score" else selected)
        score_parts.append(quality)

    maps = np.concatenate(maps_parts, axis=0)
    numeric = np.concatenate(numeric_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    scores = np.concatenate(score_parts, axis=0)
    if maps.shape[0] != len(rows):
        raise SystemExit(f"Tensor/index mismatch: {maps.shape[0]} tensor rows vs {len(rows)} CSV rows.")
    embedding_info = {"path": "", "dim": 0, "matched_rows": 0}
    if rock_embedding_npz is not None:
        embeddings, embedding_info = load_rock_embeddings(rock_embedding_npz.resolve(), rows)
        numeric = np.concatenate([numeric, embeddings], axis=1)

    map_mean = maps.mean(axis=(0, 2, 3), keepdims=True)
    map_std = maps.std(axis=(0, 2, 3), keepdims=True)
    map_std = np.where(map_std < 1e-6, 1.0, map_std)
    maps = (maps - map_mean) / map_std

    numeric_mean = numeric.mean(axis=0, keepdims=True)
    numeric_std = numeric.std(axis=0, keepdims=True)
    numeric_std = np.where(numeric_std < 1e-6, 1.0, numeric_std)
    numeric = (numeric - numeric_mean) / numeric_std

    schema = {
        "source_tensor_dir": str(tensor_dir),
        "channels": tensor_summary.get("channels", []),
        "window_m": tensor_summary.get("window_m", 0.9),
        "front_height_m": tensor_summary.get("front_height_m", 0.55),
        "numeric_features": kept_numeric_feature_names,
        "numeric_layout": "numeric_features concatenated with numeric_feature_present mask and optional rock embedding",
        "rock_embedding": embedding_info,
        "group_columns": GROUP_COLUMNS,
        "label_column": target_label_column(args.target_mode),
        "target_mode": args.target_mode,
        "quality_temperature": args.quality_temperature,
        "map_mean": map_mean.reshape(-1).tolist(),
        "map_std": map_std.reshape(-1).tolist(),
        "numeric_mean": numeric_mean.reshape(-1).tolist(),
        "numeric_std": numeric_std.reshape(-1).tolist(),
    }
    return TensorBundle(maps=maps, numeric=numeric, labels=labels, scores=scores, rows=rows, schema=schema)


def numeric_feature_selection(
    numeric_feature_names: list[Any],
    exclude_postsim_features: bool,
) -> tuple[list[int], list[str]]:
    names = [str(name) for name in numeric_feature_names]
    if exclude_postsim_features:
        forbidden = POSTSIM_NUMERIC_FEATURES.union(SUPERVISION_ONLY_FEATURES)
        keep_indices = [index for index, name in enumerate(names) if name not in forbidden]
        kept = [names[index] for index in keep_indices]
        leaked = sorted(set(kept).intersection(forbidden))
        if leaked:
            raise SystemExit(f"Post-simulation fields remained in numeric inputs: {leaked}")
        return keep_indices, kept
    return list(range(len(names))), names


def load_rock_embeddings(path: Path, rows: list[dict[str, str]]) -> tuple[np.ndarray, dict[str, Any]]:
    data = np.load(path)
    raw_embeddings = np.asarray(data["embedding"], dtype=np.float32)
    run_names = [str(value) for value in data["run_name"].tolist()]
    rock_indices = [int(value) for value in data["rock_index"].tolist()]
    by_key = {(run_name, rock_index): raw_embeddings[index] for index, (run_name, rock_index) in enumerate(zip(run_names, rock_indices))}
    embeddings = np.zeros((len(rows), raw_embeddings.shape[1] + 1), dtype=np.float32)
    matched = 0
    for row_index, row in enumerate(rows):
        key = (row.get("run_name", ""), parse_int(row.get("candidate_rock_index", "")))
        embedding = by_key.get(key)
        if embedding is None:
            continue
        embeddings[row_index, : raw_embeddings.shape[1]] = embedding
        embeddings[row_index, -1] = 1.0
        matched += 1
    return embeddings, {"path": str(path), "dim": int(raw_embeddings.shape[1]), "matched_rows": matched}


def parse_int(value: str | None) -> int:
    if value in {"", None}:
        return -1
    try:
        return int(float(value))
    except ValueError:
        return -1


def parse_weight_entries(entries: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"Invalid weight entry {entry!r}; expected key=value.")
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid weight entry {entry!r}; empty key.")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise SystemExit(f"Invalid weight value in {entry!r}.") from exc
        if not np.isfinite(value) or value <= 0.0:
            raise SystemExit(f"Invalid weight value in {entry!r}; expected positive finite number.")
        weights[key] = value
    return weights


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value in {"", None}:
        return default
    try:
        output = float(value)
    except ValueError:
        return default
    return output if np.isfinite(output) else default


def target_label_column(target_mode: str) -> str:
    if target_mode == "score":
        return "candidate_score"
    if target_mode == "risk_adjusted":
        return "candidate_score_minus_error_disturbance_velocity_risk"
    if target_mode == "structure_aware":
        return "candidate_score_minus_wall_error_drift_risk"
    if target_mode == "interlock_aware":
        return "candidate_score_minus_interlock_and_future_support_risk"
    if target_mode == "drift_guarded":
        return "candidate_score_minus_drift_velocity_support_risk"
    return "label_selected_by_pose_search"


def risk_adjusted_quality_targets(rows: list[dict[str, str]], quality: np.ndarray) -> np.ndarray:
    adjusted = np.asarray(quality, dtype=np.float32).copy()
    for index, row in enumerate(rows):
        target_error = parse_float(row.get("target_error_xy_m", ""), 0.30)
        disturbance = parse_float(row.get("placed_disturbance_xy_m", ""), 0.05)
        velocity = parse_float(row.get("velocity_inf_norm_after_place", ""), 0.0)
        height_gain = parse_float(row.get("height_gain_m", ""), 0.0)
        radial_distance = parse_float(row.get("radial_distance_m", ""), 0.0)
        risk_penalty = (
            70.0 * min(max(target_error, 0.0), 1.0)
            + 45.0 * min(max(disturbance, 0.0), 0.35)
            + 8.0 * np.log1p(max(velocity, 0.0))
            + 10.0 * max(0.0, radial_distance - 0.55)
            + 25.0 * max(0.0, 0.02 - height_gain)
        )
        adjusted[index] = float(adjusted[index] - risk_penalty)
    return adjusted


def structure_aware_quality_targets(rows: list[dict[str, str]], quality: np.ndarray) -> np.ndarray:
    adjusted = risk_adjusted_quality_targets(rows, quality)
    for index, row in enumerate(rows):
        target_x_error = abs(parse_float(row.get("target_x_error_m", ""), 0.20))
        target_y_error = abs(parse_float(row.get("target_y_error_m", ""), 0.20))
        height_gain = parse_float(row.get("height_gain_m", ""), 0.0)
        role = row.get("role", "")
        course = parse_int(row.get("course", ""))
        wall_penalty = (
            30.0 * min(max(target_x_error, 0.0), 0.50)
            + 125.0 * min(max(target_y_error, 0.0), 0.50)
            + 18.0 * max(0.0, target_y_error - 0.08)
        )
        if role in {"middle", "cap"} or course >= 1:
            wall_penalty += 28.0 * max(0.0, 0.025 - height_gain)
        if role == "cap" or course >= 3:
            wall_penalty += 24.0 * max(0.0, target_y_error - 0.06)
        adjusted[index] = float(adjusted[index] - wall_penalty)
    return adjusted


def interlock_aware_quality_targets(rows: list[dict[str, str]], quality: np.ndarray) -> np.ndarray:
    """Quality target for candidates that should remain useful to the next courses.

    The input at runtime still excludes post-simulation fields when requested.
    These terms are labels only: they encourage the support-map encoder to learn
    local interlock and future support affordances from the observed wall state.
    """
    adjusted = structure_aware_quality_targets(rows, quality)
    for index, row in enumerate(rows):
        course = parse_int(row.get("course", ""))
        role = row.get("role", "")
        if course <= 0:
            continue

        support_overlap = parse_float(row.get("support_overlap", ""), 0.0)
        support_contact_count = parse_float(row.get("support_contact_count", ""), 0.0)
        support_balance_error = parse_float(row.get("support_balance_error_m", ""), 0.16)
        target_y_error = abs(parse_float(row.get("target_y_error_m", ""), 0.16))
        height_gain = parse_float(row.get("height_gain_m", ""), 0.0)

        min_overlap = 0.24
        if course >= 2:
            min_overlap = 0.28
        if role == "cap" or course >= 3:
            min_overlap = 0.31

        desired_contacts = 1.0
        if role in {"middle", "cap"} or course >= 2:
            desired_contacts = 2.0

        interlock_penalty = (
            140.0 * max(0.0, min_overlap - support_overlap)
            + 34.0 * max(0.0, desired_contacts - support_contact_count)
            + 155.0 * min(max(support_balance_error, 0.0), 0.28)
            + 36.0 * max(0.0, 0.035 - height_gain)
        )

        if role in {"middle", "cap"} or course >= 2:
            interlock_penalty += 80.0 * max(0.0, target_y_error - 0.055)
        if role == "cap" or course >= 3:
            interlock_penalty += 42.0 * max(0.0, 0.050 - height_gain)

        interlock_reward = 10.0 * min(max(support_overlap, 0.0), 0.45) + 4.0 * min(max(support_contact_count, 0.0), 3.0)
        adjusted[index] = float(adjusted[index] + interlock_reward - interlock_penalty)
    return adjusted


def drift_guarded_quality_targets(rows: list[dict[str, str]], quality: np.ndarray) -> np.ndarray:
    """Quality target for lunar wall placements where strict success is drift-limited.

    The penalties below are labels only. When --exclude-postsim-features is used,
    runtime inputs still contain only pre-placement geometry, target, gravity,
    support-map observation, and candidate pose information.
    """
    adjusted = structure_aware_quality_targets(rows, quality)
    for index, row in enumerate(rows):
        course = parse_int(row.get("course", ""))
        role = row.get("role", "")
        gravity_ratio = parse_float(row.get("gravity_m_s2", ""), 1.624) / 9.80665
        moon_factor = 1.35 if gravity_ratio < 0.35 else 1.0

        target_y_error = abs(parse_float(row.get("target_y_error_m", ""), 0.18))
        disturbance = parse_float(row.get("placed_disturbance_xy_m", ""), 0.08)
        velocity = parse_float(row.get("velocity_inf_norm_after_place", ""), 0.0)
        support_overlap = parse_float(row.get("support_overlap", ""), 0.0)
        support_contacts = parse_float(row.get("support_contact_count", ""), 0.0)
        support_balance_error = parse_float(row.get("support_balance_error_m", ""), 0.18)
        height_gain = parse_float(row.get("height_gain_m", ""), 0.0)
        radial_distance = parse_float(row.get("radial_distance_m", ""), 0.0)

        course_factor = 1.0 + 0.22 * max(course, 0)
        if role == "middle":
            course_factor += 0.20
        elif role == "cap":
            course_factor += 0.35

        desired_overlap = 0.24 + 0.025 * max(course - 1, 0)
        if role == "cap":
            desired_overlap += 0.035
        desired_contacts = 2.0 if role in {"middle", "cap"} or course >= 2 else 1.0

        drift_penalty = moon_factor * course_factor * (
            180.0 * max(0.0, target_y_error - 0.055)
            + 92.0 * max(0.0, disturbance - 0.045)
            + 18.0 * np.log1p(max(velocity - 0.10, 0.0))
            + 150.0 * max(0.0, desired_overlap - support_overlap)
            + 36.0 * max(0.0, desired_contacts - support_contacts)
            + 180.0 * min(max(support_balance_error, 0.0), 0.25)
            + 32.0 * max(0.0, 0.035 - height_gain)
            + 22.0 * max(0.0, radial_distance - 0.48)
        )
        support_reward = 12.0 * min(max(support_overlap, 0.0), 0.45) + 5.0 * min(max(support_contacts, 0.0), 3.0)
        adjusted[index] = float(adjusted[index] + support_reward - drift_penalty)
    return adjusted


def build_groups(rows: list[dict[str, str]], labels: np.ndarray, target_mode: str) -> list[np.ndarray]:
    grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = tuple(str(row.get(column, "")) for column in GROUP_COLUMNS)
        grouped[key].append(index)
    groups = [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]
    if target_mode in QUALITY_TARGET_MODES:
        kept = []
        for group in groups:
            values = labels[group]
            if len(group) >= 2 and np.all(np.isfinite(values)) and float(np.max(values) - np.min(values)) > 1e-6:
                kept.append(group)
        return kept
    return [group for group in groups if len(group) >= 2 and np.any(labels[group] > 0.5)]


def split_groups(
    groups: list[np.ndarray],
    rows: list[dict[str, str]],
    test_fraction: float,
    rng: np.random.Generator,
    split_by_run: bool,
    requested_test_runs: list[str],
) -> dict[str, Any]:
    if requested_test_runs:
        requested = set(requested_test_runs)
        train_groups = [group for group in groups if rows[int(group[0])].get("run_name", "") not in requested]
        test_groups = [group for group in groups if rows[int(group[0])].get("run_name", "") in requested]
        if not test_groups:
            raise SystemExit(f"No rankable groups found for requested test runs: {sorted(requested)}")
        return {
            "train": train_groups,
            "test": test_groups,
            "train_runs": sorted({rows[int(group[0])].get("run_name", "") for group in train_groups}),
            "test_runs": sorted({rows[int(group[0])].get("run_name", "") for group in test_groups}),
        }
    if split_by_run:
        run_names = sorted({rows[int(group[0])].get("run_name", "") for group in groups})
        run_indices = np.arange(len(run_names))
        rng.shuffle(run_indices)
        test_run_count = max(1, int(round(len(run_names) * test_fraction)))
        test_runs = {run_names[int(index)] for index in run_indices[:test_run_count]}
        train_groups = [group for group in groups if rows[int(group[0])].get("run_name", "") not in test_runs]
        test_groups = [group for group in groups if rows[int(group[0])].get("run_name", "") in test_runs]
        return {
            "train": train_groups,
            "test": test_groups,
            "train_runs": sorted(set(run_names).difference(test_runs)),
            "test_runs": sorted(test_runs),
        }
    indices = np.arange(len(groups))
    rng.shuffle(indices)
    test_count = max(1, int(round(len(indices) * test_fraction)))
    test_ids = set(int(idx) for idx in indices[:test_count])
    return {
        "train": [group for i, group in enumerate(groups) if i not in test_ids],
        "test": [group for i, group in enumerate(groups) if i in test_ids],
        "train_runs": sorted({rows[int(group[0])].get("run_name", "") for i, group in enumerate(groups) if i not in test_ids}),
        "test_runs": sorted({rows[int(group[0])].get("run_name", "") for i, group in enumerate(groups) if i in test_ids}),
    }


def make_loader(bundle: TensorBundle, groups: list[np.ndarray], args: argparse.Namespace, shuffle: bool) -> DataLoader:
    dataset = GroupIndexDataset(groups)
    role_weights = parse_weight_entries(args.group_role_weight)
    course_weights = parse_weight_entries(args.group_course_weight)

    def collate(batch: list[np.ndarray]) -> dict[str, torch.Tensor]:
        batch_size = len(batch)
        max_count = max(len(group) for group in batch)
        maps = np.zeros((batch_size, max_count, *bundle.maps.shape[1:]), dtype=np.float32)
        numeric = np.zeros((batch_size, max_count, bundle.numeric.shape[1]), dtype=np.float32)
        labels = np.zeros((batch_size, max_count), dtype=np.float32)
        mask = np.zeros((batch_size, max_count), dtype=bool)
        row_indices = np.full((batch_size, max_count), -1, dtype=np.int64)
        group_weights = np.ones((batch_size,), dtype=np.float32)
        for batch_index, group in enumerate(batch):
            count = len(group)
            maps[batch_index, :count] = bundle.maps[group]
            numeric[batch_index, :count] = bundle.numeric[group]
            labels[batch_index, :count] = bundle.labels[group]
            mask[batch_index, :count] = True
            row_indices[batch_index, :count] = group
            first_row = bundle.rows[int(group[0])]
            role = str(first_row.get("role", ""))
            course = str(parse_int(first_row.get("course", "")))
            group_weights[batch_index] *= role_weights.get(role, 1.0)
            group_weights[batch_index] *= course_weights.get(course, 1.0)
        return {
            "maps": torch.from_numpy(maps),
            "numeric": torch.from_numpy(numeric),
            "labels": torch.from_numpy(labels),
            "mask": torch.from_numpy(mask),
            "row_indices": torch.from_numpy(row_indices),
            "group_weights": torch.from_numpy(group_weights),
        }

    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        maps, numeric, labels, mask = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            scores = score_batch(model, maps, numeric, mask)
            group_weights = batch["group_weights"].to(device, non_blocking=True)
            loss = groupwise_loss(
                scores,
                labels,
                mask,
                args.target_mode,
                args.quality_temperature,
                group_weights=group_weights,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    rows: list[dict[str, str]],
    device: torch.device,
    target_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    eval_rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    top1_hits = 0
    top3_hits = 0
    top1_regrets: list[float] = []
    top3_regrets: list[float] = []
    for batch in loader:
        maps, numeric, labels, mask = batch_to_device(batch, device)
        scores = score_batch(model, maps, numeric, mask).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        row_indices = batch["row_indices"].numpy()
        for batch_index in range(scores.shape[0]):
            valid_positions = np.flatnonzero(mask_np[batch_index])
            if len(valid_positions) == 0:
                continue
            selected_positions = target_positions(labels_np[batch_index], valid_positions, target_mode)
            if not selected_positions:
                continue
            ranked_positions = sorted(valid_positions, key=lambda pos: -float(scores[batch_index, pos]))
            top1 = set(ranked_positions[:1])
            top3 = set(ranked_positions[: min(3, len(ranked_positions))])
            selected = set(selected_positions)
            hit1 = int(bool(selected.intersection(top1)))
            hit3 = int(bool(selected.intersection(top3)))
            quality_like_mode = target_mode in QUALITY_TARGET_MODES
            true_best_score = float(np.max(labels_np[batch_index, valid_positions])) if quality_like_mode else 0.0
            top1_quality = float(labels_np[batch_index, ranked_positions[0]]) if quality_like_mode else 0.0
            top3_quality = (
                float(np.max(labels_np[batch_index, ranked_positions[: min(3, len(ranked_positions))]]))
                if quality_like_mode
                else 0.0
            )
            top1_regret = max(0.0, true_best_score - top1_quality) if quality_like_mode else 0.0
            top3_regret = max(0.0, true_best_score - top3_quality) if quality_like_mode else 0.0
            top1_hits += hit1
            top3_hits += hit3
            if quality_like_mode:
                top1_regrets.append(top1_regret)
                top3_regrets.append(top3_regret)
            first_row = rows[int(row_indices[batch_index, valid_positions[0]])]
            top1_row = rows[int(row_indices[batch_index, ranked_positions[0]])]
            item = {
                "run_name": first_row.get("run_name", ""),
                "target_name": first_row.get("target_name", ""),
                "strategy": first_row.get("strategy", ""),
                "gravity": first_row.get("gravity", ""),
                "trial": first_row.get("trial", ""),
                "slot_id": first_row.get("slot_id", ""),
                "candidate_rock_index": first_row.get("candidate_rock_index", ""),
                "candidate_count": len(valid_positions),
                "selected_candidate_ids": " ".join(
                    rows[int(row_indices[batch_index, pos])].get("candidate_id", "") for pos in selected_positions
                ),
                "net_top1_hit": hit1,
                "net_top3_hit": hit3,
                "net_top1_candidate_id": top1_row.get("candidate_id", ""),
                "net_top1_score": float(scores[batch_index, ranked_positions[0]]),
                "target_mode": target_mode,
                "true_best_quality_score": true_best_score if quality_like_mode else "",
                "net_top1_quality_score": top1_quality if quality_like_mode else "",
                "net_top1_quality_regret": top1_regret if quality_like_mode else "",
                "net_top3_quality_regret": top3_regret if quality_like_mode else "",
            }
            by_key[(item["target_name"], item["gravity"])].append(item)
            eval_rows.append(item)
    group_count = max(len(eval_rows), 1)
    by_target_gravity: dict[str, dict[str, Any]] = {}
    for (target, gravity), items in sorted(by_key.items()):
        count = max(len(items), 1)
        by_target_gravity[f"{target}|{gravity}"] = {
            "groups": len(items),
            "top1_hit_rate": sum(int(item["net_top1_hit"]) for item in items) / count,
            "top3_hit_rate": sum(int(item["net_top3_hit"]) for item in items) / count,
        }
        if target_mode in QUALITY_TARGET_MODES:
            by_target_gravity[f"{target}|{gravity}"]["mean_top1_quality_regret"] = sum(
                float(item["net_top1_quality_regret"]) for item in items
            ) / count
            by_target_gravity[f"{target}|{gravity}"]["mean_top3_quality_regret"] = sum(
                float(item["net_top3_quality_regret"]) for item in items
            ) / count
    return {
        "groups": len(eval_rows),
        "top1_hit_rate": top1_hits / group_count,
        "top3_hit_rate": top3_hits / group_count,
        "mean_top1_quality_regret": float(np.mean(top1_regrets)) if top1_regrets else 0.0,
        "mean_top3_quality_regret": float(np.mean(top3_regrets)) if top3_regrets else 0.0,
        "by_target_gravity": by_target_gravity,
    }, eval_rows


def score_batch(model: nn.Module, maps: torch.Tensor, numeric: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch_size, max_count = maps.shape[:2]
    flat_maps = maps.reshape(batch_size * max_count, *maps.shape[2:])
    flat_numeric = numeric.reshape(batch_size * max_count, numeric.shape[-1])
    flat_scores = model(flat_maps, flat_numeric)
    scores = flat_scores.reshape(batch_size, max_count).float()
    return scores.masked_fill(~mask, -1.0e4)


def target_positions(labels: np.ndarray, valid_positions: np.ndarray, target_mode: str) -> list[int]:
    if target_mode in QUALITY_TARGET_MODES:
        values = labels[valid_positions]
        if len(values) == 0 or not np.all(np.isfinite(values)):
            return []
        best = float(np.max(values))
        return [int(pos) for pos in valid_positions if float(labels[pos]) >= best - 1e-6]
    return [int(pos) for pos in valid_positions if labels[pos] > 0.5]


def groupwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    target_mode: str,
    quality_temperature: float,
    group_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if target_mode in QUALITY_TARGET_MODES:
        temperature = max(float(quality_temperature), 1e-6)
        target_logits = (labels / temperature).masked_fill(~mask, -1.0e4)
        targets = torch.softmax(target_logits, dim=1)
    else:
        positive_counts = (labels * mask.float()).sum(dim=1, keepdim=True).clamp_min(1.0)
        targets = labels / positive_counts
    log_probs = torch.log_softmax(scores, dim=1)
    per_group_loss = -(targets * log_probs).sum(dim=1)
    if group_weights is None:
        return per_group_loss.mean()
    weights = group_weights.float()
    weights = weights / weights.mean().clamp_min(1e-6)
    return (per_group_loss * weights).mean()


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    maps = batch["maps"].to(device, non_blocking=True)
    numeric = batch["numeric"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    return maps, numeric, labels, mask


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
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


def write_readme(output_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# PyTorch Support-Map CNN Ranker",
        "",
        "This model ranks candidate stone poses within each slot/candidate-rock group.",
        "",
        "Architecture:",
        "",
        "- CNN encoder over local support maps.",
        "- MLP encoder over candidate pose, stone geometry, support metrics, and missing-value mask.",
        f"- Score head trained with groupwise softmax loss; target mode: {metrics.get('target_mode', 'selected')}.",
        "",
        f"- tensor dir: `{metrics['tensor_dir']}`",
        f"- device: {metrics['device']}",
        f"- torch: {metrics['torch_version']}",
        f"- GPU: {metrics['gpu_name']}",
        f"- rows: {metrics['row_count']}",
        f"- rankable groups: {metrics['rankable_group_count']}",
        f"- test top-1 hit: {metrics['test_top1_hit_rate']:.3f}",
        f"- test top-3 hit: {metrics['test_top3_hit_rate']:.3f}",
        f"- train top-1 hit: {metrics['train_top1_hit_rate']:.3f}",
        f"- train top-3 hit: {metrics['train_top3_hit_rate']:.3f}",
    ]
    if "test_mean_top1_quality_regret" in metrics:
        lines.extend(
            [
                f"- test mean top-1 quality regret: {metrics['test_mean_top1_quality_regret']:.3f}",
                f"- test mean top-3 quality regret: {metrics['test_mean_top3_quality_regret']:.3f}",
            ]
        )
    lines.extend(
        [
            "",
            "In `selected` mode, labels imitate the current heuristic search. In `score` mode, labels use post-simulation candidate quality as supervision while keeping post-simulation fields out of the input when `--exclude-postsim-features` is set.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
