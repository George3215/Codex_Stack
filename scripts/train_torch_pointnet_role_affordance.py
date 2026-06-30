from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.pointnet_modules import PointNetBackbone, parse_channels, transform_regularizer
from scripts.train_torch_pointnet_rock_encoder import augment_cloud, load_bundle, unique_dir


ROLES = ("base", "middle", "cap")


@dataclass
class RoleBundle:
    clouds: np.ndarray
    run_name: np.ndarray
    rock_index: np.ndarray
    source_kind: np.ndarray
    cluster_label: np.ndarray
    labels: np.ndarray
    masks: np.ndarray
    obs_weights: np.ndarray
    success_counts: np.ndarray
    trial_counts: np.ndarray


class RoleAffordanceDataset(Dataset):
    def __init__(self, bundle: RoleBundle, indices: np.ndarray, augment: bool) -> None:
        self.bundle = bundle
        self.indices = indices
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        cloud = np.array(self.bundle.clouds[index], dtype=np.float32, copy=True)
        if self.augment:
            cloud = augment_cloud(cloud)
        return {
            "cloud": torch.from_numpy(cloud),
            "labels": torch.from_numpy(self.bundle.labels[index].astype(np.float32)),
            "masks": torch.from_numpy(self.bundle.masks[index].astype(np.float32)),
            "obs_weights": torch.from_numpy(self.bundle.obs_weights[index].astype(np.float32)),
            "index": torch.tensor(index, dtype=torch.long),
        }


class PointNetRoleAffordance(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden: int,
        dropout: float,
        channels: list[int],
        activation: str,
        input_transform: bool,
        feature_transform: bool,
    ) -> None:
        super().__init__()
        self.backbone = PointNetBackbone(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            channels=channels,
            activation=activation,
            input_transform=input_transform,
            feature_transform=feature_transform,
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(ROLES)),
        )

    def forward(self, cloud: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone_output = self.backbone(cloud)
        embedding = backbone_output["embedding"]
        return {
            "embedding": embedding,
            "logits": self.head(embedding),
            "input_transform": backbone_output["input_transform"],
            "feature_transform": backbone_output["feature_transform"],
        }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    configure_torch_runtime(args)
    output_dir = unique_dir(args.output.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)

    bundle = load_role_bundle(args.pointcloud_dir.resolve(), args.dataset.resolve(), args)
    labeled_indices = np.flatnonzero(bundle.masks.sum(axis=1) > 0.0)
    if labeled_indices.size < 8:
        raise SystemExit(f"Only {labeled_indices.size} labeled rocks found; need at least 8.")
    split = split_labeled_by_run(bundle.run_name, labeled_indices, args.test_fraction, np.random.default_rng(args.seed))

    train_loader = make_loader(bundle, split["train"], args, augment=args.augment, shuffle=True)
    test_loader = make_loader(bundle, split["test"], args, augment=False, shuffle=False)
    all_loader = make_loader(bundle, np.arange(bundle.clouds.shape[0]), args, augment=False, shuffle=False)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    channels = parse_channels(args.pointnet_channels, args.embedding_dim)
    model = PointNetRoleAffordance(
        input_dim=bundle.clouds.shape[2],
        embedding_dim=args.embedding_dim,
        hidden=args.hidden,
        dropout=args.dropout,
        channels=channels,
        activation=args.activation,
        input_transform=not args.no_input_transform,
        feature_transform=args.feature_transform,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = make_grad_scaler(args.amp and device.type == "cuda")
    role_weights = role_weight_tensor(args.role_weight, device)

    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, role_weights, device, args)
        scheduler.step()
        test_metrics, _ = evaluate(model, test_loader, bundle, role_weights, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_observed_accuracy": test_metrics["observed_accuracy"],
                "test_observed_f1": test_metrics["observed_f1"],
                "test_observed_mae": test_metrics["observed_mae"],
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )
        score = test_metrics["observed_f1"] - 0.25 * test_metrics["observed_mae"]
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if args.log_every > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(
                f"epoch={epoch} train_loss={train_loss:.5f} "
                f"test_f1={test_metrics['observed_f1']:.4f} "
                f"test_mae={test_metrics['observed_mae']:.4f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics, _ = evaluate(model, train_loader, bundle, role_weights, device)
    test_metrics, test_rows = evaluate(model, test_loader, bundle, role_weights, device)
    all_metrics, all_rows = evaluate(model, all_loader, bundle, role_weights, device)
    export_predictions(output_dir / "rock_role_affordance.npz", all_rows)

    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pointcloud_dir": str(args.pointcloud_dir.resolve()),
        "dataset": str(args.dataset.resolve()),
        "output_dir": str(output_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "rock_count": int(bundle.clouds.shape[0]),
        "labeled_rock_count": int(labeled_indices.size),
        "train_count": int(len(split["train"])),
        "test_count": int(len(split["test"])),
        "test_run_split": sorted(set(bundle.run_name[split["test"]].tolist())),
        "point_count": int(bundle.clouds.shape[1]),
        "input_dim": int(bundle.clouds.shape[2]),
        "roles": list(ROLES),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "embedding_dim": args.embedding_dim,
        "pointnet_channels": channels,
        "activation": args.activation,
        "input_transform": bool(not args.no_input_transform),
        "feature_transform": bool(args.feature_transform),
        "transform_reg_weight": args.transform_reg_weight,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "architecture_detail": {
            "name": "PointNetRoleAffordance",
            "stage": "rock role-affordance predictor for base/middle/cap stone selection",
            "input": "unordered normalized rock surface points; xyz or xyz+normal",
            "output": "three logits/probabilities for base, middle, and cap suitability",
            "shared_mlp_channels": channels,
            "pooling": "global max pooling over points",
            "role_head": "Linear(embedding_dim,hidden)-LayerNorm-SiLU-Dropout-Linear(hidden,3)",
        },
        "hidden": args.hidden,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "amp": bool(args.amp and device.type == "cuda"),
        "augment": bool(args.augment),
        "min_observations": args.min_observations,
        "role_weights": parse_role_weights(args.role_weight),
        "train_observed_accuracy": train_metrics["observed_accuracy"],
        "train_observed_f1": train_metrics["observed_f1"],
        "train_observed_mae": train_metrics["observed_mae"],
        "test_observed_accuracy": test_metrics["observed_accuracy"],
        "test_observed_f1": test_metrics["observed_f1"],
        "test_observed_mae": test_metrics["observed_mae"],
        "test_by_role": test_metrics["by_role"],
        "all_by_role": all_metrics["by_role"],
        "purpose": "Predict base/middle/cap role affordance from 3D PointNet rock point-cloud geometry.",
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "args": vars(args),
            "roles": ROLES,
        },
        output_dir / "pointnet_role_affordance.pt",
    )
    write_csv(output_dir / "history.csv", history)
    write_csv(output_dir / "test_predictions.csv", test_rows)
    write_csv(output_dir / "all_predictions.csv", all_rows)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_readme(output_dir, metrics)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PointNet role-affordance heads from rock point clouds.")
    parser.add_argument("--pointcloud-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="Learning dataset directory with placement_examples.csv.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--pointnet-channels", default="64,64,128", help="Comma-separated shared-MLP widths before final embedding dim.")
    parser.add_argument("--activation", choices=["relu", "silu", "gelu"], default="relu")
    parser.add_argument("--no-input-transform", action="store_true", help="Disable PointNet xyz T-Net canonical alignment.")
    parser.add_argument("--feature-transform", action="store_true", help="Enable PointNet feature T-Net after the first shared-MLP block.")
    parser.add_argument("--transform-reg-weight", type=float, default=0.001)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--disable-mkldnn", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=620)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--use-normals", action="store_true")
    parser.add_argument("--min-observations", type=int, default=1)
    parser.add_argument("--role-weight", action="append", default=[], help="Optional role=weight multiplier.")
    return parser.parse_args()


def load_role_bundle(pointcloud_dir: Path, dataset_dir: Path, args: argparse.Namespace) -> RoleBundle:
    cloud_bundle = load_bundle(pointcloud_dir, args.use_normals, cluster_family=False)
    label_stats = load_role_labels(dataset_dir / "placement_examples.csv", args.min_observations)
    labels = np.zeros((cloud_bundle.clouds.shape[0], len(ROLES)), dtype=np.float32)
    masks = np.zeros_like(labels)
    obs_weights = np.ones_like(labels)
    success_counts = np.zeros_like(labels)
    trial_counts = np.zeros_like(labels)
    for index, (run_name, rock_index) in enumerate(zip(cloud_bundle.run_name.tolist(), cloud_bundle.rock_index.tolist())):
        by_role = label_stats.get((str(run_name), int(rock_index)), {})
        for role_index, role in enumerate(ROLES):
            success, total = by_role.get(role, (0, 0))
            if total < args.min_observations:
                continue
            labels[index, role_index] = success / max(total, 1)
            masks[index, role_index] = 1.0
            obs_weights[index, role_index] = float(np.log1p(total) + 0.5)
            success_counts[index, role_index] = success
            trial_counts[index, role_index] = total
    return RoleBundle(
        clouds=cloud_bundle.clouds,
        run_name=cloud_bundle.run_name,
        rock_index=cloud_bundle.rock_index,
        source_kind=cloud_bundle.source_kind,
        cluster_label=cloud_bundle.cluster_label,
        labels=labels,
        masks=masks,
        obs_weights=obs_weights,
        success_counts=success_counts,
        trial_counts=trial_counts,
    )


def load_role_labels(path: Path, min_observations: int) -> dict[tuple[str, int], dict[str, tuple[int, int]]]:
    rows = read_csv(path)
    counts: dict[tuple[str, int], dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in rows:
        role = row.get("role", "")
        if role not in ROLES:
            continue
        run_name = row.get("run_name", "")
        rock_index = parse_int(row.get("rock_index", row.get("candidate_rock_index", "")))
        if rock_index is None:
            continue
        success = 1 if parse_int(row.get("label_success", "")) == 1 else 0
        key = (run_name, rock_index)
        counts[key][role][0] += success
        counts[key][role][1] += 1
    output: dict[tuple[str, int], dict[str, tuple[int, int]]] = {}
    for key, by_role in counts.items():
        output[key] = {
            role: (values[0], values[1])
            for role, values in by_role.items()
            if values[1] >= min_observations
        }
    return output


def split_labeled_by_run(
    run_names: np.ndarray,
    labeled_indices: np.ndarray,
    test_fraction: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    unique_runs = np.asarray(sorted(set(run_names[labeled_indices].tolist())))
    rng.shuffle(unique_runs)
    test_count = max(1, int(round(len(unique_runs) * test_fraction)))
    test_runs = set(unique_runs[:test_count].tolist())
    train = [int(index) for index in labeled_indices.tolist() if run_names[int(index)] not in test_runs]
    test = [int(index) for index in labeled_indices.tolist() if run_names[int(index)] in test_runs]
    if not train or not test:
        shuffled = labeled_indices.copy()
        rng.shuffle(shuffled)
        test_size = max(1, int(round(len(shuffled) * test_fraction)))
        return {"train": shuffled[test_size:], "test": shuffled[:test_size]}
    return {"train": np.asarray(train, dtype=np.int64), "test": np.asarray(test, dtype=np.int64)}


def make_loader(
    bundle: RoleBundle,
    indices: np.ndarray,
    args: argparse.Namespace,
    augment: bool,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        RoleAffordanceDataset(bundle, indices, augment=augment),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    role_weights: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        cloud = batch["cloud"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        obs_weights = batch["obs_weights"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(args.amp and device.type == "cuda"):
            output = model(cloud)
            logits = output["logits"]
            loss = masked_bce_loss(logits, labels, masks, obs_weights, role_weights)
            reg = torch.zeros((), dtype=loss.dtype, device=loss.device)
            input_transform = output.get("input_transform")
            feature_transform = output.get("feature_transform")
            if input_transform is not None:
                reg = reg + transform_regularizer(input_transform)
            if feature_transform is not None:
                reg = reg + transform_regularizer(feature_transform)
            loss = loss + args.transform_reg_weight * reg
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    obs_weights: torch.Tensor,
    role_weights: torch.Tensor,
) -> torch.Tensor:
    raw = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    weights = masks * obs_weights * role_weights.view(1, -1)
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    bundle: RoleBundle,
    role_weights: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    total_loss = 0.0
    loss_weight = 0.0
    for batch in loader:
        cloud = batch["cloud"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        obs_weights = batch["obs_weights"].to(device, non_blocking=True)
        output = model(cloud)
        probs = torch.sigmoid(output["logits"]).detach().cpu().numpy()
        batch_labels = labels.detach().cpu().numpy()
        batch_masks = masks.detach().cpu().numpy()
        batch_obs = obs_weights.detach().cpu().numpy()
        indices = batch["index"].detach().cpu().numpy()
        raw_loss = nn.functional.binary_cross_entropy_with_logits(output["logits"], labels, reduction="none")
        weights = masks * obs_weights * role_weights.view(1, -1)
        total_loss += float((raw_loss * weights).sum().detach().cpu())
        loss_weight += float(weights.sum().detach().cpu())
        for row_i, rock_array_index in enumerate(indices.tolist()):
            item: dict[str, Any] = {
                "run_name": str(bundle.run_name[int(rock_array_index)]),
                "rock_index": int(bundle.rock_index[int(rock_array_index)]),
                "source_kind": str(bundle.source_kind[int(rock_array_index)]),
                "cluster_label": str(bundle.cluster_label[int(rock_array_index)]),
            }
            for role_index, role in enumerate(ROLES):
                item[f"prob_{role}"] = float(probs[row_i, role_index])
                item[f"label_{role}"] = float(batch_labels[row_i, role_index])
                item[f"mask_{role}"] = int(batch_masks[row_i, role_index] > 0.5)
                item[f"success_count_{role}"] = float(bundle.success_counts[int(rock_array_index), role_index])
                item[f"trial_count_{role}"] = float(bundle.trial_counts[int(rock_array_index), role_index])
            rows.append(item)
    return metrics_from_rows(rows, total_loss / max(loss_weight, 1.0)), rows


def metrics_from_rows(rows: list[dict[str, Any]], loss: float) -> dict[str, Any]:
    observed = []
    by_role: dict[str, dict[str, float]] = {}
    for role in ROLES:
        y_true: list[int] = []
        y_prob: list[float] = []
        y_soft: list[float] = []
        for row in rows:
            if int(row.get(f"mask_{role}", 0)) != 1:
                continue
            prob = float(row[f"prob_{role}"])
            soft = float(row[f"label_{role}"])
            y_true.append(1 if soft >= 0.5 else 0)
            y_soft.append(soft)
            y_prob.append(prob)
            observed.append((1 if soft >= 0.5 else 0, prob, soft))
        by_role[role] = binary_metrics(y_true, y_prob, y_soft)
    if not observed:
        return {"loss": loss, "observed_accuracy": 0.0, "observed_f1": 0.0, "observed_mae": 0.0, "by_role": by_role}
    y_all = [item[0] for item in observed]
    p_all = [item[1] for item in observed]
    s_all = [item[2] for item in observed]
    all_metrics = binary_metrics(y_all, p_all, s_all)
    return {
        "loss": loss,
        "observed_accuracy": all_metrics["accuracy"],
        "observed_f1": all_metrics["f1"],
        "observed_mae": all_metrics["mae"],
        "by_role": by_role,
    }


def binary_metrics(y_true: list[int], y_prob: list[float], y_soft: list[float]) -> dict[str, float]:
    if not y_true:
        return {"count": 0, "positive_rate": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "mae": 0.0}
    pred = [1 if prob >= 0.5 else 0 for prob in y_prob]
    tp = sum(1 for y, p in zip(y_true, pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, pred) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(y_true, pred) if y == 0 and p == 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "count": float(len(y_true)),
        "positive_rate": float(sum(y_true) / len(y_true)),
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mae": float(np.mean(np.abs(np.asarray(y_prob) - np.asarray(y_soft)))),
    }


def export_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    np.savez_compressed(
        path,
        run_name=np.asarray([row["run_name"] for row in rows]),
        rock_index=np.asarray([int(row["rock_index"]) for row in rows], dtype=np.int32),
        source_kind=np.asarray([row["source_kind"] for row in rows]),
        cluster_label=np.asarray([row["cluster_label"] for row in rows]),
        roles=np.asarray(ROLES),
        prob=np.asarray([[float(row[f"prob_{role}"]) for role in ROLES] for row in rows], dtype=np.float32),
        label=np.asarray([[float(row[f"label_{role}"]) for role in ROLES] for row in rows], dtype=np.float32),
        mask=np.asarray([[int(row[f"mask_{role}"]) for role in ROLES] for row in rows], dtype=np.int8),
    )


def role_weight_tensor(entries: list[str], device: torch.device) -> torch.Tensor:
    parsed = parse_role_weights(entries)
    weights = [parsed.get(role, 1.0) for role in ROLES]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def parse_role_weights(entries: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"Invalid role weight {entry!r}; expected role=value.")
        role, raw_value = entry.split("=", 1)
        role = role.strip()
        if role not in ROLES:
            raise SystemExit(f"Unknown role {role!r}; expected one of {ROLES}.")
        value = float(raw_value)
        if not np.isfinite(value) or value <= 0.0:
            raise SystemExit(f"Invalid role weight {entry!r}; expected positive finite value.")
        weights[role] = value
    return weights


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


def parse_int(value: str | None) -> int | None:
    if value in {"", None}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(args: argparse.Namespace) -> None:
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(max(1, min(args.torch_threads, 4)))
    if args.disable_mkldnn and hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False


def make_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch, "amp"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def write_readme(output_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# PointNet Role-Affordance",
        "",
        "Purpose: predict whether a rock shape is suitable for base, middle, or cap positions using 3D PointNet point-cloud geometry.",
        "",
        "This is an offline stone-selection candidate. It should not be used alone for closed-loop stacking because it does not observe the current wall state.",
        "",
        "Architecture:",
        "",
        "- Input: normalized rock surface point cloud, optionally xyz+normal.",
        "- PointNet xyz T-Net canonical alignment unless disabled.",
        "- Shared 1x1 point MLP with channels recorded below.",
        "- Optional feature T-Net with orthogonality regularization.",
        "- Symmetric max pooling to a global rock embedding.",
        "- MLP role head for base/middle/cap affordance probabilities.",
        "",
        f"- pointcloud dir: `{metrics['pointcloud_dir']}`",
        f"- dataset: `{metrics['dataset']}`",
        f"- rocks: `{metrics['rock_count']}`",
        f"- labeled rocks: `{metrics['labeled_rock_count']}`",
        f"- point count: `{metrics['point_count']}`",
        f"- input dim: `{metrics['input_dim']}`",
        f"- PointNet channels: `{metrics['pointnet_channels']}`",
        f"- embedding dim: `{metrics['embedding_dim']}`",
        f"- input transform: `{metrics['input_transform']}`",
        f"- feature transform: `{metrics['feature_transform']}`",
        f"- device: `{metrics['device']}`",
        f"- test observed accuracy: `{metrics['test_observed_accuracy']:.3f}`",
        f"- test observed F1: `{metrics['test_observed_f1']:.3f}`",
        f"- test observed MAE: `{metrics['test_observed_mae']:.3f}`",
        "",
        "Outputs:",
        "",
        "- `pointnet_role_affordance.pt`: PyTorch checkpoint.",
        "- `rock_role_affordance.npz`: probabilities for all exported rocks.",
        "- `all_predictions.csv`: human-readable per-rock probabilities and observed labels.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
