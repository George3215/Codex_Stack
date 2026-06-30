from __future__ import annotations

import argparse
import csv
import json
import os
import random
from contextlib import nullcontext
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.pointnet_modules import PointNetBackbone, parse_channels, transform_regularizer


@dataclass
class RockCloudBundle:
    clouds: np.ndarray
    run_name: np.ndarray
    rock_index: np.ndarray
    source_kind: np.ndarray
    cluster_label: np.ndarray
    source_label: np.ndarray
    cluster_label_id: np.ndarray
    source_classes: list[str]
    cluster_classes: list[str]


class RockCloudDataset(Dataset):
    def __init__(self, bundle: RockCloudBundle, indices: np.ndarray, augment: bool) -> None:
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
            "source_label": torch.tensor(int(self.bundle.source_label[index]), dtype=torch.long),
            "cluster_label": torch.tensor(int(self.bundle.cluster_label_id[index]), dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }


class PointNetRockEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        source_classes: int,
        cluster_classes: int,
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
        self.source_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, source_classes),
        )
        self.cluster_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, cluster_classes),
        )

    def forward(self, cloud: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone_output = self.backbone(cloud)
        embedding = backbone_output["embedding"]
        return {
            "embedding": embedding,
            "source_logits": self.source_head(embedding),
            "cluster_logits": self.cluster_head(embedding),
            "input_transform": backbone_output["input_transform"],
            "feature_transform": backbone_output["feature_transform"],
        }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    configure_torch_runtime(args)
    pointcloud_dir = args.pointcloud_dir.resolve()
    output_dir = unique_dir(args.output.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)

    bundle = load_bundle(pointcloud_dir, args.use_normals, args.cluster_family)
    split = split_by_run(bundle.run_name, args.test_fraction, np.random.default_rng(args.seed))
    train_dataset = RockCloudDataset(bundle, split["train"], augment=args.augment)
    test_dataset = RockCloudDataset(bundle, split["test"], augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    channels = parse_channels(args.pointnet_channels, args.embedding_dim)
    model = PointNetRockEncoder(
        input_dim=bundle.clouds.shape[2],
        embedding_dim=args.embedding_dim,
        source_classes=len(bundle.source_classes),
        cluster_classes=len(bundle.cluster_classes),
        dropout=args.dropout,
        channels=channels,
        activation=args.activation,
        input_transform=not args.no_input_transform,
        feature_transform=args.feature_transform,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = make_grad_scaler(args.amp and device.type == "cuda")

    history: list[dict[str, float]] = []
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, args)
        scheduler.step()
        test_metrics = evaluate(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_source_accuracy": train_metrics["source_accuracy"],
                "train_cluster_accuracy": train_metrics["cluster_accuracy"],
                "test_source_accuracy": test_metrics["source_accuracy"],
                "test_cluster_accuracy": test_metrics["cluster_accuracy"],
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )
        score = 0.5 * test_metrics["source_accuracy"] + 0.5 * test_metrics["cluster_accuracy"]
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if args.log_every > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.5f} "
                f"test_source_acc={test_metrics['source_accuracy']:.4f} "
                f"test_cluster_acc={test_metrics['cluster_accuracy']:.4f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate(model, train_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    embedding_path = output_dir / "rock_pointnet_embeddings.npz"
    export_embeddings(model, bundle, args.batch_size, device, embedding_path)
    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pointcloud_dir": str(pointcloud_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "rock_count": int(bundle.clouds.shape[0]),
        "point_count": int(bundle.clouds.shape[1]),
        "input_dim": int(bundle.clouds.shape[2]),
        "embedding_dim": args.embedding_dim,
        "pointnet_channels": channels,
        "activation": args.activation,
        "input_transform": bool(not args.no_input_transform),
        "feature_transform": bool(args.feature_transform),
        "transform_reg_weight": args.transform_reg_weight,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "architecture_detail": {
            "name": "PointNetRockEncoder",
            "stage": "point-cloud geometry encoder for replacing hand-written rock geometry scalars",
            "input": "unordered normalized rock surface points; xyz or xyz+normal",
            "output": "global rock embedding plus source-kind and cluster-family logits",
            "shared_mlp_channels": channels,
            "pooling": "global max pooling over points",
            "source_head": "Linear(embedding_dim,256)-ReLU-Dropout-Linear(256,source_classes)",
            "cluster_head": "Linear(embedding_dim,256)-ReLU-Dropout-Linear(256,cluster_classes)",
        },
        "source_classes": bundle.source_classes,
        "cluster_classes": bundle.cluster_classes,
        "source_class_counts": dict(Counter(bundle.source_kind.tolist())),
        "cluster_class_counts": dict(Counter(bundle.cluster_label.tolist())),
        "train_count": int(len(split["train"])),
        "test_count": int(len(split["test"])),
        "test_run_split": sorted(set(bundle.run_name[split["test"]].tolist())),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "amp": bool(args.amp and device.type == "cuda"),
        "augment": bool(args.augment),
        "cluster_family": bool(args.cluster_family),
        "train_source_accuracy": train_metrics["source_accuracy"],
        "train_cluster_accuracy": train_metrics["cluster_accuracy"],
        "test_source_accuracy": test_metrics["source_accuracy"],
        "test_cluster_accuracy": test_metrics["cluster_accuracy"],
        "embedding_path": str(embedding_path),
        "architecture": "3D PointNet: optional xyz T-Net, shared 1x1 MLP over unordered point samples, optional feature T-Net, global max pooling, and multi-task classification heads.",
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        output_dir / "pointnet_rock_encoder.pt",
    )
    write_csv(output_dir / "history.csv", history)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_readme(output_dir, metrics)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch PointNet rock encoder on exported rock point clouds.")
    parser.add_argument("--pointcloud-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--pointnet-channels", default="64,64,128", help="Comma-separated shared-MLP widths before final embedding dim.")
    parser.add_argument("--activation", choices=["relu", "silu", "gelu"], default="relu")
    parser.add_argument("--no-input-transform", action="store_true", help="Disable PointNet xyz T-Net canonical alignment.")
    parser.add_argument("--feature-transform", action="store_true", help="Enable PointNet feature T-Net after the first shared-MLP block.")
    parser.add_argument("--transform-reg-weight", type=float, default=0.001)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--disable-mkldnn", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=620)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--use-normals", action="store_true", help="Concatenate sampled normals to xyz input.")
    parser.add_argument("--cluster-family", action="store_true", help="Collapse generated cluster suffixes such as _2/_3.")
    return parser.parse_args()


def load_bundle(pointcloud_dir: Path, use_normals: bool, cluster_family: bool) -> RockCloudBundle:
    cloud_parts: list[np.ndarray] = []
    run_names: list[str] = []
    rock_indices: list[int] = []
    source_kinds: list[str] = []
    cluster_labels: list[str] = []
    for path in sorted(pointcloud_dir.glob("*/rock_pointclouds.npz")):
        data = np.load(path)
        points = np.asarray(data["points"], dtype=np.float32)
        if use_normals:
            normals = np.asarray(data["normals"], dtype=np.float32)
            clouds = np.concatenate([points, normals], axis=2)
        else:
            clouds = points
        cloud_parts.append(clouds)
        count = clouds.shape[0]
        run_names.extend([path.parent.name] * count)
        rock_indices.extend(int(value) for value in data["rock_index"].tolist())
        source_kinds.extend(str(value) for value in data["source_kind"].tolist())
        cluster_labels.extend(normalize_cluster_label(str(value), cluster_family) for value in data["cluster_label"].tolist())
    if not cloud_parts:
        raise SystemExit(f"No */rock_pointclouds.npz files found under {pointcloud_dir}")
    all_clouds = np.concatenate(cloud_parts, axis=0)
    source_classes = sorted(set(source_kinds))
    cluster_classes = sorted(set(cluster_labels))
    source_map = {name: index for index, name in enumerate(source_classes)}
    cluster_map = {name: index for index, name in enumerate(cluster_classes)}
    return RockCloudBundle(
        clouds=all_clouds,
        run_name=np.asarray(run_names),
        rock_index=np.asarray(rock_indices, dtype=np.int32),
        source_kind=np.asarray(source_kinds),
        cluster_label=np.asarray(cluster_labels),
        source_label=np.asarray([source_map[name] for name in source_kinds], dtype=np.int64),
        cluster_label_id=np.asarray([cluster_map[name] for name in cluster_labels], dtype=np.int64),
        source_classes=source_classes,
        cluster_classes=cluster_classes,
    )


def normalize_cluster_label(label: str, cluster_family: bool) -> str:
    if not cluster_family:
        return label
    parts = label.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return label


def split_by_run(run_names: np.ndarray, test_fraction: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    unique_runs = np.asarray(sorted(set(run_names.tolist())))
    rng.shuffle(unique_runs)
    test_count = max(1, int(round(len(unique_runs) * test_fraction)))
    test_runs = set(unique_runs[:test_count].tolist())
    train_indices = [idx for idx, run_name in enumerate(run_names.tolist()) if run_name not in test_runs]
    test_indices = [idx for idx, run_name in enumerate(run_names.tolist()) if run_name in test_runs]
    return {
        "train": np.asarray(train_indices, dtype=np.int64),
        "test": np.asarray(test_indices, dtype=np.int64),
    }


def augment_cloud(cloud: np.ndarray) -> np.ndarray:
    angle = np.random.uniform(0.0, 2.0 * np.pi)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot = np.asarray([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    cloud[:, :3] = cloud[:, :3] @ rot.T
    if cloud.shape[1] >= 6:
        cloud[:, 3:6] = cloud[:, 3:6] @ rot.T
    cloud[:, :3] += np.random.normal(0.0, 0.005, size=cloud[:, :3].shape).astype(np.float32)
    return cloud


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    source_hits = 0
    cluster_hits = 0
    count = 0
    for batch in loader:
        cloud = batch["cloud"].to(device, non_blocking=True)
        source_label = batch["source_label"].to(device, non_blocking=True)
        cluster_label = batch["cluster_label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(args.amp and device.type == "cuda"):
            output = model(cloud)
            source_loss = nn.functional.cross_entropy(output["source_logits"], source_label)
            cluster_loss = nn.functional.cross_entropy(output["cluster_logits"], cluster_label)
            loss = 0.4 * source_loss + 0.6 * cluster_loss
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
        source_hits += int((output["source_logits"].argmax(dim=1) == source_label).sum().detach().cpu())
        cluster_hits += int((output["cluster_logits"].argmax(dim=1) == cluster_label).sum().detach().cpu())
        count += int(cloud.shape[0])
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "source_accuracy": source_hits / max(count, 1),
        "cluster_accuracy": cluster_hits / max(count, 1),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    source_hits = 0
    cluster_hits = 0
    count = 0
    for batch in loader:
        cloud = batch["cloud"].to(device, non_blocking=True)
        source_label = batch["source_label"].to(device, non_blocking=True)
        cluster_label = batch["cluster_label"].to(device, non_blocking=True)
        output = model(cloud)
        source_hits += int((output["source_logits"].argmax(dim=1) == source_label).sum().detach().cpu())
        cluster_hits += int((output["cluster_logits"].argmax(dim=1) == cluster_label).sum().detach().cpu())
        count += int(cloud.shape[0])
    return {
        "source_accuracy": source_hits / max(count, 1),
        "cluster_accuracy": cluster_hits / max(count, 1),
    }


@torch.no_grad()
def export_embeddings(model: nn.Module, bundle: RockCloudBundle, batch_size: int, device: torch.device, path: Path) -> None:
    dataset = RockCloudDataset(bundle, np.arange(bundle.clouds.shape[0]), augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    embeddings: list[np.ndarray] = []
    source_probs: list[np.ndarray] = []
    cluster_probs: list[np.ndarray] = []
    for batch in loader:
        cloud = batch["cloud"].to(device, non_blocking=True)
        output = model(cloud)
        embeddings.append(output["embedding"].detach().cpu().numpy().astype(np.float32))
        source_probs.append(torch.softmax(output["source_logits"], dim=1).detach().cpu().numpy().astype(np.float32))
        cluster_probs.append(torch.softmax(output["cluster_logits"], dim=1).detach().cpu().numpy().astype(np.float32))
    np.savez_compressed(
        path,
        embedding=np.concatenate(embeddings, axis=0),
        source_prob=np.concatenate(source_probs, axis=0),
        cluster_prob=np.concatenate(cluster_probs, axis=0),
        run_name=bundle.run_name,
        rock_index=bundle.rock_index,
        source_kind=bundle.source_kind,
        cluster_label=bundle.cluster_label,
        source_classes=np.asarray(bundle.source_classes),
        cluster_classes=np.asarray(bundle.cluster_classes),
    )


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
        "# PyTorch PointNet Rock Encoder",
        "",
        "This model learns a rock-shape embedding from exported point-cloud tensors.",
        "",
        "Architecture:",
        "",
        "- Input: normalized rock surface point cloud, optionally xyz+normal.",
        "- PointNet xyz T-Net canonical alignment unless disabled.",
        "- Shared 1x1 point MLP with channels recorded below.",
        "- Optional feature T-Net with orthogonality regularization.",
        "- Symmetric max pooling for permutation-invariant rock embedding.",
        "- Multi-task heads for generated source kind and cluster label.",
        "",
        f"- point-cloud dir: `{metrics['pointcloud_dir']}`",
        f"- device: {metrics['device']}",
        f"- torch: {metrics['torch_version']}",
        f"- GPU: {metrics['gpu_name']}",
        f"- rocks: {metrics['rock_count']}",
        f"- points per rock: {metrics['point_count']}",
        f"- input dim: {metrics['input_dim']}",
        f"- PointNet channels: {metrics['pointnet_channels']}",
        f"- embedding dim: {metrics['embedding_dim']}",
        f"- input transform: {metrics['input_transform']}",
        f"- feature transform: {metrics['feature_transform']}",
        f"- test source-kind accuracy: {metrics['test_source_accuracy']:.3f}",
        f"- test cluster-label accuracy: {metrics['test_cluster_accuracy']:.3f}",
        "",
        "These embeddings are intended to replace hand-written rock geometry scalars in later stack placement networks. Pretrained external PointNet++/Point Transformer weights can be added as a separate backbone if their license and domain transfer are appropriate.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
