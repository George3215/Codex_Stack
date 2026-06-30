from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PointNetSpec:
    input_dim: int
    embedding_dim: int
    channels: list[int]
    activation: str
    input_transform: bool
    feature_transform: bool
    feature_transform_dim: int


def parse_channels(spec: str, embedding_dim: int, default: str = "64,64,128") -> list[int]:
    raw = spec.strip() or default
    channels = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not channels:
        channels = [64, 64, 128]
    if channels[-1] != embedding_dim:
        channels.append(int(embedding_dim))
    return channels


def make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class TransformNet(nn.Module):
    """PointNet T-Net for learned canonical alignment."""

    def __init__(self, k: int, activation: str = "relu") -> None:
        super().__init__()
        self.k = int(k)
        self.conv = nn.Sequential(
            nn.Conv1d(self.k, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            make_activation(activation),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            make_activation(activation),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            make_activation(activation),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            make_activation(activation),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            make_activation(activation),
            nn.Linear(256, self.k * self.k),
        )
        nn.init.zeros_(self.fc[-1].weight)
        identity = torch.eye(self.k, dtype=torch.float32).reshape(-1)
        with torch.no_grad():
            self.fc[-1].bias.copy_(identity)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, K, N]
        x = self.conv(features)
        x = torch.max(x, dim=2).values
        transform = self.fc(x).reshape(-1, self.k, self.k)
        return transform


class PointNetBackbone(nn.Module):
    """3D PointNet backbone: T-Net, shared MLP, symmetric max pooling."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        channels: list[int],
        activation: str = "relu",
        input_transform: bool = True,
        feature_transform: bool = False,
    ) -> None:
        super().__init__()
        if input_dim < 3:
            raise ValueError("PointNetBackbone expects at least xyz channels.")
        self.spec = PointNetSpec(
            input_dim=int(input_dim),
            embedding_dim=int(embedding_dim),
            channels=[int(value) for value in channels],
            activation=activation,
            input_transform=bool(input_transform),
            feature_transform=bool(feature_transform),
            feature_transform_dim=int(channels[0]),
        )
        self.input_tnet = TransformNet(3, activation) if input_transform else None
        self.blocks = nn.ModuleList()
        previous = int(input_dim)
        for channel in channels:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(previous, int(channel), kernel_size=1),
                    nn.BatchNorm1d(int(channel)),
                    make_activation(activation),
                )
            )
            previous = int(channel)
        self.feature_tnet = TransformNet(channels[0], activation) if feature_transform else None

    def forward(self, cloud: torch.Tensor) -> dict[str, torch.Tensor | None]:
        # cloud: [B, N, C]. The first three channels are xyz; optional next three are normals.
        input_transform = None
        if self.input_tnet is not None:
            xyz_features = cloud[:, :, :3].transpose(1, 2).contiguous()
            input_transform = self.input_tnet(xyz_features)
            cloud = apply_xyz_transform(cloud, input_transform)

        x = cloud.transpose(1, 2).contiguous()
        feature_transform = None
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index == 0 and self.feature_tnet is not None:
                feature_transform = self.feature_tnet(x)
                x = torch.bmm(feature_transform, x)
        embedding = torch.max(x, dim=2).values
        return {
            "embedding": embedding,
            "input_transform": input_transform,
            "feature_transform": feature_transform,
        }


def apply_xyz_transform(cloud: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    xyz = torch.bmm(cloud[:, :, :3], transform)
    if cloud.shape[2] < 6:
        return torch.cat([xyz, cloud[:, :, 3:]], dim=2) if cloud.shape[2] > 3 else xyz
    normals = torch.bmm(cloud[:, :, 3:6], transform)
    tail = cloud[:, :, 6:]
    return torch.cat([xyz, normals, tail], dim=2)


def transform_regularizer(transform: torch.Tensor | None) -> torch.Tensor:
    if transform is None:
        return torch.tensor(0.0)
    k = transform.shape[1]
    identity = torch.eye(k, dtype=transform.dtype, device=transform.device).unsqueeze(0)
    diff = torch.bmm(transform, transform.transpose(1, 2)) - identity
    return torch.mean(torch.norm(diff, dim=(1, 2)))
