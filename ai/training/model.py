"""NIMA model: pretrained CNN backbone + 10-bin score distribution head.

``forward`` returns ``(dist, embedding)`` from one pass: ``dist`` [N, 10] is the softmax
score distribution, ``embedding`` [N, 1280] is the pooled backbone feature *before* dropout,
reused at inference time for similarity grouping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models

NUM_BINS = 10

# name -> (constructor, ImageNet weights, pooled feature size)
BACKBONES: dict[str, tuple[Any, Any, int]] = {
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V2, 1280),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
}


class NIMA(nn.Module):
    def __init__(
        self, backbone: str = "mobilenet_v2", pretrained: bool = True, dropout: float = 0.75
    ) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}, choose from {sorted(BACKBONES)}")
        ctor, weights, dim = BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)

        self.backbone_name = backbone
        self.dropout = dropout
        self.embedding_dim = dim
        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, NUM_BINS))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = torch.flatten(self.pool(self.features(x)), 1)
        dist = torch.softmax(self.head(embedding), dim=1)
        return dist, embedding

    def freeze_backbone(self, frozen: bool) -> None:
        for p in self.features.parameters():
            p.requires_grad = not frozen

    def config(self) -> dict[str, Any]:
        return {"backbone": self.backbone_name, "dropout": self.dropout}


def dist_mean_std(dist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std of score distributions over bins 1..10."""
    bins = torch.arange(1, NUM_BINS + 1, dtype=dist.dtype, device=dist.device)
    mean = (dist * bins).sum(dim=1)
    std = ((dist * (bins - mean[:, None]) ** 2).sum(dim=1)).sqrt()
    return mean, std


def load_checkpoint(
    path: Path, map_location: str | torch.device = "cpu"
) -> tuple[NIMA, dict[str, Any]]:
    """Rebuild a :class:`NIMA` from a ``best.pt`` / ``last.pt`` written by ``train.py``."""
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=True)
    model = NIMA(ckpt["backbone"], pretrained=False, dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["model"])
    return model, ckpt
