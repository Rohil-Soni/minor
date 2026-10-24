"""Earth Mover's Distance loss for score distributions (NIMA, r=2)."""

from __future__ import annotations

import torch


def emd_loss(
    pred: torch.Tensor, target: torch.Tensor, r: int = 2, reduction: str = "mean"
) -> torch.Tensor:
    """``(mean_k |CDF_pred - CDF_target|^r)^(1/r)`` per sample, averaged over the batch.

    Computed in float32 so it stays stable under AMP.
    """
    cdf_diff = torch.cumsum(pred.float(), dim=1) - torch.cumsum(target.float(), dim=1)
    per_sample = cdf_diff.abs().pow(r).mean(dim=1).pow(1.0 / r)
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"unknown reduction {reduction!r}")
