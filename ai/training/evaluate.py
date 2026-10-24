"""Evaluate a checkpoint: SRCC, PLCC, binary accuracy (mean >= 5) and EMD.

Run from ``ai/``::

    python -m training.evaluate --checkpoint checkpoints/best.pt \\
        --splits data/ava_splits --images <dir> --out models/metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from training.dataset_ava import DEFAULT_WORKERS, AVADataset
from training.model import NIMA, NUM_BINS, load_checkpoint

log = logging.getLogger(__name__)

GOOD_THRESHOLD = 5.0
# plan §5 Phase 3: below this something is wrong (normalisation / labels)
SRCC_SANITY = 0.45
SRCC_TARGET = 0.50


@torch.no_grad()
def predict(
    model: NIMA,
    loader: DataLoader,
    device: torch.device,
    amp: bool = False,
    limit_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and target distributions, both ``[N, 10]`` float64."""
    model.eval()
    preds, targets = [], []
    channels_last = device.type == "cuda"
    for i, (images, dists) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            pred, _ = model(images)
        preds.append(pred.float().cpu().numpy())
        targets.append(dists.numpy())
    return np.concatenate(preds).astype(np.float64), np.concatenate(targets).astype(np.float64)


def _corr(fn, a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None  # undefined for constant input
    return float(fn(a, b).statistic)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    bins = np.arange(1, NUM_BINS + 1)
    pred_mean, target_mean = pred @ bins, target @ bins
    cdf_diff = np.cumsum(pred, axis=1) - np.cumsum(target, axis=1)
    emd = np.sqrt(np.mean(cdf_diff**2, axis=1)).mean()
    accuracy = np.mean((pred_mean >= GOOD_THRESHOLD) == (target_mean >= GOOD_THRESHOLD))
    return {
        "srcc": _corr(stats.spearmanr, pred_mean, target_mean),
        "plcc": _corr(stats.pearsonr, pred_mean, target_mean),
        "accuracy": float(accuracy),
        "emd": float(emd),
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("data/ava_splits"))
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--out", type=Path, default=Path("models/metrics.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint)
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)

    dataset = AVADataset(args.splits / f"{args.split}.csv", args.images, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    pred, target = predict(model, loader, device, amp=device.type == "cuda")
    metrics = compute_metrics(pred, target)

    result = {
        "split": args.split,
        "n": len(dataset),
        **metrics,
        "backbone": ckpt["backbone"],
        "epoch": ckpt["epoch"],
        "checkpoint": str(args.checkpoint),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    srcc = metrics["srcc"]
    if srcc is None:
        log.warning("SRCC undefined (constant predictions?)")
    elif srcc < SRCC_SANITY:
        log.warning("SRCC %.3f < %.2f: check normalisation and labels", srcc, SRCC_SANITY)
    elif srcc < SRCC_TARGET:
        log.warning("SRCC %.3f is below the %.2f target", srcc, SRCC_TARGET)
    return result


if __name__ == "__main__":
    main()
