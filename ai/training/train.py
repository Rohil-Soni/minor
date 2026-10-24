"""Train NIMA on AVA splits from ``training.dataset_ava prepare``.

Stage A (``--epochs-head``): backbone frozen, head only, Adam lr 1e-3.
Stage B (``--epochs-full``): everything trainable, backbone lr ~5e-6, head lr 1e-4, cosine decay,
early stopping on val EMD.

Every epoch writes ``last.pt`` (for ``--resume``) and, when val EMD improves, ``best.pt``.
Per-epoch losses go to ``train_log.csv``. Run from ``ai/``::

    python -m training.train --splits data/ava_splits --images <dir> --out checkpoints
    python -m training.train ... --resume auto     # continue after a Kaggle/Colab timeout
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.dataset_ava import DEFAULT_SEED, DEFAULT_WORKERS, load_splits
from training.evaluate import compute_metrics, predict
from training.losses import emd_loss
from training.model import NIMA

log = logging.getLogger(__name__)

LOG_FIELDS = ["epoch", "stage", "train_emd", "val_emd", "val_srcc", "lr_backbone", "lr_head", "s"]
# changing these between a run and its --resume changes the lr schedule mid-way
SCHEDULE_ARGS = ("epochs_head", "epochs_full", "lr_head", "lr_full_head", "lr_backbone")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--splits", type=Path, required=True, help="output dir of dataset_ava prepare")
    p.add_argument("--images", type=Path, required=True, help="AVA image folder")
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    p.add_argument("--backbone", default="mobilenet_v2", help="mobilenet_v2 or efficientnet_b0")
    p.add_argument("--epochs-head", type=int, default=5)
    p.add_argument("--epochs-full", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64, help="reduce if CUDA runs out of memory")
    p.add_argument("--lr-head", type=float, default=1e-3, help="stage A head lr")
    p.add_argument("--lr-full-head", type=float, default=1e-4, help="stage B head lr")
    p.add_argument("--lr-backbone", type=float, default=5e-6, help="stage B backbone lr")
    p.add_argument("--patience", type=int, default=3, help="stage B epochs without val gain")
    p.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default="auto", help="auto, cpu or cuda")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    p.add_argument("--resume", help="path to last.pt, or 'auto' for <out>/last.pt if present")
    p.add_argument(
        "--stop-after",
        type=int,
        help="end this session after N epochs (split a long run across sessions; use --resume)",
    )
    p.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
    p.add_argument("--limit-batches", type=int, help="cap batches per epoch (smoke tests)")
    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def stage_for(epoch: int, args: argparse.Namespace) -> str:
    return "head" if epoch < args.epochs_head else "full"


def build_optimizer(
    model: NIMA, stage: str, args: argparse.Namespace
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    if stage == "head":
        model.freeze_backbone(True)
        return torch.optim.Adam(model.head.parameters(), lr=args.lr_head), None
    model.freeze_backbone(False)
    optimizer = torch.optim.Adam(
        [
            {"params": model.features.parameters(), "lr": args.lr_backbone},
            {"params": model.head.parameters(), "lr": args.lr_full_head},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs_full)
    )
    return optimizer, scheduler


def train_one_epoch(
    model: NIMA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    stage: str,
    limit_batches: int | None,
) -> float:
    model.train()
    if stage == "head":
        model.features.eval()  # frozen backbone: keep BatchNorm running stats fixed

    total, count = 0.0, 0
    for i, (images, dists) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        dists = dists.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            pred, _ = model(images)
        loss = emd_loss(pred, dists)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += loss.item() * len(images)
        count += len(images)
    return total / max(count, 1)


def _save(obj: dict[str, Any], path: Path) -> None:
    """Write via a temp file so a session killed mid-save never leaves a broken checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _plain_args(args: argparse.Namespace) -> dict[str, Any]:
    # torch.load(weights_only=True) only accepts primitive types
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    seed_everything(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    log.info("device %s, amp %s", device, amp)

    datasets = load_splits(args.splits, args.images)
    generator = torch.Generator()
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        datasets["train"], shuffle=True, drop_last=True, generator=generator, **loader_kwargs
    )
    val_loader = DataLoader(datasets["val"], **loader_kwargs)
    log.info("train %d / val %d images", len(datasets["train"]), len(datasets["val"]))

    model = NIMA(args.backbone, pretrained=not args.no_pretrained).to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)

    args.out.mkdir(parents=True, exist_ok=True)
    last_path, best_path = args.out / "last.pt", args.out / "best.pt"
    log_path = args.out / "train_log.csv"

    start_epoch, best_val, bad_epochs = 0, float("inf"), 0
    resume_path = last_path if args.resume == "auto" else args.resume
    ckpt = None
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        if ckpt["backbone"] != args.backbone:
            raise SystemExit(f"checkpoint backbone {ckpt['backbone']!r} != --backbone")
        model.load_state_dict(ckpt["model"])
        start_epoch, best_val, bad_epochs = ckpt["epoch"] + 1, ckpt["best_val"], ckpt["bad_epochs"]
        log.info(
            "resumed from %s at epoch %d (best val EMD %.4f)", resume_path, start_epoch, best_val
        )
        changed = [
            k
            for k in SCHEDULE_ARGS
            if ckpt["args"].get(k) is not None and ckpt["args"][k] != getattr(args, k)
        ]
        if changed:
            log.warning(
                "resuming with different %s than the checkpoint; the saved lr schedule is kept",
                ", ".join(changed),
            )
        if ckpt.get("stopped_early"):
            log.info("that run already stopped early; nothing left to train")
            start_epoch = args.epochs_head + args.epochs_full
    elif args.resume and args.resume != "auto":
        raise SystemExit(f"--resume checkpoint not found: {args.resume}")
    elif log_path.exists():
        log_path.unlink()  # fresh run

    total_epochs = args.epochs_head + args.epochs_full
    stage, optimizer, scheduler = None, None, None
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    for epoch in range(start_epoch, total_epochs):
        if stage_for(epoch, args) != stage:
            stage = stage_for(epoch, args)
            optimizer, scheduler = build_optimizer(model, stage, args)
            if ckpt is not None and ckpt["stage"] == stage:
                optimizer.load_state_dict(ckpt["optimizer"])
                if scheduler is not None and ckpt["scheduler"] is not None:
                    scheduler.load_state_dict(ckpt["scheduler"])
                scaler.load_state_dict(ckpt["scaler"])
            ckpt = None
            log.info("stage %s from epoch %d", stage, epoch)

        t0 = time.perf_counter()
        generator.manual_seed(args.seed + epoch)  # same shuffle order with or without resume
        train_emd = train_one_epoch(
            model, train_loader, optimizer, scaler, device, amp, stage, args.limit_batches
        )
        pred, target = predict(model, val_loader, device, amp, args.limit_batches)
        val = compute_metrics(pred, target)
        lrs = [g["lr"] for g in optimizer.param_groups]
        if scheduler is not None:
            scheduler.step()

        improved = val["emd"] < best_val
        if improved:
            best_val, bad_epochs = val["emd"], 0
        elif stage == "full":
            bad_epochs += 1
        stop = stage == "full" and bad_epochs >= args.patience

        state = {
            **model.config(),
            "model": model.state_dict(),
            "epoch": epoch,
            "stage": stage,
            "val_metrics": val,
            "best_val": best_val,
            "bad_epochs": bad_epochs,
            "stopped_early": stop,
            "args": _plain_args(args),
        }
        if improved:
            _save(state, best_path)
        _save(
            {
                **state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict(),
            },
            last_path,
        )

        seconds = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            "stage": stage,
            "train_emd": round(train_emd, 5),
            "val_emd": round(val["emd"], 5),
            "val_srcc": None if val["srcc"] is None else round(val["srcc"], 4),
            "lr_backbone": lrs[0] if stage == "full" else 0.0,
            "lr_head": lrs[-1],
            "s": round(seconds, 1),
        }
        new_file = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        log.info(
            "epoch %d/%d [%s] train EMD %.4f | val EMD %.4f SRCC %s%s | %.0fs",
            epoch + 1,
            total_epochs,
            stage,
            train_emd,
            val["emd"],
            row["val_srcc"],
            " *best*" if improved else "",
            seconds,
        )

        if stop:
            log.info("early stopping: no val improvement for %d epochs", bad_epochs)
            break
        if args.stop_after is not None and epoch + 1 - start_epoch >= args.stop_after:
            log.info("--stop-after %d reached; continue with --resume auto", args.stop_after)
            break

    summary = {"best_val_emd": best_val, "best": str(best_path), "log": str(log_path)}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
