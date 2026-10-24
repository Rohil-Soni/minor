"""AVA dataset: ``AVA.txt`` parsing, fixed-seed subset and splits, and a torch ``Dataset``.

Works with any AVA copy laid out as ``AVA.txt`` plus a folder (possibly nested) of
``<image_id>.jpg`` files, e.g. a Kaggle dataset mounted read-only under ``/kaggle/input``.

Run from ``ai/``::

    # pick a 40k subset, drop missing/corrupt images, write train/val/test CSVs
    python -m training.dataset_ava prepare --ava-txt <AVA.txt> --images <dir> \\
        --out data/ava_splits --subset 40000

    # acceptance check: dataset sizes, one batch shape, distribution sums
    python -m training.dataset_ava check --splits data/ava_splits --images <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from photoholic_ai.preprocess import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE

log = logging.getLogger(__name__)

NUM_BINS = 10
VOTE_COLS = [f"v{i}" for i in range(1, NUM_BINS + 1)]
PROB_COLS = [f"p{i}" for i in range(1, NUM_BINS + 1)]
SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg"}
DEFAULT_SEED = 42
DEFAULT_WORKERS = 0 if os.name == "nt" else 2


def parse_ava_txt(path: Path) -> pd.DataFrame:
    """Parse ``AVA.txt`` into ``image_id, p1..p10, mean, num_votes``.

    Columns in the file: index, image_id, 10 vote counts, 2 semantic tags, challenge id.
    Rows with zero votes and duplicate ids are dropped.
    """
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    if raw.shape[1] < 2 + NUM_BINS:
        raise ValueError(
            f"{path}: expected at least {2 + NUM_BINS} columns, got {raw.shape[1]}. "
            "Is this the AVA.txt label file?"
        )

    ids = raw.iloc[:, 1].astype(np.int64).to_numpy()
    votes = raw.iloc[:, 2 : 2 + NUM_BINS].to_numpy(dtype=np.float64)
    total = votes.sum(axis=1)
    keep = total > 0
    if (~keep).any():
        log.warning("dropping %d rows with zero votes", int((~keep).sum()))

    probs = votes[keep] / total[keep, None]
    df = pd.DataFrame(probs, columns=PROB_COLS)
    df.insert(0, "image_id", ids[keep])
    df["mean"] = probs @ np.arange(1, NUM_BINS + 1)
    df["num_votes"] = total[keep].astype(np.int64)

    dupes = df["image_id"].duplicated()
    if dupes.any():
        log.warning("dropping %d duplicate image ids", int(dupes.sum()))
        df = df[~dupes]
    return df.reset_index(drop=True)


def index_images(image_dir: Path) -> dict[int, str]:
    """Map image id -> path relative to ``image_dir`` (posix) for every ``<id>.jpg`` below it."""
    index: dict[int, str] = {}
    for root, _dirs, files in os.walk(image_dir):
        for name in files:
            stem, ext = os.path.splitext(name)
            if ext.lower() in IMAGE_EXTS and stem.isdigit():
                rel = Path(root, name).relative_to(image_dir).as_posix()
                index.setdefault(int(stem), rel)
    return index


def is_readable(path: Path) -> bool:
    """True if the JPEG fully decodes (``draft`` keeps this cheap; catches truncated files)."""
    try:
        with Image.open(path) as im:
            im.draft("RGB", (64, 64))
            im.load()
        return True
    except Exception:  # PIL raises OSError, SyntaxError, ValueError, DecompressionBombError...
        return False


def _chunks(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def select_subset(
    labels: pd.DataFrame,
    index: dict[int, str],
    image_dir: Path,
    subset: int | None,
    seed: int = DEFAULT_SEED,
    verify: bool = True,
    workers: int = 8,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fixed-seed random sample of ``subset`` readable images (all if ``subset`` is None)."""
    available = labels[labels["image_id"].isin(index.keys())].copy()
    available["path"] = available["image_id"].map(index)
    stats = {"labelled": len(labels), "missing": len(labels) - len(available), "corrupt": 0}

    rng = np.random.default_rng(seed)
    shuffled = available.iloc[rng.permutation(len(available))]
    target = len(shuffled) if subset is None else min(subset, len(shuffled))

    if not verify:
        return shuffled.iloc[:target].reset_index(drop=True), stats

    # verify in shuffled order until we have enough good images, so the result only
    # depends on the seed and the files on disk
    kept: list[int] = []
    rows = list(range(len(shuffled)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk in _chunks(rows, 512):
            paths = [image_dir / shuffled["path"].iat[i] for i in chunk]
            for i, ok in zip(chunk, pool.map(is_readable, paths), strict=True):
                if ok:
                    kept.append(i)
                else:
                    stats["corrupt"] += 1
                    log.debug("corrupt image: %s", shuffled["path"].iat[i])
                if len(kept) == target:
                    break
            if len(kept) == target:
                break
    return shuffled.iloc[kept].reset_index(drop=True), stats


def make_splits(
    selected: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    test_ids: set[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Split 90/5/5 with a fixed seed, or use ``test_ids`` (standard AVA test list) as test."""
    rng = np.random.default_rng(seed + 1)
    df = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
    n = len(df)

    if test_ids is not None:
        is_test = df["image_id"].isin(test_ids)
        if not is_test.any():
            raise ValueError("none of the selected images are in the test list")
        test, rest = df[is_test], df[~is_test]
    else:
        n_test = round(n * test_frac)
        test, rest = df.iloc[:n_test], df.iloc[n_test:]

    n_val = round(n * val_frac)
    val, train = rest.iloc[:n_val], rest.iloc[n_val:]
    return {"train": train, "val": val, "test": test}


def prepare(
    ava_txt: Path,
    image_dir: Path,
    out_dir: Path,
    subset: int | None = 40_000,
    seed: int = DEFAULT_SEED,
    test_list: Path | None = None,
    verify: bool = True,
    workers: int = 8,
) -> dict:
    """Parse labels, pick the subset, write ``{split}.csv`` + ``meta.json`` to ``out_dir``."""
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    t0 = time.perf_counter()

    labels = parse_ava_txt(Path(ava_txt))
    log.info("parsed %d labelled images from %s", len(labels), ava_txt)
    index = index_images(image_dir)
    log.info("found %d candidate image files under %s", len(index), image_dir)

    selected, stats = select_subset(labels, index, image_dir, subset, seed, verify, workers)
    log.info(
        "selected %d images (skipped %d missing, %d corrupt)",
        len(selected),
        stats["missing"],
        stats["corrupt"],
    )
    if subset is not None and len(selected) < subset:
        log.warning("only %d usable images, fewer than requested subset %d", len(selected), subset)

    test_ids = None
    if test_list is not None:
        text = Path(test_list).read_text(encoding="utf-8")
        test_ids = {int(tok) for tok in text.split() if tok.isdigit()}

    splits = make_splits(selected, seed, test_ids=test_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["image_id", "path", *PROB_COLS, "mean", "num_votes"]
    for name, df in splits.items():
        df[columns].to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")

    meta = {
        "seed": seed,
        "subset_requested": subset,
        "test_split": "standard_list" if test_ids is not None else "random",
        "sizes": {name: len(df) for name, df in splits.items()},
        **stats,
        "ava_txt": str(ava_txt),
        "image_dir": str(image_dir),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def build_transforms(train: bool, input_size: int = INPUT_SIZE) -> T.Compose:
    """Train: resize 256 -> random crop 224 -> flip. Eval/inference: resize straight to 224.

    No crop at eval: composition matters for aesthetics. Inference must use the eval path.
    """
    bilinear = T.InterpolationMode.BILINEAR
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        resize = round(input_size * 256 / 224)
        return T.Compose(
            [
                T.Resize((resize, resize), interpolation=bilinear, antialias=True),
                T.RandomCrop(input_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ]
        )
    return T.Compose(
        [
            T.Resize((input_size, input_size), interpolation=bilinear, antialias=True),
            T.ToTensor(),
            normalize,
        ]
    )


class AVADataset(Dataset):
    """Yields ``(image [3,H,W] float32, score distribution [10] float32)``."""

    def __init__(
        self, split_csv: Path, image_dir: Path, train: bool, input_size: int = INPUT_SIZE
    ) -> None:
        df = pd.read_csv(split_csv, encoding="utf-8")
        self.image_ids = df["image_id"].to_numpy()
        self.paths = [Path(image_dir) / p for p in df["path"]]
        self.dists = torch.from_numpy(df[PROB_COLS].to_numpy(dtype=np.float32))
        self.transform = build_transforms(train, input_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), self.dists[i]


def load_splits(
    splits_dir: Path, image_dir: Path, input_size: int = INPUT_SIZE
) -> dict[str, AVADataset]:
    """Datasets for train (augmented) and val/test (eval transforms) from ``prepare`` output."""
    splits_dir = Path(splits_dir)
    return {
        name: AVADataset(
            splits_dir / f"{name}.csv", image_dir, train=name == "train", input_size=input_size
        )
        for name in SPLITS
    }


def _cmd_prepare(args: argparse.Namespace) -> None:
    meta = prepare(
        args.ava_txt,
        args.images,
        args.out,
        subset=None if args.subset <= 0 else args.subset,
        seed=args.seed,
        test_list=args.test_list,
        verify=not args.no_verify,
        workers=args.workers,
    )
    print(json.dumps(meta, indent=2))


def _cmd_check(args: argparse.Namespace) -> None:
    datasets = load_splits(args.splits, args.images)
    for name, ds in datasets.items():
        print(f"{name:>5}: {len(ds)} images")

    loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    images, dists = next(iter(loader))
    sums = dists.sum(dim=1)
    print(f"batch images: {list(images.shape)} {images.dtype}")
    print(f"batch dists:  {list(dists.shape)} sums min={sums.min():.6f} max={sums.max():.6f}")
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4):
        raise SystemExit("distribution sums are not ~1")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="pick subset and write split CSVs")
    p.add_argument("--ava-txt", type=Path, required=True, help="path to AVA.txt")
    p.add_argument("--images", type=Path, required=True, help="folder containing <id>.jpg")
    p.add_argument("--out", type=Path, default=Path("data/ava_splits"))
    p.add_argument("--subset", type=int, default=40_000, help="number of images; <=0 means all")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--test-list", type=Path, help="standard AVA test id list (one id per line)")
    p.add_argument("--no-verify", action="store_true", help="skip decoding check of each image")
    p.add_argument("--workers", type=int, default=8, help="threads for the decoding check")
    p.set_defaults(func=_cmd_prepare)

    c = sub.add_parser("check", help="print split sizes and one batch")
    c.add_argument("--splits", type=Path, default=Path("data/ava_splits"))
    c.add_argument("--images", type=Path, required=True)
    c.add_argument("--batch-size", type=int, default=8)
    c.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    c.set_defaults(func=_cmd_check)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
