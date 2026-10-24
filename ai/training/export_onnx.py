"""Export a NIMA checkpoint to ONNX with a sidecar JSON and an onnxruntime parity check.

Outputs ``<out>.onnx`` (input ``input`` [N,3,224,224] float32; outputs ``dist`` [N,10] and
``embedding`` [N,1280]; dynamic batch) and ``<out>.json`` with the preprocessing settings the
runtime must use, the model version, file sha256 and test metrics.

Run from ``ai/``::

    python -m training.export_onnx --checkpoint checkpoints/best.pt --metrics models/metrics.json

    # no trained weights yet: ImageNet backbone + untrained head, so the pipeline can run
    # locally (embeddings/grouping are meaningful, aesthetic scores are not)
    python -m training.export_onnx --untrained
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from photoholic_ai.preprocess import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE
from training.dataset_ava import AVADataset
from training.model import NIMA, NUM_BINS, load_checkpoint

log = logging.getLogger(__name__)

OPSET = 17
PARITY_ATOL = 1e-4
INPUT_NAME = "input"
OUTPUT_NAMES = ["dist", "embedding"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export(model: NIMA, out: Path, opset: int = OPSET) -> None:
    model.eval()
    dummy = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        out,
        input_names=[INPUT_NAME],
        output_names=OUTPUT_NAMES,
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
        opset_version=opset,
        external_data=False,  # one self-contained ~9 MB file
        dynamo=True,
    )
    onnx.checker.check_model(onnx.load(str(out)))


def parity_inputs(splits: Path | None, images: Path | None, n: int, seed: int) -> torch.Tensor:
    """``n`` random test-split images (eval transforms), or random tensors without data."""
    if splits is None or images is None:
        return torch.randn(
            n, 3, INPUT_SIZE, INPUT_SIZE, generator=torch.Generator().manual_seed(seed)
        )
    ds = AVADataset(Path(splits) / "test.csv", images, train=False)
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    return torch.stack([ds[int(i)][0] for i in idx])


def parity_check(model: NIMA, onnx_path: Path, x: torch.Tensor) -> dict[str, float]:
    """Max abs difference between PyTorch and onnxruntime, for a batch and for batch size 1."""
    model.eval()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref_dist, ref_emb = (t.numpy() for t in model(x))
    dist, emb = sess.run(OUTPUT_NAMES, {INPUT_NAME: x.numpy()})
    single, _ = sess.run(OUTPUT_NAMES, {INPUT_NAME: x[:1].numpy()})
    return {
        "dist": float(np.abs(dist - ref_dist).max()),
        "embedding": float(np.abs(emb - ref_emb).max()),
        "dist_batch1": float(np.abs(single - ref_dist[:1]).max()),
    }


def build_sidecar(
    model: NIMA,
    onnx_path: Path,
    version: str,
    trained: bool,
    source: str,
    metrics: dict[str, Any] | None,
    parity: dict[str, float],
    opset: int,
) -> dict[str, Any]:
    return {
        "model": {
            "name": f"nima_{model.backbone_name.replace('_', '')}",
            "version": version,
            "backbone": model.backbone_name,
            "trained": trained,
            "source": source,
        },
        "onnx": {
            "file": onnx_path.name,
            "sha256": sha256_file(onnx_path),
            "opset": opset,
            "input": {"name": INPUT_NAME, "shape": ["N", 3, INPUT_SIZE, INPUT_SIZE]},
            "outputs": {
                "dist": ["N", NUM_BINS],
                "embedding": ["N", model.embedding_dim],
            },
        },
        # must match training.dataset_ava.build_transforms(train=False)
        "preprocess": {
            "input_size": INPUT_SIZE,
            "resize": "bilinear_antialias",
            "color": "RGB",
            "layout": "NCHW",
            "scale": 1 / 255,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "metrics": metrics,
        "parity_max_abs_diff": parity,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="best.pt from training.train")
    src.add_argument(
        "--untrained",
        action="store_true",
        help="ImageNet backbone + random head (placeholder until a trained model exists)",
    )
    p.add_argument("--out", type=Path, default=Path("models/aesthetic_v1.onnx"))
    p.add_argument("--metrics", type=Path, help="metrics.json from training.evaluate")
    p.add_argument("--version", default=None, help="default 1.0.0, or 0.0.0-untrained")
    p.add_argument("--backbone", default="mobilenet_v2", help="for --untrained only")
    p.add_argument("--splits", type=Path, help="use test images for the parity check")
    p.add_argument("--images", type=Path)
    p.add_argument("--opset", type=int, default=OPSET)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true", help="random backbone (tests only)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("onnx_ir").setLevel(logging.WARNING)  # very chatty optimiser passes

    if args.untrained:
        model = NIMA(args.backbone, pretrained=not args.no_pretrained)
        torch.manual_seed(args.seed)
        source = "untrained head" + ("" if args.no_pretrained else ", imagenet backbone")
        version = args.version or "0.0.0-untrained"
    else:
        model, _ = load_checkpoint(args.checkpoint)
        source = str(args.checkpoint)
        version = args.version or "1.0.0"

    metrics = None
    if args.metrics is not None:
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))

    export(model, args.out, args.opset)
    x = parity_inputs(args.splits, args.images, n=8, seed=args.seed)
    parity = parity_check(model, args.out, x)
    log.info("parity max abs diff: %s", parity)
    if max(parity["dist"], parity["dist_batch1"]) >= PARITY_ATOL:
        raise SystemExit(f"parity check failed: {parity} (limit {PARITY_ATOL})")

    sidecar = build_sidecar(
        model, args.out, version, not args.untrained, source, metrics, parity, args.opset
    )
    sidecar_path = args.out.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size_mb:.1f} MB) and {sidecar_path}")
    if args.untrained:
        log.warning("UNTRAINED model: aesthetic scores are placeholders until you export best.pt")
    return sidecar


if __name__ == "__main__":
    main()
