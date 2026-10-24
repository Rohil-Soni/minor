"""Aesthetic scoring with ONNX Runtime.

One forward pass per image gives a 10-bin score distribution (-> mean, std) and an
L2-normalised embedding used for grouping. Images are decoded on a thread pool and run
through the model in batches. A file that fails to decode is recorded as
``status: "decode_error"`` and skipped, never crashing the run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from photoholic_ai import bridge
from photoholic_ai.imaging import ImageDecodeError, load_rgb
from photoholic_ai.preprocess import PreprocessSpec, to_array

log = logging.getLogger(__name__)

# best first; filtered by what this onnxruntime build provides
PREFERRED_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
NUM_BINS = 10
BINS = np.arange(1, NUM_BINS + 1, dtype=np.float64)


@dataclass
class ScoreResult:
    path: str
    dist: np.ndarray | None = None
    mean: float | None = None
    std: float | None = None
    embedding: np.ndarray | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def pick_providers(requested: Sequence[str] | None = None) -> list[str]:
    available = ort.get_available_providers()
    if requested:
        missing = [p for p in requested if p not in available]
        if missing:
            raise ValueError(f"providers not available: {missing} (have {available})")
        return list(requested)
    return [p for p in PREFERRED_PROVIDERS if p in available]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dist_stats(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std over bins 1..10 for ``[N, 10]`` distributions."""
    dist = dist.astype(np.float64)
    mean = dist @ BINS
    std = np.sqrt((dist * (BINS[None, :] - mean[:, None]) ** 2).sum(axis=1))
    return mean, std


def l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


class Scorer:
    def __init__(
        self,
        onnx_path: Path,
        sidecar_path: Path | None = None,
        providers: Sequence[str] | None = None,
        batch_size: int = 32,
        workers: int | None = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.is_file():
            raise FileNotFoundError(
                f"model not found: {self.onnx_path}. Export one with "
                "`python -m training.export_onnx` (see ai/training/README.md)."
            )
        sidecar_path = Path(sidecar_path or self.onnx_path.with_suffix(".json"))
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"model sidecar not found: {sidecar_path}")
        self.sidecar: dict[str, Any] = json.loads(sidecar_path.read_text(encoding="utf-8"))

        sha = _sha256(self.onnx_path)
        expected = self.sidecar.get("onnx", {}).get("sha256")
        if expected and expected != sha:
            raise ValueError(
                f"{sidecar_path.name} was written for a different model file "
                f"(sha256 {expected[:12]}... != {sha[:12]}...). Re-export both together."
            )
        self.spec = PreprocessSpec.from_sidecar(self.sidecar)
        self.batch_size = batch_size
        self.workers = workers or min(8, os.cpu_count() or 4)

        self.session = ort.InferenceSession(
            str(self.onnx_path), providers=pick_providers(providers)
        )
        self.input_name = self.session.get_inputs()[0].name
        model = self.sidecar.get("model", {})
        self.model_info = {
            "name": model.get("name"),
            "version": model.get("version"),
            "trained": model.get("trained"),
            "onnx_sha256": sha,
            "providers": self.session.get_providers(),
        }
        if model.get("trained") is False:
            log.warning(
                "model %s is UNTRAINED: aesthetic scores are placeholders", model.get("version")
            )

    def _decode(self, path: str) -> np.ndarray | str:
        try:
            return to_array(load_rgb(Path(path)), self.spec)
        except ImageDecodeError as e:
            return str(e)

    def run_batch(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dist, emb = self.session.run(["dist", "embedding"], {self.input_name: batch})
        return dist, emb

    def score_paths(
        self, paths: Sequence[str], progress: Callable[[int], None] | None = None
    ) -> list[ScoreResult]:
        results: list[ScoreResult] = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for start in range(0, len(paths), self.batch_size):
                chunk = list(paths[start : start + self.batch_size])
                decoded = list(pool.map(self._decode, chunk))
                good = [i for i, d in enumerate(decoded) if isinstance(d, np.ndarray)]
                chunk_results = [ScoreResult(path=p) for p in chunk]
                for i, d in enumerate(decoded):
                    if isinstance(d, str):
                        log.warning("decode error: %s", d)
                        chunk_results[i].error = d
                if good:
                    dist, emb = self.run_batch(np.stack([decoded[i] for i in good]))
                    mean, std = dist_stats(dist)
                    emb = l2_normalise(emb)
                    for j, i in enumerate(good):
                        r = chunk_results[i]
                        r.dist, r.mean, r.std, r.embedding = dist[j], mean[j], std[j], emb[j]
                results.extend(chunk_results)
                if progress is not None:
                    progress(len(chunk))
        return results


def score_filtered(
    filtered: bridge.Record,
    scorer: Scorer,
    run_dir: Path | None = None,
    progress: Callable[[int], None] | None = None,
) -> tuple[bridge.Record, np.ndarray]:
    """Score every image of a loaded ``filtered_images.json``; optionally write the outputs."""
    images = filtered["images"]
    results = scorer.score_paths([r["path"] for r in images], progress)

    records, embeddings = [], []
    for rec, res in zip(images, results, strict=True):
        out = {
            "id": rec["id"],
            "path": rec["path"],
            "sharpness_score": rec["sharpness_score"],
            "aesthetic_mean": None,
            "aesthetic_std": None,
            "aesthetic_dist": None,
            "embedding_index": None,
            "status": "ok" if res.ok else "decode_error",
        }
        if res.ok:
            out["aesthetic_mean"] = round(float(res.mean), 4)
            out["aesthetic_std"] = round(float(res.std), 4)
            out["aesthetic_dist"] = [round(float(p), 5) for p in res.dist]
            out["embedding_index"] = len(embeddings)
            embeddings.append(res.embedding)
        else:
            out["error"] = res.error
        records.append(out)

    dim = scorer.sidecar.get("onnx", {}).get("outputs", {}).get("embedding", [None, 1280])[1]
    emb = np.stack(embeddings) if embeddings else np.zeros((0, dim), dtype=np.float32)
    scored = {
        "schema_version": bridge.SCHEMA_VERSION,
        "model": scorer.model_info,
        "images": records,
    }
    if run_dir is not None:
        bridge.write_scored(run_dir, scored, emb)
    return scored, emb
