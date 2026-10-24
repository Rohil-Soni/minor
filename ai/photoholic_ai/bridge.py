"""Bridge files between pipeline stages (plan §3).

- ``filtered_images.json``: Rust engine -> Python. Schema v1 envelope, or the legacy v0 bare
  array, which is wrapped as ``{"schema_version": 0, "images": [...], "rejected": []}``
  with ids assigned here.
- ``scored_images.json`` + ``embeddings.npy``: scorer -> grouping/selection.
- ``results.json``: final output (written by selection).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

Record = dict[str, Any]

FILTERED_NAME = "filtered_images.json"
SCORED_NAME = "scored_images.json"
EMBEDDINGS_NAME = "embeddings.npy"
RESULTS_NAME = "results.json"
SCHEMA_VERSION = 1

EXIF_FIELDS = (
    "arw_path",
    "focal_length",
    "iso",
    "shutter_speed",
    "date_time",
    "date_time_original",
    "subsec_time_original",
)


class BridgeError(ValueError):
    """A bridge file is missing, malformed or from an unsupported schema version."""


def _fail(path: Path, msg: str) -> BridgeError:
    return BridgeError(f"{path}: {msg}")


def _read_json(path: Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise BridgeError(f"{path}: file not found. Run the previous pipeline stage first.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise _fail(path, f"not valid JSON ({e})") from e


def write_json(path: Path, obj: Any) -> None:
    """UTF-8 JSON via a temp file, so readers never see a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _check_image(path: Path, i: int, rec: Any) -> Record:
    if not isinstance(rec, dict):
        raise _fail(path, f"images[{i}] is not an object")
    if not isinstance(rec.get("path"), str) or not rec["path"]:
        raise _fail(path, f"images[{i}] has no 'path'")
    score = rec.get("sharpness_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise _fail(path, f"images[{i}] ({rec['path']}) has no numeric 'sharpness_score'")
    out = {field: rec.get(field) for field in EXIF_FIELDS}
    out.update(rec)
    out["sharpness_score"] = float(score)
    return out


def load_filtered(path: Path) -> Record:
    """Load ``filtered_images.json`` (v1 envelope or legacy v0 bare array), validated."""
    path = Path(path)
    data = _read_json(path)

    if isinstance(data, list):
        log.info("%s: legacy v0 bare array, wrapping and assigning ids", path)
        images = [_check_image(path, i, r) for i, r in enumerate(data)]
        images.sort(key=lambda r: r["path"])
        for n, rec in enumerate(images, start=1):
            rec["id"] = f"img_{n:06d}"
        return {"schema_version": 0, "images": images, "rejected": []}

    if not isinstance(data, dict):
        raise _fail(path, "expected a JSON object or array at the top level")
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise _fail(path, f"unsupported schema_version {version!r} (expected {SCHEMA_VERSION})")
    if not isinstance(data.get("images"), list):
        raise _fail(path, "'images' must be a list")

    images = [_check_image(path, i, r) for i, r in enumerate(data["images"])]
    ids = [r.get("id") for r in images]
    if any(not isinstance(i, str) or not i for i in ids):
        raise _fail(path, "every image needs a string 'id' in schema v1")
    if len(set(ids)) != len(ids):
        raise _fail(path, "image ids are not unique")

    rejected = data.get("rejected", [])
    if not isinstance(rejected, list):
        raise _fail(path, "'rejected' must be a list")
    return {**data, "images": images, "rejected": rejected}


def write_scored(run_dir: Path, scored: Record, embeddings: np.ndarray) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / EMBEDDINGS_NAME, embeddings.astype(np.float32, copy=False))
    write_json(run_dir / SCORED_NAME, scored)


def load_scored(run_dir: Path) -> tuple[Record, np.ndarray]:
    """``scored_images.json`` and ``embeddings.npy``, checked against each other."""
    run_dir = Path(run_dir)
    path = run_dir / SCORED_NAME
    scored = _read_json(path)
    if not isinstance(scored, dict) or scored.get("schema_version") != SCHEMA_VERSION:
        raise _fail(path, f"expected schema_version {SCHEMA_VERSION}")
    if not isinstance(scored.get("images"), list):
        raise _fail(path, "'images' must be a list")

    emb_path = run_dir / EMBEDDINGS_NAME
    if not emb_path.is_file():
        raise BridgeError(f"{emb_path}: file not found")
    embeddings = np.load(emb_path)
    used = [r["embedding_index"] for r in scored["images"] if r.get("embedding_index") is not None]
    if embeddings.ndim != 2 or sorted(used) != list(range(len(embeddings))):
        raise _fail(path, f"embedding indices do not match {emb_path.name} {embeddings.shape}")
    return scored, embeddings
