"""Decision matrix: rank each group and pick its best photo (plan Phase 7).

Sharpness (raw Laplacian variance, unbounded) and aesthetics (1..10) are on different scales,
so they are normalised before combining:

- ``sharpness_norm``: ``log1p(sharpness)`` min-max scaled *within the group*
  (all equal, or a group of one -> 1.0)
- ``aesthetic_norm``: ``(aesthetic_mean - 1) / 9`` on the absolute scale, so a group of
  mediocre shots stays mediocre; a missing score contributes 0
- ``final_score = w_sharpness * sharpness_norm + w_aesthetic * aesthetic_norm``

Ties: higher aesthetic mean, then lower aesthetic std, then earlier capture time, then id.

The action stage is safe by default (``none``). ``copy`` copies each group's best (JPEG and
paired ARW) into ``<input_dir>/photoholic_picks/``; ``move`` moves the others into
``<input_dir>/photoholic_rejects/``. Files are never deleted or overwritten, and every
copy/move is logged to ``actions.json`` so it can be undone.
"""

from __future__ import annotations

import filecmp
import json
import logging
import math
import shutil
from collections.abc import Sequence
from pathlib import Path

from photoholic_ai import bridge
from photoholic_ai.config import Config
from photoholic_ai.grouping import parse_capture_time

log = logging.getLogger(__name__)

PICKS_DIR = "photoholic_picks"
REJECTS_DIR = "photoholic_rejects"
ACTIONS_NAME = "actions.json"


def _sharpness_norm(values: Sequence[float]) -> list[float]:
    logs = [math.log1p(max(v, 0.0)) for v in values]
    lo, hi = min(logs), max(logs)
    if hi - lo < 1e-12:
        return [1.0] * len(logs)
    return [(v - lo) / (hi - lo) for v in logs]


def _aesthetic_norm(mean: float | None) -> float | None:
    if mean is None:
        return None
    return min(max((mean - 1.0) / 9.0, 0.0), 1.0)


def rank_group(members: Sequence[bridge.Record], cfg: Config) -> list[bridge.Record]:
    """Score and rank one group. ``members`` need ``id``, ``sharpness_score`` and optionally
    ``aesthetic_mean``, ``aesthetic_std``, ``capture_time``. Returns new dicts, best first."""
    sharp = _sharpness_norm([m["sharpness_score"] for m in members])
    rows = []
    for m, s in zip(members, sharp, strict=True):
        a = _aesthetic_norm(m.get("aesthetic_mean"))
        final = cfg.w_sharpness * s + cfg.w_aesthetic * (a or 0.0)
        rows.append({**m, "sharpness_norm": s, "aesthetic_norm": a, "final_score": final})

    def key(r: bridge.Record) -> tuple:
        mean, std, t = r.get("aesthetic_mean"), r.get("aesthetic_std"), r.get("capture_time")
        return (
            -r["final_score"],
            -(mean if mean is not None else -math.inf),
            std if std is not None else math.inf,
            t if t is not None else math.inf,
            r["id"],
        )

    rows.sort(key=key)
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank
    return rows


def _round(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(x, nd)


def select(
    filtered: bridge.Record,
    scored: bridge.Record,
    groups: Sequence[Sequence[str]],
    cfg: Config,
) -> bridge.Record:
    """Build ``results.json`` (plan §3.3) from the bridge files and the grouping."""
    by_id = {r["id"]: r for r in filtered["images"]}
    scores = {r["id"]: r for r in scored["images"]}

    clusters = []
    for cluster_id, ids in enumerate(groups):
        members = []
        for image_id in ids:
            src, sc = by_id[image_id], scores[image_id]
            members.append(
                {
                    "id": image_id,
                    "path": src["path"],
                    "arw_path": src.get("arw_path"),
                    "sharpness_score": src["sharpness_score"],
                    "aesthetic_mean": sc.get("aesthetic_mean"),
                    "aesthetic_std": sc.get("aesthetic_std"),
                    "capture_time": parse_capture_time(src),
                }
            )
        ranked = rank_group(members, cfg)
        clusters.append(
            {
                "cluster_id": cluster_id,
                "best_id": ranked[0]["id"],
                "members": [
                    {
                        "id": r["id"],
                        "path": r["path"],
                        "arw_path": r["arw_path"],
                        "sharpness_norm": _round(r["sharpness_norm"]),
                        "aesthetic_norm": _round(r["aesthetic_norm"]),
                        "final_score": _round(r["final_score"]),
                        "rank": r["rank"],
                    }
                    for r in ranked
                ],
            }
        )

    unscored = [
        {"id": r["id"], "path": r["path"], "status": r["status"], "error": r.get("error")}
        for r in scored["images"]
        if r["status"] != "ok"
    ]
    rejected = filtered.get("rejected", [])
    n_scored = len(scored["images"]) - len(unscored)
    stats = {
        "input": len(filtered["images"]) + len(rejected),
        "rejected_sharpness": sum(r.get("reason") == "low_sharpness" for r in rejected),
        "rejected_decode": sum(r.get("reason") == "decode_error" for r in rejected),
        "decode_errors": len(unscored),
        "scored": n_scored,
        "clusters": len(clusters),
        # share of scored photos that are not a group's best pick
        "reduction_pct": round(100.0 * (1 - len(clusters) / n_scored), 1) if n_scored else 0.0,
    }
    return {
        "schema_version": bridge.SCHEMA_VERSION,
        "weights": {"sharpness": cfg.w_sharpness, "aesthetic": cfg.w_aesthetic},
        "model": scored.get("model"),
        "clusters": clusters,
        "unscored": unscored,
        "rejected": rejected,
        "stats": stats,
    }


def _relative(src: Path, input_dir: Path) -> Path:
    """Path below ``input_dir`` (keeps sub-folders), or just the name for outside files."""
    try:
        return src.resolve().relative_to(input_dir.resolve())
    except ValueError:
        return Path(src.name)


def _destination(src: Path, input_dir: Path, target_root: Path) -> Path:
    """Keep the sub-folder layout under ``target_root``; add ``_1``, ``_2``... on clashes."""
    dest = target_root / _relative(src, input_dir)
    n = 1
    while dest.exists():
        dest = dest.with_name(f"{src.stem}_{n}{src.suffix}")
        n += 1
    return dest


def _files(member: bridge.Record) -> list[Path]:
    paths = [Path(member["path"])]
    if member.get("arw_path"):
        paths.append(Path(member["arw_path"]))
    return paths


def apply_action(
    results: bridge.Record, input_dir: Path, action: str, run_dir: Path | None = None
) -> list[dict[str, str]]:
    """Copy best picks or move the rest. Never deletes, never overwrites."""
    if action == "none":
        return []
    if action not in ("copy", "move"):
        raise ValueError(f"unknown action {action!r}")

    input_dir = Path(input_dir)
    root = input_dir / (PICKS_DIR if action == "copy" else REJECTS_DIR)
    done: list[dict[str, str]] = []
    for cluster in results["clusters"]:
        for member in cluster["members"]:
            is_best = member["id"] == cluster["best_id"]
            if is_best != (action == "copy"):
                continue
            for src in _files(member):
                if not src.is_file():
                    log.warning("skipping missing file %s", src)
                    continue
                if action == "copy":
                    existing = root / _relative(src, input_dir)
                    if existing.is_file() and filecmp.cmp(src, existing, shallow=False):
                        continue  # already picked in an earlier run
                dest = _destination(src, input_dir, root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if action == "copy":
                    shutil.copy2(src, dest)
                else:
                    shutil.move(src, dest)
                done.append({"action": action, "src": str(src), "dst": str(dest)})

    log.info("%s: %d files -> %s", action, len(done), root)
    if run_dir is not None and done:
        log_path = Path(run_dir) / ACTIONS_NAME
        previous = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else []
        bridge.write_json(log_path, previous + done)
    return done


def write_results(run_dir: Path, results: bridge.Record) -> Path:
    path = Path(run_dir) / bridge.RESULTS_NAME
    bridge.write_json(path, results)
    return path
