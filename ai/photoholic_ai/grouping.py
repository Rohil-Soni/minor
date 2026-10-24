"""Group burst / near-duplicate shots using capture time and embedding similarity (plan Phase 6).

Two images are joined (union-find, so groups chain A~B~C) when either

- both have capture times ``<= burst_gap_s`` apart **and** cosine similarity
  ``>= sim_threshold_timed``, or
- cosine similarity ``>= sim_threshold_untimed`` (re-takes / near-duplicates), but when both
  have capture times only if they are ``<= session_gap_h`` apart, so the same view shot on
  different days (e.g. the moon every night) is not merged into one group.

Embeddings are L2-normalised, so cosine similarity is a dot product. Images that match
nothing form groups of one.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np

from photoholic_ai.config import Config

log = logging.getLogger(__name__)

METHODS = ("union_find", "agglomerative")
_BLOCK = 1024  # rows of the similarity matrix computed at once
# EXIF "2026:06:01 10:15:22"; also the "2026-06-01 10:15:22" display form of older engine output
_DATETIME = re.compile(r"^(\d{4})[:-](\d{2})[:-](\d{2})[ T](\d{2}):(\d{2}):(\d{2})")


def parse_capture_time(rec: dict) -> float | None:
    """Seconds (naive local camera time, as a POSIX-style number) or None.

    Uses ``date_time_original`` + ``subsec_time_original``, falling back to ``date_time``.
    """
    for field, subsec_field in (
        ("date_time_original", "subsec_time_original"),
        ("date_time", None),
    ):
        value = rec.get(field)
        if not isinstance(value, str):
            continue
        m = _DATETIME.match(value.strip())
        if not m:
            continue
        try:
            dt = datetime(*map(int, m.groups()), tzinfo=timezone.utc)
        except ValueError:  # e.g. the "0000:00:00 00:00:00" placeholder
            continue
        seconds = dt.timestamp()
        subsec = rec.get(subsec_field) if subsec_field else None
        if isinstance(subsec, str) and subsec.strip().isdigit():
            seconds += float("0." + subsec.strip())
        return seconds
    return None


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _union_find_labels(
    embeddings: np.ndarray, times: Sequence[float | None], cfg: Config
) -> np.ndarray:
    n = len(embeddings)
    t = np.array([np.nan if x is None else x for x in times], dtype=np.float64)
    uf = _UnionFind(n)
    for start in range(0, n, _BLOCK):
        rows = slice(start, min(start + _BLOCK, n))
        sim = embeddings[rows] @ embeddings.T
        dt = np.abs(t[rows, None] - t[None, :])  # NaN when either photo has no time
        close = dt <= cfg.burst_gap_s  # NaN -> False
        other_session = dt > cfg.session_gap_h * 3600  # NaN -> False: untimed stays eligible
        edges = (close & (sim >= cfg.sim_threshold_timed)) | (
            (sim >= cfg.sim_threshold_untimed) & ~other_session
        )
        for i, j in zip(*np.nonzero(edges), strict=True):
            i += start
            if i < j:
                uf.union(int(i), int(j))
    return np.array([uf.find(i) for i in range(n)])


def _agglomerative_labels(embeddings: np.ndarray, cfg: Config) -> np.ndarray:
    """Average-linkage clustering on cosine distance; ignores capture time."""
    from sklearn.cluster import AgglomerativeClustering

    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=1.0 - cfg.sim_threshold_untimed,
    )
    return model.fit_predict(embeddings)


def group_images(
    ids: Sequence[str],
    embeddings: np.ndarray,
    times: Sequence[float | None],
    cfg: Config | None = None,
    method: str = "union_find",
) -> list[list[str]]:
    """Groups of ids. Groups and members keep input order (sorted by first member)."""
    cfg = cfg or Config()
    if not len(ids) == len(embeddings) == len(times):
        raise ValueError("ids, embeddings and times must have the same length")
    if method not in METHODS:
        raise ValueError(f"unknown grouping method {method!r}, choose from {METHODS}")
    if len(ids) == 0:
        return []

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if method == "union_find":
        labels = _union_find_labels(embeddings, times, cfg)
    else:
        labels = _agglomerative_labels(embeddings, cfg)

    groups: dict[int, list[str]] = {}
    for label, image_id in zip(labels, ids, strict=True):
        groups.setdefault(int(label), []).append(image_id)
    result = list(groups.values())  # dict keeps first-seen order
    log.info("grouped %d images into %d groups", len(ids), len(result))
    return result
