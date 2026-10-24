"""Per-image report (``report.csv``) and optional EDA charts.

One CSV row per input image, including those rejected for blur or that failed to decode, with
all scores, its group, rank and EXIF. Charts (needs matplotlib): aesthetic histogram and
sharpness vs ISO / shutter speed.
"""

from __future__ import annotations

import logging
import re
from fractions import Fraction
from pathlib import Path

import pandas as pd

from photoholic_ai import bridge

log = logging.getLogger(__name__)

REPORT_NAME = "report.csv"
COLUMNS = [
    "id", "path", "arw_path", "status", "cluster_id", "cluster_size", "rank", "is_best",
    "sharpness_score", "sharpness_norm", "aesthetic_mean", "aesthetic_std", "aesthetic_norm",
    "final_score", "iso", "shutter_s", "focal_length_mm", "date_time_original", "error",
]  # fmt: skip
_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:/\d+)?")


def parse_number(value: object) -> float | None:
    """First number in an EXIF display string: ``"400"``, ``"1/250 s"``, ``"35 mm"``."""
    if value is None:
        return None
    m = _NUMBER.search(str(value))
    if not m:
        return None
    try:
        return float(Fraction(m.group()))
    except (ValueError, ZeroDivisionError):
        return None


def build_table(
    filtered: bridge.Record, scored: bridge.Record, results: bridge.Record
) -> pd.DataFrame:
    scores = {r["id"]: r for r in scored["images"]}
    placement = {}
    for cluster in results["clusters"]:
        for member in cluster["members"]:
            placement[member["id"]] = (cluster, member)

    rows = []
    for rec in filtered["images"]:
        sc = scores.get(rec["id"], {})
        cluster, member = placement.get(rec["id"], (None, {}))
        if cluster is None:
            status = sc.get("status", "not_scored")
        else:
            status = "best" if cluster["best_id"] == rec["id"] else "duplicate"
        rows.append(
            {
                "id": rec["id"],
                "path": rec["path"],
                "arw_path": rec.get("arw_path"),
                "status": status,
                "cluster_id": cluster["cluster_id"] if cluster else None,
                "cluster_size": len(cluster["members"]) if cluster else None,
                "rank": member.get("rank"),
                "is_best": status == "best",
                "sharpness_score": rec["sharpness_score"],
                "sharpness_norm": member.get("sharpness_norm"),
                "aesthetic_mean": sc.get("aesthetic_mean"),
                "aesthetic_std": sc.get("aesthetic_std"),
                "aesthetic_norm": member.get("aesthetic_norm"),
                "final_score": member.get("final_score"),
                "iso": parse_number(rec.get("iso")),
                "shutter_s": parse_number(rec.get("shutter_speed")),
                "focal_length_mm": parse_number(rec.get("focal_length")),
                "date_time_original": rec.get("date_time_original"),
                "error": sc.get("error"),
            }
        )
    for rej in filtered.get("rejected", []):
        rows.append(
            {
                "path": rej.get("path"),
                "status": f"rejected_{rej.get('reason', 'unknown')}",
                "is_best": False,
                "sharpness_score": rej.get("sharpness_score"),
            }
        )
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.astype({"cluster_id": "Int64", "cluster_size": "Int64", "rank": "Int64"})


def write_csv(df: pd.DataFrame, run_dir: Path) -> Path:
    path = Path(run_dir) / REPORT_NAME
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def write_charts(df: pd.DataFrame, run_dir: Path) -> list[Path]:
    """PNG charts for the EDA deliverable. Returns the files written (none without data)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping charts (pip install matplotlib)")
        return []

    out_dir = Path(run_dir) / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    scored = df.dropna(subset=["aesthetic_mean"])
    if not scored.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(scored["aesthetic_mean"], bins=20, range=(1, 10), color="#4C78A8")
        best = scored[scored["is_best"]]
        if not best.empty:
            ax.hist(best["aesthetic_mean"], bins=20, range=(1, 10), color="#F58518",
                    label="best picks")  # fmt: skip
            ax.legend()
        ax.set(xlabel="aesthetic score (mean)", ylabel="images", title="Aesthetic scores")
        written.append(_save(fig, out_dir / "aesthetic_hist.png"))

    for column, label, log_x in (("iso", "ISO", True), ("shutter_s", "shutter (s)", True)):
        data = df.dropna(subset=[column, "sharpness_score"])
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        rejected = data["status"].str.startswith("rejected")
        ax.scatter(data.loc[~rejected, column], data.loc[~rejected, "sharpness_score"],
                   s=14, alpha=0.7, label="kept", color="#4C78A8")  # fmt: skip
        ax.scatter(data.loc[rejected, column], data.loc[rejected, "sharpness_score"],
                   s=14, alpha=0.7, label="rejected (blur)", color="#E45756")  # fmt: skip
        ax.set(xlabel=label, ylabel="sharpness (Laplacian variance)",
               title=f"Sharpness vs {label}", yscale="log")  # fmt: skip
        if log_x:
            ax.set_xscale("log")
        ax.legend()
        written.append(_save(fig, out_dir / f"sharpness_vs_{column}.png"))
    return written


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
