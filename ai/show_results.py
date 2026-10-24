"""Print a finished run's results in the terminal (reads results.json; touches no photos).

python ai/show_results.py "D:\\Photos\\Trip\\.photoholic"           # summary + largest groups
python ai/show_results.py runs\\Photos_v2 --groups 20               # more groups
python ai/show_results.py runs\\Photos_v2 --group 5                 # every photo in group 5
python ai/show_results.py runs\\Photos_v2 --rejected                # blur / decode rejects
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from photoholic_ai import bridge


def name(path: str | None) -> str:
    # works for Windows paths on any OS
    return path.replace("\\", "/").rsplit("/", 1)[-1] if path else "-"


def fmt(x: float | None, nd: int = 2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def table(headers: list[str], rows: list[list[str]], right: set[int]) -> str:
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)
    ]

    def line(cells: list[str]) -> str:
        return "  ".join(
            c.rjust(w) if i in right else c.ljust(w)
            for i, (c, w) in enumerate(zip(cells, widths, strict=True))
        )

    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])


def load(run_dir: Path) -> tuple[dict, dict]:
    results_path = run_dir / bridge.RESULTS_NAME
    if not results_path.is_file():
        raise SystemExit(f"no {bridge.RESULTS_NAME} in {run_dir}. Run ai/run_pipeline.py first.")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    scored_path = run_dir / bridge.SCORED_NAME
    scores = {}
    if scored_path.is_file():
        scored = json.loads(scored_path.read_text(encoding="utf-8"))
        scores = {r["id"]: r for r in scored["images"]}
    return results, scores


def show_summary(results: dict) -> None:
    s = results["stats"]
    w = results["weights"]
    model = results.get("model") or {}
    print("PHOTOHOLIC RESULTS")
    print("=" * 50)
    rows = [
        ("Input photos", s["input"]),
        ("Rejected as blurry", s["rejected_sharpness"]),
        ("Could not be read", s["rejected_decode"] + s["decode_errors"]),
        ("Scored by the AI model", s["scored"]),
        ("Groups = best picks kept", s["clusters"]),
        ("Near-duplicates set aside", s["scored"] - s["clusters"]),
        ("Reduction (of scored)", f"{s['reduction_pct']}%"),
    ]
    for label, value in rows:
        print(f"  {label:<28}{value:>10}")
    print(f"  {'Weights (sharp / aesthetic)':<28}{w['sharpness']:>5} / {w['aesthetic']}")
    if model:
        trained = "trained" if model.get("trained") else "UNTRAINED placeholder"
        print(f"  {'Model':<28}{model.get('name')} {model.get('version')} ({trained})")
    sizes = Counter(len(c["members"]) for c in results["clusters"])
    singles = sizes.pop(1, 0)
    multi = sum(sizes.values())
    print(
        f"\n  {singles} single shots, {multi} groups of 2+ "
        f"(largest {max((len(c['members']) for c in results['clusters']), default=0)})"
    )


def show_groups(results: dict, scores: dict, n: int) -> None:
    clusters = sorted(results["clusters"], key=lambda c: -len(c["members"]))[:n]
    rows = []
    for c in clusters:
        best = c["members"][0]
        sc = scores.get(best["id"], {})
        rows.append([
            str(c["cluster_id"]), str(len(c["members"])), name(best["path"]),
            fmt(best["final_score"]), fmt(best["sharpness_norm"]), fmt(sc.get("aesthetic_mean")),
            name(c["members"][-1]["path"]) if len(c["members"]) > 1 else "",
        ])  # fmt: skip
    print(f"\nLARGEST {len(rows)} GROUPS (best pick = rank 1)")
    print(table(["group", "photos", "best pick", "final", "sharp", "aesth", "worst in group"],
                rows, right={0, 1, 3, 4, 5}))  # fmt: skip


def show_group(results: dict, scores: dict, cluster_id: int) -> None:
    cluster = next((c for c in results["clusters"] if c["cluster_id"] == cluster_id), None)
    if cluster is None:
        raise SystemExit(f"no group {cluster_id} (groups are 0..{len(results['clusters']) - 1})")
    rows = []
    for m in cluster["members"]:
        sc = scores.get(m["id"], {})
        rows.append([
            str(m["rank"]), name(m["path"]), fmt(m["final_score"]), fmt(m["sharpness_norm"]),
            fmt(m["aesthetic_norm"]), fmt(sc.get("aesthetic_mean")),
            "BEST" if m["id"] == cluster["best_id"] else "",
        ])  # fmt: skip
    print(f"\nGROUP {cluster_id}: {len(rows)} photos")
    print(table(["rank", "photo", "final", "sharp_norm", "aes_norm", "aes_mean", ""],
                rows, right={0, 2, 3, 4, 5}))  # fmt: skip
    print(
        "\nfinal = w_sharp * sharp_norm + w_aes * aes_norm  (sharp_norm is relative to this group)"
    )


def show_rejected(results: dict, n: int) -> None:
    rejected = sorted(results.get("rejected", []), key=lambda r: r.get("sharpness_score") or 0)
    unscored = results.get("unscored", [])
    print(
        f"\nREJECTED ({len(rejected)} by the engine, {len(unscored)} failed to decode in scoring)"
    )
    rows = [[name(r["path"]), r.get("reason", "?"), fmt(r.get("sharpness_score"), 1)]
            for r in rejected[:n]]  # fmt: skip
    rows += [[name(r["path"]), "decode_error", "-"] for r in unscored[:n]]
    print(table(["photo", "reason", "sharpness"], rows, right={2}))
    if len(rejected) > n:
        print(f"... and {len(rejected) - n} more (see report.csv)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", type=Path, help="the .photoholic folder (or --run-dir) of a run")
    p.add_argument("--groups", type=int, default=10, help="how many of the largest groups")
    p.add_argument("--group", type=int, help="show every photo of one group")
    p.add_argument("--rejected", action="store_true", help="list rejected photos")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # non-ASCII file names on Windows consoles
    results, scores = load(args.run_dir)
    if args.group is not None:
        show_group(results, scores, args.group)
    elif args.rejected:
        show_rejected(results, args.groups)
    else:
        show_summary(results)
        show_groups(results, scores, args.groups)
        print(
            f"\nMore: --group <id> for one group, --rejected for rejects, "
            f"report.csv for every photo ({args.run_dir})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
