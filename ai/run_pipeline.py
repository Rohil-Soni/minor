"""Photoholic end to end: Rust engine -> aesthetic scoring -> grouping -> selection -> report.

    python ai/run_pipeline.py "D:\\Photos\\Trip"
    python ai/run_pipeline.py "D:\\Photos\\Trip" --action copy      # copy best picks
    python ai/run_pipeline.py "D:\\Photos\\Trip" --skip-engine      # reuse filtered_images.json

All outputs go to ``<input_dir>/.photoholic/`` (``filtered_images.json``,
``scored_images.json``, ``embeddings.npy``, ``results.json``, ``report.csv``, ``charts/``).
Photos are only read, unless ``--action copy|move`` is given (never delete, never overwrite).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm

from photoholic_ai import bridge, grouping, report, selection
from photoholic_ai.config import ACTIONS, load_config
from photoholic_ai.scorer import Scorer, score_filtered

log = logging.getLogger("photoholic")

AI_DIR = Path(__file__).resolve().parent
REPO_DIR = AI_DIR.parent
EXE = "image-reader.exe" if os.name == "nt" else "image-reader"
DEFAULT_ENGINE = REPO_DIR / "image-reader" / "target" / "release" / EXE
DEFAULT_MODEL = AI_DIR / "models" / "aesthetic_v1.onnx"
RUN_DIR = ".photoholic"


class PipelineError(Exception):
    """A stage failed in a way the user can fix (missing engine/model, bad input)."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cull a photo folder: keep the best shot of every burst / near-duplicate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_dir", type=Path, help="folder of JPEG/ARW photos (searched recursively)")
    p.add_argument("--engine", type=Path, default=DEFAULT_ENGINE, help="Rust image-reader binary")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="aesthetic ONNX model")
    p.add_argument("--run-dir", type=Path, help=f"outputs (default <input_dir>/{RUN_DIR})")
    p.add_argument("--config", type=Path, help="config.toml with overrides")
    p.add_argument("--skip-engine", action="store_true", help="reuse filtered_images.json")
    p.add_argument("--min-sharpness", type=float)
    p.add_argument("--burst-gap", type=float, dest="burst_gap_s", help="seconds")
    p.add_argument("--sim", type=float, dest="sim_threshold_timed", help="cosine, shots in a burst")
    p.add_argument("--sim-untimed", type=float, dest="sim_threshold_untimed")
    p.add_argument("--w-sharp", type=float, dest="w_sharpness")
    p.add_argument("--w-aes", type=float, dest="w_aesthetic")
    p.add_argument("--action", choices=ACTIONS, help="none (default), copy best, move rejects")
    p.add_argument("--grouping", choices=grouping.METHODS, default="union_find")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--charts", action="store_true", help="write EDA charts (needs matplotlib)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    # one weight implies the other
    if args.w_sharpness is not None and args.w_aesthetic is None:
        args.w_aesthetic = round(1.0 - args.w_sharpness, 6)
    elif args.w_aesthetic is not None and args.w_sharpness is None:
        args.w_sharpness = round(1.0 - args.w_aesthetic, 6)
    return args


@contextmanager
def stage(name: str, timings: dict[str, float]):
    log.info("== %s", name)
    t0 = time.perf_counter()
    yield
    timings[name] = time.perf_counter() - t0


def run_engine(engine: Path, input_dir: Path, out: Path, min_sharpness: float) -> None:
    if not engine.is_file():
        raise PipelineError(
            f"Rust engine not found: {engine}\n"
            "Build it with `cargo build --release` in image-reader/, pass --engine, "
            "or use --skip-engine to reuse an existing filtered_images.json."
        )
    cmd = [str(engine), str(input_dir), "--out", str(out), "--min-sharpness", str(min_sharpness)]
    log.debug("running %s", cmd)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise PipelineError(f"Rust engine failed with exit code {e.returncode}") from e


def run(argv: list[str] | None = None) -> bridge.Record:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise PipelineError(f"input folder not found: {input_dir}")
    run_dir = (args.run_dir or input_dir / RUN_DIR).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = run_dir / bridge.FILTERED_NAME

    overrides = {
        k: getattr(args, k)
        for k in ("min_sharpness", "burst_gap_s", "sim_threshold_timed", "sim_threshold_untimed",
                  "w_sharpness", "w_aesthetic", "action", "batch_size")
    }  # fmt: skip
    cfg = load_config(args.config, **overrides)
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    with stage("engine", timings):
        if args.skip_engine:
            log.info("skipping engine, reusing %s", filtered_path)
        else:
            run_engine(args.engine, input_dir, filtered_path, cfg.min_sharpness)
        filtered = bridge.load_filtered(filtered_path)

    with stage("score", timings):
        scorer = Scorer(args.model, batch_size=cfg.batch_size)
        with tqdm(total=len(filtered["images"]), desc="scoring", unit="img") as bar:
            scored, embeddings = score_filtered(filtered, scorer, run_dir, progress=bar.update)

    with stage("group", timings):
        by_id = {r["id"]: r for r in filtered["images"]}
        ok = [r for r in scored["images"] if r["status"] == "ok"]
        groups = grouping.group_images(
            [r["id"] for r in ok],
            embeddings[[r["embedding_index"] for r in ok]] if ok else embeddings,
            [grouping.parse_capture_time(by_id[r["id"]]) for r in ok],
            cfg,
            method=args.grouping,
        )

    with stage("select", timings):
        results = selection.select(filtered, scored, groups, cfg)
        selection.write_results(run_dir, results)
        actions = selection.apply_action(results, input_dir, cfg.action, run_dir)

    with stage("report", timings):
        table = report.build_table(filtered, scored, results)
        csv_path = report.write_csv(table, run_dir)
        charts = report.write_charts(table, run_dir) if args.charts else []

    total = time.perf_counter() - t_start
    print_summary(results, timings, total, run_dir, csv_path, charts, cfg.action, actions)
    return results


def print_summary(results, timings, total, run_dir, csv_path, charts, action, actions) -> None:
    s = results["stats"]
    model = results.get("model") or {}
    lines = [
        "",
        "Photoholic summary",
        f"  input images        {s['input']}",
        f"  rejected (blur)     {s['rejected_sharpness']}",
        f"  failed to decode    {s['rejected_decode'] + s['decode_errors']}",
        f"  scored              {s['scored']}",
        f"  groups (best picks) {s['clusters']}",
        f"  reduction           {s['reduction_pct']}%",
        "  time                "
        + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items())
        + f" | total {total:.1f}s",
        f"  results             {run_dir / bridge.RESULTS_NAME}",
        f"  report              {csv_path}",
    ]
    if charts:
        lines.append(f"  charts              {charts[0].parent}")
    if action != "none":
        lines.append(f"  action              {action}: {len(actions)} files")
    if model.get("trained") is False:
        lines.append("  NOTE: untrained model - aesthetic scores are placeholders")
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    verbose = "-v" in (argv or sys.argv) or "--verbose" in (argv or sys.argv)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        run(argv)
    except (PipelineError, bridge.BridgeError, FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
