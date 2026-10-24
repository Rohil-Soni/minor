# Photoholic — Logs

## 2026-06-01 — Engine design

Summary: started designing the Photoholic pipeline and selecting crates.

### Pipeline Overview
The engine is split into four pipeline stages for throughput and predictable resource usage.

#### Stage 1 — Discovery (I/O bound)
- Single thread scans directories and groups RAW/JPEG pairs into a `FileGroup`.
- Sends `FileGroup` structs down a channel to the decoding stage.

#### Stage 2 — Decoding & Resizing (CPU bound)
- A `rayon` thread pool reads files into memory.
- JPEGs are decoded; RAWs extract embedded JPEG slices and decode.
- Resize to CNN input (e.g., 224x224) using a fast resampling (Nearest or Bilinear).

#### Stage 3 — Tensor Conversion & Inference (compute bound)
- 1–2 threads convert resized images to normalized tensors.
- Run inference via `ort` (ONNX Runtime).

#### Stage 4 — Action (I/O bound)
- Single thread receives scores, applies culling logic, and moves/deletes `FileGroup` pairs on disk.

### Decisions / Rationale
- Keep I/O stages single-threaded to avoid filesystem contention.
- Use `rayon` for CPU-parallel image decode and resize.
- Keep model inference isolated to a small pool to control GPU/CPU usage.

### Dependencies (selected crates)
Add these to `Cargo.toml` under `[dependencies]`:

```toml
walkdir = "2.5"           # I/O and discovery
crossbeam-channel = "0.5" # concurrency pipeline
rayon = "1.10"            # CPU parallelism
zune-jpeg = "0.4"         # high-speed JPEG decoding
ort = "2.0"               # ONNX Runtime for inference
```

---
Notes by Raghav

## 2026-10-24 — Rust ↔ Python bridge (Phase 1)

Summary: made the engine drivable from the Python orchestrator and formalised `filtered_images.json` as a versioned contract (see `AESTHETIC_NN_PLAN.md` §3.1).

### Changes
- CLI: `image-reader <INPUT_DIR> [--out <JSON_PATH>] [--min-sharpness 15.0]` replaces the hard-coded path. Default output is `<INPUT_DIR>/.photoholic/filtered_images.json`; `discovered_images.txt` is written next to it. Missing input dir → exit code 2.
- Images without readable EXIF are kept with `null` EXIF fields (previously silently dropped).
- Added `date_time_original` and `subsec_time_original` for burst grouping. Date fields are now the raw EXIF strings (`2026:06:01 10:15:22`).
- Images below the threshold (`low_sharpness`) or that fail to decode (`decode_error`) are listed in a `rejected` array.
- Output is a v1 envelope (`schema_version`, `generated_at`, `source_dir`, `min_sharpness`, `images`, `rejected`); images sorted by path with stable ids `img_000001…`.
- Discovery skips `.photoholic/`, `photoholic_picks/`, `photoholic_rejects/` so run outputs are never re-scanned.
- Sharpness algorithm unchanged.

### Decisions / Rationale
- `ort` is now an optional `onnx` feature (off by default): its prebuilt binaries are MSVC-only, and the team builds with the GNU toolchain (rustup `x86_64-pc-windows-gnu` + WinLibs MinGW-w64), no Visual Studio needed. Not used until Phase 10.
- Added `humantime` for the RFC 3339 `generated_at` timestamp instead of hand-rolling date math.
- Windows Smart App Control blocks cargo build scripts (os error 4551); it must be off to build.

## 2026-10-24 — Python environment & skeleton (Phase 0)

Summary: set up the `ai/` package layout from `AESTHETIC_NN_PLAN.md` §4, with dependencies and a test harness.

### Changes
- `ai/photoholic_ai/` package (bridge, imaging, preprocess, scorer, grouping, selection, report) and `ai/training/` (dataset, model, losses, train, evaluate, export); modules are docstring stubs naming the phase that implements them.
- `config.py` implemented: frozen `Config` dataclass with the §6 defaults, validation (weights sum to 1, action ∈ none/copy/move — never delete), optional `config.toml` + CLI overrides via `load_config`.
- `requirements.txt` (runtime), `requirements-train.txt` (torch etc.), `requirements-dev.txt` (pytest, ruff, black); major versions pinned.
- `pyproject.toml` with pytest, black and ruff config (line length 100, py310).
- `ai/tests/test_config.py`: 8 tests.
- `.gitignore`: model weights, checkpoints, datasets, Python caches.

### Setup
```powershell
py -3.12 -m venv ai\.venv
ai\.venv\Scripts\python.exe -m pip install -r ai\requirements-dev.txt
ai\.venv\Scripts\python.exe -m pytest ai/tests
```

## 2026-10-24 — AVA dataset preparation (Phase 2)

Summary: data pipeline for training the aesthetic model on a random 40k AVA subset, runnable on Kaggle/Colab GPUs.

### Changes
- `training/dataset_ava.py`:
  - parses `AVA.txt` → normalised 10-bin vote distributions (drops zero-vote rows and duplicates)
  - finds `<id>.jpg` recursively under the image folder
  - fixed-seed random subset (default 40k), skipping missing files and images that fail to decode
  - 90/5/5 train/val/test split, or the standard AVA test list via `--test-list`
  - writes `{train,val,test}.csv` (paths relative to the image folder, so splits move between machines) and `meta.json`
  - `AVADataset` + `load_splits` for torch; CLI commands `prepare` and `check`
- Transforms: train = resize 256 → random crop 224 → flip; eval = resize straight to 224 (no crop, composition matters). Inference must use the eval transforms.
- `photoholic_ai/preprocess.py` now holds `INPUT_SIZE`, `IMAGENET_MEAN`, `IMAGENET_STD`, the one definition training imports and the ONNX sidecar will carry.
- `training/README.md`: Kaggle / Colab / local setup.
- `tests/test_dataset_ava.py`: 7 tests on a synthetic AVA (missing, corrupt and zero-vote cases, determinism, test list, batch shapes).

### Decisions / Rationale
- Subset of 30–50k (default 40k) instead of full AVA (~32 GB); training on Kaggle, where AVA is already hosted, so nothing large is downloaded locally.
- On Kaggle/Colab, install `requirements.txt` + `onnx` only: their CUDA torch must not be replaced by `requirements-train.txt`.

## 2026-10-24 — Model & training (Phase 3)

Summary: NIMA model, EMD loss, a resumable two-stage training loop for Kaggle/Colab GPUs, and test-set evaluation.

### Changes
- `training/model.py`: `NIMA`, a pretrained backbone (`mobilenet_v2` default, `efficientnet_b0` option) → global average pool → **embedding (1280)** → Dropout(0.75) → Linear(10) → softmax. `forward` returns `(dist, embedding)`, with the embedding taken before dropout. Also `dist_mean_std` and `load_checkpoint`. MobileNetV2 has 2.24M params.
- `training/losses.py`: squared EMD (r=2) on the CDFs, computed in float32 so it is AMP-safe.
- `training/train.py`:
  - Stage A: head only, frozen backbone and BatchNorm, Adam 1e-3.
  - Stage B: full network, backbone 5e-6, head 1e-4, cosine decay, early stop after 3 epochs without val improvement.
  - AMP and channels-last on CUDA; everything seeded.
  - `last.pt` saved every epoch via temp file + rename; `best.pt` on val-EMD improvement; `train_log.csv` per epoch.
  - `--resume auto` keeps the optimizer, scheduler and scaler state, and shuffling is seeded per epoch, so a resumed run matches an uninterrupted one. `--stop-after N` splits a run across sessions. Changing schedule args on resume gives a warning.
- `training/evaluate.py`: SRCC, PLCC, binary accuracy (mean ≥ 5) and EMD on a split; writes `metrics.json` and warns below SRCC 0.45/0.50.
- `training/README.md`: Kaggle train/evaluate steps and the defaults table.
- Tests: 12 new (EMD values, mean/std, model shapes for both backbones, embedding-before-dropout, freezing, metrics, a CPU train → interrupt → resume → evaluate smoke run). 27 total.

### Pending
- Real training run on Kaggle (40k subset) and `ai/models/metrics.json` with SRCC ≥ 0.50.

## 2026-10-24 — ONNX export (Phase 4)

Summary: export the model to a single ONNX file plus a sidecar JSON that pins the preprocessing.

### Changes
- `training/export_onnx.py`:
  - input `input` [N,3,224,224]; outputs `dist` [N,10] and `embedding` [N,1280]; dynamic batch, opset 17
  - uses the dynamo exporter (needs `onnxscript`, added to `requirements-train.txt`)
  - `onnx.checker` + onnxruntime parity check (batch of 8 and batch of 1; fails above 1e-4)
- Sidecar `aesthetic_v1.json`: model name/version/backbone, `trained` flag, sha256, opset, I/O shapes, preprocessing (size, bilinear antialias, RGB, NCHW, 1/255, ImageNet mean/std), test metrics, parity diffs.
- `--untrained`: ImageNet backbone + random head, version `0.0.0-untrained`, `trained: false`. A placeholder so the full pipeline runs locally before the Kaggle model exists. Its embeddings (grouping) are meaningful; its aesthetic scores are not.
- `.gitignore`: `ai/models/aesthetic_*.json` (sidecars travel with the `.onnx`).

### Result
- `aesthetic_v1.onnx` is 9.1 MB; parity max abs diff on `dist` 6e-8.

## 2026-10-24 — Inference runtime (Phase 5)

Summary: score photos with the ONNX model, without torch (runtime only needs `requirements.txt`).

### Changes
- `bridge.py`: loads and validates `filtered_images.json` (v1 envelope; legacy v0 bare array is wrapped and given ids sorted by path), missing EXIF fields become `null`; clear `BridgeError` messages for malformed files; atomic UTF-8 JSON writes; `scored_images.json` + `embeddings.npy` write/load with an index consistency check.
- `imaging.py`: JPEG via `Image.draft` (fast DCT-scaled decode, both sides ≥ 448 px) + EXIF orientation; ARW via the rawpy embedded preview, rotated by the raw orientation flag, with a half-size development as fallback; any failure → `ImageDecodeError`.
- `preprocess.py`: `PreprocessSpec` read from the model sidecar (rejects unsupported settings); `to_array` resizes with Pillow bilinear, which antialiases exactly like torchvision `Resize`. A test checks it equals the training eval transform within 1e-5.
- `scorer.py`: `Scorer` picks CUDA → DirectML → CPU providers, checks the sidecar sha256 matches the `.onnx`, decodes on a thread pool, batches of 32, computes mean/std and L2-normalised embeddings; `score_filtered` writes the §3.2 files with `status: decode_error` for broken images.
- Tests: bridge (v0/v1/12 malformed cases), preprocess (hand-computed pixel, training parity), imaging (EXIF rotation, draft, corrupt JPEG/ARW), scorer (dummy ONNX built in the test, batching, decode errors, sha mismatch).

### Result
- 100 camera-sized JPEGs (6000×4000, 91 MB) + 1 corrupt file scored in 2.6 s on CPU (26 ms/image; target < 30 s). The corrupt file was recorded as `decode_error`.
- Not tested: real Sony ARW files (no samples available).

## 2026-10-24 — Grouping (Phase 6)

Summary: put burst / near-duplicate shots of the same moment into one group.

### Changes
- `grouping.py`:
  - `parse_capture_time`: `DateTimeOriginal` + `SubSecTimeOriginal` (e.g. `"45"` → +0.45 s), falling back to `DateTime`; accepts EXIF `:` and the older `-` format; `0000:00:00` placeholders → `None`.
  - `group_images` (union-find): join when both are timed, ≤ `burst_gap_s` (10 s) apart and cosine ≥ `sim_threshold_timed` (0.80), or whenever cosine ≥ `sim_threshold_untimed` (0.92).
  - Similarity computed in 1024-row blocks so memory stays bounded; groups keep input order; singletons are groups of one.
  - `method="agglomerative"`: sklearn average-linkage on cosine distance as an alternative (ignores time).
- Tests: two tight clusters + an outlier → 3 groups; the time gate in 6 cases; `burst_gap_s` from config; chaining across a burst; agglomerative; capture-time parsing.

### Pending
- Tune thresholds on a real photo folder (plan **[ASK]**).

## 2026-10-24 — Decision matrix & selection (Phase 7)

Summary: rank each group and pick one best photo, with scale-safe scores and a safe action stage.

### Changes
- `selection.py`:
  - `rank_group`:
    - `sharpness_norm` = min-max of `log1p(sharpness)` within the group (equal values or a group of one → 1.0)
    - `aesthetic_norm` = `(mean − 1) / 9` on the absolute scale, clipped; missing → contributes 0
    - `final = 0.4·sharp + 0.6·aes`
    - ties: higher aesthetic mean → lower std → earlier capture time → id
  - `select` builds `results.json` (§3.3), plus `unscored` (decode errors), `rejected` and stats. `reduction_pct` is the share of scored photos that are not a group's best pick.
  - `apply_action`: `none` (default) / `copy` (best JPEG + paired ARW → `photoholic_picks/`, keeping sub-folders; re-runs skip identical files) / `move` (non-best → `photoholic_rejects/`). Never deletes or overwrites (adds `_1`, `_2`); every operation is logged to `.photoholic/actions.json` for undo.
- Tests: plan edge cases (group of one, equal sharpness, missing aesthetic), log scaling, huge sharpness not swamping aesthetics, weights, tie-breaks, results schema/stats, action safety (none, copy idempotent, no overwrite, move).

## 2026-10-24 — Orchestrator CLI & report (Phase 8)

Summary: one command runs the whole pipeline and writes `results.json` + `report.csv` (+ charts).

### Changes
- `ai/run_pipeline.py "<folder>"`:
  - runs the Rust engine (`subprocess.run` with list args, no shell) → score → group → select → action → report
  - flags: `--engine`, `--model`, `--run-dir`, `--config`, `--skip-engine`, `--min-sharpness`, `--burst-gap`, `--sim`, `--sim-untimed`, `--w-sharp/--w-aes` (one implies the other), `--action`, `--grouping`, `--batch-size`, `--charts`
  - prints a summary with per-stage timings; clear one-line errors (exit 1) for a missing folder, engine or model, or bad JSON
- `photoholic_ai/report.py`:
  - `report.csv`: one row per input image incl. blur rejects and decode errors, with status (best/duplicate/rejected_…/decode_error), group, rank, all scores and numeric EXIF (`"1/250 s"` → 0.004)
  - `--charts`: aesthetic histogram and sharpness vs ISO / shutter (matplotlib, optional)
- `ai/main.py`: thin wrapper around `run_pipeline.main`, so `python ai/main.py <folder>` works.

### Result (acceptance, synthetic burst shoot)
- 100 × 24 MP JPEGs (648 MB), 27 scenes in bursts of 1–6, real EXIF times, 6 blurred frames, 2 without EXIF, 1 corrupt file.
- **33.8 s end to end on CPU** (target < 60 s): engine 18.6 s, score 4.6 s, report 10.6 s (first-run matplotlib font cache).
- 27 groups = 27 scenes, all pure, none split. All 6 blurred frames rejected; no blurred frame picked; both no-EXIF photos kept; corrupt file → `decode_error`. Reduction 71.3%.
- These are synthetic scenes, so real-photo threshold tuning is still pending.

## 2026-10-24 — Tests & docs (Phase 9)

Summary: end-to-end tests and README setup/usage; the whole pipeline is verified locally.

### Changes
- `tests/test_e2e.py`:
  - generated burst shoot (3 scenes × 2 frames + a blurred copy) → full pipeline with the dummy ONNX model
  - one test uses handmade engine output; one runs the **real Rust binary** with `--action copy` and checks the 3 picks are copied and the originals are byte-for-byte unchanged
  - a missing model exits with code 1
- README Quick Start: prerequisites (GNU toolchain, Smart App Control note), build, model (trained or `--untrained` placeholder), usage, outputs, options, decision matrix, tests.
- Report charts: no empty "rejected" legend entry (engine rejects carry no EXIF).

### Result
- 109 tests pass (pytest, ~80 s); ruff + black clean; `cargo build --release` clean.
- Final synthetic run (100 × 24 MP): **26.4 s end to end**; 27/27 scenes grouped exactly, 6/6 blurred frames rejected, no-EXIF photos kept, corrupt file reported.

### Still open
- Train on Kaggle and export the real model (SRCC ≥ 0.50).
- Tune grouping thresholds and weights on a real photo folder; test real Sony ARW files.
- Decide how to share the ~9 MB model (Git LFS / GitHub Release / drive link).

## 2026-10-24 — Tuning on real photos (Nikon D5600, 1,221 JPEGs)

Summary: first run on a real shoot. Grouping defaults confirmed; blur filter redesigned (approved by Rohil).

### Grouping
- All 1,221 photos have capture times. Consecutive shots 0–5 s apart: median cosine 0.88, p10 ≈ 0.79–0.82. Random (different-scene) pairs: p99.9 = 0.903, so `sim_threshold_untimed` 0.92 is safe and 0.90 would start merging scenes.
- Shooting style is re-takes of one subject every ~1.5–2 s with changing framing, so the defaults (10 s / 0.80 / 0.92) group **per subject** (e.g. 26 poster shots over 68 s → 1 group). Stricter settings (5 s / 0.85) split those groups into uneven fragments, not clean compositions.
- **Decision: keep the defaults** (one keeper per subject).

### Blur filter (engine, `calculate_sharpness`)
- Problem: whole-frame Laplacian variance at full resolution rejected **421/1221 (34%)**, including sharp photos with large plain areas (statue against sky, poles, every moon shot, night portraits). Scores overlapped: a misfocused poster scored 14.5, a sharp statue 13.3, so no threshold could separate them.
- New algorithm: greyscale → downscale to 2048 px long side → same 4-neighbour |Laplacian| → variance per tile on a 12×12 grid → **mean of the 2 sharpest tiles**. Resolution-independent; a sharp subject anywhere in the frame counts.
- Prototyped in Python on all 1,221 photos against hand-labelled examples; separation between blurry and sharp improved from overlapping to a 6.8× gap. 2048 px means slight focus misses only visible at 100% are not "ruined"; those are still ranked within their group.
- Default `--min-sharpness` 15 → **10** (Rust and `config.py`).
- Result: rejects **106 (8.7%)**; 315 rescued (moon series, night portraits, statue/poles), 0 newly rejected; Rust matches the prototype (Spearman 0.9998); engine stage 75 s → 56 s.
- Rust unit tests: flat image = 0, tiny images don't panic (the old loop underflowed on height 0), blur lowers the score, a sharp patch on a plain background scores like full texture, downscale keeps aspect.
- Borderline: a slightly soft moon (DSC_9778) and a sharp one (DSC_9795) both score 10.2; they are in the same burst, so selection decides.

## 2026-10-24 — GUI (branch `gui`)

Summary: local web GUI to review a finished run: all groups with all their photos, and a side-by-side compare of any photo against its group's pick.

### Changes
- `app/server.py`:
  - stdlib `ThreadingHTTPServer` + Pillow, no new dependencies
  - binds 127.0.0.1 only; serves only photos listed in the run (looked up by id, so no path traversal)
  - thumbnails (360 px) and previews (2048 px) are cached in `<run>/gui_cache/`; originals are served for 100% zoom
  - `POST /api/pick` saves manual picks to `<run>/user_picks.json` (choosing the automatic pick again removes the override); nothing is moved or deleted
- `app/static/` (vanilla HTML/CSS/JS, no build step, so it can be wrapped in Tauri later):
  - **Groups view**: every group with all photos together, pick first (gold) and rank badges; filters (bursts / all / singles / changed by you), sort (time / size), file-name search, thumbnail size
  - **Compare view**: pick vs any photo; zoom/pan synced on normalised image coordinates (wheel zooms around the cursor, drag, double-click, `Z` 100% from the original file, `F` fit); score table with ▲/▼ vs the pick; filmstrip; keyboard (← → photo, ↑ ↓ group, P pick, Esc)
  - **Rejected view**: blur rejects sorted by score, with a full-size viewer
- `ai/tests/test_gui_server.py`: 14 tests on a real pipeline run (payload, resized/original images, 404/400 for unknown ids, traversal and bad sizes, pick save/validate/restore/restart).
- Checked in the browser on the 1,221-photo run: no console errors.

### Found while testing
- The 54-photo moon group spans **4.1 days**: moon shots from different nights pass the "cosine ≥ 0.92 at any time" rule.

## 2026-10-24 — Grouping: one session cap for near-duplicates (branch `gui`)

Summary: the 0.92 near-duplicate rule no longer joins photos from different sessions (decided by Rohil).

### Changes
- New setting `session_gap_h` (default **3 h**, CLI `--session-gap`): when both photos have capture times, the `sim_threshold_untimed` rule only joins them if they are ≤ 3 h apart. Photos without a capture time keep the old behaviour. Must be ≥ `burst_gap_s`.
- GUI server ignores saved picks that no longer belong to their group (groups change when the pipeline is re-run).
- Tests: the 3 h boundary, same view two days apart, three nights of moon frames → one group per night (one group with `session_gap_h=72`), untimed still grouped, config validation, stale picks.

### Result on the 1,221-photo run
- The 4.1-day moon group became one group per night (e.g. 09-27: 4, 09-28: 11, 09-29: 13, 10-01: 5 photos), each with its own pick.
- Longest group span is now 5.4 min. Groups 454 → 459; nothing else changed noticeably.