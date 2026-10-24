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