
Photoholic — AI-Driven Image Culling Engine

Project: An AI-driven, performance-optimized image culling system that automates selection of the "best" photo from a set of shots by combining objective sharpness metrics with learned aesthetic scoring and redundancy filtering.

Authors: Raghav Maheshwari, Ajinkya Borwankar, Rohil Soni

Commit message format
---------------------

1) Type — When to use it
- `feat` : Introducing a brand new feature or capability.
- `fix` : Patching a bug or error.
- `perf` : Code changes tied directly to improving performance/throughput.
- `refactor` : Rewriting code without changing its external behavior (e.g., renaming variables, splitting functions).
- `chore` : Routine tasks, updating dependencies, or changing configuration files.
- `docs` : Writing or updating documentation, READMEs, or comments.

2) The scope
This is the most critical part for a monorepo. The scope strictly defines where the change happened. Since you have three distinct domains, your team should agree on three primary scopes:
- `(engine)`: Anything inside the image-reader/ Rust folder.
- `(ai)`: Anything inside the ai/ Python folder.
- `(gui)`: Anything inside the app/ React/Tauri folder.
- `(repo)`: Root-level changes, like updating the main .gitignore.

3) The description
The description must be written in the imperative mood. Think of it as giving a command to the codebase.

Header format
-------------
`type(scope): concise description in the imperative mood`

Overview
--------
Photoholic replaces manual culling with a pipelined, data-driven engine that ranks images by technical sharpness and aesthetic quality, and groups near-duplicate "burst" frames for bulk decisions.

Key Features
- High-throughput Rust engine for disk I/O and preprocessing
- Objective sharpness scoring using Variance of Laplacian (DSP)
- Learned aesthetic scoring via an ONNX model (MobileNetV2 / NIMA style)
- Redundancy / burst clustering using embedding similarity or DBSCAN
- Configurable decision matrix to promote the best shot per cluster

Technical Pipeline
------------------
Stage 1 — Discovery (I/O bound)
- Single-threaded scanner walks folders, mines EXIF, and groups RAW/JPEG pairs into `FileGroup` items to feed the pipeline.

Stage 2 — Decoding & Resizing (CPU bound)
- A `rayon` thread pool performs fast decode of JPEGs or extracts embedded JPEGs from RAWs, then resizes thumbnails to the CNN input size (e.g., 224×224) using a fast resizer.

Stage 3 — Tensor Conversion & Inference (compute bound)
- Convert resized images to normalized tensors and run inference with ONNX Runtime (`ort`) to obtain aesthetic/likability scores.

Stage 4 — Action (I/O bound)
- Apply culling logic (Decision Matrix) to rank and move/delete files on disk; perform cluster-level selection for burst sequences.

Decision Matrix
----------------
Final Rank = Sharpness Score + Aesthetic Score

Clusters (Burst Logic)
- Images are embedded and clustered (temporal proximity + embedding similarity). The highest-ranked image in each cluster is kept; the rest are flagged for deletion or review.

Tech Stack
----------
- Language: Rust (performance engine) and Python (model training / analytics)
- ML: PyTorch for training, ONNX for inference
- Analytics: Pandas, Matplotlib for reports

Selected Rust crates
- I/O and Discovery: `walkdir = "2.5"`
- Concurrency / Pipeline: `crossbeam-channel = "0.5"`, `rayon = "1.10"`
- High-speed decoding: `zune-jpeg = "0.4"`
- ONNX runtime: `ort = "2.0"`

Project Deliverables
- Culling Engine: CLI/GUI to sort 100+ images in under ~60s on a modern machine
- Redundancy Filter: Reduce dataset volume by ~40–60% by clustering near-duplicates
- EDA Dashboard: Post-process reports showing camera/shooting performance statistics

Quick Start (Windows)
---------------------

### 1. Prerequisites
- **Rust** via [rustup](https://rustup.rs) with the GNU toolchain (no Visual Studio needed):
  `rustup default stable-x86_64-pc-windows-gnu`, plus a MinGW-w64 GCC for linking:
  `winget install BrechtSanders.WinLibs.POSIX.UCRT`
- **Python 3.10+** (3.12 tested): `winget install Python.Python.3.12`
- If `cargo build` fails with *"An Application Control policy has blocked this file
  (os error 4551)"*, Windows Smart App Control is blocking Cargo's build scripts.

### 2. Build the engine and install Python packages
```powershell
cd image-reader
cargo build --release
cd ..\ai
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Get the aesthetic model
The pipeline needs `ai/models/aesthetic_v1.onnx` + its sidecar `aesthetic_v1.json`
(not in git, ~9 MB).

- **Trained model**: train on Kaggle (see [ai/training/README.md](ai/training/README.md)),
  then export: `python -m training.export_onnx --checkpoint checkpoints/best.pt --metrics models/metrics.json`
- **Placeholder** until then (needs `requirements-train.txt`): ImageNet backbone with an
  untrained score head. Grouping works; aesthetic scores are meaningless.
  ```powershell
  .venv\Scripts\python.exe -m pip install -r requirements-train.txt
  .venv\Scripts\python.exe -m training.export_onnx --untrained
  ```

### 4. Cull a folder
```powershell
ai\.venv\Scripts\python.exe ai\run_pipeline.py "D:\Photos\Trip"              # report only
ai\.venv\Scripts\python.exe ai\run_pipeline.py "D:\Photos\Trip" --charts     # + EDA charts
ai\.venv\Scripts\python.exe ai\run_pipeline.py "D:\Photos\Trip" --action copy
```

Everything is written to `D:\Photos\Trip\.photoholic\`:

| File | Contents |
|------|----------|
| `filtered_images.json` | Rust engine output: sharp images + EXIF, `rejected` list |
| `scored_images.json`, `embeddings.npy` | aesthetic score distribution, mean/std, 1280-d embedding |
| `results.json` | groups, ranked members, best pick per group, stats |
| `report.csv` | one row per photo: status, group, rank, all scores, ISO/shutter |
| `charts/` | aesthetic histogram, sharpness vs ISO / shutter (`--charts`) |
| `actions.json` | every file copied/moved (for undo) |

Photos are **never deleted or overwritten**. `--action copy` copies each group's best
(JPEG + paired ARW) to `photoholic_picks\`; `--action move` moves the rest to
`photoholic_rejects\`. Default is `none` (report only).

Main options (defaults in `ai/photoholic_ai/config.py`, or a `--config config.toml`):

| Flag | Default | Meaning |
|------|---------|---------|
| `--min-sharpness` | 10 | blur threshold: Laplacian variance of the sharpest region (2048 px copy, top 2 of 12×12 tiles) |
| `--burst-gap` | 10 | max seconds between frames of a burst |
| `--sim` / `--sim-untimed` | 0.80 / 0.92 | cosine similarity to group (in a burst / otherwise) |
| `--w-sharp` / `--w-aes` | 0.4 / 0.6 | decision-matrix weights |
| `--skip-engine` | | reuse an existing `filtered_images.json` |
| `--grouping agglomerative` | | sklearn clustering instead of time + similarity |

Decision matrix: `final = w_sharp · sharpness_norm + w_aes · aesthetic_norm`, with sharpness
log-scaled and min-max normalised **within each group** (so its unbounded raw value cannot
swamp the aesthetic score) and aesthetics on the absolute 1–10 scale.

Show a finished run in the terminal (no re-run needed):
```powershell
ai\.venv\Scripts\python.exe ai\show_results.py "D:\Photos\Trip\.photoholic"            # summary + largest groups
ai\.venv\Scripts\python.exe ai\show_results.py "D:\Photos\Trip\.photoholic" --group 3  # one group, ranked
ai\.venv\Scripts\python.exe ai\show_results.py "D:\Photos\Trip\.photoholic" --rejected
```
For a step-by-step demo (e.g. for evaluation) see [DEMO.md](DEMO.md).

### GUI: review groups and compare with the pick
```powershell
ai\.venv\Scripts\python.exe app\server.py "D:\Photos\Trip\.photoholic"
```
Opens `http://127.0.0.1:8765/` in your browser (local only, nothing is uploaded):
- **Groups**: every group with all of its photos together, the pick first (gold). Filter
  bursts / singles / changed, sort by time or size, search a file name.
- **Compare**: click any photo to see it side by side with its group's pick. Zoom and pan
  are synced (wheel, drag, `Z` = 100%, `F` = fit), scores show ▲/▼ against the pick, the
  filmstrip switches photos (`←` `→`), `↑` `↓` change group.
- **Make this the pick** (`P`) saves your choice to `user_picks.json` in the run folder;
  photos are never moved or deleted.
- **Rejected**: the photos dropped as blurry, to double-check the filter.

### 5. Tests
```powershell
ai\.venv\Scripts\python.exe -m pip install -r ai\requirements-dev.txt -r ai\requirements-train.txt
ai\.venv\Scripts\python.exe -m pytest ai\tests
```


'''This Readme Was Built With The help of AI to save time'''