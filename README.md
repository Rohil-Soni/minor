
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

Quick Start (build)
-------------------
To build the engine (Rust):

```powershell
cd path\to\repo\image-reader
cargo build --release
```


'''This Readme Was Built With The help of AI to save time'''