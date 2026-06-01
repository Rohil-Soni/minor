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