06/01/2026
Started on the engine.
The Photoholic engine will be divided into four stages for pipelining:

Stage 1: Discovery (I/O bound) — 1 thread scans the directory, groups RAW/JPEG pairs, and sends the `FileGroup` struct down a channel.

Stage 2: Decoding & Resizing (CPU bound) — a `rayon` thread pool receives the path and reads the file into memory. If the file is a JPEG it is decoded; if it is a RAW, the embedded JPEG byte slice is extracted and decoded. The image is then resized to the CNN input size (e.g., 224x224) using a fast algorithm such as Nearest Neighbor or Bilinear.

Stage 3: Tensor Conversion & Inference (compute bound) — 1 to 2 threads convert the resized image into a normalized tensor and push it through the `ort` ONNX model.

Stage 4: Action (I/O bound) — 1 thread takes the output scores, executes the culling logic, and moves or deletes the `FileGroup` pairs on disk.

For all stages the following Rust crates have been decided:

# I/O and Discovery
walkdir = "2.5"

# Concurrency Pipeline
crossbeam-channel = "0.5"
rayon = "1.10"

# High-Speed Decoding
zune-jpeg = "0.4"

# for CNN loop with Microsoft ONNX Runtime
ort = "2.0"

- Raghav