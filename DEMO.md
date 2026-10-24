# Photoholic — demo guide

Run the full pipeline on a photo folder and show the results in a terminal.
Commands are for **Windows PowerShell**, run from the repo folder (`minor\`).
Your photos are only read; outputs go to the folder given with `--run-dir`.

## A. On a laptop that is already set up (about 2 minutes)

```powershell
# 1. cull a folder (Rust blur filter -> AI scoring -> grouping -> best pick per group)
ai\.venv\Scripts\python.exe ai\run_pipeline.py "C:\Users\rohil\Pictures\Photos" --run-dir runs\demo --charts

# 2. show the results
ai\.venv\Scripts\python.exe ai\show_results.py runs\demo                # summary + largest groups
ai\.venv\Scripts\python.exe ai\show_results.py runs\demo --group 0       # one group, ranked
ai\.venv\Scripts\python.exe ai\show_results.py runs\demo --rejected     # blurry / unreadable

# 3. open the per-photo report and charts
start runs\demo\report.csv
start runs\demo\charts
```

To show it **without waiting** for the 1–2 minute run, use a run that already exists:
`ai\.venv\Scripts\python.exe ai\show_results.py runs\Photos_v2`

Optional: `--action copy` copies each group's best photo (and its RAW) into
`<folder>\photoholic_picks\`. It never deletes or overwrites anything.

## B. On a fresh Windows PC (about 20 minutes, mostly downloads)

1. Install the tools (once):
   ```powershell
   winget install Git.Git
   winget install Python.Python.3.12
   winget install Rustlang.Rustup
   winget install BrechtSanders.WinLibs.POSIX.UCRT
   ```
   Close and reopen PowerShell, then: `rustup default stable-x86_64-pc-windows-gnu`.
   If `cargo build` says *"An Application Control policy has blocked this file"*,
   Windows Smart App Control is blocking it (Windows Security → App & browser control).

2. Get the code and build:
   ```powershell
   git clone https://github.com/Techusic/minor.git
   cd minor\image-reader
   cargo build --release
   cd ..\ai
   py -3.12 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Get the model (`ai\models\aesthetic_v1.onnx` + `.json`, ~9 MB, not in git):
   - copy both files from a teammate's `ai\models\` folder, **or**
   - make the placeholder (downloads PyTorch, ~500 MB):
     ```powershell
     .venv\Scripts\python.exe -m pip install -r requirements-train.txt
     .venv\Scripts\python.exe -m training.export_onnx --untrained
     cd ..
     ```

4. Run steps 1–3 of part A with your own photo folder.

5. Optional, prove it works: `ai\.venv\Scripts\python.exe -m pip install -r ai\requirements-dev.txt`
   then `ai\.venv\Scripts\python.exe -m pytest ai\tests` (109 tests) and
   `cd image-reader; cargo test --release` (Rust tests).

## What to point out

| Step | Where it shows | What it does |
|------|----------------|--------------|
| Rust engine | "Rejected as blurry" | Pairs JPEG/RAW, reads EXIF, scores the **sharpest region** of each photo (Laplacian variance on 12×12 tiles of a 2048 px copy) and drops ruined shots |
| AI model | `aes_mean` column | MobileNetV2 (NIMA) predicts a 1–10 score distribution and a 1280-d embedding in one pass (ONNX Runtime, ~25 ms/photo on CPU) |
| Grouping | "groups of 2+" | Burst/near-duplicate shots: within 10 s **and** embedding cosine ≥ 0.80, or cosine ≥ 0.92 at any time |
| Selection | `--group N` | `final = 0.4 · sharpness (relative within group) + 0.6 · aesthetic`; rank 1 is kept |
| Report | `report.csv`, `charts\` | Every photo with status, group, rank, scores, ISO/shutter (EDA) |

Measured on 1,221 Nikon D5600 photos (24 MP): 82 s end to end on a laptop CPU,
106 blurry rejects, 454 best picks (59% fewer photos to look at).

**Be upfront about this:** until the model is trained on AVA (Kaggle, see
`ai/training/README.md`), the aesthetic scores come from an untrained placeholder. The
summary prints `UNTRAINED placeholder`, and within a group the pick mostly follows
sharpness. Blur filtering and grouping do not depend on the model being trained.
