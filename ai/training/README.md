# Training the aesthetic model (NIMA on AVA)

We train on a **random 40k subset of AVA** on a free cloud GPU (Kaggle or Colab).
Nothing here hard-codes paths: every script takes the data locations as arguments.
All commands run from the `ai/` folder.

| Step | Command | Phase |
|------|---------|-------|
| Pick subset + write splits | `python -m training.dataset_ava prepare ...` | 2 |
| Sanity-check a batch | `python -m training.dataset_ava check ...` | 2 |
| Train (resumable) | `python -m training.train ...` | 3 |
| Evaluate on test | `python -m training.evaluate ...` | 3 |
| Export ONNX | `python -m training.export_onnx ...` | 4 |

## Expected data layout

Any AVA copy works as long as it has:

- `AVA.txt`: one line per image: `index image_id v1..v10 tag1 tag2 challenge_id`
- a folder of `<image_id>.jpg` files (sub-folders are fine; they are searched recursively)
- optional: the standard test list (e.g. `aesthetics_image_lists/generic_test.jpgl`, one id per line)

`prepare` skips ids with no image file and images that fail to decode, and logs both counts
to `meta.json`. The split is fixed-seed (`--seed 42`): train 90% / val 5% / test 5%, or the
standard test list if you pass `--test-list`.

## Kaggle (recommended: AVA is already hosted there)

1. **Create → New Notebook.** In the right panel: *Accelerator* → GPU (T4 or P100), *Internet* → on.
2. **Add Input** → search `AVA aesthetic` → add a dataset that contains `AVA.txt` and the images.
3. Find the paths (dataset folder names differ between uploads):

   ```bash
   !find /kaggle/input -name AVA.txt
   !find /kaggle/input -name "*.jpg" | head -3
   ```

4. Get the code and the extra packages. Kaggle already has a CUDA build of torch/torchvision,
   so **don't** install `requirements-train.txt` there (it would replace torch):

   ```bash
   !git clone https://github.com/Techusic/minor.git
   %cd minor/ai
   !pip install -q -r requirements.txt onnx
   ```

5. Prepare the 40k subset. Outputs go to `/kaggle/working` (the input is read-only):

   ```bash
   !python -m training.dataset_ava prepare \
       --ava-txt /kaggle/input/<dataset>/AVA.txt \
       --images  /kaggle/input/<dataset>/images \
       --out     /kaggle/working/ava_splits \
       --subset 40000 --workers 4
   !python -m training.dataset_ava check \
       --splits /kaggle/working/ava_splits \
       --images /kaggle/input/<dataset>/images --num-workers 2
   ```

   `check` must print `batch images: [8, 3, 224, 224]` and distribution sums of 1.000000.

6. Train (about 5 head epochs + up to 12 full epochs; on a T4 roughly 5–10 min per epoch for 36k images):

   ```bash
   !python -m training.train \
       --splits /kaggle/working/ava_splits \
       --images /kaggle/input/<dataset>/images \
       --out    /kaggle/working/checkpoints \
       --batch-size 64 --num-workers 4
   ```

   - Writes `last.pt` every epoch, `best.pt` when val EMD improves, and `train_log.csv`.
   - Out of GPU memory → `--batch-size 32`. Uses one GPU (on "T4 x2" the second one sits idle).
   - Session timed out? Re-run the **same command** with `--resume auto`. It continues from
     `last.pt` with the same lr schedule and shuffling. To split on purpose, add `--stop-after N`.
   - Keep the outputs: *Save Version → Save & Run All*, or download `/kaggle/working/checkpoints`.
     Kaggle sessions stop after 12 h, and `/kaggle/working` is wiped when an interactive session ends.

7. Evaluate on the held-out test split:

   ```bash
   !python -m training.evaluate \
       --checkpoint /kaggle/working/checkpoints/best.pt \
       --splits /kaggle/working/ava_splits \
       --images /kaggle/input/<dataset>/images \
       --out    /kaggle/working/metrics.json
   ```

   Prints SRCC, PLCC, binary accuracy (mean ≥ 5) and EMD. Target: **SRCC ≥ 0.50**
   (about 0.55–0.65 is normal for MobileNetV2 on a 40k subset). Below 0.45 something is wrong;
   check the labels and normalisation. Copy `metrics.json` to `ai/models/metrics.json` in the repo.

### Training defaults (plan §5 Phase 3)

| Stage | Epochs | Trainable | Learning rate |
|-------|--------|-----------|---------------|
| A | `--epochs-head 5` | head only (backbone + BatchNorm frozen) | Adam 1e-3 |
| B | `--epochs-full 12` | everything | backbone 5e-6, head 1e-4, cosine decay; early stop after 3 epochs with no val gain |

AMP (float16) and channels-last are on automatically on CUDA (`--no-amp` to disable).
`--backbone efficientnet_b0` swaps the backbone.

## Google Colab

Colab has no AVA copy, so the images must be copied in first. Put them on the **local disk**
(`/content`), not on Google Drive: reading 40k small files from Drive is very slow.

1. *Runtime → Change runtime type* → GPU.
2. Copy AVA into `/content/ava`, for example with the Kaggle CLI (upload your own
   `kaggle.json` API token when prompted) or by unzipping archives you have stored in Drive.
3. Same commands as Kaggle steps 4–5, with `/content/ava/...` for the inputs and
   `/content/ava_splits` for `--out`. Copy checkpoints to Drive at the end to keep them.

## Local (CPU only)

Useful for testing the code, too slow for real training:

```powershell
ai\.venv\Scripts\python.exe -m pip install -r ai\requirements-train.txt
cd ai
.venv\Scripts\python.exe -m training.dataset_ava prepare --ava-txt D:\AVA\AVA.txt --images D:\AVA\images --out data\ava_splits --subset 40000
```

`ai/data/` and `ai/checkpoints/` are gitignored.
