"""End to end: generated photos -> (Rust engine) -> score -> group -> select -> report."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import ExifTags, Image, ImageFilter

import run_pipeline
from photoholic_ai import bridge

SCENES = [(200, 40, 40), (40, 160, 60), (30, 60, 200)]


def make_shoot(folder: Path) -> dict[str, int]:
    """3 scenes x 2 burst frames (textured, so sharp) + 1 blurred copy. Returns name -> scene."""
    rng = np.random.default_rng(0)
    folder.mkdir(parents=True)
    truth = {}
    for s, colour in enumerate(SCENES):
        for f in range(2):
            arr = np.clip(np.array(colour) + rng.normal(0, 25, (120, 160, 3)), 0, 255)
            img = Image.fromarray(arr.astype(np.uint8))
            exif = Image.Exif()
            exif.get_ifd(ExifTags.IFD.Exif)[
                ExifTags.Base.DateTimeOriginal
            ] = f"2026:06:01 10:{s * 5:02d}:{f:02d}"
            name = f"DSC{s}{f}.JPG"
            img.save(folder / name, exif=exif, quality=92)
            truth[name] = s
    blurred = Image.open(folder / "DSC00.JPG").filter(ImageFilter.GaussianBlur(8))
    blurred.save(folder / "DSC00_blur.JPG")
    return truth


def handmade_filtered(folder: Path, truth: dict[str, int]) -> None:
    """What the Rust engine would write, for machines without the engine built."""
    images = [
        {"id": f"img_{i:06d}", "path": str(folder / name), "arw_path": None,
         "sharpness_score": 500.0 + i,
         "date_time_original": f"2026:06:01 10:{truth[name] * 5:02d}:0{i % 2}"}
        for i, name in enumerate(sorted(truth), start=1)
    ]  # fmt: skip
    rejected = [{"path": str(folder / "DSC00_blur.JPG"), "reason": "low_sharpness",
                 "sharpness_score": 2.0}]  # fmt: skip
    run_dir = folder / ".photoholic"
    bridge.write_json(
        run_dir / bridge.FILTERED_NAME,
        {"schema_version": 1, "source_dir": str(folder), "min_sharpness": 15.0,
         "images": images, "rejected": rejected},
    )  # fmt: skip


def check_results(folder: Path, truth: dict[str, int]) -> dict:
    run_dir = folder / ".photoholic"
    results = json.loads((run_dir / bridge.RESULTS_NAME).read_text(encoding="utf-8"))

    assert results["schema_version"] == 1
    assert set(results) >= {"weights", "clusters", "stats", "unscored", "rejected"}
    for cluster in results["clusters"]:
        assert set(cluster) == {"cluster_id", "best_id", "members"}
        assert cluster["members"][0]["id"] == cluster["best_id"]
        assert [m["rank"] for m in cluster["members"]] == list(
            range(1, len(cluster["members"]) + 1)
        )
        for m in cluster["members"]:
            assert 0.0 <= m["final_score"] <= 1.0
            assert set(m) >= {"id", "path", "sharpness_norm", "aesthetic_norm", "final_score"}

    groups = sorted(sorted(truth[Path(m["path"]).name] for m in c["members"])
                    for c in results["clusters"])  # fmt: skip
    assert groups == [[0, 0], [1, 1], [2, 2]]  # bursts grouped by scene
    assert [Path(r["path"]).name for r in results["rejected"]] == ["DSC00_blur.JPG"]
    assert results["stats"]["clusters"] == 3 and results["stats"]["scored"] == 6

    with (run_dir / "report.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 7
    assert sum(r["is_best"] == "True" for r in rows) == 3
    for name in ("scored_images.json", "embeddings.npy", "filtered_images.json"):
        assert (run_dir / name).is_file()
    return results


def test_pipeline_with_handmade_engine_output(tmp_path, dummy_model):
    folder = tmp_path / "My Shoot"
    truth = make_shoot(folder)
    handmade_filtered(folder, truth)

    code = run_pipeline.main([str(folder), "--skip-engine", "--model", str(dummy_model)])
    assert code == 0
    check_results(folder, truth)


@pytest.mark.skipif(not run_pipeline.DEFAULT_ENGINE.is_file(), reason="Rust engine not built")
def test_full_pipeline_with_rust_engine_and_copy(tmp_path, dummy_model):
    folder = tmp_path / "Trip 2026"
    truth = make_shoot(folder)
    originals = {p.name: p.read_bytes() for p in folder.iterdir()}

    code = run_pipeline.main([str(folder), "--model", str(dummy_model), "--action", "copy"])
    assert code == 0
    results = check_results(folder, truth)

    picks = sorted(p.name for p in (folder / "photoholic_picks").iterdir())
    best = sorted(Path(m["path"]).name for c in results["clusters"] for m in c["members"]
                  if m["id"] == c["best_id"])  # fmt: skip
    assert picks == best and len(picks) == 3
    # the photos themselves are never modified
    assert {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()} == originals


def test_missing_model_fails_cleanly(tmp_path):
    folder = tmp_path / "f"
    folder.mkdir()
    shutil.copy(Path(__file__), folder / "not_a_photo.txt")
    handmade_filtered(folder, {})
    assert (
        run_pipeline.main([str(folder), "--skip-engine", "--model", str(tmp_path / "x.onnx")]) == 1
    )
