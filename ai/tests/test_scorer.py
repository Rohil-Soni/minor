import json
import shutil

import numpy as np
import pytest
from PIL import Image

from photoholic_ai import bridge
from photoholic_ai.scorer import Scorer, dist_stats, l2_normalise, pick_providers, score_filtered


def make_photos(folder, colours):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, c in enumerate(colours):
        p = folder / f"IMG_{i:03d}.jpg"
        Image.new("RGB", (320, 240), c).save(p)
        paths.append(str(p))
    return paths


def test_dist_stats():
    dist = np.zeros((2, 10))
    dist[0, :] = 0.1
    dist[1, 6] = 1.0
    mean, std = dist_stats(dist)
    np.testing.assert_allclose(mean, [5.5, 7.0])
    np.testing.assert_allclose(std, [np.sqrt(8.25), 0.0])


def test_l2_normalise_handles_zero_rows():
    out = l2_normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0]])


def test_pick_providers_cpu_always_available():
    assert pick_providers()[-1] == "CPUExecutionProvider"
    with pytest.raises(ValueError, match="not available"):
        pick_providers(["MadeUpExecutionProvider"])


def test_score_paths_batches_and_decode_errors(dummy_model, tmp_path):
    paths = make_photos(tmp_path / "my photos", [(250, 0, 0), (0, 250, 0), (0, 0, 250), (9, 9, 9)])
    broken = tmp_path / "my photos" / "broken.jpg"
    broken.write_bytes(b"\xff\xd8 nope")
    paths.insert(2, str(broken))

    scorer = Scorer(dummy_model, batch_size=2, workers=2)
    seen = []
    results = scorer.score_paths(paths, progress=seen.append)

    assert [r.path for r in results] == paths  # order kept across batches
    assert seen == [2, 2, 1]
    assert [r.ok for r in results] == [True, True, False, True, True]
    assert "broken.jpg" in results[2].error
    for r in (r for r in results if r.ok):
        assert r.dist.shape == (10,) and r.dist.sum() == pytest.approx(1.0, abs=1e-5)
        assert 1.0 <= r.mean <= 10.0 and r.std >= 0
        assert np.linalg.norm(r.embedding) == pytest.approx(1.0, abs=1e-5)
    assert not np.allclose(results[0].dist, results[1].dist)  # different colours differ


def test_same_image_same_score_regardless_of_batch(dummy_model, tmp_path):
    paths = make_photos(tmp_path / "p", [(10, 120, 200)] * 3)
    one = Scorer(dummy_model, batch_size=1).score_paths(paths)
    three = Scorer(dummy_model, batch_size=3).score_paths(paths)
    np.testing.assert_allclose(one[0].dist, three[2].dist, atol=1e-6)


def test_score_filtered_writes_bridge_files(dummy_model, tmp_path):
    paths = make_photos(tmp_path / "in", [(255, 255, 0), (0, 255, 255)])
    bad = tmp_path / "in" / "bad.jpg"
    bad.write_bytes(b"")
    filtered = {
        "schema_version": 1,
        "images": [
            {"id": "img_000001", "path": paths[0], "sharpness_score": 100.0},
            {"id": "img_000002", "path": str(bad), "sharpness_score": 50.0},
            {"id": "img_000003", "path": paths[1], "sharpness_score": 30.0},
        ],
        "rejected": [],
    }
    run_dir = tmp_path / "in" / ".photoholic"
    scored, emb = score_filtered(filtered, Scorer(dummy_model), run_dir)

    loaded, loaded_emb = bridge.load_scored(run_dir)
    assert loaded == json.loads(json.dumps(scored))
    np.testing.assert_array_equal(loaded_emb, emb)
    assert emb.shape == (2, 1280)

    ok, err, ok2 = loaded["images"]
    assert (ok["status"], ok["embedding_index"], len(ok["aesthetic_dist"])) == ("ok", 0, 10)
    assert err["status"] == "decode_error" and err["aesthetic_mean"] is None
    assert err["embedding_index"] is None and "bad.jpg" in err["error"]
    assert ok2["embedding_index"] == 1
    assert loaded["model"]["name"] == "dummy" and len(loaded["model"]["onnx_sha256"]) == 64


def test_sidecar_sha_mismatch_rejected(dummy_model, tmp_path):
    copy = tmp_path / "m.onnx"
    shutil.copy(dummy_model, copy)
    sidecar = json.loads(dummy_model.with_suffix(".json").read_text(encoding="utf-8"))
    sidecar["onnx"]["sha256"] = "0" * 64
    copy.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="different model file"):
        Scorer(copy)


def test_missing_model_has_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_onnx"):
        Scorer(tmp_path / "nope.onnx")
