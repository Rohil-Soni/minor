import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from tests.conftest import CORRUPT_ID, MISSING_ID, N_IMAGES, ZERO_VOTE_ID  # noqa: E402
from training import dataset_ava as ava  # noqa: E402


def test_parse_normalises_and_drops_zero_votes(fake_ava):
    ava_txt, _ = fake_ava
    df = ava.parse_ava_txt(ava_txt)

    assert ZERO_VOTE_ID not in set(df["image_id"])
    assert len(df) == N_IMAGES + 2
    np.testing.assert_allclose(df[ava.PROB_COLS].sum(axis=1), 1.0)
    uniform = df[df["image_id"] == MISSING_ID].iloc[0]
    assert uniform["mean"] == pytest.approx(5.5)


def test_parse_rejects_wrong_file(tmp_path):
    bad = tmp_path / "not_ava.txt"
    bad.write_text("1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="AVA.txt"):
        ava.parse_ava_txt(bad)


def test_prepare_skips_missing_and_corrupt(fake_ava, tmp_path):
    ava_txt, image_dir = fake_ava
    meta = ava.prepare(ava_txt, image_dir, tmp_path / "splits", subset=None, workers=2)

    assert meta["missing"] == 1
    assert meta["corrupt"] == 1
    assert sum(meta["sizes"].values()) == N_IMAGES
    assert json.loads((tmp_path / "splits" / "meta.json").read_text(encoding="utf-8")) == meta

    frames = {s: pd.read_csv(tmp_path / "splits" / f"{s}.csv") for s in ava.SPLITS}
    ids = [set(f["image_id"]) for f in frames.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert {MISSING_ID, CORRUPT_ID}.isdisjoint(ids[0] | ids[1] | ids[2])
    assert all("\\" not in p for p in frames["train"]["path"])  # portable relative paths


def test_prepare_subset_is_deterministic(fake_ava, tmp_path):
    ava_txt, image_dir = fake_ava
    a = ava.prepare(ava_txt, image_dir, tmp_path / "a", subset=20, seed=7, workers=2)
    ava.prepare(ava_txt, image_dir, tmp_path / "b", subset=20, seed=7, workers=2)
    ava.prepare(ava_txt, image_dir, tmp_path / "c", subset=20, seed=8, workers=2)

    assert sum(a["sizes"].values()) == 20
    train = {d: pd.read_csv(tmp_path / d / "train.csv")["image_id"].tolist() for d in "abc"}
    assert train["a"] == train["b"]
    assert train["a"] != train["c"]


def test_prepare_uses_standard_test_list(fake_ava, tmp_path):
    ava_txt, image_dir = fake_ava
    test_list = tmp_path / "test.jpgl"
    test_list.write_text("1000\n1001\n1002\n", encoding="utf-8")

    meta = ava.prepare(
        ava_txt, image_dir, tmp_path / "splits", subset=None, test_list=test_list, workers=2
    )

    assert meta["test_split"] == "standard_list"
    test = pd.read_csv(tmp_path / "splits" / "test.csv")
    assert set(test["image_id"]) == {1000, 1001, 1002}


def test_dataset_batch_shapes(fake_ava, tmp_path):
    ava_txt, image_dir = fake_ava
    ava.prepare(ava_txt, image_dir, tmp_path / "splits", subset=None, workers=2)
    datasets = ava.load_splits(tmp_path / "splits", image_dir)

    images, dists = next(iter(torch.utils.data.DataLoader(datasets["train"], batch_size=4)))
    assert images.shape == (4, 3, 224, 224)
    assert images.dtype == torch.float32
    assert dists.shape == (4, 10)
    torch.testing.assert_close(dists.sum(dim=1), torch.ones(4), atol=1e-5, rtol=0)


def test_eval_transform_is_deterministic_resize():
    img = Image.fromarray(np.full((30, 90, 3), 128, dtype=np.uint8))
    tf = ava.build_transforms(train=False)
    x1, x2 = tf(img), tf(img)

    assert x1.shape == (3, 224, 224)
    torch.testing.assert_close(x1, x2)
    expected = (128 / 255 - ava.IMAGENET_MEAN[0]) / ava.IMAGENET_STD[0]
    assert x1[0, 112, 112].item() == pytest.approx(expected, abs=1e-5)
