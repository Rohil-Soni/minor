import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training import dataset_ava, evaluate, train  # noqa: E402
from training.losses import emd_loss  # noqa: E402
from training.model import NIMA, dist_mean_std, load_checkpoint  # noqa: E402


def one_hot(index: int) -> torch.Tensor:
    t = torch.zeros(1, 10)
    t[0, index] = 1.0
    return t


def test_emd_zero_for_identical():
    p = torch.softmax(torch.randn(4, 10), dim=1)
    assert emd_loss(p, p).item() == pytest.approx(0.0, abs=1e-7)


def test_emd_one_bin_shift():
    # CDFs differ by 1 in exactly one of 10 positions -> sqrt(1/10)
    assert emd_loss(one_hot(4), one_hot(5)).item() == pytest.approx(math.sqrt(0.1))
    # two bins apart -> sqrt(2/10); further apart costs more
    assert emd_loss(one_hot(4), one_hot(6)).item() == pytest.approx(math.sqrt(0.2))


def test_emd_reduction_none():
    p, t = torch.cat([one_hot(0), one_hot(9)]), torch.cat([one_hot(0), one_hot(0)])
    per = emd_loss(p, t, reduction="none")
    assert per.shape == (2,)
    assert per[0].item() == pytest.approx(0.0)
    assert per[1].item() == pytest.approx(math.sqrt(0.9))


def test_dist_mean_std():
    uniform = torch.full((1, 10), 0.1)
    mean, std = dist_mean_std(uniform)
    assert mean.item() == pytest.approx(5.5)
    assert std.item() == pytest.approx(math.sqrt(8.25))
    mean, std = dist_mean_std(one_hot(6))
    assert (mean.item(), std.item()) == (pytest.approx(7.0), pytest.approx(0.0))


@pytest.mark.parametrize("backbone", ["mobilenet_v2", "efficientnet_b0"])
def test_model_outputs(backbone):
    model = NIMA(backbone, pretrained=False).eval()
    dist, emb = model(torch.randn(2, 3, 224, 224))
    assert dist.shape == (2, 10)
    assert emb.shape == (2, 1280)
    torch.testing.assert_close(dist.sum(dim=1), torch.ones(2))


def test_embedding_is_before_dropout():
    model = NIMA(pretrained=False).train()
    dropout = model.head[0]
    assert isinstance(dropout, torch.nn.Dropout) and dropout.p == 0.75

    seen = {}
    dropout.register_forward_hook(lambda m, inp, out: seen.update(inp=inp[0]))
    _, emb = model(torch.randn(2, 3, 224, 224))
    assert seen["inp"] is emb  # returned embedding is the dropout *input*


def test_freeze_backbone():
    model = NIMA(pretrained=False)
    model.freeze_backbone(True)
    assert not any(p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    model.freeze_backbone(False)
    assert all(p.requires_grad for p in model.features.parameters())


def test_unknown_backbone():
    with pytest.raises(ValueError, match="unknown backbone"):
        NIMA("resnet9000", pretrained=False)


def test_compute_metrics_perfect_and_constant():
    rng = np.random.default_rng(0)
    target = rng.dirichlet(np.ones(10), size=50)
    perfect = evaluate.compute_metrics(target, target)
    assert perfect["srcc"] == pytest.approx(1.0)
    assert perfect["plcc"] == pytest.approx(1.0)
    assert perfect["accuracy"] == 1.0
    assert perfect["emd"] == pytest.approx(0.0)

    constant = evaluate.compute_metrics(np.full_like(target, 0.1), target)
    assert constant["srcc"] is None and constant["plcc"] is None


def _read_log(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_train_resume_evaluate_smoke(fake_ava, tmp_path):
    ava_txt, image_dir = fake_ava
    splits = tmp_path / "splits"
    dataset_ava.prepare(ava_txt, image_dir, splits, subset=None, workers=2)
    out = tmp_path / "ckpt dir"
    common = [
        "--splits", str(splits), "--images", str(image_dir), "--out", str(out),
        "--batch-size", "4", "--limit-batches", "2", "--num-workers", "0",
        "--no-pretrained", "--device", "cpu", "--epochs-head", "1", "--epochs-full", "2",
    ]  # fmt: skip

    # session 1 ends after 2 of 3 epochs, like a Kaggle timeout
    train.main([*common, "--stop-after", "2"])
    assert (out / "best.pt").exists() and (out / "last.pt").exists()
    assert [r["stage"] for r in _read_log(out / "train_log.csv")] == ["head", "full"]

    # session 2 continues where it stopped instead of starting over
    train.main([*common, "--resume", "auto"])
    rows = _read_log(out / "train_log.csv")
    assert [int(r["epoch"]) for r in rows] == [0, 1, 2]
    assert [r["stage"] for r in rows] == ["head", "full", "full"]
    # cosine schedule carried over: 5e-6 at stage start, then half way down
    assert float(rows[1]["lr_backbone"]) == pytest.approx(5e-6)
    assert float(rows[2]["lr_backbone"]) == pytest.approx(2.5e-6)

    # a finished run has nothing left to do
    train.main([*common, "--resume", "auto"])
    assert len(_read_log(out / "train_log.csv")) == 3

    model, ckpt = load_checkpoint(out / "last.pt")
    assert ckpt["epoch"] == 2 and isinstance(model, NIMA)

    metrics_path = tmp_path / "models" / "metrics.json"
    result = evaluate.main(
        [
            "--checkpoint", str(out / "best.pt"), "--splits", str(splits),
            "--images", str(image_dir), "--out", str(metrics_path),
            "--batch-size", "4", "--num-workers", "0", "--device", "cpu",
        ]
    )  # fmt: skip
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved == result
    assert saved["split"] == "test" and saved["n"] > 0
    assert {"srcc", "plcc", "accuracy", "emd"} <= saved.keys()


def test_resume_missing_checkpoint_errors(fake_ava, tmp_path):
    ava_txt, image_dir = fake_ava
    dataset_ava.prepare(ava_txt, image_dir, tmp_path / "s", subset=None, workers=2)
    with pytest.raises(SystemExit, match="not found"):
        train.main(
            [
                "--splits", str(tmp_path / "s"), "--images", str(image_dir),
                "--out", str(tmp_path / "o"), "--no-pretrained", "--device", "cpu",
                "--resume", str(tmp_path / "nope.pt"),
            ]
        )  # fmt: skip
