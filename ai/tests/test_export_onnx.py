import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from training import export_onnx  # noqa: E402
from training.model import NIMA  # noqa: E402


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("export dir")
    ckpt_path = tmp / "best.pt"
    model = NIMA(pretrained=False)
    torch.save({**model.config(), "model": model.state_dict(), "epoch": 3}, ckpt_path)
    metrics = tmp / "metrics.json"
    metrics.write_text(json.dumps({"srcc": 0.6, "split": "test"}), encoding="utf-8")
    out = tmp / "aesthetic_v1.onnx"
    sidecar = export_onnx.main(
        ["--checkpoint", str(ckpt_path), "--out", str(out), "--metrics", str(metrics)]
    )
    return out, sidecar, model


def test_sidecar_contents(exported):
    out, sidecar, _ = exported
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8")) == sidecar
    assert sidecar["model"]["trained"] is True
    assert sidecar["model"]["version"] == "1.0.0"
    assert sidecar["onnx"]["sha256"] == export_onnx.sha256_file(out)
    assert sidecar["preprocess"]["input_size"] == 224
    assert sidecar["preprocess"]["mean"] == [0.485, 0.456, 0.406]
    assert sidecar["metrics"]["srcc"] == 0.6
    assert sidecar["parity_max_abs_diff"]["dist"] < export_onnx.PARITY_ATOL


@pytest.mark.parametrize("batch", [1, 5])
def test_onnx_dynamic_batch_matches_torch(exported, batch):
    out, _, model = exported
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["input"]
    assert [o.name for o in sess.get_outputs()] == ["dist", "embedding"]

    x = np.random.default_rng(batch).standard_normal((batch, 3, 224, 224), dtype=np.float32)
    dist, emb = sess.run(None, {"input": x})
    assert dist.shape == (batch, 10) and emb.shape == (batch, 1280)
    with torch.no_grad():
        ref_dist, _ = model.eval()(torch.from_numpy(x))
    np.testing.assert_allclose(dist, ref_dist.numpy(), atol=1e-4)


def test_untrained_export_is_labelled(tmp_path):
    sidecar = export_onnx.main(
        ["--untrained", "--no-pretrained", "--out", str(tmp_path / "m.onnx")]
    )
    assert sidecar["model"]["trained"] is False
    assert sidecar["model"]["version"] == "0.0.0-untrained"
    assert sidecar["metrics"] is None
