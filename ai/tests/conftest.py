import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from photoholic_ai.preprocess import IMAGENET_MEAN, IMAGENET_STD

N_IMAGES = 40
EMB_DIM = 1280


def build_dummy_onnx(path: Path, seed: int = 0) -> Path:
    """Tiny model with the real I/O contract: ``input`` [N,3,224,224] -> ``dist`` [N,10],
    ``embedding`` [N,1280]. Per-channel mean colour -> linear layers. Writes the sidecar too."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(seed)
    w_dist = rng.standard_normal((3, 10)).astype(np.float32) * 3
    w_emb = rng.standard_normal((3, EMB_DIM)).astype(np.float32)
    b_emb = rng.standard_normal(EMB_DIM).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "w_dist"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["dist"], axis=1),
            helper.make_node("MatMul", ["flat", "w_emb"], ["emb_raw"]),
            helper.make_node("Add", ["emb_raw", "b_emb"], ["embedding"]),
        ],
        "dummy_nima",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, 224, 224])],
        [
            helper.make_tensor_value_info("dist", TensorProto.FLOAT, ["N", 10]),
            helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["N", EMB_DIM]),
        ],
        [
            numpy_helper.from_array(w_dist, "w_dist"),
            numpy_helper.from_array(w_emb, "w_emb"),
            numpy_helper.from_array(b_emb, "b_emb"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))

    sidecar = {
        "model": {"name": "dummy", "version": "0.0.0-test", "trained": True},
        "onnx": {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "outputs": {"dist": ["N", 10], "embedding": ["N", EMB_DIM]},
        },
        "preprocess": {
            "input_size": 224,
            "resize": "bilinear_antialias",
            "color": "RGB",
            "layout": "NCHW",
            "scale": 1 / 255,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def dummy_model(tmp_path_factory) -> Path:
    return build_dummy_onnx(tmp_path_factory.mktemp("model dir") / "dummy.onnx")


ZERO_VOTE_ID = 9001
MISSING_ID = 9002
CORRUPT_ID = 9003


@pytest.fixture()
def fake_ava(tmp_path: Path) -> tuple[Path, Path]:
    """AVA.txt + nested image folder with one zero-vote row, one missing and one corrupt image."""
    rng = np.random.default_rng(0)
    image_dir = tmp_path / "ava images"
    lines = []
    for n, image_id in enumerate(range(1000, 1000 + N_IMAGES), start=1):
        votes = rng.integers(0, 30, size=10)
        votes[4] += 1  # never all-zero
        lines.append(f"{n} {image_id} {' '.join(map(str, votes))} 1 2 100")
        sub = image_dir / f"part{image_id % 3}"
        sub.mkdir(parents=True, exist_ok=True)
        pixels = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(sub / f"{image_id}.jpg", quality=90)

    lines.append(f"98 {ZERO_VOTE_ID} {' '.join(['0'] * 10)} 1 2 100")
    lines.append(f"99 {MISSING_ID} {' '.join(['5'] * 10)} 1 2 100")
    lines.append(f"100 {CORRUPT_ID} {' '.join(['5'] * 10)} 1 2 100")
    (image_dir / "part0" / f"{CORRUPT_ID}.jpg").write_bytes(b"\xff\xd8\xff not a jpeg")

    ava_txt = tmp_path / "AVA.txt"
    ava_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ava_txt, image_dir
