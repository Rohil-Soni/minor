import numpy as np
import pytest
from PIL import Image

from photoholic_ai.preprocess import IMAGENET_MEAN, IMAGENET_STD, PreprocessSpec, to_array


def test_shape_and_dtype():
    x = to_array(Image.new("RGB", (640, 427), (1, 2, 3)))
    assert x.shape == (3, 224, 224)
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]


def test_hand_computed_pixel():
    rgb = (10, 200, 30)
    x = to_array(Image.new("RGB", (300, 500), rgb))
    for c in range(3):
        expected = (rgb[c] / 255 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        assert x[c, 100, 50] == pytest.approx(expected, abs=1e-6)


def test_greyscale_and_rgba_become_rgb():
    assert to_array(Image.new("L", (50, 50), 128)).shape == (3, 224, 224)
    assert to_array(Image.new("RGBA", (50, 50), (1, 2, 3, 0))).shape == (3, 224, 224)


def test_matches_training_eval_transform():
    """Inference preprocessing must equal what the model saw at validation time."""
    pytest.importorskip("torch")
    from training.dataset_ava import build_transforms

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (480, 720, 3), dtype=np.uint8))
    expected = build_transforms(train=False)(img).numpy()
    np.testing.assert_allclose(to_array(img), expected, atol=1e-5)


def test_spec_from_sidecar():
    sidecar = {
        "preprocess": {
            "input_size": 256,
            "resize": "bilinear_antialias",
            "color": "RGB",
            "layout": "NCHW",
            "scale": 1 / 255,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.25, 0.25, 0.25],
        }
    }
    spec = PreprocessSpec.from_sidecar(sidecar)
    assert spec == PreprocessSpec(256, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    assert to_array(Image.new("RGB", (10, 10)), spec).shape == (3, 256, 256)


@pytest.mark.parametrize("key, value", [("resize", "nearest"), ("layout", "NHWC"), ("scale", 1.0)])
def test_spec_rejects_unsupported(key, value):
    pre = {"input_size": 224, "mean": [0, 0, 0], "std": [1, 1, 1], key: value}
    with pytest.raises(ValueError, match="unsupported"):
        PreprocessSpec.from_sidecar({"preprocess": pre})
