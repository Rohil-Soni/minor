"""Turn an RGB image into model input: resize to 224x224 (bilinear, antialiased), scale to
[0, 1], ImageNet-normalise, CHW float32.

Must match ``training.dataset_ava.build_transforms(train=False)``. The settings are read
from the model's sidecar JSON so training and inference cannot drift. The constants below
are the single definition shared by training, the ONNX sidecar written at export, and inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SUPPORTED = {"resize": "bilinear_antialias", "color": "RGB", "layout": "NCHW"}


@dataclass(frozen=True)
class PreprocessSpec:
    input_size: int = INPUT_SIZE
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    @classmethod
    def from_sidecar(cls, sidecar: Path | dict[str, Any]) -> PreprocessSpec:
        if not isinstance(sidecar, dict):
            sidecar = json.loads(Path(sidecar).read_text(encoding="utf-8"))
        pre = sidecar["preprocess"]
        for key, expected in SUPPORTED.items():
            if pre.get(key, expected) != expected:
                raise ValueError(f"unsupported preprocess {key}={pre[key]!r} (need {expected!r})")
        if abs(pre.get("scale", 1 / 255) - 1 / 255) > 1e-12:
            raise ValueError(f"unsupported preprocess scale {pre['scale']}")
        return cls(int(pre["input_size"]), tuple(pre["mean"]), tuple(pre["std"]))


DEFAULT_SPEC = PreprocessSpec()


def to_array(img: Image.Image, spec: PreprocessSpec = DEFAULT_SPEC) -> np.ndarray:
    """``[3, size, size]`` float32. Pillow's BILINEAR resize antialiases when downscaling,
    exactly like torchvision ``Resize`` on PIL images."""
    size = spec.input_size
    img = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.asarray(spec.mean, dtype=np.float32)) / np.asarray(spec.std, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))
