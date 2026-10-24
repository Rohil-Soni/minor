"""Resize to the model input size and ImageNet-normalise to NCHW float32, reading size and
mean/std from the model sidecar JSON so it always matches training. Implemented in Phase 5.

The constants below are the single definition shared by training (``training.dataset_ava``),
the ONNX sidecar written at export, and inference.
"""

INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
