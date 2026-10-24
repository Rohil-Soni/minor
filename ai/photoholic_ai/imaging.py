"""Open photos as RGB PIL images for the model.

- JPEG (and other Pillow formats): fast downscaled decode with ``Image.draft``, then the EXIF
  orientation is applied.
- Sony ARW: the embedded JPEG preview via rawpy (same idea as ``image-reader/src/arw.rs``),
  rotated by the raw's orientation flag; falls back to a half-size raw development.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image, ImageOps

log = logging.getLogger(__name__)

RAW_EXTS = {".arw"}
# JPEG DCT scaling decodes at >= this size; plenty for a 224 px model input
DRAFT_SIZE = (448, 448)

# rawpy/LibRaw ``sizes.flip`` -> PIL transpose
_FLIP = {
    3: Image.Transpose.ROTATE_180,
    5: Image.Transpose.ROTATE_90,  # 90 deg counter-clockwise
    6: Image.Transpose.ROTATE_270,  # 90 deg clockwise
}


class ImageDecodeError(Exception):
    """The file could not be decoded into an image."""


def _open_pillow(source: Path | io.BytesIO, draft: tuple[int, int] | None) -> Image.Image:
    with Image.open(source) as im:
        if draft is not None:
            im.draft("RGB", draft)  # no-op for non-JPEG formats
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def _open_arw(path: Path, draft: tuple[int, int] | None) -> Image.Image:
    import rawpy  # imported lazily: only needed for raw files

    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            log.debug("%s: no usable embedded preview, developing raw", path)
            return Image.fromarray(raw.postprocess(half_size=True, use_camera_wb=True))

        if thumb.format == rawpy.ThumbFormat.JPEG:
            img = _open_pillow(io.BytesIO(thumb.data), draft)
        else:
            img = Image.fromarray(thumb.data).convert("RGB")
        transpose = _FLIP.get(raw.sizes.flip)
        return img.transpose(transpose) if transpose is not None else img


def load_rgb(path: Path, draft: tuple[int, int] | None = DRAFT_SIZE) -> Image.Image:
    """Decode ``path`` to an upright RGB image. Raises :class:`ImageDecodeError`."""
    path = Path(path)
    try:
        if path.suffix.lower() in RAW_EXTS:
            return _open_arw(path, draft)
        return _open_pillow(path, draft)
    except ImageDecodeError:
        raise
    except Exception as e:  # PIL/rawpy raise many types for broken files
        raise ImageDecodeError(f"{path}: {type(e).__name__}: {e}") from e
