import pytest
from PIL import Image

from photoholic_ai.imaging import ImageDecodeError, load_rgb


def test_exif_orientation_is_applied(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW to display
    Image.new("RGB", (400, 200), (200, 10, 10)).save(path, exif=exif)

    img = load_rgb(path)
    assert img.size == (200, 400)
    assert img.mode == "RGB"


def test_draft_downscales_large_jpeg(tmp_path):
    path = tmp_path / "big photo.JPG"
    Image.new("RGB", (4000, 3000), (1, 2, 3)).save(path, quality=80)

    w, h = load_rgb(path).size
    # JPEG DCT scaling picks the smallest scale keeping both sides >= 448: 1/4 here
    assert (w, h) == (1000, 750)
    assert load_rgb(path, draft=None).size == (4000, 3000)


def test_png_and_greyscale(tmp_path):
    path = tmp_path / "g.png"
    Image.new("L", (30, 20), 77).save(path)
    img = load_rgb(path)
    assert img.mode == "RGB" and img.getpixel((0, 0)) == (77, 77, 77)


@pytest.mark.parametrize("name, data", [("broken.jpg", b"\xff\xd8\xff junk"), ("empty.jpg", b"")])
def test_corrupt_raises_decode_error(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    with pytest.raises(ImageDecodeError, match=name):
        load_rgb(path)


def test_missing_and_corrupt_arw_raise_decode_error(tmp_path):
    pytest.importorskip("rawpy")
    bad = tmp_path / "DSC0001.ARW"
    bad.write_bytes(b"not a raw file")
    with pytest.raises(ImageDecodeError, match="DSC0001.ARW"):
        load_rgb(bad)
    with pytest.raises(ImageDecodeError):
        load_rgb(tmp_path / "missing.jpg")
