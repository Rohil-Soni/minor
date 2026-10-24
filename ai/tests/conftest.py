from pathlib import Path

import numpy as np
import pytest
from PIL import Image

N_IMAGES = 40
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
