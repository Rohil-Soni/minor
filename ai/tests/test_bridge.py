import json

import numpy as np
import pytest

from photoholic_ai import bridge
from photoholic_ai.bridge import BridgeError

WIN_A = "D:\\Photos\\Trip\\DSC0002.JPG"
WIN_B = "D:\\Photos\\Trip\\DSC0001.JPG"


def write(tmp_path, obj, name="filtered_images.json"):
    path = tmp_path / name
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")
    return path


def v1(**overrides):
    data = {
        "schema_version": 1,
        "generated_at": "2026-10-24T08:44:00Z",
        "source_dir": "D:\\Photos\\Trip",
        "min_sharpness": 15.0,
        "images": [
            {"id": "img_000001", "path": WIN_B, "sharpness_score": 312.77, "iso": "400"},
            {"id": "img_000002", "path": WIN_A, "sharpness_score": 20, "iso": None},
        ],
        "rejected": [{"path": "D:\\x.JPG", "reason": "low_sharpness", "sharpness_score": 9.1}],
    }
    data.update(overrides)
    return data


def test_load_v1_fills_missing_exif_fields(tmp_path):
    out = bridge.load_filtered(write(tmp_path, v1()))
    assert out["schema_version"] == 1
    first = out["images"][0]
    assert first["path"] == WIN_B  # backslashes preserved
    assert first["iso"] == "400"
    assert first["date_time_original"] is None and first["arw_path"] is None
    assert out["images"][1]["sharpness_score"] == 20.0
    assert out["rejected"][0]["reason"] == "low_sharpness"


def test_load_v0_bare_array_wraps_and_assigns_ids(tmp_path):
    legacy = [
        {"path": WIN_A, "arw_path": None, "sharpness_score": 50.0, "date_time": "x"},
        {"path": WIN_B, "arw_path": None, "sharpness_score": 60.0, "date_time": "y"},
    ]
    out = bridge.load_filtered(write(tmp_path, legacy))
    assert out["schema_version"] == 0
    assert out["rejected"] == []
    assert [(r["id"], r["path"]) for r in out["images"]] == [
        ("img_000001", WIN_B),  # sorted by path, like the Rust engine
        ("img_000002", WIN_A),
    ]


@pytest.mark.parametrize(
    "content, match",
    [
        ("{not json", "not valid JSON"),
        ('"just a string"', "top level"),
        (v1(schema_version=2), "unsupported schema_version 2"),
        ({"images": []}, "unsupported schema_version None"),
        (v1(images="nope"), "'images' must be a list"),
        (v1(images=[{"id": "a", "sharpness_score": 1}]), "has no 'path'"),
        (v1(images=[{"id": "a", "path": "p", "sharpness_score": "1"}]), "numeric"),
        (v1(images=[{"id": "a", "path": "p", "sharpness_score": True}]), "numeric"),
        (v1(images=[{"path": "p", "sharpness_score": 1}]), "string 'id'"),
        (
            v1(images=[{"id": "a", "path": "p", "sharpness_score": 1}] * 2),
            "not unique",
        ),
        (v1(rejected={}), "'rejected' must be a list"),
        ([1, 2], "not an object"),
    ],
)
def test_malformed_input_rejected(tmp_path, content, match):
    with pytest.raises(BridgeError, match=match):
        bridge.load_filtered(write(tmp_path, content))


def test_missing_file_has_clear_message(tmp_path):
    with pytest.raises(BridgeError, match="Run the previous pipeline stage"):
        bridge.load_filtered(tmp_path / "nope.json")


def test_scored_roundtrip_and_index_check(tmp_path):
    scored = {
        "schema_version": 1,
        "model": {"name": "m"},
        "images": [
            {"id": "a", "embedding_index": 0, "status": "ok"},
            {"id": "b", "embedding_index": None, "status": "decode_error"},
            {"id": "c", "embedding_index": 1, "status": "ok"},
        ],
    }
    emb = np.eye(2, 4, dtype=np.float64)
    bridge.write_scored(tmp_path, scored, emb)
    loaded, loaded_emb = bridge.load_scored(tmp_path)
    assert loaded == scored
    assert loaded_emb.dtype == np.float32 and loaded_emb.shape == (2, 4)

    np.save(tmp_path / bridge.EMBEDDINGS_NAME, np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(BridgeError, match="embedding indices"):
        bridge.load_scored(tmp_path)


def test_write_json_is_utf8(tmp_path):
    path = tmp_path / "sub" / "r.json"
    bridge.write_json(path, {"path": "D:\\Fotos\\Café\\ü.jpg"})
    assert "Café" in path.read_text(encoding="utf-8")
    assert not (tmp_path / "sub" / "r.json.tmp").exists()
