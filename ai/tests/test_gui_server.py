"""The GUI server (app/server.py) against a real pipeline run on generated photos."""

import json
import sys
import threading
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

import run_pipeline
from tests.test_e2e import handmade_filtered, make_shoot

APP_DIR = Path(__file__).resolve().parents[2] / "app"
sys.path.insert(0, str(APP_DIR))
import server  # noqa: E402


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, dummy_model):
    folder = tmp_path_factory.mktemp("gui") / "Shoot"
    truth = make_shoot(folder)
    handmade_filtered(folder, truth)
    assert run_pipeline.main([str(folder), "--skip-engine", "--model", str(dummy_model)]) == 0
    return folder / ".photoholic"


@pytest.fixture()
def live(run_dir):
    (run_dir / server.PICKS_NAME).unlink(missing_ok=True)
    srv = server.make_server(server.Run(run_dir), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base
    srv.shutdown()
    srv.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.headers["Content-Type"], r.read()


def post(url, obj):
    req = urllib.request.Request(
        url, json.dumps(obj).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def status_of(fn, *args):
    try:
        fn(*args)
    except urllib.error.HTTPError as e:
        return e.code
    return 200


def test_payload_lists_every_group_and_photo(live):
    _, ctype, body = get(live + "/api/run")
    data = json.loads(body)
    assert ctype.startswith("application/json")
    assert len(data["groups"]) == 3 and sum(len(g["members"]) for g in data["groups"]) == 6
    for g in data["groups"]:
        assert g["pick_id"] == g["best_id"] == g["members"][0]["id"]
        assert {
            "name",
            "rank",
            "final",
            "sharp_norm",
            "aes_mean",
            "clock",
            "sharpness",
        } <= set(g["members"][0])
    assert [r["name"] for r in data["rejected"]] == ["DSC00_blur.JPG"]
    assert data["stats"]["clusters"] == 3


@pytest.mark.parametrize("size, limit", [("thumb", 360), ("preview", 2048)])
def test_resized_images(live, size, limit):
    data = json.loads(get(live + "/api/run")[2])
    image_id = data["groups"][0]["members"][0]["id"]
    status, ctype, body = get(f"{live}/img/{image_id}?size={size}")
    assert status == 200 and ctype == "image/jpeg"
    assert max(Image.open(BytesIO(body)).size) <= limit


def test_full_image_is_the_original_file(live, run_dir):
    data = json.loads(get(live + "/api/run")[2])
    run = server.Run(run_dir)
    for image_id in (data["groups"][0]["members"][0]["id"], data["rejected"][0]["id"]):
        assert get(f"{live}/img/{image_id}?size=full")[2] == run.files[image_id].read_bytes()


@pytest.mark.parametrize(
    "path, code",
    [
        ("/img/nope?size=thumb", 404),
        ("/img/img_000001?size=huge", 404),
        ("/img/..%2F..%2Fsecret?size=full", 400),
        ("/../server.py", 404),
        ("/static/../../server.py", 404),
        ("/does-not-exist.js", 404),
    ],
)
def test_only_run_photos_and_static_files_are_served(live, path, code):
    assert status_of(get, live + path) == code


def test_static_page(live):
    status, ctype, body = get(live + "/")
    assert status == 200 and ctype.startswith("text/html")
    assert b"Photoholic" in body


def test_pick_is_saved_validated_and_restorable(live, run_dir):
    group = json.loads(get(live + "/api/run")[2])["groups"][0]
    other = group["members"][1]["id"]

    assert post(live + "/api/pick", {"group": group["id"], "id": other})["picks"] == {
        str(group["id"]): other
    }
    saved = json.loads((run_dir / server.PICKS_NAME).read_text(encoding="utf-8"))
    assert saved["picks"] == {str(group["id"]): other}
    assert saved["files"][str(group["id"])].endswith(".JPG")
    assert json.loads(get(live + "/api/run")[2])["groups"][0]["pick_id"] == other

    # choosing the automatic pick again removes the override
    assert post(live + "/api/pick", {"group": group["id"], "id": group["best_id"]})["picks"] == {}

    other_group = json.loads(get(live + "/api/run")[2])["groups"][1]
    assert (
        status_of(
            post,
            live + "/api/pick",
            {"group": group["id"], "id": other_group["best_id"]},
        )
        == 400
    )
    assert status_of(post, live + "/api/pick", {"group": 999, "id": "x"}) == 400
    assert status_of(post, live + "/api/pick", {"nope": 1}) == 400


def test_picks_survive_a_restart(run_dir):
    run = server.Run(run_dir)
    group = run.results["clusters"][0]
    run.set_pick(group["cluster_id"], group["members"][1]["id"])
    reloaded = server.Run(run_dir).payload()["groups"][0]
    assert reloaded["pick_id"] == group["members"][1]["id"]
    (run_dir / server.PICKS_NAME).unlink()


def test_missing_run_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run ai/run_pipeline.py first"):
        server.Run(tmp_path)
