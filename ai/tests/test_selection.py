import json
import math
from pathlib import Path

import pytest

from photoholic_ai.config import Config
from photoholic_ai.selection import ACTIONS_NAME, apply_action, rank_group, select

CFG = Config()


def m(id_, sharp, aes=5.0, std=1.0, t=None):
    return {"id": id_, "sharpness_score": sharp, "aesthetic_mean": aes, "aesthetic_std": std,
            "capture_time": t}  # fmt: skip


def test_group_of_one_gets_full_sharpness():
    (only,) = rank_group([m("a", 3.0, aes=7.0)], CFG)
    assert only["sharpness_norm"] == 1.0
    assert only["aesthetic_norm"] == pytest.approx(6 / 9)
    assert only["final_score"] == pytest.approx(0.4 + 0.6 * 6 / 9)
    assert only["rank"] == 1


def test_equal_sharpness_in_group_all_one():
    ranked = rank_group([m("a", 50.0, aes=4.0), m("b", 50.0, aes=6.0)], CFG)
    assert [r["sharpness_norm"] for r in ranked] == [1.0, 1.0]
    assert [r["id"] for r in ranked] == ["b", "a"]  # decided by aesthetics


def test_sharpness_is_log_scaled_min_max_within_group():
    ranked = {r["id"]: r for r in rank_group([m("lo", 15), m("mid", 150), m("hi", 3000)], CFG)}
    assert ranked["lo"]["sharpness_norm"] == 0.0
    assert ranked["hi"]["sharpness_norm"] == 1.0
    expected = (math.log1p(150) - math.log1p(15)) / (math.log1p(3000) - math.log1p(15))
    assert ranked["mid"]["sharpness_norm"] == pytest.approx(expected)


def test_huge_sharpness_cannot_swamp_aesthetics():
    # raw sharpness 100x higher, but far worse aesthetics: aesthetics win at 0.4/0.6
    ranked = rank_group([m("sharp", 5000, aes=2.0), m("pretty", 50, aes=9.0)], CFG)
    assert ranked[0]["id"] == "pretty"
    sharp_first = rank_group(ranked, Config(w_sharpness=0.9, w_aesthetic=0.1))
    assert sharp_first[0]["id"] == "sharp"


def test_missing_aesthetic_contributes_zero():
    ranked = rank_group([m("scored", 10, aes=5.5), m("unknown", 10, aes=None, std=None)], CFG)
    unknown = next(r for r in ranked if r["id"] == "unknown")
    assert unknown["aesthetic_norm"] is None
    assert unknown["final_score"] == pytest.approx(0.4)
    assert ranked[0]["id"] == "scored"


def test_aesthetic_norm_is_absolute_and_clipped():
    ranked = {r["id"]: r for r in rank_group([m("a", 1, aes=1.0), m("b", 1, aes=10.5)], CFG)}
    assert ranked["a"]["aesthetic_norm"] == 0.0
    assert ranked["b"]["aesthetic_norm"] == 1.0


def test_tie_breaks():
    # identical scores -> earlier capture time, then lower std, then id
    same = [m("late", 10, aes=6.0, std=1.0, t=20.0), m("early", 10, aes=6.0, std=1.0, t=10.0)]
    assert [r["id"] for r in rank_group(same, CFG)] == ["early", "late"]
    steady = [m("wide", 10, aes=6.0, std=2.0), m("narrow", 10, aes=6.0, std=0.5)]
    assert rank_group(steady, CFG)[0]["id"] == "narrow"
    untimed = [m("b", 10, t=None), m("a", 10, t=None)]
    assert [r["id"] for r in rank_group(untimed, CFG)] == ["a", "b"]


def fixture_run(tmp_path):
    """Three photos (two a burst, one alone), one decode error, two blur rejects."""
    folder = tmp_path / "Trip 2026"
    (folder / "day1").mkdir(parents=True)
    files = {}
    for name in ["DSC1.JPG", "DSC1.ARW", "DSC2.JPG", "day1/DSC3.JPG", "bad.JPG"]:
        p = folder / name
        p.write_bytes(name.encode() * 10)
        files[name] = str(p)

    filtered = {
        "schema_version": 1,
        "images": [
            {"id": "img_000001", "path": files["DSC1.JPG"], "arw_path": files["DSC1.ARW"],
             "sharpness_score": 400.0, "date_time_original": "2026:06:01 10:00:00"},
            {"id": "img_000002", "path": files["DSC2.JPG"], "arw_path": None,
             "sharpness_score": 40.0, "date_time_original": "2026:06:01 10:00:01"},
            {"id": "img_000003", "path": files["day1/DSC3.JPG"], "arw_path": None,
             "sharpness_score": 99.0},
            {"id": "img_000004", "path": files["bad.JPG"], "sharpness_score": 50.0},
        ],
        "rejected": [
            {"path": "x", "reason": "low_sharpness", "sharpness_score": 3.0},
            {"path": "y", "reason": "decode_error", "sharpness_score": None},
        ],
    }  # fmt: skip
    scored = {
        "schema_version": 1,
        "model": {"name": "dummy"},
        "images": [
            {"id": "img_000001", "path": files["DSC1.JPG"], "aesthetic_mean": 6.0,
             "aesthetic_std": 1.0, "embedding_index": 0, "status": "ok"},
            {"id": "img_000002", "path": files["DSC2.JPG"], "aesthetic_mean": 6.2,
             "aesthetic_std": 1.0, "embedding_index": 1, "status": "ok"},
            {"id": "img_000003", "path": files["day1/DSC3.JPG"], "aesthetic_mean": 4.0,
             "aesthetic_std": 1.0, "embedding_index": 2, "status": "ok"},
            {"id": "img_000004", "path": files["bad.JPG"], "aesthetic_mean": None,
             "aesthetic_std": None, "embedding_index": None, "status": "decode_error",
             "error": "boom"},
        ],
    }  # fmt: skip
    groups = [["img_000001", "img_000002"], ["img_000003"]]
    return folder, files, select(filtered, scored, groups, CFG)


def test_results_schema_and_stats(tmp_path):
    _, _, results = fixture_run(tmp_path)
    assert results["schema_version"] == 1
    assert results["weights"] == {"sharpness": 0.4, "aesthetic": 0.6}

    burst, single = results["clusters"]
    assert burst["cluster_id"] == 0 and burst["best_id"] == "img_000001"
    assert [x["rank"] for x in burst["members"]] == [1, 2]
    assert set(burst["members"][0]) >= {
        "id", "path", "sharpness_norm", "aesthetic_norm", "final_score", "rank"
    }  # fmt: skip
    assert single["best_id"] == "img_000003" and single["members"][0]["sharpness_norm"] == 1.0

    assert results["unscored"] == [
        {"id": "img_000004", "path": results["unscored"][0]["path"],
         "status": "decode_error", "error": "boom"}
    ]  # fmt: skip
    assert results["stats"] == {
        "input": 6, "rejected_sharpness": 1, "rejected_decode": 1, "decode_errors": 1,
        "scored": 3, "clusters": 2, "reduction_pct": 33.3,
    }  # fmt: skip
    json.dumps(results)  # serialisable


def test_action_none_touches_nothing(tmp_path):
    folder, _, results = fixture_run(tmp_path)
    before = sorted(p.relative_to(folder) for p in folder.rglob("*"))
    assert apply_action(results, folder, "none") == []
    assert sorted(p.relative_to(folder) for p in folder.rglob("*")) == before


def test_copy_best_with_raw_keeps_originals_and_is_idempotent(tmp_path):
    folder, files, results = fixture_run(tmp_path)
    run_dir = folder / ".photoholic"
    done = apply_action(results, folder, "copy", run_dir)

    picks = folder / "photoholic_picks"
    assert sorted(p.relative_to(picks).as_posix() for p in picks.rglob("*.*")) == [
        "DSC1.ARW", "DSC1.JPG", "day1/DSC3.JPG"
    ]  # fmt: skip
    assert all(Path(f).exists() for f in files.values())  # originals untouched
    assert len(done) == 3

    assert apply_action(results, folder, "copy", run_dir) == []  # same files: nothing new
    log = json.loads((run_dir / ACTIONS_NAME).read_text(encoding="utf-8"))
    assert [e["dst"] for e in log] == [d["dst"] for d in done]


def test_copy_never_overwrites_a_different_file(tmp_path):
    folder, _, results = fixture_run(tmp_path)
    picks = folder / "photoholic_picks"
    picks.mkdir()
    (picks / "DSC1.JPG").write_bytes(b"someone else's file")

    apply_action(results, folder, "copy")
    assert (picks / "DSC1.JPG").read_bytes() == b"someone else's file"
    assert (picks / "DSC1_1.JPG").read_bytes() == b"DSC1.JPG" * 10


def test_move_non_best_to_rejects(tmp_path):
    folder, files, results = fixture_run(tmp_path)
    clash = folder / "photoholic_rejects" / "DSC2.JPG"
    clash.parent.mkdir()
    clash.write_bytes(b"older reject")

    done = apply_action(results, folder, "move", folder / ".photoholic")

    assert [d["src"] for d in done] == [files["DSC2.JPG"]]
    assert not (folder / "DSC2.JPG").exists()
    assert clash.read_bytes() == b"older reject"  # not overwritten
    assert (folder / "photoholic_rejects" / "DSC2_1.JPG").exists()
    assert (folder / "DSC1.JPG").exists() and (folder / "day1" / "DSC3.JPG").exists()


def test_unknown_action():
    with pytest.raises(ValueError, match="unknown action"):
        apply_action({"clusters": []}, ".", "delete")
