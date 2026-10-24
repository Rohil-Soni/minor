import json

import pytest

import show_results


@pytest.fixture()
def run_dir(tmp_path):
    results = {
        "schema_version": 1,
        "weights": {"sharpness": 0.4, "aesthetic": 0.6},
        "model": {"name": "dummy", "version": "0.0.0-untrained", "trained": False},
        "clusters": [
            {"cluster_id": 0, "best_id": "a", "members": [
                {"id": "a", "path": "D:\\P\\DSC1.JPG", "rank": 1, "final_score": 0.7,
                 "sharpness_norm": 1.0, "aesthetic_norm": 0.5},
                {"id": "b", "path": "D:\\P\\DSC2.JPG", "rank": 2, "final_score": 0.3,
                 "sharpness_norm": 0.0, "aesthetic_norm": 0.5},
            ]},
            {"cluster_id": 1, "best_id": "c", "members": [
                {"id": "c", "path": "D:\\P\\Café.JPG", "rank": 1, "final_score": 0.6,
                 "sharpness_norm": 1.0, "aesthetic_norm": None},
            ]},
        ],
        "unscored": [{"id": "d", "path": "D:\\P\\bad.JPG", "status": "decode_error"}],
        "rejected": [
            {"path": "D:\\P\\blur.JPG", "reason": "low_sharpness", "sharpness_score": 3.2}
        ],
        "stats": {"input": 5, "rejected_sharpness": 1, "rejected_decode": 0, "decode_errors": 1,
                  "scored": 3, "clusters": 2, "reduction_pct": 33.3},
    }  # fmt: skip
    scored = {"schema_version": 1, "images": [{"id": "a", "aesthetic_mean": 5.5}]}
    (tmp_path / "results.json").write_text(json.dumps(results), encoding="utf-8")
    (tmp_path / "scored_images.json").write_text(json.dumps(scored), encoding="utf-8")
    return tmp_path


def test_summary_and_groups(run_dir, capsys):
    assert show_results.main([str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "Input photos" in out and "UNTRAINED placeholder" in out
    assert "1 single shots, 1 groups of 2+ (largest 2)" in out
    assert "DSC1.JPG" in out and "5.50" in out  # best pick with its aesthetic mean
    assert "D:\\P" not in out  # file names only


def test_one_group_and_rejected(run_dir, capsys):
    show_results.main([str(run_dir), "--group", "1"])
    assert "Café.JPG" in capsys.readouterr().out
    show_results.main([str(run_dir), "--rejected"])
    out = capsys.readouterr().out
    assert "blur.JPG" in out and "low_sharpness" in out and "bad.JPG" in out


def test_errors(run_dir, tmp_path):
    with pytest.raises(SystemExit, match="no group 9"):
        show_results.main([str(run_dir), "--group", "9"])
    with pytest.raises(SystemExit, match="Run ai/run_pipeline.py first"):
        show_results.main([str(tmp_path / "empty")])
