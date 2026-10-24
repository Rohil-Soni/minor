import pytest

from photoholic_ai import report


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1/250 s", 0.004),
        ("2.5 s", 2.5),
        ("35 mm", 35.0),
        ("400", 400.0),
        (400, 400.0),
        ("1/0 s", None),
        ("n/a", None),
        (None, None),
    ],
)
def test_parse_number(value, expected):
    assert report.parse_number(value) == (pytest.approx(expected) if expected else expected)


def sample():
    filtered = {
        "images": [
            {"id": "a", "path": "D:\\P\\a.JPG", "arw_path": "D:\\P\\a.ARW",
             "sharpness_score": 300.0, "iso": "400", "shutter_speed": "1/250 s",
             "focal_length": "35 mm", "date_time_original": "2026:06:01 10:00:00"},
            {"id": "b", "path": "D:\\P\\b.JPG", "sharpness_score": 100.0, "iso": "800"},
            {"id": "c", "path": "D:\\P\\c.JPG", "sharpness_score": 90.0},
        ],
        "rejected": [
            {"path": "D:\\P\\blur.JPG", "reason": "low_sharpness", "sharpness_score": 4.0}
        ],
    }  # fmt: skip
    scored = {
        "images": [
            {"id": "a", "status": "ok", "aesthetic_mean": 6.1, "aesthetic_std": 1.2},
            {"id": "b", "status": "ok", "aesthetic_mean": 5.0, "aesthetic_std": 1.5},
            {"id": "c", "status": "decode_error", "error": "boom"},
        ]
    }
    results = {
        "clusters": [
            {"cluster_id": 0, "best_id": "a", "members": [
                {"id": "a", "rank": 1, "sharpness_norm": 1.0, "aesthetic_norm": 0.57,
                 "final_score": 0.74},
                {"id": "b", "rank": 2, "sharpness_norm": 0.0, "aesthetic_norm": 0.44,
                 "final_score": 0.27},
            ]}
        ]
    }  # fmt: skip
    return filtered, scored, results


def test_build_table_one_row_per_input_image():
    df = report.build_table(*sample())
    assert list(df.columns) == report.COLUMNS
    assert df["status"].tolist() == ["best", "duplicate", "decode_error", "rejected_low_sharpness"]
    a = df.iloc[0]
    assert (a["cluster_id"], a["cluster_size"], a["rank"], a["is_best"]) == (0, 2, 1, True)
    assert (a["iso"], a["shutter_s"], a["focal_length_mm"]) == (400.0, 0.004, 35.0)
    assert a["arw_path"] == "D:\\P\\a.ARW"
    assert df.iloc[2]["error"] == "boom"
    assert df.iloc[3]["sharpness_score"] == 4.0


def test_csv_and_charts(tmp_path):
    pytest.importorskip("matplotlib")
    df = report.build_table(*sample())
    csv_path = report.write_csv(df, tmp_path)
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("id,path,arw_path")

    charts = report.write_charts(df, tmp_path)
    assert sorted(p.name for p in charts) == [
        "aesthetic_hist.png", "sharpness_vs_iso.png", "sharpness_vs_shutter_s.png"
    ]  # fmt: skip
    assert all(p.stat().st_size > 1000 for p in charts)
