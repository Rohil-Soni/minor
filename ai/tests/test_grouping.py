import numpy as np
import pytest

from photoholic_ai.config import Config
from photoholic_ai.grouping import group_images, parse_capture_time


def unit(dim=64, seed=0):
    v = np.random.default_rng(seed).standard_normal(dim)
    return v / np.linalg.norm(v)


def pair_with_similarity(sim, seed=0):
    """Two unit vectors with exactly cosine similarity ``sim``."""
    a = unit(seed=seed)
    b = unit(seed=seed + 100)
    b = b - (b @ a) * a
    b /= np.linalg.norm(b)
    return np.stack([a, sim * a + np.sqrt(1 - sim**2) * b])


def jitter(center, n, scale, seed):
    rng = np.random.default_rng(seed)
    v = center + rng.standard_normal((n, len(center))) * scale
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_two_tight_clusters_and_outlier():
    emb = np.vstack(
        [jitter(unit(seed=1), 4, 0.02, 1), jitter(unit(seed=2), 3, 0.02, 2), unit(seed=3)]
    )
    ids = [f"img_{i}" for i in range(len(emb))]
    order = [0, 4, 1, 5, 7, 2, 6, 3]  # interleaved input order
    groups = group_images([ids[i] for i in order], emb[order], [None] * len(emb))

    assert sorted(map(sorted, groups)) == [
        ["img_0", "img_1", "img_2", "img_3"],
        ["img_4", "img_5", "img_6"],
        ["img_7"],
    ]
    assert groups[0][0] == "img_0" and groups[1][0] == "img_4"  # input order kept


@pytest.mark.parametrize(
    "sim, dt, grouped",
    [
        (0.85, 3.0, True),  # burst: close in time, moderately similar
        (0.85, 60.0, False),  # same similarity, too far apart in time
        (0.85, None, False),  # no time: needs the stricter threshold
        (0.95, 3600.0, True),  # near-duplicate re-take an hour later, same session
        (0.95, 2.9 * 3600, True),
        (0.95, 3.1 * 3600, False),  # beyond the 3 h session gap
        (0.99, 2 * 86400, False),  # same view two days later stays separate
        (0.95, None, True),
        (0.70, 1.0, False),  # close in time but a different scene
    ],
)
def test_time_gated_thresholds(sim, dt, grouped):
    emb = pair_with_similarity(sim)
    times = [1000.0, None if dt is None else 1000.0 + dt]
    groups = group_images(["a", "b"], emb, times)
    assert (len(groups) == 1) is grouped


def test_same_view_on_different_nights_gets_one_group_per_night():
    # the moon: near-identical frames, a few per night, three nights apart
    moon = unit(seed=7)
    emb = jitter(moon, 9, 0.005, seed=3)
    assert (emb @ emb.T).min() > 0.99
    nights = [n * 86400 + 21 * 3600 + k * 40.0 for n in range(3) for k in range(3)]
    ids = [f"night{n}_{k}" for n in range(3) for k in range(3)]

    groups = group_images(ids, emb, nights)
    assert sorted(map(sorted, groups)) == [[f"night{n}_{k}" for k in range(3)] for n in range(3)]
    # a longer session setting merges them again
    assert len(group_images(ids, emb, nights, Config(session_gap_h=72))) == 1


def test_untimed_photos_still_group_by_similarity():
    emb = pair_with_similarity(0.95)
    assert len(group_images(["a", "b"], emb, [1000.0, None])) == 1
    assert len(group_images(["a", "b"], emb, [None, None])) == 1


def test_burst_gap_from_config():
    emb = pair_with_similarity(0.85)
    assert len(group_images(["a", "b"], emb, [0.0, 8.0], Config(burst_gap_s=10))) == 1
    assert len(group_images(["a", "b"], emb, [0.0, 8.0], Config(burst_gap_s=5))) == 2


def test_chaining_through_a_burst():
    # a~b and b~c within the gap, a and c 16 s apart: one group via b
    a, b = pair_with_similarity(0.9, seed=5)
    c = 0.9 * b + np.sqrt(1 - 0.81) * unit(seed=9)
    c /= np.linalg.norm(c)
    groups = group_images(["a", "b", "c"], np.stack([a, b, c]), [0.0, 8.0, 16.0])
    assert groups == [["a", "b", "c"]]


def test_singletons_and_empty():
    emb = np.eye(3, 8)
    assert group_images(["x", "y", "z"], emb, [0.0, 1.0, 2.0]) == [["x"], ["y"], ["z"]]
    assert group_images([], np.zeros((0, 8)), []) == []


def test_agglomerative_option():
    emb = np.vstack([jitter(unit(seed=1), 3, 0.01, 1), jitter(unit(seed=2), 2, 0.01, 2)])
    groups = group_images(list("abcde"), emb, [None] * 5, method="agglomerative")
    assert sorted(map(sorted, groups)) == [["a", "b", "c"], ["d", "e"]]
    assert group_images(["solo"], emb[:1], [None], method="agglomerative") == [["solo"]]


def test_bad_inputs():
    with pytest.raises(ValueError, match="same length"):
        group_images(["a"], np.zeros((2, 4)), [None])
    with pytest.raises(ValueError, match="unknown grouping method"):
        group_images(["a"], np.zeros((1, 4)), [None], method="kmeans")


def test_parse_capture_time():
    base = parse_capture_time({"date_time_original": "2026:06:01 10:15:22"})
    assert parse_capture_time(
        {"date_time_original": "2026:06:01 10:15:22", "subsec_time_original": "45"}
    ) == pytest.approx(base + 0.45)
    assert parse_capture_time(
        {"date_time_original": "2026:06:01 10:15:23", "subsec_time_original": "045"}
    ) == pytest.approx(base + 1.045)
    # falls back to date_time; accepts the older "-" display format
    assert (
        parse_capture_time({"date_time_original": None, "date_time": "2026-06-01 10:15:22"}) == base
    )
    assert parse_capture_time({"date_time_original": "0000:00:00 00:00:00"}) is None
    assert parse_capture_time({"date_time_original": "garbage", "date_time": None}) is None
    assert parse_capture_time({}) is None
