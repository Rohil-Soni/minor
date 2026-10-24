from pathlib import Path

import pytest

from photoholic_ai import config
from photoholic_ai.config import Config, load_config


def test_defaults_match_plan():
    cfg = Config()
    assert cfg.min_sharpness == 10.0
    assert cfg.input_size == 224
    assert cfg.batch_size == 32
    assert cfg.burst_gap_s == 10.0
    assert cfg.sim_threshold_timed == 0.80
    assert cfg.sim_threshold_untimed == 0.92
    assert (cfg.w_sharpness, cfg.w_aesthetic) == (0.4, 0.6)
    assert cfg.action == "none"


def test_toml_then_overrides(tmp_path: Path):
    toml = tmp_path / "sub dir" / "config.toml"
    toml.parent.mkdir()
    toml.write_text("burst_gap_s = 5\nw_sharpness = 0.5\nw_aesthetic = 0.5\n", encoding="utf-8")

    cfg = load_config(toml, burst_gap_s=3.0, action=None)

    assert cfg.burst_gap_s == 3.0  # override beats file
    assert cfg.w_sharpness == 0.5
    assert cfg.action == "none"  # None override is ignored


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"w_sharpness": 0.5}, "sum to 1"),
        ({"action": "delete"}, "action must be one of"),
        ({"sim_threshold_timed": 1.5}, "cosine similarity"),
        ({"min_sharpness": -1}, "min_sharpness"),
    ],
)
def test_invalid_values_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Config(**kwargs)


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown config keys: bogus"):
        load_config(bogus=1)


def test_delete_is_never_an_action():
    assert "delete" not in config.ACTIONS
