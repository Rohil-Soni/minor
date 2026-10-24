"""Pipeline settings: single source of thresholds and weights (plan §6).

Defaults live on :class:`Config`. An optional ``config.toml`` can override any
field; CLI flags are applied on top with :func:`dataclasses.replace`.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

log = logging.getLogger(__name__)

ACTIONS = ("none", "copy", "move")


@dataclass(frozen=True)
class Config:
    min_sharpness: float = 15.0
    input_size: int = 224
    batch_size: int = 32
    burst_gap_s: float = 10.0
    sim_threshold_timed: float = 0.80
    sim_threshold_untimed: float = 0.92
    w_sharpness: float = 0.4
    w_aesthetic: float = 0.6
    action: str = "none"

    def __post_init__(self) -> None:
        if self.min_sharpness < 0:
            raise ValueError(f"min_sharpness must be >= 0, got {self.min_sharpness}")
        if self.input_size <= 0 or self.batch_size <= 0:
            raise ValueError("input_size and batch_size must be positive")
        if self.burst_gap_s < 0:
            raise ValueError(f"burst_gap_s must be >= 0, got {self.burst_gap_s}")
        for name in ("sim_threshold_timed", "sim_threshold_untimed"):
            value = getattr(self, name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a cosine similarity in [-1, 1], got {value}")
        if self.w_sharpness < 0 or self.w_aesthetic < 0:
            raise ValueError("weights must be non-negative")
        if not math.isclose(self.w_sharpness + self.w_aesthetic, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"w_sharpness + w_aesthetic must sum to 1, got "
                f"{self.w_sharpness} + {self.w_aesthetic}"
            )
        if self.action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {self.action!r}")


def load_config(path: Path | None = None, **overrides: Any) -> Config:
    """Build a :class:`Config` from defaults, an optional TOML file, then ``overrides``.

    ``overrides`` whose value is ``None`` are ignored, so unset CLI flags can be
    passed straight through.
    """
    values: dict[str, Any] = {}
    if path is not None:
        with Path(path).open("rb") as f:
            values.update(tomllib.load(f))
        log.debug("loaded config from %s", path)
    values.update({k: v for k, v in overrides.items() if v is not None})

    known = {f.name for f in dataclasses.fields(Config)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    return Config(**values)
