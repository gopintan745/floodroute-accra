"""Shared planning and realization costs for graded flood severity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CostModelConfig:
    """Validated cost parameters for dry, shallow, and deep edge states."""

    quality_penalty_weight: float = 1.0
    shallow_multiplier: float = 2.0
    shallow_penalty_s: float = 0.0
    deep_multiplier: float = 3.0
    deep_penalty_s: float = 600.0

    @classmethod
    def from_config(cls, config: dict) -> "CostModelConfig":
        cost_model = config.get("cost_model", {})
        flood_model = config.get(
            "flood_model",
            config.get("flood_risk", {}).get("flood_model", {}),
        )
        level_costs = flood_model.get("level_costs", {})
        shallow = level_costs.get("shallow", {})
        deep = level_costs.get("deep", {})
        return cls(
            quality_penalty_weight=float(cost_model.get("quality_penalty_weight", 1.0)),
            shallow_multiplier=float(shallow.get("multiplier", 2.0)),
            shallow_penalty_s=float(shallow.get("penalty_s", 0.0)),
            deep_multiplier=float(deep.get("multiplier", 3.0)),
            deep_penalty_s=float(deep.get("penalty_s", 600.0)),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "CostModelConfig":
        with Path(path).open(encoding="utf-8") as config_file:
            return cls.from_config(yaml.safe_load(config_file) or {})


def base_edge_time(
    base_travel_time: float,
    traffic_multiplier: float,
    road_quality_score: float,
    quality_penalty_weight: float,
) -> float:
    """Compute travel time from base time, deterministic traffic, and quality."""
    base_travel_time = float(base_travel_time)
    traffic_multiplier = float(traffic_multiplier)
    road_quality_score = float(road_quality_score)
    quality_penalty_weight = float(quality_penalty_weight)
    if base_travel_time < 0:
        raise ValueError("base_travel_time must be non-negative")
    if traffic_multiplier < 1.0:
        raise ValueError("traffic_multiplier must be at least 1.0")
    if not 0.0 <= road_quality_score <= 1.0:
        raise ValueError("road_quality_score must be in [0, 1]")
    if quality_penalty_weight < 0:
        raise ValueError("quality_penalty_weight must be non-negative")
    return base_travel_time * traffic_multiplier * (
        1.0 + (1.0 - road_quality_score) * quality_penalty_weight
    )


def level_cost(base_time: float, level: int, config: CostModelConfig) -> float:
    """Return travel cost for severity level 0=dry, 1=shallow, 2=deep."""
    base_time = float(base_time)
    if base_time < 0:
        raise ValueError("base_time must be non-negative")
    if isinstance(level, bool) or level not in (0, 1, 2):
        raise ValueError(f"level must be 0, 1, or 2, got {level!r}")
    if level == 0:
        multiplier, penalty = 1.0, 0.0
    elif level == 1:
        multiplier, penalty = config.shallow_multiplier, config.shallow_penalty_s
    else:
        multiplier, penalty = config.deep_multiplier, config.deep_penalty_s
    if multiplier < 0 or penalty < 0:
        raise ValueError("severity multipliers and penalties must be non-negative")
    return base_time * multiplier + penalty


def realized_cost(base_time: float, true_level: int, config: CostModelConfig) -> float:
    """Cost an edge using its realized severity."""
    return level_cost(base_time, true_level, config)


def expected_cost(base_time: float, belief, config: CostModelConfig) -> float:
    """Exact expected edge cost under a dry/shallow/deep belief vector."""
    probabilities = tuple(float(value) for value in belief)
    if len(probabilities) != 3:
        raise ValueError("belief must contain probabilities for dry, shallow, and deep")
    if any(value < 0 for value in probabilities):
        raise ValueError("belief probabilities must be non-negative")
    if abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError("belief probabilities must sum to 1")
    return sum(
        probability * level_cost(base_time, level, config)
        for level, probability in enumerate(probabilities)
    )
