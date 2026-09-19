"""
Shared cost model for RL agent reward and Dijkstra baselines.

This module provides a single source of truth for computing effective travel time
on a graph edge, ensuring the RL environment's reward function and the classical
baselines (Dijkstra/A*) use identical cost calculations for fair comparison.

Cost function:
    effective_travel_time = base_travel_time * traffic_multiplier * quality_multiplier * flood_multiplier

where:
    quality_multiplier = 1 + (1 - road_quality_score) * quality_penalty_weight
    flood_multiplier   = 1 + flood_weight * flood_susceptibility

All inputs are expected to be non-negative. The function is pure and stateless.
"""

from dataclasses import dataclass
from typing import Optional
import yaml
from pathlib import Path


@dataclass(frozen=True)
class CostModelConfig:
    """Configuration for the cost model, loaded from config.yaml."""
    flood_weight: float
    quality_penalty_weight: float

    @classmethod
    def from_config(cls, config: dict) -> "CostModelConfig":
        cm = config.get("cost_model", {})
        return cls(
            flood_weight=cm.get("flood_weight", 0.5),
            quality_penalty_weight=cm.get("quality_penalty_weight", 1.0),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "CostModelConfig":
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls.from_config(config)


def effective_travel_time(
    base_travel_time: float,       # seconds, from graph edge 'travel_time'
    traffic_multiplier: float,     # from traffic_profile.parquet lookup
    flood_susceptibility: float,   # 0-1, static per edge (flood_risk)
    road_quality_score: float,     # 0-1, static per edge
    flood_weight: float,           # config: cost_model.flood_weight
    quality_penalty_weight: float, # config: cost_model.quality_penalty_weight
) -> float:
    """
    Compute effective travel time for an edge under current conditions.

    Args:
        base_travel_time: Free-flow travel time in seconds (from OSM length/speed).
        traffic_multiplier: Congestion multiplier >= 1.0 (1.0 = free flow).
        flood_susceptibility: Flood risk score in [0, 1] (1 = highest risk).
        road_quality_score: Road quality score in [0, 1] (1 = best quality).
        flood_weight: Weight controlling flood penalty magnitude (>= 0).
        quality_penalty_weight: Weight controlling quality penalty magnitude (>= 0).

    Returns:
        Effective travel time in seconds (>= base_travel_time).

    Raises:
        ValueError: If any input is negative or flood/quality scores outside [0, 1].
    """
    # Validate inputs
    if base_travel_time < 0:
        raise ValueError(f"base_travel_time must be >= 0, got {base_travel_time}")
    if traffic_multiplier < 1.0:
        raise ValueError(f"traffic_multiplier must be >= 1.0, got {traffic_multiplier}")
    if not 0.0 <= flood_susceptibility <= 1.0:
        raise ValueError(f"flood_susceptibility must be in [0, 1], got {flood_susceptibility}")
    if not 0.0 <= road_quality_score <= 1.0:
        raise ValueError(f"road_quality_score must be in [0, 1], got {road_quality_score}")
    if flood_weight < 0:
        raise ValueError(f"flood_weight must be >= 0, got {flood_weight}")
    if quality_penalty_weight < 0:
        raise ValueError(f"quality_penalty_weight must be >= 0, got {quality_penalty_weight}")

    # Quality multiplier: 1.0 at best quality (score=1), up to 1+weight at worst (score=0)
    quality_multiplier = 1.0 + (1.0 - road_quality_score) * quality_penalty_weight

    # Flood multiplier: 1.0 at zero risk (score=0), up to 1+weight at max risk (score=1)
    flood_multiplier = 1.0 + flood_weight * flood_susceptibility

    return base_travel_time * traffic_multiplier * quality_multiplier * flood_multiplier


def effective_travel_time_from_config(
    base_travel_time: float,
    traffic_multiplier: float,
    flood_susceptibility: float,
    road_quality_score: float,
    config: CostModelConfig,
) -> float:
    """Convenience wrapper using a CostModelConfig object."""
    return effective_travel_time(
        base_travel_time=base_travel_time,
        traffic_multiplier=traffic_multiplier,
        flood_susceptibility=flood_susceptibility,
        road_quality_score=road_quality_score,
        flood_weight=config.flood_weight,
        quality_penalty_weight=config.quality_penalty_weight,
    )


def compute_edge_cost(
    edge_data: dict,
    traffic_multiplier: float,
    config: CostModelConfig,
) -> float:
    """
    Compute effective travel time for a graph edge using its stored attributes.

    Expected edge_data keys:
        - 'travel_time': base free-flow travel time in seconds (from OSMnx)
        - 'flood_risk': flood susceptibility in [0, 1] (from build_flood_layer.py)
        - 'road_quality_score': quality score in [0, 1] (from build_graph_costs.py)

    Missing keys are treated as zero-risk / best-quality (conservative for routing).
    """
    base_tt = edge_data.get("travel_time", 0.0)
    flood_risk = edge_data.get("flood_risk", 0.0)
    rq_score = edge_data.get("road_quality_score", 1.0)

    # Handle NaN values from flood risk sampling
    import math
    if math.isnan(flood_risk):
        flood_risk = 0.0
    if math.isnan(rq_score):
        rq_score = 1.0

    return effective_travel_time_from_config(
        base_travel_time=base_tt,
        traffic_multiplier=traffic_multiplier,
        flood_susceptibility=flood_risk,
        road_quality_score=rq_score,
        config=config,
    )


# Example usage / quick test
if __name__ == "__main__":
    # Quick sanity check
    config = CostModelConfig(flood_weight=0.5, quality_penalty_weight=1.0)

    # Free-flow, no flood, perfect quality -> should equal base
    assert effective_travel_time(100, 1.0, 0.0, 1.0, 0.5, 1.0) == 100.0

    # 2x traffic, no flood, perfect quality -> 200
    assert effective_travel_time(100, 2.0, 0.0, 1.0, 0.5, 1.0) == 200.0

    # Free-flow, max flood (0.5 weight), perfect quality -> 150
    assert effective_travel_time(100, 1.0, 1.0, 1.0, 0.5, 1.0) == 150.0

    # Free-flow, no flood, worst quality (1.0 weight) -> 200
    assert effective_travel_time(100, 1.0, 0.0, 0.0, 0.5, 1.0) == 200.0

    # Combined: 2x traffic, max flood, worst quality -> 100 * 2 * 2 * 1.5 = 600
    assert effective_travel_time(100, 2.0, 1.0, 0.0, 0.5, 1.0) == 600.0

    print("All sanity checks passed.")
    print(f"Config: {config}")
