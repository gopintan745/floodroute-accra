"""Expected-cost and full-information routing baseline tests."""

import networkx as nx
import numpy as np
import pandas as pd

from src.baselines.dijkstra_dynamic import (
    annotate_dynamic_costs,
    build_traffic_lookup,
    dynamic_shortest_path,
    dynamic_shortest_path_with_replanning,
    oracle_shortest_path,
)
from src.common.cost_model import (
    CostModelConfig,
    base_edge_time,
    expected_cost,
    level_cost,
    realized_cost,
)
from src.env.hazard_simulator import HazardSimulator


def _graph():
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"
    for node, x, y in [
        (1, 0.0, 0.0), (2, .003, 0.0), (3, .006, 0.0),
        (4, .003, .01), (5, .006, .01),
    ]:
        graph.add_node(node, x=x, y=y)
    attrs = {"road_quality_score": 1.0, "highway_class": "primary", "length": 1.0}
    graph.add_edge(1, 2, key=0, travel_time=5.0, flood_susceptibility=0.0, **attrs)
    graph.add_edge(2, 3, key=0, travel_time=5.0, flood_susceptibility=1.0, **attrs)
    graph.add_edge(2, 4, key=0, travel_time=8.0, flood_susceptibility=0.0, **attrs)
    graph.add_edge(4, 5, key=0, travel_time=8.0, flood_susceptibility=0.0, **attrs)
    graph.add_edge(5, 3, key=0, travel_time=8.0, flood_susceptibility=0.0, **attrs)
    return graph


def _profile():
    return pd.DataFrame([
        {"highway_class": "primary", "hour": hour, "is_weekend": weekend,
         "multiplier": 1.0}
        for hour in range(24) for weekend in (False, True)
    ])


def _config():
    return {
        "env": {"train_flood_day_rate": 1.0, "reveal_radius_hops": 1,
                "max_episode_steps": 20, "step_penalty": 0.1},
        "flood_model": {
            "signal_mode": "exact", "intensity_weight_alpha": 0.5,
            "deep_fraction_min": 0.1, "deep_fraction_max": 0.7,
            "mid_episode_event_base_rate": 0.0,
            "level_costs": {
                "shallow": {"multiplier": 2.0, "penalty_s": 0.0},
                "deep": {"multiplier": 3.0, "penalty_s": 600.0},
            },
        },
        "cost_model": {"quality_penalty_weight": 1.0},
    }


def _sim(graph, levels=None):
    sim = HazardSimulator(graph, {}, _config(), "train", seed=7)
    scenario = {"is_flood_day": True, "intensity": 0.5}
    if levels:
        scenario["edge_levels"] = levels
    sim.reset_episode(scenario)
    return sim


def test_base_level_expected_and_realized_costs():
    cfg = CostModelConfig()
    assert base_edge_time(100, 1.2, .8, 1.0) == 144.0
    assert level_cost(100, 0, cfg) == 100.0
    assert level_cost(100, 1, cfg) == 200.0
    assert level_cost(100, 2, cfg) == 900.0
    assert realized_cost(100, 2, cfg) == 900.0
    assert expected_cost(100, [.6, .3, .1], cfg) == 210.0


def test_expected_cost_annotation_uses_beliefs():
    graph = _graph()
    lookup = build_traffic_lookup(_profile())
    cfg = CostModelConfig()
    beliefs = {(2, 3, 0): np.array([.6, .3, .1])}

    annotate_dynamic_costs(graph, lookup, 12, False, cfg, beliefs)

    expected = expected_cost(5.0, [.6, .3, .1], cfg)
    assert graph.edges[2, 3, 0]["dynamic_cost"] == expected
    assert graph.edges[1, 2, 0]["dynamic_cost"] == 5.0


def test_dynamic_planner_uses_local_posterior_and_detours_deep_edge():
    graph = _graph()
    hazard = _sim(graph, {(2, 3, 0): 2})
    result = dynamic_shortest_path(
        graph, _profile(), (0.0, 0.0), (.006, 0.0), 12, False,
        CostModelConfig(), hazard_sim=hazard,
    )

    assert result["path"] == [1, 2, 4, 5, 3]
    assert result["total_expected_cost"] == 29.0


def test_replanning_returns_realized_level_cost_and_revisit_count():
    graph = _graph()
    hazard = _sim(graph, {(2, 3, 0): 1})
    result = dynamic_shortest_path_with_replanning(
        graph, hazard, build_traffic_lookup(_profile()), (0.0, 0.0), (.006, 0.0),
        12, False, CostModelConfig(), reveal_radius_hops=1, max_steps=20,
    )

    assert result["reached_destination"] is True
    assert result["path"] == [1, 2, 3]
    assert result["total_realized_time"] == 15.0
    assert result["total_revisits"] == 0


def test_oracle_uses_realized_levels():
    graph = _graph()
    hazard = _sim(graph, {(2, 3, 0): 2})
    result = oracle_shortest_path(
        graph, hazard, build_traffic_lookup(_profile()), (0.0, 0.0), (.006, 0.0),
        12, False, CostModelConfig(),
    )

    assert result["path"] == [1, 2, 4, 5, 3]
    assert result["total_realized_time"] == 29.0


def test_traffic_lookup_missing_values_fall_back_to_free_flow():
    graph = _graph()
    annotate_dynamic_costs(
        graph, {}, 14, False, CostModelConfig(),
        {(2, 3, 0): [1.0, 0.0, 0.0]},
    )
    assert graph.edges[2, 3, 0]["dynamic_cost"] == 5.0
