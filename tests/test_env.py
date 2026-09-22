"""Project-specific regression tests for the routing environment."""

import networkx as nx

from src.env.accra_routing_env import AccraRoutingEnv
from src.env.graph_wrapper import GraphWrapper


def _make_cardinal_graph():
    graph = nx.MultiDiGraph()
    nodes = {
        "c": {"x": 0.0, "y": 0.0},
        "n": {"x": 0.0, "y": 1.0},
        "e": {"x": 1.0, "y": 0.0},
        "s": {"x": 0.0, "y": -1.0},
        "w": {"x": -1.0, "y": 0.0},
    }
    graph.add_nodes_from((node, data) for node, data in nodes.items())
    for src, dst in [("c", "n"), ("c", "e"), ("c", "s"), ("c", "w")]:
        graph.add_edge(src, dst, key=0, travel_time=10.0, highway_class="residential", flood_risk=0.0, road_quality_score=1.0)
    return graph


def test_graph_wrapper_action_mask_is_bearing_sorted_and_padded():
    graph = _make_cardinal_graph()
    wrapper = GraphWrapper(graph, max_degree=4)

    mask = wrapper.action_mask("c")
    assert mask == [True, True, True, True]
    assert wrapper.action_to_edge("c", 0) == ("c", "n", 0)
    assert wrapper.action_to_edge("c", 1) == ("c", "e", 0)
    assert wrapper.action_to_edge("c", 2) == ("c", "s", 0)
    assert wrapper.action_to_edge("c", 3) == ("c", "w", 0)


def test_env_smoke_and_masks_are_exposed_for_sb3():
    env = AccraRoutingEnv(
        graph_path="data/processed/road_graph_full.graphml",
        traffic_profile_path="data/processed/traffic_profile.parquet",
        climatology_path="data/processed/rainfall_climatology.json",
        config={
            "env": {
                "max_episode_steps": 5,
                "flood_penalty": 50,
                "step_penalty": 0.1,
                "reveal_radius_hops": 1,
                "train_flood_day_rate": 0.0,
                "mid_episode_event_base_rate": 0.0,
                "max_degree": None,
            },
            "cost_model": {"flood_weight": 0.5, "quality_penalty_weight": 1.0},
        },
        mode="train",
    )

    obs, info = env.reset(seed=123)
    assert "current_node_features" in obs
    assert "destination_node_features" in obs
    assert "local_edge_features" in obs
    assert len(env.action_masks()) == env.action_space.n
    assert sum(env.action_masks()) > 0
    assert "origin" in info
    assert "destination" in info
