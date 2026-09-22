"""Project-specific regression tests for the routing environment."""

import networkx as nx
import numpy as np
from gymnasium import spaces
from sb3_contrib import MaskablePPO

from src.env.accra_routing_env import AccraRoutingEnv
from src.env.hazard_simulator import HazardSimulator
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
        graph.add_edge(
            src,
            dst,
            key=0,
            travel_time=10.0,
            highway_class="residential",
            flood_risk=0.0,
            road_quality_score=1.0,
        )
    return graph


def _make_env_with_graph(graph, *, max_episode_steps=5, step_penalty=0.1, flood_penalty=50.0):
    env = AccraRoutingEnv(
        graph_path="data/processed/road_graph_full.graphml",
        traffic_profile_path="data/processed/traffic_profile.parquet",
        climatology_path="data/processed/rainfall_climatology.json",
        config={
            "env": {
                "max_episode_steps": max_episode_steps,
                "flood_penalty": flood_penalty,
                "step_penalty": step_penalty,
                "reveal_radius_hops": 1,
                "train_flood_day_rate": 0.0,
                "mid_episode_event_base_rate": 0.0,
                "max_degree": None,
            },
            "cost_model": {"flood_weight": 0.5, "quality_penalty_weight": 1.0},
        },
        mode="train",
    )
    env.graph = graph
    env.graph_wrapper = GraphWrapper(graph, max_degree=env.graph_wrapper.max_degree)
    env.action_space = spaces.Discrete(env.graph_wrapper.max_degree)
    env.traffic_lookup = {}
    env.hazard_sim.graph = graph
    env.hazard_sim.flood_susceptibility = {
        (u, v, key): HazardSimulator._edge_susceptibility(data)
        for u, v, key, data in graph.edges(keys=True, data=True)
    }
    env.hazard_sim.flooded_edges = set()
    env.hazard_sim.current_position = None
    env.hazard_sim._episode_reset = False
    env.hazard_sim.step = 0
    env.hazard_sim.is_flood_day = False
    env.observation_space = spaces.Dict(
        {
            "current_node_features": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "destination_node_features": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "local_edge_features": spaces.Box(low=-np.inf, high=np.inf, shape=(env.graph_wrapper.max_degree, 4), dtype=np.float32),
        }
    )
    env.episode_context = {"hour": 0, "is_weekend": False, "is_flood_day": False}
    env.hazard_sim.reset_episode()
    env.hazard_sim.is_flood_day = False
    env.hazard_sim.flooded_edges = set()
    return env


def test_graph_wrapper_action_mask_is_bearing_sorted_and_padded():
    graph = _make_cardinal_graph()
    wrapper = GraphWrapper(graph, max_degree=4)

    mask = wrapper.action_mask("c")
    assert mask == [True, True, True, True]
    assert wrapper.action_to_edge("c", 0) == ("c", "n", 0)
    assert wrapper.action_to_edge("c", 1) == ("c", "e", 0)
    assert wrapper.action_to_edge("c", 2) == ("c", "s", 0)
    assert wrapper.action_to_edge("c", 3) == ("c", "w", 0)


def test_action_index_consistency_across_nodes():
    graph = nx.MultiDiGraph()
    graph.add_node("a", x=0.0, y=0.0)
    graph.add_node("north_a", x=0.0, y=2.0)
    graph.add_node("east_a", x=2.0, y=0.0)
    graph.add_node("b", x=10.0, y=0.0)
    graph.add_node("north_b", x=10.0, y=2.0)
    graph.add_node("west_b", x=8.0, y=0.0)

    graph.add_edge("a", "north_a", key=0, travel_time=10.0, highway_class="residential", flood_risk=0.0, road_quality_score=1.0)
    graph.add_edge("a", "east_a", key=0, travel_time=10.0, highway_class="residential", flood_risk=0.0, road_quality_score=1.0)
    graph.add_edge("b", "north_b", key=0, travel_time=10.0, highway_class="residential", flood_risk=0.0, road_quality_score=1.0)
    graph.add_edge("b", "west_b", key=0, travel_time=10.0, highway_class="residential", flood_risk=0.0, road_quality_score=1.0)

    wrapper = GraphWrapper(graph, max_degree=2)
    north_index = 0
    assert wrapper.action_to_edge("a", north_index) == ("a", "north_a", 0)
    assert wrapper.action_to_edge("b", north_index) == ("b", "north_b", 0)


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

    model = MaskablePPO("MultiInputPolicy", env, n_steps=8, verbose=0)
    action, _ = model.predict(obs, action_masks=env.action_masks())
    assert 0 <= int(action) < env.action_space.n


def test_partial_observability_boundary():
    graph = nx.MultiDiGraph()
    for node, coords in {
        "a": (0.0, 0.0),
        "b": (0.0, 1.0),
        "c": (0.0, 2.0),
        "d": (0.0, 3.0),
    }.items():
        graph.add_node(node, x=coords[0], y=coords[1])
    graph.add_edge("a", "b", key=0, travel_time=10.0, highway_class="residential", flood_risk=0.2, road_quality_score=1.0)
    graph.add_edge("b", "c", key=0, travel_time=10.0, highway_class="residential", flood_risk=0.2, road_quality_score=1.0)
    graph.add_edge("c", "d", key=0, travel_time=10.0, highway_class="residential", flood_risk=0.25, road_quality_score=1.0)

    env = _make_env_with_graph(graph)
    env.reset(seed=0)
    env._current_node = "a"
    env.hazard_sim.current_position = "a"
    env.hazard_sim.flooded_edges = {("a", "b", 0)}

    local_feature = env._edge_feature_vector(("a", "b", 0), 0, False)
    outside_feature = env._edge_feature_vector(("c", "d", 0), 0, False)
    assert local_feature[0] == 1.0
    assert outside_feature[0] == 0.25


def test_reward_reduces_correctly_on_boring_episode():
    graph = nx.MultiDiGraph()
    for node, coords in {"start": (0.0, 0.0), "mid": (0.0, 1.0), "far": (0.0, 2.0)}.items():
        graph.add_node(node, x=coords[0], y=coords[1])
    graph.add_edge("start", "mid", key=0, travel_time=12.0, highway_class="residential", flood_risk=0.0, road_quality_score=1.0)
    graph.add_edge("mid", "far", key=0, travel_time=12.0, highway_class="residential", flood_risk=0.0, road_quality_score=1.0)

    env = _make_env_with_graph(graph, step_penalty=0.1)
    env.reset(seed=0)
    env.traffic_lookup = {}
    env._current_node = "start"
    env._destination_node = "far"
    env.hazard_sim.is_flood_day = False
    env.hazard_sim.flooded_edges = set()
    env.hazard_sim.set_current_position("start")
    env.episode_context["is_flood_day"] = False

    _, reward, terminated, truncated, _ = env.step(0)
    assert terminated is False
    assert truncated is False
    assert reward == -12.0 - 0.1


def test_termination_vs_truncation():
    graph = nx.MultiDiGraph()
    for node, coords in {"start": (0.0, 0.0), "mid": (0.0, 1.0), "goal": (0.0, 2.0)}.items():
        graph.add_node(node, x=coords[0], y=coords[1])
    graph.add_edge("start", "mid", key=0, travel_time=5.0, highway_class="residential", flood_risk=0.0, road_quality_score=1.0)
    graph.add_edge("mid", "goal", key=0, travel_time=5.0, highway_class="residential", flood_risk=0.0, road_quality_score=1.0)

    env = _make_env_with_graph(graph, max_episode_steps=1, step_penalty=0.0)
    env.reset(seed=0)
    env._current_node = "start"
    env._destination_node = "goal"
    env.hazard_sim.set_current_position("start")
    env.hazard_sim.is_flood_day = False
    env.hazard_sim.flooded_edges = set()

    _, reward, terminated, truncated, info = env.step(0)
    assert terminated is False
    assert truncated is True
    assert info["truncated"] is True
    assert info["terminated"] is False

    env2 = _make_env_with_graph(graph, max_episode_steps=10, step_penalty=0.0)
    env2.reset(seed=0)
    env2._current_node = "start"
    env2._destination_node = "goal"
    env2.hazard_sim.set_current_position("start")
    env2.hazard_sim.is_flood_day = False
    env2.hazard_sim.flooded_edges = set()

    _, reward2, terminated2, truncated2, info2 = env2.step(0)
    assert terminated2 is False
    assert truncated2 is False

    env2._current_node = "mid"
    env2._destination_node = "goal"
    env2.hazard_sim.set_current_position("mid")
    _, reward3, terminated3, truncated3, info3 = env2.step(0)
    assert terminated3 is True
    assert truncated3 is False
    assert info3["terminated"] is True
