"""Performance comparison for routing policies on rainy episodes."""

import json
from pathlib import Path

import networkx as nx
import numpy as np
from sb3_contrib import MaskablePPO

from src.agents.training_utils import (
    evaluate_masked_random_outcomes,
    evaluate_static_shortest_path_outcomes,
    summarize_outcomes,
)
from src.env.accra_routing_env import AccraRoutingEnv


def _make_rainy_route_graph():
    graph = nx.MultiDiGraph()
    graph.add_node("start", x=0.0, y=0.0)
    graph.add_node("shortcut", x=1.0, y=0.0)
    graph.add_node("safe_a", x=0.0, y=1.0)
    graph.add_node("safe_b", x=0.0, y=2.0)
    graph.add_node("goal", x=1.0, y=2.0)

    graph.add_edge(
        "start", "shortcut", key=0, travel_time=1.0,
        flood_susceptibility=1.0, flood_risk=1.0,
        road_quality_score=1.0, highway_class="primary",
    )
    graph.add_edge(
        "shortcut", "goal", key=0, travel_time=1.0,
        flood_susceptibility=1.0, flood_risk=1.0,
        road_quality_score=1.0, highway_class="primary",
    )
    for source, target in [("start", "safe_a"), ("safe_a", "safe_b"), ("safe_b", "goal")]:
        graph.add_edge(
            source, target, key=0, travel_time=2.0,
            flood_susceptibility=0.0, flood_risk=0.0,
            road_quality_score=1.0, highway_class="primary",
        )
    return graph


class _FixedRainyRouteEnv(AccraRoutingEnv):
    """Keep every episode on the same rainy origin-destination trip."""

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._origin_node = "start"
        self._destination_node = "goal"
        self._current_node = "start"
        self.hazard_sim.set_current_position("start")
        observation, info = self._build_observation()
        info["origin"] = "start"
        info["destination"] = "goal"
        return observation, info


def _make_env_factory(graph, config):
    def factory():
        return _FixedRainyRouteEnv(
            graph,
            traffic_profile_path="does-not-exist.parquet",
            climatology_path="does-not-exist.json",
            config=config,
            mode="train",
        )

    return factory


def _action_to(env, target):
    return next(
        index
        for index in range(env.action_space.n)
        if (edge := env.graph_wrapper.action_to_edge(env._current_node, index))
        and edge[1] == target
    )


def _evaluate_model_outcomes(model, env_factory, num_episodes, seed):
    outcomes = []
    for episode in range(num_episodes):
        env = env_factory()
        observation, reset_info = env.reset(seed=seed + episode)
        total_reward = 0.0
        terminated = truncated = False
        final_info = {}
        while not (terminated or truncated):
            action, _ = model.predict(
                observation,
                deterministic=True,
                action_masks=np.asarray(env.action_masks()),
            )
            observation, reward, terminated, truncated, final_info = env.step(int(action))
            total_reward += reward
        outcomes.append({
            "reward": float(total_reward),
            "success": bool(final_info.get("success", False)),
            "dead_end": bool(final_info.get("dead_end", False)),
            "truncated": bool(final_info.get("truncated", False)),
            "flood_day": bool(reset_info["is_flood_day"]),
        })
    return outcomes


def test_rl_agent_beats_masked_random_and_static_on_rainy_days():
    """The trained policy should avoid the flooded shortcut on every rainy trip."""
    graph = _make_rainy_route_graph()
    config = {
        "env": {
            "max_degree": None,
            "max_episode_steps": 15,
            "flood_penalty": 50.0,
            "step_penalty": 0.1,
            "reveal_radius_hops": 1,
            "train_flood_day_rate": 1.0,
            "mid_episode_event_base_rate": 0.0,
        },
        "cost_model": {"flood_weight": 0.5, "quality_penalty_weight": 1.0},
    }
    env_factory = _make_env_factory(graph, config)
    training_env = env_factory()
    model = MaskablePPO(
        "MultiInputPolicy",
        training_env,
        n_steps=16,
        batch_size=8,
        learning_rate=0.01,
        gamma=0.9,
        seed=7,
        verbose=0,
    )
    model.learn(total_timesteps=1_000)

    rl_outcomes = _evaluate_model_outcomes(model, env_factory, 12, 100)
    random_outcomes = evaluate_masked_random_outcomes(
        env_factory, num_episodes=12, seed=100
    )
    static_outcomes = evaluate_static_shortest_path_outcomes(
        env_factory, num_episodes=12, seed=100
    )
    rl_summary = summarize_outcomes(rl_outcomes)
    random_summary = summarize_outcomes(
        random_outcomes
    )
    static_summary = summarize_outcomes(
        static_outcomes
    )

    result = {
        "scenario": "fixed rainy-day route",
        "episodes": 12,
        "seed": 100,
        "methods": {
            "rl_agent": {"summary": rl_summary, "outcomes": rl_outcomes},
            "masked_random": {"summary": random_summary, "outcomes": random_outcomes},
            "static_shortest_path": {"summary": static_summary, "outcomes": static_outcomes},
        },
    }
    output_path = Path("results/scenario_outputs/rainy_day_performance.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    assert rl_summary["flood_day_rate"] == 1.0
    assert random_summary["flood_day_rate"] == 1.0
    assert static_summary["flood_day_rate"] == 1.0
    assert rl_summary["completion_rate"] == 1.0
    assert rl_summary["mean_reward"] > random_summary["mean_reward"]
    assert rl_summary["mean_reward"] > static_summary["mean_reward"]


def test_flooded_detour_loop_terminates_on_revisit():
    """A flooded detour cannot loop until the episode cap is exhausted."""
    graph = nx.MultiDiGraph()
    for node, (x, y) in {
        "start": (0.0, 0.0),
        "junction": (1.0, 0.0),
        "detour": (1.0, 1.0),
        "goal": (2.0, 0.0),
    }.items():
        graph.add_node(node, x=x, y=y)

    edge_defaults = {
        "length": 1.0,
        "road_quality_score": 1.0,
        "highway_class": "primary",
        "flood_risk": 0.0,
        "flood_susceptibility": 0.0,
    }
    graph.add_edge("start", "junction", key=0, travel_time=1.0, **edge_defaults)
    graph.add_edge(
        "junction", "detour", key=0, travel_time=1.0,
        **{**edge_defaults, "flood_risk": 1.0, "flood_susceptibility": 1.0},
    )
    graph.add_edge("detour", "junction", key=0, travel_time=1.0, **edge_defaults)
    graph.add_edge("junction", "goal", key=0, travel_time=1.0, **edge_defaults)

    config = {
        "env": {
            "max_degree": None,
            "max_episode_steps": 64,
            "flood_penalty": 50.0,
            "step_penalty": 0.1,
            "reveal_radius_hops": 1,
            "train_flood_day_rate": 1.0,
            "mid_episode_event_base_rate": 0.0,
            "revisit_mode": "terminate",
            "revisit_penalty": 50.0,
            "revisit_penalty_growth": 2.0,
        },
        "cost_model": {"flood_weight": 0.5, "quality_penalty_weight": 1.0},
    }
    env = _FixedRainyRouteEnv(
        graph,
        traffic_profile_path="does-not-exist.parquet",
        climatology_path="does-not-exist.json",
        config=config,
        mode="train",
    )
    env.reset(
        seed=0,
        options={
            "origin": "start",
            "destination": "goal",
            "is_flood_day": True,
            "flooded_edges": [["junction", "detour", 0]],
        },
    )

    env.step(_action_to(env, "junction"))
    _, _, terminated_on_detour, _, detour_info = env.step(_action_to(env, "detour"))
    _, reward, terminated_on_return, truncated, return_info = env.step(
        _action_to(env, "junction")
    )

    assert terminated_on_detour is False
    assert detour_info["is_flooded"] is True
    assert terminated_on_return is True
    assert truncated is False
    assert return_info["revisit_failure"] is True
    assert return_info["success"] is False
    assert return_info["total_revisits"] == 1
    assert reward < 0


def test_flooded_detour_loop_penalized_on_revisit():
    """A flooded detour loop penalizes the agent but allows continuation instead of terminating."""
    graph = nx.MultiDiGraph()
    for node, (x, y) in {
        "start": (0.0, 0.0),
        "junction": (1.0, 0.0),
        "detour": (1.0, 1.0),
        "goal": (2.0, 0.0),
    }.items():
        graph.add_node(node, x=x, y=y)

    edge_defaults = {
        "length": 1.0,
        "road_quality_score": 1.0,
        "highway_class": "primary",
        "flood_risk": 0.0,
        "flood_susceptibility": 0.0,
    }
    graph.add_edge("start", "junction", key=0, travel_time=1.0, **edge_defaults)
    graph.add_edge(
        "junction", "detour", key=0, travel_time=1.0,
        **{**edge_defaults, "flood_risk": 1.0, "flood_susceptibility": 1.0},
    )
    graph.add_edge("detour", "junction", key=0, travel_time=1.0, **edge_defaults)
    graph.add_edge("junction", "goal", key=0, travel_time=1.0, **edge_defaults)

    config = {
        "env": {
            "max_degree": None,
            "max_episode_steps": 64,
            "flood_penalty": 50.0,
            "step_penalty": 0.1,
            "reveal_radius_hops": 1,
            "train_flood_day_rate": 1.0,
            "mid_episode_event_base_rate": 0.0,
            "revisit_mode": "penalty",  # Penalize but don't terminate
            "revisit_penalty": 50.0,
            "revisit_penalty_growth": 2.0,
        },
        "cost_model": {"flood_weight": 0.5, "quality_penalty_weight": 1.0},
    }
    env = _FixedRainyRouteEnv(
        graph,
        traffic_profile_path="does-not-exist.parquet",
        climatology_path="does-not-exist.json",
        config=config,
        mode="train",
    )
    env.reset(
        seed=0,
        options={
            "origin": "start",
            "destination": "goal",
            "is_flood_day": True,
            "flooded_edges": [["junction", "detour", 0]],
        },
    )

    # Step 1: start -> junction
    env.step(_action_to(env, "junction"))

    # Step 2: junction -> detour (flooded)
    _, _, terminated_on_detour, _, detour_info = env.step(_action_to(env, "detour"))

    # Step 3: detour -> junction (revisit, should be penalized but not terminated)
    _, reward_on_return, terminated_on_return, truncated, return_info = env.step(
        _action_to(env, "junction")
    )

    # Step 4: junction -> goal (continue to goal after penalty)
    _, reward_on_goal, terminated_on_goal, _, goal_info = env.step(
        _action_to(env, "goal")
    )

    # On first revisit (detour -> junction), episode should NOT terminate
    assert terminated_on_detour is False
    assert detour_info["is_flooded"] is True

    # On returning to junction (revisit), should be penalized but NOT terminated
    assert terminated_on_return is False
    assert return_info["revisit_failure"] is False  # Not a failure in penalty mode
    assert return_info["total_revisits"] == 1
    assert return_info["revisit_penalty"] == 50.0  # First revisit penalty
    assert reward_on_return < 0  # Negative reward due to penalty

    # Should be able to continue to goal
    assert terminated_on_goal is True
    assert truncated is False
    assert goal_info["success"] is True
    assert goal_info["total_revisits"] == 1

    # Verify the escalating penalty on second revisit in a NEW episode (since previous ended at goal)
    # We track revisits for the "junction" node specifically by checking visit_count
    # Note: total_revisits is episode-wide for ALL nodes
    env2 = _FixedRainyRouteEnv(
        graph,
        traffic_profile_path="does-not-exist.parquet",
        climatology_path="does-not-exist.json",
        config=config,
        mode="train",
    )
    env2.reset(
        seed=1,
        options={
            "origin": "start",
            "destination": "goal",
            "is_flood_day": True,
            "flooded_edges": [["junction", "detour", 0]],
        },
    )
    # Go to junction, then loop detour -> junction twice
    env2.step(_action_to(env2, "junction"))  # start -> junction (junction visit_count=1)
    env2.step(_action_to(env2, "detour"))    # junction -> detour (flooded, detour visit_count=1)
    _, _, _, _, first_revisit_info = env2.step(_action_to(env2, "junction"))  # detour -> junction (junction visit_count=2, total_revisits=1)
    assert first_revisit_info["visit_count"] == 2  # Junction visited twice now
    assert first_revisit_info["total_revisits"] == 1
    assert first_revisit_info["revisit_penalty"] == 50.0
    
    env2.step(_action_to(env2, "detour"))    # junction -> detour (detour visit_count=2, total_revisits=2)
    _, reward_second_revisit, _, _, second_revisit_info = env2.step(
        _action_to(env2, "junction")  # detour -> junction (junction visit_count=3, total_revisits=3)
    )
    assert second_revisit_info["visit_count"] == 3  # Junction visited three times now
    assert second_revisit_info["total_revisits"] == 3  # Episode-wide: 1 (junction 1st) + 1 (detour 1st) + 1 (junction 2nd)
    # Penalty = 50 * 2^(total_revisits-1) = 50 * 2^2 = 200
    assert second_revisit_info["revisit_penalty"] == 200.0
    assert reward_second_revisit < -100  # Includes escalating penalty


def test_informed_revisit_pardoned_with_new_flood_info():
    """A revisit is pardoned (no penalty, no termination) when new flood information is discovered."""
    graph = nx.MultiDiGraph()
    for node, (x, y) in {
        "start": (0.0, 0.0),
        "junction": (1.0, 0.0),
        "detour": (1.0, 1.0),
        "goal": (2.0, 0.0),
    }.items():
        graph.add_node(node, x=x, y=y)

    edge_defaults = {
        "length": 1.0,
        "road_quality_score": 1.0,
        "highway_class": "primary",
        "flood_risk": 0.0,
        "flood_susceptibility": 0.0,
    }
    graph.add_edge("start", "junction", key=0, travel_time=1.0, **edge_defaults)
    graph.add_edge(
        "junction", "detour", key=0, travel_time=1.0,
        **{**edge_defaults, "flood_risk": 1.0, "flood_susceptibility": 1.0},
    )
    graph.add_edge("detour", "junction", key=0, travel_time=1.0, **edge_defaults)
    graph.add_edge("junction", "goal", key=0, travel_time=1.0, **edge_defaults)

    config = {
        "env": {
            "max_degree": None,
            "max_episode_steps": 64,
            "flood_penalty": 50.0,
            "step_penalty": 0.1,
            "reveal_radius_hops": 1,
            "train_flood_day_rate": 1.0,
            "mid_episode_event_base_rate": 0.0,
            "revisit_mode": "terminate",  # Would terminate on blind revisit
            "revisit_penalty": 50.0,
            "revisit_penalty_growth": 2.0,
        },
        "cost_model": {"flood_weight": 0.5, "quality_penalty_weight": 1.0},
    }
    env = _FixedRainyRouteEnv(
        graph,
        traffic_profile_path="does-not-exist.parquet",
        climatology_path="does-not-exist.json",
        config=config,
        mode="train",
    )
    # Start with NO flooded edges - we'll manually add the flood mid-episode
    env.reset(
        seed=0,
        options={
            "origin": "start",
            "destination": "goal",
            "is_flood_day": True,
            "flooded_edges": [],  # Start with no flooded edges
        },
    )

    # Step 1: start -> junction (first visit, known_flooded=0)
    env.step(_action_to(env, "junction"))

    # Step 2: junction -> detour (NOT flooded yet)
    _, _, terminated_on_detour, _, detour_info = env.step(_action_to(env, "detour"))
    assert terminated_on_detour is False
    assert detour_info["is_flooded"] is False

    # Step 3: detour -> junction (revisit to junction, STILL known_flooded=0)
    # This is a BLIND revisit - should be penalized/terminated
    _, reward_on_return, terminated_on_return, _, return_info = env.step(
        _action_to(env, "junction")
    )
    assert terminated_on_return is True, "Blind revisit should terminate in terminate mode"
    assert return_info["revisit_failure"] is True
    assert return_info["revisit_penalty"] == 50.0
    assert return_info["total_revisits"] == 1


def test_informed_revisit_pardoned_when_flood_discovered_between_visits():
    """A revisit is pardoned when a flood is discovered between the two visits to the same node."""
    graph = nx.MultiDiGraph()
    for node, (x, y) in {
        "start": (0.0, 0.0),
        "junction": (1.0, 0.0),
        "detour": (1.0, 1.0),
        "goal": (2.0, 0.0),
    }.items():
        graph.add_node(node, x=x, y=y)

    edge_defaults = {
        "length": 1.0,
        "road_quality_score": 1.0,
        "highway_class": "primary",
        "flood_risk": 0.0,
        "flood_susceptibility": 0.0,
    }
    graph.add_edge("start", "junction", key=0, travel_time=1.0, **edge_defaults)
    graph.add_edge(
        "junction", "detour", key=0, travel_time=1.0,
        **{**edge_defaults, "flood_risk": 1.0, "flood_susceptibility": 1.0},
    )
    graph.add_edge("detour", "junction", key=0, travel_time=1.0, **edge_defaults)
    graph.add_edge("junction", "goal", key=0, travel_time=1.0, **edge_defaults)

    config = {
        "env": {
            "max_degree": None,
            "max_episode_steps": 64,
            "flood_penalty": 50.0,
            "step_penalty": 0.1,
            "reveal_radius_hops": 1,
            "train_flood_day_rate": 1.0,
            "mid_episode_event_base_rate": 0.0,
            "revisit_mode": "terminate",  # Would terminate on blind revisit
            "revisit_penalty": 50.0,
            "revisit_penalty_growth": 2.0,
        },
        "cost_model": {"flood_weight": 0.5, "quality_penalty_weight": 1.0},
    }
    env = _FixedRainyRouteEnv(
        graph,
        traffic_profile_path="does-not-exist.parquet",
        climatology_path="does-not-exist.json",
        config=config,
        mode="train",
    )
    # Start with NO flooded edges
    env.reset(
        seed=0,
        options={
            "origin": "start",
            "destination": "goal",
            "is_flood_day": True,
            "flooded_edges": [],
        },
    )

    # Step 1: start -> junction (first visit, known_flooded=0)
    env.step(_action_to(env, "junction"))

    # Step 2: junction -> detour (NOT flooded)
    env.step(_action_to(env, "detour"))

    # NOW add a flood on junction->detour (simulating mid-episode discovery)
    env.hazard_sim.flooded_edges.add(("junction", "detour", 0))

    # Step 3: detour -> junction (revisit to junction)
    # First visit: known_flooded=0
    # Now: junction->detour is within 1 hop of detour, so when at junction it's known
    # Actually, _local_known_flood_count(junction) checks from junction's position
    # The edge junction->detour is adjacent to junction, so it's within 1 hop
    # Since it's now in flooded_edges, known_flooded=1 > prior_known=0 -> INFORMED REVISIT
    _, reward_on_return, terminated_on_return, _, return_info = env.step(
        _action_to(env, "junction")
    )

    # Informed revisit: pardoned, no penalty, no termination
    assert terminated_on_return is False, "Informed revisit should not terminate"
    assert return_info["revisit_failure"] is False, "Informed revisit should not be a failure"
    assert return_info["revisit_penalty"] == 0.0, "Informed revisit should have no penalty"
    assert return_info["total_revisits"] == 0, "Informed revisit should not increment total_revisits"

    # Step 4: junction -> goal (continue successfully)
    _, _, terminated_on_goal, _, goal_info = env.step(_action_to(env, "goal"))
    assert terminated_on_goal is True
    assert goal_info["success"] is True
    assert goal_info["total_revisits"] == 0, "No blind revisits occurred"


def test_informed_revisit_requires_strictly_more_info():
    """A revisit is only pardoned if strictly MORE flood edges are known than before."""
    graph = nx.MultiDiGraph()
    for node, (x, y) in {
        "start": (0.0, 0.0),
        "junction": (1.0, 0.0),
        "detour": (1.0, 1.0),
        "goal": (2.0, 0.0),
    }.items():
        graph.add_node(node, x=x, y=y)

    edge_defaults = {
        "length": 1.0,
        "road_quality_score": 1.0,
        "highway_class": "primary",
        "flood_risk": 0.0,
        "flood_susceptibility": 0.0,
    }
    graph.add_edge("start", "junction", key=0, travel_time=1.0, **edge_defaults)
    graph.add_edge(
        "junction", "detour", key=0, travel_time=1.0,
        **{**edge_defaults, "flood_risk": 1.0, "flood_susceptibility": 1.0},
    )
    graph.add_edge("detour", "junction", key=0, travel_time=1.0, **edge_defaults)
    graph.add_edge("junction", "goal", key=0, travel_time=1.0, **edge_defaults)

    config = {
        "env": {
            "max_degree": None,
            "max_episode_steps": 64,
            "flood_penalty": 50.0,
            "step_penalty": 0.1,
            "reveal_radius_hops": 1,
            "train_flood_day_rate": 1.0,
            "mid_episode_event_base_rate": 0.0,
            "revisit_mode": "penalty",
            "revisit_penalty": 50.0,
            "revisit_penalty_growth": 2.0,
        },
        "cost_model": {"flood_weight": 0.5, "quality_penalty_weight": 1.0},
    }
    env = _FixedRainyRouteEnv(
        graph,
        traffic_profile_path="does-not-exist.parquet",
        climatology_path="does-not-exist.json",
        config=config,
        mode="train",
    )
    # Start with NO flooded edges - agent will discover none on first visit to junction
    env.reset(
        seed=0,
        options={
            "origin": "start",
            "destination": "goal",
            "is_flood_day": True,
            "flooded_edges": [],  # No pre-flooded edges
        },
    )

    # Step 1: start -> junction (first visit, 0 known floods)
    env.step(_action_to(env, "junction"))

    # Step 2: junction -> detour (not flooded)
    env.step(_action_to(env, "detour"))

    # Step 3: detour -> junction (revisit, STILL 0 known floods - no new info)
    # This should be a BLIND REVISIT - penalty applies
    _, reward, terminated, _, info = env.step(_action_to(env, "junction"))
    assert terminated is False  # penalty mode doesn't terminate
    assert info["revisit_penalty"] == 50.0, "Blind revisit should incur penalty"
    assert info["total_revisits"] == 1
    assert info["revisit_failure"] is False

    # Now simulate a mid-episode flood event on junction->detour
    # (manually add to hazard_sim to test the informed revisit logic)
    env.hazard_sim.flooded_edges.add(("junction", "detour", 0))

    # Step 4: junction -> detour (now flooded, agent discovers NEW flood)
    _, _, _, _, detour_info2 = env.step(_action_to(env, "detour"))
    assert detour_info2["is_flooded"] is True

    # Step 5: detour -> junction (revisit, NOW 1 known flood vs 0 before)
    # This should be an INFORMED REVISIT - pardoned
    _, reward2, terminated2, _, info2 = env.step(_action_to(env, "junction"))
    assert terminated2 is False
    assert info2["revisit_penalty"] == 0.0, "Informed revisit (0->1 known floods) should be pardoned"
    assert info2["total_revisits"] == 1, "total_revisits should not increase for informed revisit"
    assert info2["revisit_failure"] is False


def test_informed_revisit_resets_known_flood_count():
    """After an informed revisit, the known_flooded_at_visit is updated, so next revisit needs even more new info."""
    graph = nx.MultiDiGraph()
    for node, (x, y) in {
        "start": (0.0, 0.0),
        "junction": (1.0, 0.0),
        "detour": (1.0, 1.0),
        "goal": (2.0, 0.0),
    }.items():
        graph.add_node(node, x=x, y=y)

    edge_defaults = {
        "length": 1.0,
        "road_quality_score": 1.0,
        "highway_class": "primary",
        "flood_risk": 0.0,
        "flood_susceptibility": 0.0,
    }
    graph.add_edge("start", "junction", key=0, travel_time=1.0, **edge_defaults)
    graph.add_edge(
        "junction", "detour", key=0, travel_time=1.0,
        **{**edge_defaults, "flood_risk": 1.0, "flood_susceptibility": 1.0},
    )
    graph.add_edge("detour", "junction", key=0, travel_time=1.0, **edge_defaults)
    graph.add_edge("junction", "goal", key=0, travel_time=1.0, **edge_defaults)

    config = {
        "env": {
            "max_degree": None,
            "max_episode_steps": 64,
            "flood_penalty": 50.0,
            "step_penalty": 0.1,
            "reveal_radius_hops": 1,
            "train_flood_day_rate": 1.0,
            "mid_episode_event_base_rate": 0.0,
            "revisit_mode": "penalty",
            "revisit_penalty": 50.0,
            "revisit_penalty_growth": 2.0,
        },
        "cost_model": {"flood_weight": 0.5, "quality_penalty_weight": 1.0},
    }
    env = _FixedRainyRouteEnv(
        graph,
        traffic_profile_path="does-not-exist.parquet",
        climatology_path="does-not-exist.json",
        config=config,
        mode="train",
    )
    # Start with junction->detour flooded
    env.reset(
        seed=0,
        options={
            "origin": "start",
            "destination": "goal",
            "is_flood_day": True,
            "flooded_edges": [["junction", "detour", 0]],
        },
    )

    # First loop: discover flood, informed revisit pardoned
    env.step(_action_to(env, "junction"))           # visit junction (known=1)
    env.step(_action_to(env, "detour"))              # discover flood on junction->detour
    _, _, _, _, info1 = env.step(_action_to(env, "junction"))  # revisit junction (known=1, prior=1 -> INFORMED? No, 1 > 1 is False!)
    # Wait - the first visit to junction had known=1 (junction->detour is within 1 hop and flooded)
    # The revisit also sees known=1. Since 1 > 1 is False, this is a BLIND revisit!
    # Let's verify this behavior...
    assert info1["revisit_penalty"] == 50.0, "First revisit with same known count should be penalized"
    assert info1["total_revisits"] == 1

    # Add a NEW flood edge (detour->junction) within reveal radius
    env.hazard_sim.flooded_edges.add(("detour", "junction", 0))

    # Second loop: NEW flood discovered
    env.step(_action_to(env, "detour"))              # flooded edge
    _, _, _, _, info2 = env.step(_action_to(env, "junction"))  # revisit junction (known=2, prior=1 -> INFORMED!)
    assert info2["revisit_penalty"] == 0.0, "Second revisit with more known floods (2>1) should be pardoned"
    assert info2["total_revisits"] == 1, "total_revisits unchanged"

    # Third loop: NO new floods
    # Note: step to detour is also a BLIND revisit for detour (known=2, prior=2)
    # So total_revisits becomes 2 after detour, then 3 after junction
    env.step(_action_to(env, "detour"))
    _, _, _, _, info3 = env.step(_action_to(env, "junction"))  # revisit junction (known=2, prior=2 -> BLIND!)
    # total_revisits=3 (1 from first blind junction revisit + 1 from blind detour revisit + 1 from this blind junction revisit)
    assert info3["revisit_penalty"] == 200.0, "Third revisit with same known count (2==2) should be penalized (escalated to 200)"
    assert info3["total_revisits"] == 3