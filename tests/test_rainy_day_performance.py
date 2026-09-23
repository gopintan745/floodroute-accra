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
            "max_episode_steps": 10,
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