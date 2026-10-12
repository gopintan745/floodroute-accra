"""Shared helpers for the Phase 5 MaskablePPO training entrypoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np
import yaml
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from src.env.accra_routing_env import AccraRoutingEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
GRAPH_PATH = REPO_ROOT / "data" / "processed" / "road_graph_full.graphml"
TRAFFIC_PROFILE_PATH = REPO_ROOT / "data" / "processed" / "traffic_profile.parquet"
CLIMATOLOGY_PATH = REPO_ROOT / "data" / "processed" / "rainfall_climatology.json"
RESULTS_LOGS_DIR = REPO_ROOT / "results" / "logs"
MODELS_DIR = REPO_ROOT / "models"

logger = logging.getLogger(__name__)


def _travel_time_weight(_source, _target, data) -> float:
    """Read GraphML travel-time attributes without relying on string coercion."""
    if isinstance(data, dict) and "travel_time" in data:
        return float(data["travel_time"])
    if isinstance(data, dict):
        return min(float(edge.get("travel_time", 0.0)) for edge in data.values())
    return 0.0


def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def load_graph(graph_path: str | Path = GRAPH_PATH) -> nx.MultiDiGraph:
    return nx.read_graphml(Path(graph_path))


def resolve_center_node(graph: nx.MultiDiGraph, configured_node=None):
    if configured_node is not None:
        if configured_node not in graph:
            raise ValueError(f"Configured sanity-check center node {configured_node!r} is not in the graph")
        return configured_node

    target_x, target_y = -0.23, 5.56
    return min(
        graph.nodes,
        key=lambda node: (
            (float(graph.nodes[node].get("x", 0.0)) - target_x) ** 2
            + (float(graph.nodes[node].get("y", 0.0)) - target_y) ** 2
        ),
    )


def make_env_factory(
    graph: nx.MultiDiGraph | str | Path,
    config: dict | str | Path,
    mode: str = "train",
) -> Callable[[], AccraRoutingEnv]:
    if isinstance(graph, (str, Path)):
        graph = load_graph(graph)
    if isinstance(config, (str, Path)):
        config = load_config(config)

    def make_env() -> AccraRoutingEnv:
        return AccraRoutingEnv(
            graph,
            TRAFFIC_PROFILE_PATH,
            CLIMATOLOGY_PATH,
            config,
            mode=mode,
        )

    return make_env


def make_normalized_vec_env(
    graph: nx.MultiDiGraph | str | Path,
    config: dict | str | Path,
    n_envs: int,
    mode: str = "train",
) -> VecNormalize:
    if isinstance(graph, (str, Path)):
        graph = load_graph(graph)
    if isinstance(config, (str, Path)):
        config = load_config(config)

    vec_env = make_vec_env(make_env_factory(graph, config, mode), n_envs=n_envs)
    clip_reward = float(
        config.get("training", {}).get("ppo", {}).get("clip_reward", 10.0)
    )
    if clip_reward <= 0:
        raise ValueError("training.ppo.clip_reward must be positive")
    return VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_reward=clip_reward,
    )


def save_training_artifacts(model, vec_env: VecNormalize, output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "model"
    vecnormalize_path = output_path / "vecnormalize.pkl"
    model.save(model_path)
    vec_env.save(vecnormalize_path)
    return model_path, vecnormalize_path


def _scenario_record(env, seed: int, reset_info: dict) -> dict:
    return {
        "seed": int(seed),
        "origin": reset_info["origin"],
        "destination": reset_info["destination"],
        "month": int(reset_info["month"]),
        "hour": int(reset_info["hour"]),
        "is_weekend": bool(reset_info["is_weekend"]),
        "is_flood_day": bool(reset_info["is_flood_day"]),
        "intensity": float(reset_info["intensity"]),
        "edge_levels": [
            {"edge": list(edge), "level": int(level)}
            for edge, level in sorted(env.hazard_sim.true_levels.items(), key=lambda item: repr(item[0]))
            if level > 0
        ],
    }


def make_evaluation_scenarios(
    env_factory: Callable[[], AccraRoutingEnv],
    num_episodes: int = 20,
    seed: int = 42,
    flood_only: bool = False,
    decision_relevant: bool = False,
    max_attempts: int = 200_000,
) -> list[dict]:
    """Create replayable natural, flood-stratified, or decision-relevant trips."""
    scenarios = []
    rng = np.random.default_rng(seed)
    attempts = 0
    while len(scenarios) < num_episodes and attempts < max_attempts:
        candidate_seed = int(rng.integers(0, 2**31))
        env = env_factory()
        _, info = env.reset(seed=candidate_seed)
        record = _scenario_record(env, candidate_seed, info)
        attempts += 1
        accepted = not flood_only or record["is_flood_day"]
        if accepted and decision_relevant:
            try:
                route = nx.shortest_path(
                    env.graph,
                    record["origin"],
                    record["destination"],
                    weight=_travel_time_weight,
                )
                wet_edges = {
                    tuple(item["edge"])
                    for item in record["edge_levels"]
                    if int(item["level"]) > 0
                }
                accepted = record["is_flood_day"] and any(
                    any((u, v, key) in wet_edges for key in env.graph.get_edge_data(u, v))
                    for u, v in zip(route[:-1], route[1:])
                )
            except nx.NetworkXNoPath:
                accepted = False
        if accepted:
            scenarios.append(record)
        env.close()
    if len(scenarios) < num_episodes:
        raise RuntimeError(
            f"Generated {len(scenarios)}/{num_episodes} requested scenarios after {attempts} attempts"
        )
    return scenarios


def _run_episode(env, action_selector, seed: int, scenario: dict | None = None) -> dict:
    """Run one ordinary Gymnasium episode and collect comparable outcomes."""
    observation, reset_info = env.reset(
        seed=seed,
        options={"scenario": scenario} if scenario is not None else None,
    )
    total_reward = 0.0
    realized_time = 0.0
    wet_edges = deep_edges = 0
    terminated = truncated = False
    final_info = {}
    while not (terminated or truncated):
        action = int(action_selector(env, observation))
        observation, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        realized_time += float(final_info.get("realized_travel_time", 0.0))
        wet_edges += int(final_info.get("level", 0) > 0)
        deep_edges += int(final_info.get("level", 0) == 2)
    return {
        "reward": float(total_reward),
        "realized_travel_time": float(realized_time),
        "success": bool(final_info.get("success", False)),
        "dead_end": bool(final_info.get("dead_end", False)),
        "truncated": bool(final_info.get("truncated", False)),
        "wet_edge_traversals": int(wet_edges),
        "deep_edge_traversals": int(deep_edges),
        "deep_traversal_episode": deep_edges > 0,
        "total_revisits": int(final_info.get("total_revisits", 0)),
        "flood_day": bool(reset_info.get("is_flood_day", False)),
    }


def _scores_from_outcomes(outcomes: list[dict]) -> list[float]:
    return [float(outcome["reward"]) for outcome in outcomes]


def evaluate_masked_random(
    env_factory: Callable[[], AccraRoutingEnv],
    num_episodes: int = 20,
    seed: int = 42,
    scenarios: list[dict] | None = None,
) -> list[float]:
    return _scores_from_outcomes(
        evaluate_masked_random_outcomes(env_factory, num_episodes, seed, scenarios)
    )


def evaluate_masked_random_outcomes(
    env_factory: Callable[[], AccraRoutingEnv],
    num_episodes: int = 20,
    seed: int = 42,
    scenarios: list[dict] | None = None,
) -> list[dict]:
    scenarios = scenarios or make_evaluation_scenarios(env_factory, num_episodes, seed)
    outcomes = []
    for episode, scenario in enumerate(scenarios):
        env = env_factory()
        rng = np.random.default_rng(seed + episode)
        outcomes.append(_run_episode(
            env,
            lambda current_env, _observation: int(
                rng.choice(np.flatnonzero(current_env.action_masks()))
            ),
            seed + episode,
            scenario,
        ))
    return outcomes


def evaluate_model(
    model,
    graph: nx.MultiDiGraph,
    config: dict,
    vecnormalize_path: str | Path,
    num_episodes: int = 20,
    seed: int = 42,
    scenarios: list[dict] | None = None,
) -> list[float]:
    outcomes = evaluate_model_outcomes(
        model, graph, config, vecnormalize_path, num_episodes, seed, scenarios
    )
    return [float(item["reward"]) for item in outcomes]


def _evaluate_model_scenarios(
    model, graph, config, vecnormalize_path, scenarios: list[dict]
) -> list[dict]:
    eval_env = make_vec_env(make_env_factory(graph, config, mode="eval"), n_envs=1)
    eval_env = VecNormalize.load(vecnormalize_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    base_env = eval_env.venv.envs[0]
    outcomes = []
    for scenario in scenarios:
        raw_observation, reset_info = base_env.reset(options={"scenario": scenario})
        observations = eval_env.normalize_obs({
            key: np.expand_dims(value, axis=0)
            for key, value in raw_observation.items()
        })
        total_reward = 0.0
        realized_time = 0.0
        wet_edges = deep_edges = 0
        for _ in range(int(config.get("env", {}).get("max_episode_steps", 200))):
            masks = get_action_masks(eval_env)
            action, _ = model.predict(observations, deterministic=True, action_masks=masks)
            raw_observation, reward, terminated, truncated, info = base_env.step(int(action[0]))
            observations = eval_env.normalize_obs({
                key: np.expand_dims(value, axis=0)
                for key, value in raw_observation.items()
            })
            total_reward += float(reward)
            realized_time += float(info.get("realized_travel_time", 0.0))
            wet_edges += int(info.get("level", 0) > 0)
            deep_edges += int(info.get("level", 0) == 2)
            final_info = info
            if terminated or truncated:
                break
        outcomes.append({
            "reward": total_reward,
            "realized_travel_time": realized_time,
            "success": bool(final_info.get("success", False)),
            "dead_end": bool(final_info.get("dead_end", False)),
            "truncated": bool(final_info.get("truncated", False)),
            "flood_day": bool(reset_info.get("is_flood_day", False)),
            "wet_edge_traversals": wet_edges,
            "deep_edge_traversals": deep_edges,
            "deep_traversal_episode": deep_edges > 0,
            "total_revisits": int(final_info.get("total_revisits", 0)),
        })
    eval_env.close()
    return outcomes


def evaluate_model_outcomes(
    model,
    graph: nx.MultiDiGraph,
    config: dict,
    vecnormalize_path: str | Path,
    num_episodes: int = 20,
    seed: int = 42,
    scenarios: list[dict] | None = None,
) -> list[dict]:
    if scenarios is None:
        factory = make_env_factory(graph, config, mode="eval")
        scenarios = make_evaluation_scenarios(factory, num_episodes, seed)
    return _evaluate_model_scenarios(model, graph, config, vecnormalize_path, scenarios)


def summarize_outcomes(outcomes: list[dict]) -> dict:
    if not outcomes:
        return {"episodes": 0, "completion_rate": 0.0}
    return {
        "episodes": len(outcomes),
        "completion_rate": float(np.mean([item["success"] for item in outcomes])),
        "dead_end_rate": float(np.mean([item["dead_end"] for item in outcomes])),
        "truncation_rate": float(np.mean([item["truncated"] for item in outcomes])),
        "flood_day_rate": float(np.mean([item["flood_day"] for item in outcomes])),
        "mean_reward": float(np.mean([item["reward"] for item in outcomes])),
    }


def evaluate_static_shortest_path(
    env_factory: Callable[[], AccraRoutingEnv],
    num_episodes: int = 20,
    seed: int = 42,
    scenarios: list[dict] | None = None,
) -> list[float]:
    return _scores_from_outcomes(
        evaluate_static_shortest_path_outcomes(env_factory, num_episodes, seed, scenarios)
    )


def _static_action_selector(env, _observation):
    path = nx.shortest_path(
        env.graph,
        env._current_node,
        env._destination_node,
        weight=_travel_time_weight,
    )
    target = path[1]
    return next(
        index for index in range(env.action_space.n)
        if env.graph_wrapper.action_to_edge(env._current_node, index)
        and env.graph_wrapper.action_to_edge(env._current_node, index)[1] == target
    )


def evaluate_static_shortest_path_outcomes(
    env_factory: Callable[[], AccraRoutingEnv],
    num_episodes: int = 20,
    seed: int = 42,
    scenarios: list[dict] | None = None,
) -> list[dict]:
    scenarios = scenarios or make_evaluation_scenarios(env_factory, num_episodes, seed)
    outcomes = []
    for episode, scenario in enumerate(scenarios):
        env = env_factory()
        try:
            outcomes.append(_run_episode(env, _static_action_selector, seed + episode, scenario))
        except nx.NetworkXNoPath:
            outcomes.append({
                "reward": 0.0,
                "realized_travel_time": float("inf"),
                "success": False,
                "dead_end": False,
                "truncated": False,
                "flooded_edges": 0,
                "flood_day": False,
            })
    return outcomes


def summarize_sanity_checks(
    trained_scores: list[float],
    random_scores: list[float],
    static_scores: list[float],
) -> dict:
    result = {
        "trained_mean_reward": float(np.mean(trained_scores)),
        "masked_random_mean_reward": float(np.mean(random_scores)),
        "static_shortest_path_mean_reward": float(np.mean(static_scores)),
    }
    result["trained_beats_masked_random"] = (
        result["trained_mean_reward"] > result["masked_random_mean_reward"]
    )
    result["trained_beats_static_shortest_path"] = (
        result["trained_mean_reward"] > result["static_shortest_path_mean_reward"]
    )
    return result
