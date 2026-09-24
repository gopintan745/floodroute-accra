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
    graph: nx.MultiDiGraph,
    config: dict,
    mode: str = "train",
) -> Callable[[], AccraRoutingEnv]:
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
    graph: nx.MultiDiGraph,
    config: dict,
    n_envs: int,
    mode: str = "train",
) -> VecNormalize:
    vec_env = make_vec_env(make_env_factory(graph, config, mode), n_envs=n_envs)
    return VecNormalize(vec_env, norm_obs=True, norm_reward=True)


def save_training_artifacts(model, vec_env: VecNormalize, output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "model"
    vecnormalize_path = output_path / "vecnormalize.pkl"
    model.save(model_path)
    vec_env.save(vecnormalize_path)
    return model_path, vecnormalize_path


def _run_episode(env, action_selector, seed: int) -> dict:
    """Run one ordinary Gymnasium episode and collect comparable outcomes."""
    observation, reset_info = env.reset(seed=seed)
    total_reward = 0.0
    realized_time = 0.0
    flooded_edges = 0
    terminated = truncated = False
    final_info = {}
    while not (terminated or truncated):
        action = int(action_selector(env, observation))
        observation, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        realized_time += float(final_info.get("realized_travel_time", 0.0))
        flooded_edges += int(final_info.get("is_flooded", False))
    return {
        "reward": float(total_reward),
        "realized_travel_time": float(realized_time),
        "success": bool(final_info.get("success", False)),
        "dead_end": bool(final_info.get("dead_end", False)),
        "truncated": bool(final_info.get("truncated", False)),
        "flooded_edges": int(flooded_edges),
        "flood_day": bool(reset_info.get("is_flood_day", False)),
    }


def _scores_from_outcomes(outcomes: list[dict]) -> list[float]:
    return [float(outcome["reward"]) for outcome in outcomes]


def evaluate_masked_random(
    env_factory: Callable[[], AccraRoutingEnv],
    num_episodes: int = 20,
    seed: int = 42,
) -> list[float]:
    return _scores_from_outcomes(
        evaluate_masked_random_outcomes(env_factory, num_episodes, seed)
    )


def evaluate_masked_random_outcomes(
    env_factory: Callable[[], AccraRoutingEnv],
    num_episodes: int = 20,
    seed: int = 42,
) -> list[dict]:
    outcomes = []
    for episode in range(num_episodes):
        env = env_factory()
        rng = np.random.default_rng(seed + episode)
        outcomes.append(_run_episode(
            env,
            lambda current_env, _observation: int(
                rng.choice(np.flatnonzero(current_env.action_masks()))
            ),
            seed + episode,
        ))
    return outcomes


def evaluate_model(
    model,
    graph: nx.MultiDiGraph,
    config: dict,
    vecnormalize_path: str | Path,
    num_episodes: int = 20,
    seed: int = 42,
) -> list[float]:
    eval_env = make_vec_env(make_env_factory(graph, config, mode="eval"), n_envs=1)
    eval_env = VecNormalize.load(vecnormalize_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    scores = []
    for episode in range(num_episodes):
        eval_env.seed(seed + episode)
        observations = eval_env.reset()
        total_reward = 0.0
        for _ in range(int(config.get("env", {}).get("max_episode_steps", 200))):
            masks = get_action_masks(eval_env)
            action, _ = model.predict(observations, deterministic=True, action_masks=masks)
            observations, rewards, dones, _ = eval_env.step(action)
            total_reward += float(rewards[0])
            if dones[0]:
                break
        scores.append(total_reward)
    eval_env.close()
    return scores


def evaluate_model_outcomes(
    model,
    graph: nx.MultiDiGraph,
    config: dict,
    vecnormalize_path: str | Path,
    num_episodes: int = 20,
    seed: int = 42,
) -> list[dict]:
    eval_env = make_vec_env(make_env_factory(graph, config, mode="eval"), n_envs=1)
    eval_env = VecNormalize.load(vecnormalize_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    outcomes = []
    for episode in range(num_episodes):
        eval_env.seed(seed + episode)
        observations = eval_env.reset()
        total_reward = 0.0
        final_info = {}
        for _ in range(int(config.get("env", {}).get("max_episode_steps", 200))):
            action, _ = model.predict(
                observations,
                deterministic=True,
                action_masks=get_action_masks(eval_env),
            )
            observations, rewards, dones, infos = eval_env.step(action)
            total_reward += float(rewards[0])
            final_info = infos[0]
            if dones[0]:
                break
        outcomes.append({
            "reward": float(total_reward),
            "success": bool(final_info.get("success", False)),
            "dead_end": bool(final_info.get("dead_end", False)),
            "truncated": bool(final_info.get("truncated", False)),
            "flood_day": bool(final_info.get("is_flood_day", False)),
        })
    eval_env.close()
    return outcomes


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
) -> list[float]:
    return _scores_from_outcomes(
        evaluate_static_shortest_path_outcomes(env_factory, num_episodes, seed)
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
) -> list[dict]:
    outcomes = []
    for episode in range(num_episodes):
        env = env_factory()
        try:
            outcomes.append(_run_episode(env, _static_action_selector, seed + episode))
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
