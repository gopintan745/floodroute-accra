"""Decision-curve diagnostic for risk-assessment vs flood-avoidance behavior.

Evaluates and plots an agent's or planner's probability of choosing a wet shortcut
over a safe detour as a function of the break-even margin:
    margin = detour_cost - expected_shortcut_cost

A rational expected-cost replanner switches routes at margin = 0 (step function).
A risk-assessing RL policy should flip near zero, distinguishing risk assessment
from naive water avoidance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np

from src.baselines.dijkstra_dynamic import (
    build_traffic_lookup,
    dynamic_shortest_path_with_replanning,
)
from src.common.cost_model import CostModelConfig, expected_cost
from src.env.accra_routing_env import AccraRoutingEnv


def make_parametrized_toy_graph(detour_time: float, shortcut_base_time: float = 10.0) -> nx.MultiDiGraph:
    """Create a decision graph:
        start -> junction (decision point)
                   ├── shortcut -> goal (risky, fixed base time)
                   └── safe_a -> safe_b -> goal (safe, variable detour time)
    """
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"
    coords = {
        "start": (0.0, 0.0),
        "junction": (1.0, 0.0),
        "shortcut": (2.0, 0.0),
        "safe_a": (1.0, 1.0),
        "safe_b": (2.0, 1.0),
        "goal": (3.0, 0.0),
    }
    for node, (x, y) in coords.items():
        graph.add_node(node, x=x, y=y)

    defaults = {
        "length": 1.0,
        "road_quality_score": 1.0,
        "highway_class": "primary",
        "flood_susceptibility": 0.0,
        "flood_risk": 0.0,
    }
    graph.add_edge("start", "junction", key=0, travel_time=1.0, **defaults)

    # Shortcut: two edges, total base time = 2 * shortcut_base_time
    graph.add_edge(
        "junction", "shortcut", key=0, travel_time=shortcut_base_time,
        **{**defaults, "flood_susceptibility": 0.8},
    )
    graph.add_edge(
        "shortcut", "goal", key=0, travel_time=shortcut_base_time,
        **{**defaults, "flood_susceptibility": 0.8},
    )

    # Detour: split into two edges, total time = detour_time
    half_detour = max(0.1, float(detour_time) / 2.0)
    graph.add_edge("junction", "safe_a", key=0, travel_time=half_detour, **defaults)
    graph.add_edge("safe_a", "safe_b", key=0, travel_time=half_detour, **defaults)
    graph.add_edge("safe_b", "goal", key=0, travel_time=0.1, **defaults)
    return graph


def compute_expected_shortcut_cost(
    shortcut_base_time: float,
    belief: np.ndarray,
    cost_cfg: CostModelConfig,
) -> float:
    """Expected total time of junction -> shortcut -> goal."""
    single_edge_expected = expected_cost(shortcut_base_time, belief, cost_cfg)
    return 2.0 * single_edge_expected


def evaluate_decision_point(
    env: AccraRoutingEnv,
    action_selector: Callable[[AccraRoutingEnv, dict], int],
    shortcut_target: str = "shortcut",
) -> bool:
    """Run from start to junction and check if the chosen next node is the shortcut."""
    obs, _ = env.reset(options={"origin": "start", "destination": "goal"})
    # Step 1: start -> junction
    action = action_selector(env, obs)
    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        return False
    # At junction: make route choice
    action = action_selector(env, obs)
    edge = env.graph_wrapper.action_to_edge(env._current_node, action)
    return bool(edge and edge[1] == shortcut_target)


def replanner_action_selector(env: AccraRoutingEnv) -> int:
    """Select action using dynamic replanning baseline."""
    traffic_lookup = build_traffic_lookup(env.traffic_profile)
    origin_pt = (float(env.graph.nodes[env._current_node]["x"]), float(env.graph.nodes[env._current_node]["y"]))
    dest_pt = (float(env.graph.nodes[env._destination_node]["x"]), float(env.graph.nodes[env._destination_node]["y"]))
    plan = dynamic_shortest_path_with_replanning(
        graph=env.graph,
        hazard_sim=env.hazard_sim,
        traffic_lookup=traffic_lookup,
        origin=origin_pt,
        destination=dest_pt,
        hour=env.episode_context["hour"],
        is_weekend=env.episode_context["is_weekend"],
        cost_cfg=env.cost_cfg,
        reveal_radius_hops=env.hazard_sim.reveal_radius_hops,
        max_steps=10,
    )
    if not plan["path"] or len(plan["path"]) < 2:
        return 0
    next_node = plan["path"][1]
    return next(
        idx for idx in range(env.action_space.n)
        if (edge := env.graph_wrapper.action_to_edge(env._current_node, idx))
        and edge[1] == next_node
    )


def sweep_decision_margins(
    detour_margins: list[float] | np.ndarray,
    shortcut_base_time: float = 10.0,
    shortcut_level: int = 1,
    cost_cfg: CostModelConfig | None = None,
    num_trials_per_margin: int = 10,
    model=None,
    vec_env=None,
) -> dict:
    """Sweep detour margin = detour_time - expected_shortcut_cost.

    Returns dict mapping margin to choice probability of the wet shortcut.
    """
    cost_cfg = cost_cfg or CostModelConfig()
    # In exact signal mode, level 1 (shallow) belief is [0, 1, 0]
    belief = np.zeros(3, dtype=np.float64)
    belief[shortcut_level] = 1.0
    expected_shortcut = compute_expected_shortcut_cost(shortcut_base_time, belief, cost_cfg)

    margins_data = []
    for margin in detour_margins:
        detour_time = max(0.2, expected_shortcut + float(margin))
        actual_margin = detour_time - expected_shortcut
        graph = make_parametrized_toy_graph(detour_time, shortcut_base_time)
        config = {
            "env": {
                "max_degree": None,
                "max_episode_steps": 20,
                "step_penalty": 0.1,
                "reveal_radius_hops": 1,
                "train_flood_day_rate": 1.0,
                "mid_episode_event_base_rate": 0.0,
                "revisit_mode": "terminate",
            },
            "cost_model": {"quality_penalty_weight": 1.0},
            "flood_model": {
                "signal_mode": "exact",
                "level_costs": {
                    "shallow": {"multiplier": cost_cfg.shallow_multiplier, "penalty_s": cost_cfg.shallow_penalty_s},
                    "deep": {"multiplier": cost_cfg.deep_multiplier, "penalty_s": cost_cfg.deep_penalty_s},
                },
            },
        }

        # Replanner baseline choice
        env = AccraRoutingEnv(
            graph,
            "does-not-exist.parquet",
            "does-not-exist.json",
            config=config,
            mode="train",
        )
        env.reset(options={
            "origin": "start",
            "destination": "goal",
            "is_flood_day": True,
            "edge_levels": {("junction", "shortcut", 0): shortcut_level, ("shortcut", "goal", 0): shortcut_level},
        })
        replanner_chose_shortcut = evaluate_decision_point(
            env, lambda e, _obs: replanner_action_selector(e)
        )

        # Policy model choice (if provided)
        model_shortcut_prob = None
        if model is not None and vec_env is not None:
            chosen = 0
            for trial in range(num_trials_per_margin):
                env.reset(seed=trial, options={
                    "origin": "start",
                    "destination": "goal",
                    "is_flood_day": True,
                    "edge_levels": {("junction", "shortcut", 0): shortcut_level, ("shortcut", "goal", 0): shortcut_level},
                })
                def model_selector(current_env, current_obs):
                    norm_obs = vec_env.normalize_obs({
                        k: np.expand_dims(v, 0) for k, v in current_obs.items()
                    })
                    act, _ = model.predict(norm_obs, deterministic=True, action_masks=current_env.action_masks())
                    return int(act[0])
                if evaluate_decision_point(env, model_selector):
                    chosen += 1
            model_shortcut_prob = float(chosen / num_trials_per_margin)

        margins_data.append({
            "margin": float(actual_margin),
            "detour_time": float(detour_time),
            "expected_shortcut_time": float(expected_shortcut),
            "replanner_chose_shortcut": bool(replanner_chose_shortcut),
            "model_shortcut_prob": model_shortcut_prob,
        })

    return {
        "shortcut_base_time": shortcut_base_time,
        "shortcut_level": shortcut_level,
        "expected_shortcut_time": expected_shortcut,
        "data": margins_data,
    }


def save_decision_curve_results(results: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return output_path

