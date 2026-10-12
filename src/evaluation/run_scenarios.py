"""Generate and persist reproducible evaluation scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.agents.training_utils import (
  CLIMATOLOGY_PATH,
  GRAPH_PATH,
  TRAFFIC_PROFILE_PATH,
  load_config,
  load_graph,
)
from src.evaluation.compare_results import compare_results, write_comparison, write_summary
from src.env.accra_routing_env import AccraRoutingEnv
from src.baselines.dijkstra_dynamic import (
  build_traffic_lookup,
  dynamic_shortest_path_with_replanning,
  oracle_shortest_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_OUTPUT_DIR = REPO_ROOT / "results" / "scenario_outputs"
SCENARIO_PATH = SCENARIO_OUTPUT_DIR / "scenarios.json"
RESULTS_PATH = SCENARIO_OUTPUT_DIR / "scenario_results.json"


def _try_build_scenario(graph, env_ctor_args, config, seed, mode="eval") -> dict | None:
  """Dry-run one candidate seed and reject unreachable sampled pairs."""
  env = AccraRoutingEnv(graph, *env_ctor_args, config, mode=mode)
  _, info = env.reset(seed=seed)
  if not nx.has_path(graph, info["origin"], info["destination"]):
    return None
  edge_levels = [
    {"edge": list(edge), "level": int(level)}
    for edge, level in sorted(env.hazard_sim.true_levels.items(), key=lambda item: repr(item[0]))
    if level > 0
  ]
  return {
    "seed": int(seed),
    "origin": info["origin"],
    "destination": info["destination"],
    "month": int(info["month"]),
    "hour": int(info["hour"]),
    "is_weekend": bool(info["is_weekend"]),
    "is_flood_day": bool(info["is_flood_day"]),
    "intensity": float(info["intensity"]),
    "edge_levels": edge_levels,
  }


def generate_scenarios(
  graph,
  env_ctor_args,
  config,
  num_scenarios,
  num_flood_scenarios,
  seed,
  num_decision_relevant_scenarios=None,
  max_attempts=200_000,
) -> dict:
  """Generate natural, flood, and decision-relevant replayable scenarios."""
  rng = np.random.default_rng(seed)

  def is_decision_relevant(record):
    if not record["is_flood_day"]:
      return False
    try:
      route = nx.shortest_path(
        graph, record["origin"], record["destination"],
        weight=lambda _u, _v, edge_data: min(
          float(edge.get("travel_time", 0.0)) for edge in edge_data.values()
        ),
      )
    except nx.NetworkXNoPath:
      return False
    wet_edges = {tuple(item["edge"]) for item in record["edge_levels"]}
    return any(
      any((u, v, key) in wet_edges for key in graph.get_edge_data(u, v, default={}))
      for u, v in zip(route[:-1], route[1:])
    )

  def draw_until(target_count, predicate=lambda _record: True):
    results, attempts = [], 0
    while len(results) < target_count and attempts < max_attempts:
      candidate = int(rng.integers(0, 2**31))
      record = _try_build_scenario(graph, env_ctor_args, config, candidate)
      attempts += 1
      if record and predicate(record):
        results.append(record)
    if len(results) < target_count:
      raise RuntimeError(
        f"Only found {len(results)}/{target_count} valid scenarios in {attempts} attempts "
        "- check graph reachability or climatology flood rate."
      )
    return results

  return {
    "natural": draw_until(int(num_scenarios)),
    "flood_stratified": draw_until(
      int(num_flood_scenarios), lambda record: record["is_flood_day"]
    ),
    "decision_relevant": draw_until(
      int(num_decision_relevant_scenarios if num_decision_relevant_scenarios is not None else num_flood_scenarios),
      is_decision_relevant,
    ),
  }


def save_scenarios(scenarios: dict, output_path: str | Path = SCENARIO_PATH) -> Path:
  """Write scenario definitions as readable JSON."""
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(scenarios, indent=2), encoding="utf-8")
  return output_path


def _make_env(graph, env_ctor_args, config):
  return AccraRoutingEnv(graph, *env_ctor_args, config, mode="eval")


def _run_env_policy(env, scenario, action_selector) -> dict:
  observation, reset_info = env.reset(options={"scenario": scenario})
  total_reward = 0.0
  realized_time = 0.0
  wet_edges = 0
  deep_edges = 0
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
    "scenario_seed": int(scenario["seed"]),
    "reward": float(total_reward),
    "realized_travel_time": float(realized_time),
    "success": bool(final_info.get("success", False)),
    "dead_end": bool(final_info.get("dead_end", False)),
    "truncated": bool(final_info.get("truncated", False)),
    "wet_edge_traversals": wet_edges,
    "deep_edge_traversals": deep_edges,
    "deep_traversal_episode": deep_edges > 0,
    "total_revisits": int(final_info.get("total_revisits", 0)),
    "flood_day": bool(reset_info["is_flood_day"]),
  }


def _static_action_selector(env, observation):
  path = nx.shortest_path(
    env.graph,
    env._current_node,
    env._destination_node,
    weight=lambda _u, _v, data: min(
      float(edge.get("travel_time", 0.0)) for edge in data.values()
    ),
  )
  target = path[1]
  return next(
    index for index in range(env.action_space.n)
    if env.graph_wrapper.action_to_edge(env._current_node, index)
    and env.graph_wrapper.action_to_edge(env._current_node, index)[1] == target
  )


def _evaluate_dynamic(env, scenario) -> dict:
  env.reset(options={"scenario": scenario})
  env.graph.graph.setdefault("crs", "EPSG:4326")
  result = dynamic_shortest_path_with_replanning(
    graph=env.graph,
    hazard_sim=env.hazard_sim,
    traffic_lookup=build_traffic_lookup(env.traffic_profile),
    origin=(float(env.graph.nodes[scenario["origin"]]["x"]), float(env.graph.nodes[scenario["origin"]]["y"])),
    destination=(float(env.graph.nodes[scenario["destination"]]["x"]), float(env.graph.nodes[scenario["destination"]]["y"])),
    hour=int(scenario["hour"]),
    is_weekend=bool(scenario["is_weekend"]),
    cost_cfg=env.cost_cfg,
    reveal_radius_hops=env.hazard_sim.reveal_radius_hops,
    max_steps=int(env.config.get("env", {}).get("max_episode_steps", 200)),
  )
  return {
    "scenario_seed": int(scenario["seed"]),
    "reward": float(-result["total_realized_time"] - result.get("revisit_penalty", 0.0)),
    "realized_travel_time": float(result["total_realized_time"]),
    "success": bool(result["reached_destination"]),
    "dead_end": False,
    "truncated": not bool(result["reached_destination"]) and not bool(result.get("revisit_failure")),
    "revisit_failure": bool(result.get("revisit_failure", False)),
    "wet_edge_traversals": int(result.get("wet_edge_traversals", 0)),
    "deep_edge_traversals": int(result.get("deep_edge_traversals", 0)),
    "deep_traversal_episode": bool(result.get("deep_edge_traversals", 0)),
    "total_revisits": int(result.get("total_revisits", 0)),
    "flood_day": bool(scenario["is_flood_day"]),
  }


def _evaluate_oracle(env, scenario) -> dict:
  env.reset(options={"scenario": scenario})
  result = oracle_shortest_path(
    graph=env.graph,
    hazard_sim=env.hazard_sim,
    traffic_lookup=build_traffic_lookup(env.traffic_profile),
    origin=(float(env.graph.nodes[scenario["origin"]]["x"]), float(env.graph.nodes[scenario["origin"]]["y"])),
    destination=(float(env.graph.nodes[scenario["destination"]]["x"]), float(env.graph.nodes[scenario["destination"]]["y"])),
    hour=int(scenario["hour"]),
    is_weekend=bool(scenario["is_weekend"]),
    cost_cfg=env.cost_cfg,
  )
  levels = [env.hazard_sim.true_level(*edge) for edge in result.get("edges_traversed", [])]
  return {
    "scenario_seed": int(scenario["seed"]),
    "reward": float(-result["total_realized_time"]),
    "realized_travel_time": float(result["total_realized_time"]),
    "success": bool(result["reached_destination"]),
    "dead_end": False,
    "truncated": False,
    "wet_edge_traversals": sum(level > 0 for level in levels),
    "deep_edge_traversals": sum(level == 2 for level in levels),
    "deep_traversal_episode": 2 in levels,
    "total_revisits": 0,
    "flood_day": bool(scenario["is_flood_day"]),
  }


def _evaluate_rl_scenario(model, vec_env, base_env, scenario) -> dict:
  """Evaluate a fixed scenario with training-time observation normalization."""
  raw_observation, reset_info = base_env.reset(options={"scenario": scenario})
  observation = vec_env.normalize_obs({
    key: np.expand_dims(value, axis=0)
    for key, value in raw_observation.items()
  })
  total_reward = 0.0
  realized_time = 0.0
  wet_edges = deep_edges = 0
  terminated = truncated = False
  final_info = {}
  while not (terminated or truncated):
    action, _ = model.predict(
      observation,
      deterministic=True,
      action_masks=get_action_masks(vec_env),
    )
    raw_observation, reward, terminated, truncated, final_info = base_env.step(int(action[0]))
    observation = vec_env.normalize_obs({
      key: np.expand_dims(value, axis=0)
      for key, value in raw_observation.items()
    })
    total_reward += float(reward)
    realized_time += float(final_info.get("realized_travel_time", 0.0))
    wet_edges += int(final_info.get("level", 0) > 0)
    deep_edges += int(final_info.get("level", 0) == 2)
  return {
    "scenario_seed": int(scenario["seed"]),
    "reward": float(total_reward),
    "realized_travel_time": float(realized_time),
    "success": bool(final_info.get("success", False)),
    "dead_end": bool(final_info.get("dead_end", False)),
    "truncated": bool(final_info.get("truncated", False)),
    "wet_edge_traversals": wet_edges,
    "deep_edge_traversals": int(deep_edges),
    "deep_traversal_episode": deep_edges > 0,
    "total_revisits": int(final_info.get("total_revisits", 0)),
    "flood_day": bool(reset_info["is_flood_day"]),
  }
def evaluate_scenarios(
  graph,
  env_ctor_args,
  config,
  scenarios: dict,
  model=None,
  vecnormalize_path: str | Path | None = None,
  seed: int = 42,
) -> dict:
  """Evaluate fixed scenarios with masked random, static, and optional RL policies."""
  records = [
    *scenarios.get("natural", []),
    *scenarios.get("flood_stratified", []),
    *scenarios.get("decision_relevant", []),
  ]
  methods = {
    "masked_random": [],
    "static_shortest_path": [],
    "dynamic_replanning": [],
    "oracle": [],
  }
  if model is not None:
    methods["rl_agent"] = []
    if vecnormalize_path is None:
      raise ValueError("vecnormalize_path is required when evaluating an RL model")
    rl_vec_env = DummyVecEnv([lambda: _make_env(graph, env_ctor_args, config)])
    rl_vec_env = VecNormalize.load(vecnormalize_path, rl_vec_env)
    rl_vec_env.training = False
    rl_vec_env.norm_reward = False
    rl_base_env = rl_vec_env.venv.envs[0]
  for index, scenario in enumerate(records):
    rng = np.random.default_rng(seed + index)
    oracle_env = _make_env(graph, env_ctor_args, config)
    oracle_outcome = _evaluate_oracle(oracle_env, scenario)
    oracle_env.close()
    oracle_time = oracle_outcome["realized_travel_time"]
    methods["oracle"].append(oracle_outcome)

    env = _make_env(graph, env_ctor_args, config)
    methods["masked_random"].append(
      _run_env_policy(
        env,
        scenario,
        lambda current_env, _obs: int(rng.choice(np.flatnonzero(current_env.action_masks()))),
      )
    )
    env.close()

    env = _make_env(graph, env_ctor_args, config)
    methods["static_shortest_path"].append(_run_env_policy(env, scenario, _static_action_selector))
    env.close()

    env = _make_env(graph, env_ctor_args, config)
    methods["dynamic_replanning"].append(_evaluate_dynamic(env, scenario))
    env.close()

    if model is not None:
      methods["rl_agent"].append(
        _evaluate_rl_scenario(model, rl_vec_env, rl_base_env, scenario)
      )
    for outcomes in methods.values():
      outcome = outcomes[-1]
      outcome["oracle_realized_time"] = float(oracle_time)
      outcome["regret_vs_oracle"] = max(
        0.0, float(outcome.get("realized_travel_time", 0.0)) - float(oracle_time)
      )
  if model is not None:
    rl_vec_env.close()
  return {"scenarios": scenarios, "methods": methods}


def save_results(results: dict, output_path: str | Path = RESULTS_PATH) -> Path:
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
  return output_path


def main(
  config_path: str | Path | None = None,
  model_path: str | Path | None = None,
  vecnormalize_path: str | Path | None = None,
) -> Path:
  config = load_config(config_path)
  evaluation = config.get("evaluation", {})
  graph = load_graph(GRAPH_PATH)
  env_ctor_args = (str(TRAFFIC_PROFILE_PATH), str(CLIMATOLOGY_PATH))
  scenarios = generate_scenarios(
    graph,
    env_ctor_args,
    config,
    num_scenarios=int(evaluation.get("num_scenarios", 100)),
    num_flood_scenarios=int(evaluation.get("num_flood_scenarios", 30)),
    seed=int(evaluation.get("seed", 42)),
    num_decision_relevant_scenarios=int(
      evaluation.get("num_decision_relevant_scenarios", evaluation.get("num_flood_scenarios", 30))
    ),
  )
  save_scenarios(scenarios)
  model = MaskablePPO.load(model_path) if model_path else None
  results = evaluate_scenarios(
    graph,
    env_ctor_args,
    config,
    scenarios,
    model=model,
    vecnormalize_path=vecnormalize_path,
  )
  results_path = save_results(results)
  comparison = compare_results(results)
  write_comparison(comparison)
  write_summary(comparison)
  return results_path


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=None)
  parser.add_argument("--model", default=None)
  parser.add_argument("--vecnormalize", default=None)
  args = parser.parse_args()
  print(main(args.config, args.model, args.vecnormalize))
