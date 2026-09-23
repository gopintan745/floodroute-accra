"""Generate and persist reproducible evaluation scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks

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
  return {
    "seed": int(seed),
    "origin": info["origin"],
    "destination": info["destination"],
    "month": int(info["month"]),
    "hour": int(info["hour"]),
    "is_weekend": bool(info["is_weekend"]),
    "is_flood_day": bool(info["is_flood_day"]),
    "flooded_edges": [list(edge) for edge in sorted(env.hazard_sim.flooded_edges, key=str)],
    "flood_events": [
      {"step": int(step), "edges": [list(edge) for edge in edges]}
      for step, edges in sorted(env.hazard_sim.event_schedule.items())
      if edges
    ],
  }


def generate_scenarios(
  graph,
  env_ctor_args,
  config,
  num_scenarios,
  num_flood_scenarios,
  seed,
  max_attempts=200_000,
) -> dict:
  """Generate natural and flood-stratified, replayable scenario records."""
  rng = np.random.default_rng(seed)

  def draw_until(target_count, flood_only):
    results, attempts = [], 0
    while len(results) < target_count and attempts < max_attempts:
      candidate = int(rng.integers(0, 2**31))
      record = _try_build_scenario(graph, env_ctor_args, config, candidate)
      attempts += 1
      if record and (not flood_only or record["is_flood_day"]):
        results.append(record)
    if len(results) < target_count:
      raise RuntimeError(
        f"Only found {len(results)}/{target_count} valid scenarios in {attempts} attempts "
        "- check graph reachability or climatology flood rate."
      )
    return results

  return {
    "natural": draw_until(int(num_scenarios), flood_only=False),
    "flood_stratified": draw_until(int(num_flood_scenarios), flood_only=True),
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
    "scenario_seed": int(scenario["seed"]),
    "reward": float(total_reward),
    "realized_travel_time": float(realized_time),
    "success": bool(final_info.get("success", False)),
    "dead_end": bool(final_info.get("dead_end", False)),
    "truncated": bool(final_info.get("truncated", False)),
    "flooded_edges": int(flooded_edges),
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
    "reward": float(-result["total_realized_time"]),
    "realized_travel_time": float(result["total_realized_time"]),
    "success": bool(result["reached_destination"]),
    "dead_end": False,
    "truncated": not bool(result["reached_destination"]),
    "flooded_edges": None,
    "flood_day": bool(scenario["is_flood_day"]),
  }
def evaluate_scenarios(
  graph,
  env_ctor_args,
  config,
  scenarios: dict,
  model=None,
  seed: int = 42,
) -> dict:
  """Evaluate fixed scenarios with masked random, static, and optional RL policies."""
  records = [*scenarios.get("natural", []), *scenarios.get("flood_stratified", [])]
  methods = {
    "masked_random": [],
    "static_shortest_path": [],
    "dynamic_replanning": [],
  }
  if model is not None:
    methods["rl_agent"] = []
  for index, scenario in enumerate(records):
    rng = np.random.default_rng(seed + index)
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
      env = _make_env(graph, env_ctor_args, config)
      methods["rl_agent"].append(
        _run_env_policy(
          env,
          scenario,
          lambda current_env, obs: model.predict(
            obs, deterministic=True, action_masks=get_action_masks(current_env)
          )[0],
        )
      )
      env.close()
  return {"scenarios": scenarios, "methods": methods}


def save_results(results: dict, output_path: str | Path = RESULTS_PATH) -> Path:
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
  return output_path


def main(config_path: str | Path | None = None) -> Path:
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
  )
  save_scenarios(scenarios)
  results = evaluate_scenarios(graph, env_ctor_args, config, scenarios)
  results_path = save_results(results)
  comparison = compare_results(results)
  write_comparison(comparison)
  write_summary(comparison)
  return results_path


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=None)
  args = parser.parse_args()
  print(main(args.config))
