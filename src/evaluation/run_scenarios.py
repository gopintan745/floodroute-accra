"""Generate and persist reproducible evaluation scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np

from src.agents.training_utils import (
  CLIMATOLOGY_PATH,
  GRAPH_PATH,
  TRAFFIC_PROFILE_PATH,
  load_config,
  load_graph,
)
from src.env.accra_routing_env import AccraRoutingEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_OUTPUT_DIR = REPO_ROOT / "results" / "scenario_outputs"
SCENARIO_PATH = SCENARIO_OUTPUT_DIR / "scenarios.json"


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
  return save_scenarios(scenarios)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=None)
  args = parser.parse_args()
  print(main(args.config))
