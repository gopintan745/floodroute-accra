"""Train MaskablePPO on the full study-area graph with VecNormalize."""

import argparse
import json
import logging
from pathlib import Path

from sb3_contrib import MaskablePPO

from src.agents.training_utils import (
    GRAPH_PATH,
    MODELS_DIR,
    RESULTS_LOGS_DIR,
    load_config,
    load_graph,
    make_normalized_vec_env,
    save_training_artifacts,
)

logger = logging.getLogger(__name__)


def main(config_path: str | Path | None = None):
  config = load_config(config_path)
  sanity_report_path = MODELS_DIR / "sanity_smallgraph" / "sanity_checks.json"
  if not sanity_report_path.exists():
    raise RuntimeError(
      "Missing small-graph sanity report; run train_smallgraph_sanity.py first."
    )
  sanity_report = json.loads(sanity_report_path.read_text(encoding="utf-8"))
  if not sanity_report.get("trained_beats_masked_random", False):
    raise RuntimeError(
      "Small-graph sanity checks did not pass; full-graph PPO is blocked."
    )
  graph = load_graph(GRAPH_PATH)
  ppo_config = config.get("training", {}).get("ppo", {})
  n_envs = int(ppo_config.get("n_envs", 4))
  vec_env = make_normalized_vec_env(graph, config, n_envs=n_envs, mode="train")
  model = MaskablePPO(
    "MultiInputPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=str(RESULTS_LOGS_DIR),
  )
  model.learn(total_timesteps=int(ppo_config.get("total_timesteps", 1_000_000)))
  paths = save_training_artifacts(model, vec_env, MODELS_DIR / "ppo_full")
  vec_env.close()
  logger.info("Saved full PPO model to %s and VecNormalize stats to %s", *paths)
  return paths


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=None)
  logging.basicConfig(level=logging.INFO)
  main(parser.parse_args().config)
