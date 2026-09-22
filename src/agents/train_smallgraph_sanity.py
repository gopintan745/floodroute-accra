"""Train MaskablePPO on a deterministic small real-graph sanity subgraph."""

import argparse
import json
import logging
from pathlib import Path

import networkx as nx
from sb3_contrib import MaskablePPO

from src.agents.training_utils import (
    GRAPH_PATH,
    MODELS_DIR,
    RESULTS_LOGS_DIR,
    evaluate_masked_random,
    evaluate_masked_random_outcomes,
    evaluate_model,
    evaluate_model_outcomes,
    evaluate_static_shortest_path,
    evaluate_static_shortest_path_outcomes,
    load_config,
    load_graph,
    make_env_factory,
    make_normalized_vec_env,
    resolve_center_node,
    save_training_artifacts,
    summarize_outcomes,
    summarize_sanity_checks,
)

logger = logging.getLogger(__name__)


def main(config_path: str | Path | None = None):
    config = load_config(config_path)
    full_graph = load_graph(GRAPH_PATH)
    sanity_config = config.get("training", {}).get("sanity_check", {})
    center_node = resolve_center_node(full_graph, sanity_config.get("center_node"))
    subgraph = nx.ego_graph(
        full_graph,
        center_node,
        radius=int(sanity_config.get("subgraph_radius", 3)),
    ).copy()
    logger.info("Sanity graph: center=%s nodes=%d edges=%d", center_node, subgraph.number_of_nodes(), subgraph.number_of_edges())

    vec_env = make_normalized_vec_env(subgraph, config, n_envs=4, mode="train")
    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=str(RESULTS_LOGS_DIR),
    )
    model.learn(total_timesteps=int(sanity_config.get("total_timesteps", 50_000)))

    output_dir = MODELS_DIR / "sanity_smallgraph"
    _, vecnormalize_path = save_training_artifacts(model, vec_env, output_dir)
    vec_env.close()

    eval_config = dict(config)
    eval_factory = make_env_factory(subgraph, eval_config, mode="eval")
    random_scores = evaluate_masked_random(eval_factory, seed=int(config.get("evaluation", {}).get("seed", 42)))
    static_scores = evaluate_static_shortest_path(eval_factory, seed=int(config.get("evaluation", {}).get("seed", 42)))
    trained_scores = evaluate_model(model, subgraph, eval_config, vecnormalize_path, seed=int(config.get("evaluation", {}).get("seed", 42)))
    checks = summarize_sanity_checks(trained_scores, random_scores, static_scores)
    checks["trained_eval_outcomes"] = summarize_outcomes(
        evaluate_model_outcomes(model, subgraph, eval_config, vecnormalize_path, seed=int(config.get("evaluation", {}).get("seed", 42)))
    )
    checks["masked_random_eval_outcomes"] = summarize_outcomes(
        evaluate_masked_random_outcomes(eval_factory, seed=int(config.get("evaluation", {}).get("seed", 42)))
    )
    checks["static_shortest_path_eval_outcomes"] = summarize_outcomes(
        evaluate_static_shortest_path_outcomes(eval_factory, seed=int(config.get("evaluation", {}).get("seed", 42)))
    )
    for name, value in checks.items():
        logger.info("sanity_check.%s=%s", name, value)
    report_path = output_dir / "sanity_checks.json"
    report_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    (RESULTS_LOGS_DIR / "completion_rate_report.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    if not checks["trained_beats_masked_random"]:
        raise RuntimeError(
            "Sanity training did not beat the masked-random baseline; "
            "do not start full-graph PPO until the environment or reward is debugged."
        )
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    logging.basicConfig(level=logging.INFO)
    main(parser.parse_args().config)
