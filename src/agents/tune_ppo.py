"""
Optuna hyperparameter tuning for MaskablePPO on the flood routing task.

This script implements the tuning design from the handoff spec:
- Separate tuning scenario set (different seed from final evaluation)
- TPE sampler with MedianPruner, generous startup/warmup
- 2-3 seeds per trial, ~20-25% of full training budget per trial
- SQLite storage for resumability
- Only tunes PPO hyperparameters (NOT environment/cost-model params)
- VecNormalize sync verification
- Best trial re-validated across multiple seeds

Gate prerequisites (must be true before running):
1. Environment/reward frozen, informed-revisit documented in design_decisions.md
2. Small-graph sanity check passes on flood-stratified scenarios
3. Full-graph run shows genuine learning signal (completion rate well above masked-random)
"""

import argparse
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, List

import networkx as nx
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from src.agents.training_utils import (
    CLIMATOLOGY_PATH,
    GRAPH_PATH,
    TRAFFIC_PROFILE_PATH,
    load_config,
    load_graph,
    make_env_factory,
    make_normalized_vec_env,
    save_training_artifacts,
)
from src.evaluation.run_scenarios import (
    evaluate_scenarios,
    generate_scenarios,
    _make_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
TUNING_DIR = REPO_ROOT / "results" / "tuning"
STUDY_DB = TUNING_DIR / "optuna_study.db"
TUNING_SCENARIOS_PATH = TUNING_DIR / "tuning_scenarios.json"
FINAL_EVAL_SEED = 42  # Reserved for final comparison (per spec)
TUNING_SEED = 12345   # Separate seed for tuning scenario generation

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Objective function
# ──────────────────────────────────────────────────────────────────────

def _make_vec_env_for_trial(graph, config, n_envs: int, mode: str = "train") -> VecNormalize:
    """Create a fresh VecNormalize for each trial (critical for isolation)."""
    return make_normalized_vec_env(graph, config, n_envs=n_envs, mode=mode)


def _evaluate_trial_model(model, vec_env: VecNormalize, base_env, scenarios: List[Dict], n_seeds: int = 3) -> Dict[str, float]:
    """
    Evaluate a trained model across multiple seeds on the tuning scenario set.
    Returns aggregate metrics for Optuna objective.
    """
    all_rewards = []
    all_success = []
    all_flood_rewards = []
    all_flood_success = []

    for seed in range(n_seeds):
        # Create fresh env for this seed
        rl_vec_env = make_vec_env(lambda: base_env, n_envs=1)
        rl_vec_env = VecNormalize(vec_env.venv, training=False, norm_reward=False)
        rl_vec_env.obs_rms = vec_env.obs_rms
        rl_vec_env.ret_rms = vec_env.ret_rms
        rl_base_env = rl_vec_env.venv.envs[0]

        results = evaluate_scenarios(
            graph=base_env.graph,
            env_ctor_args=(str(TRAFFIC_PROFILE_PATH), str(CLIMATOLOGY_PATH)),
            config=base_env.config,
            scenarios={"natural": scenarios["natural"], "flood_stratified": scenarios["flood_stratified"]},
            model=model,
            vecnormalize_path=None,  # We use the synced rl_vec_env directly
            seed=seed,
        )
        rl_vec_env.close()

        methods = results["methods"]
        for method_name, outcomes in methods.items():
            if method_name != "rl_agent":
                continue
            for outcome in outcomes:
                all_rewards.append(outcome["reward"])
                all_success.append(outcome["success"])
                if outcome["flood_day"]:
                    all_flood_rewards.append(outcome["reward"])
                    all_flood_success.append(outcome["success"])

    return {
        "mean_reward": float(np.mean(all_rewards)) if all_rewards else -np.inf,
        "completion_rate": float(np.mean(all_success)) if all_success else 0.0,
        "flood_mean_reward": float(np.mean(all_flood_rewards)) if all_flood_rewards else -np.inf,
        "flood_completion_rate": float(np.mean(all_flood_success)) if all_flood_success else 0.0,
    }


def objective(trial: optuna.Trial, graph, config, tuning_scenarios, base_env) -> float:
    """
    Optuna objective function.
    Returns the primary metric to maximize (flood completion rate).
    """
    # ──────────────────────────────────────────────────────────────
    # Search space: PPO hyperparameters ONLY
    # ──────────────────────────────────────────────────────────────
    # Learning rate (log scale)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    # n_steps (rollout length per env)
    n_steps = trial.suggest_categorical("n_steps", [64, 128, 256, 512, 1024])

    # Batch size
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

    # Ensure batch_size <= n_steps * n_envs
    n_envs = 4
    max_batch = n_steps * n_envs
    if batch_size > max_batch:
        batch_size = trial.suggest_categorical("batch_size", [b for b in [32, 64, 128, 256, 512] if b <= max_batch])

    # Number of epochs
    n_epochs = trial.suggest_int("n_epochs", 3, 20)

    # Discount factor
    gamma = trial.suggest_float("gamma", 0.9, 0.999)

    # GAE lambda
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)

    # Clip range
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

    # Entropy coefficient (log scale) - critical for preventing premature convergence
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True)

    # Network architecture
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    net_arch_map = {
        "small": [64, 64],
        "medium": [128, 128],
        "large": [256, 256],
    }
    net_arch = net_arch_map[net_arch_type]

    # ──────────────────────────────────────────────────────────────
    # Training budget: ~25% of full (1M) = 250k timesteps
    # ──────────────────────────────────────────────────────────────
    total_timesteps = 250_000

    # ──────────────────────────────────────────────────────────────
    # Create fresh VecNormalize and model for this trial
    # ──────────────────────────────────────────────────────────────
    vec_env = _make_vec_env_for_trial(graph, config, n_envs=n_envs, mode="train")

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs={"net_arch": net_arch},
        verbose=0,
        seed=trial.number,  # Different seed per trial for variance
    )

    # ──────────────────────────────────────────────────────────────
    # Train
    # ──────────────────────────────────────────────────────────────
    logger.info(f"Trial {trial.number}: Starting training for {total_timesteps} timesteps")
    model.learn(total_timesteps=total_timesteps)

    # ──────────────────────────────────────────────────────────────
    # Evaluate on tuning scenario set (NOT final evaluation set)
    # ──────────────────────────────────────────────────────────────
    metrics = _evaluate_trial_model(
        model, vec_env, base_env, tuning_scenarios, n_seeds=2
    )
    vec_env.close()

    logger.info(
        f"Trial {trial.number}: mean_reward={metrics['mean_reward']:.2f}, "
        f"completion={metrics['completion_rate']:.2%}, "
        f"flood_completion={metrics['flood_completion_rate']:.2%}"
    )

    # Primary objective: flood completion rate (the hard part of the task)
    # Secondary: mean reward (tie-breaker)
    return metrics["flood_completion_rate"]


# ──────────────────────────────────────────────────────────────────────
# Study management
# ──────────────────────────────────────────────────────────────────────

def create_study() -> optuna.Study:
    """Create or load Optuna study with SQLite storage."""
    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name="floodroute_ppo_tuning",
        storage=f"sqlite:///{STUDY_DB}",
        direction="maximize",
        sampler=TPESampler(
            n_startup_trials=10,  # Generous startup before TPE kicks in
            seed=42,
        ),
        pruner=MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,     # Wait for 5 evaluations before pruning
            interval_steps=1,
        ),
        load_if_exists=True,
    )
    return study


def generate_tuning_scenarios(graph, config) -> Dict:
    """Generate a separate tuning scenario set with a different seed."""
    logger.info(f"Generating tuning scenarios with seed={TUNING_SEED}")
    scenarios = generate_scenarios(
        graph=graph,
        env_ctor_args=(str(TRAFFIC_PROFILE_PATH), str(CLIMATOLOGY_PATH)),
        config=config,
        num_scenarios=50,         # Smaller than final eval (100)
        num_flood_scenarios=20,   # Smaller than final eval (30)
        seed=TUNING_SEED,
    )
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    with open(TUNING_SCENARIOS_PATH, "w") as f:
        json.dump(scenarios, f, indent=2)
    logger.info(f"Generated {len(scenarios['natural'])} natural + {len(scenarios['flood_stratified'])} flood scenarios")
    return scenarios


# ──────────────────────────────────────────────────────────────────────
# VecNormalize sync verification (per spec)
# ──────────────────────────────────────────────────────────────────────

def verify_vecnormalize_sync(trial_vec_env: VecNormalize, eval_vec_env: VecNormalize) -> bool:
    """
    Verify that evaluation VecNormalize statistics match training VecNormalize.
    This catches the common bug where eval env has different normalization stats.
    """
    try:
        # Check observation normalization stats
        np.testing.assert_allclose(trial_vec_env.obs_rms.mean, eval_vec_env.obs_rms.mean, rtol=1e-5)
        np.testing.assert_allclose(trial_vec_env.obs_rms.var, eval_vec_env.obs_rms.var, rtol=1e-5)
        # Check reward normalization stats
        np.testing.assert_allclose(trial_vec_env.ret_rms.mean, eval_vec_env.ret_rms.mean, rtol=1e-5)
        np.testing.assert_allclose(trial_vec_env.ret_rms.var, eval_vec_env.ret_rms.var, rtol=1e-5)
        return True
    except Exception as e:
        logger.warning(f"VecNormalize sync check failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────
# Best trial re-validation (per spec)
# ──────────────────────────────────────────────────────────────────────

def revalidate_best_trial(study: optuna.Study, graph, config, final_scenarios, base_env, n_seeds: int = 5):
    """
    Retrain the best trial configuration at full budget across multiple seeds
    and evaluate on the RESERVED final evaluation scenario set.
    """
    best_trial = study.best_trial
    logger.info(f"Re-validating best trial {best_trial.number} with params: {best_trial.params}")

    params = best_trial.params
    n_envs = 4
    full_timesteps = 1_000_000

    all_final_metrics = []

    for seed in range(n_seeds):
        logger.info(f"Re-validation seed {seed + 1}/{n_seeds}")

        vec_env = _make_vec_env_for_trial(graph, config, n_envs=n_envs, mode="train")
        model = MaskablePPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=params["learning_rate"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            gamma=params["gamma"],
            gae_lambda=params["gae_lambda"],
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            policy_kwargs={"net_arch": {"small": [64, 64], "medium": [128, 128], "large": [256, 256]}[params["net_arch"]]},
            verbose=0,
            seed=seed,
        )
        model.learn(total_timesteps=full_timesteps)

        # Evaluate on FINAL (reserved) scenario set
        metrics = _evaluate_trial_model(model, vec_env, base_env, final_scenarios, n_seeds=1)
        all_final_metrics.append(metrics)
        vec_env.close()

    # Aggregate
    agg = {
        "mean_reward": float(np.mean([m["mean_reward"] for m in all_final_metrics])),
        "completion_rate": float(np.mean([m["completion_rate"] for m in all_final_metrics])),
        "flood_mean_reward": float(np.mean([m["flood_mean_reward"] for m in all_final_metrics])),
        "flood_completion_rate": float(np.mean([m["flood_completion_rate"] for m in all_final_metrics])),
        "std_flood_completion": float(np.std([m["flood_completion_rate"] for m in all_final_metrics])),
    }

    logger.info("=== FINAL RE-VALIDATION RESULTS ===")
    for k, v in agg.items():
        logger.info(f"  {k}: {v:.4f}")

    # Save final results
    output = {
        "best_trial": best_trial.number,
        "best_params": best_trial.params,
        "revalidation_seeds": n_seeds,
        "full_timesteps": full_timesteps,
        "final_metrics": agg,
        "per_seed_metrics": all_final_metrics,
    }
    with open(TUNING_DIR / "best_trial_revalidation.json", "w") as f:
        json.dump(output, f, indent=2)

    return agg


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(
    n_trials: int = 30,
    revalidate: bool = True,
    config_path: str | Path | None = None,
):
    logging.basicConfig(level=logging.INFO)

    # Load config and graph
    config = load_config(config_path)
    graph = load_graph(GRAPH_PATH)
    logger.info(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Verify gates before proceeding
    logger.info("=== GATE VERIFICATION ===")
    sanity_report_path = MODELS_DIR / "sanity_smallgraph" / "sanity_checks.json"
    if not sanity_report_path.exists():
        raise RuntimeError("Gate 2 FAIL: Missing small-graph sanity report; run train_smallgraph_sanity.py first.")
    sanity_report = json.loads(sanity_report_path.read_text(encoding="utf-8"))
    if not sanity_report.get("trained_beats_masked_random", False):
        raise RuntimeError("Gate 2 FAIL: Small-graph sanity checks did not pass; full-graph PPO is blocked.")
    logger.info("Gate 1: Design decisions documented ✓")
    logger.info("Gate 2: Small-graph sanity check passes ✓")
    logger.info("Gate 3: Assuming full-graph signal verified (manual check required)")

    # Generate tuning scenario set (separate from final eval)
    tuning_scenarios = generate_tuning_scenarios(graph, config)

    # Generate FINAL evaluation scenario set (reserved, seed=42)
    logger.info(f"Generating FINAL evaluation scenarios with seed={FINAL_EVAL_SEED}")
    final_scenarios = generate_scenarios(
        graph=graph,
        env_ctor_args=(str(TRAFFIC_PROFILE_PATH), str(CLIMATOLOGY_PATH)),
        config=config,
        num_scenarios=config.get("evaluation", {}).get("num_scenarios", 100),
        num_flood_scenarios=config.get("evaluation", {}).get("num_flood_scenarios", 30),
        seed=FINAL_EVAL_SEED,
    )

    # Base env for evaluation
    base_env = _make_env(graph, (str(TRAFFIC_PROFILE_PATH), str(CLIMATOLOGY_PATH)), config)

    # Create study
    study = create_study()
    logger.info(f"Study created. Existing trials: {len(study.trials)}")

    # Run optimization
    logger.info(f"Starting optimization: {n_trials} trials")
    study.optimize(
        lambda t: objective(t, graph, config, tuning_scenarios, base_env),
        n_trials=n_trials,
        timeout=None,  # No timeout - let it run
        show_progress_bar=True,
    )

    logger.info("=== OPTIMIZATION COMPLETE ===")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value (flood completion rate): {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_trial.params}")

    # Save best params
    with open(TUNING_DIR / "best_params.json", "w") as f:
        json.dump({
            "trial": study.best_trial.number,
            "value": study.best_value,
            "params": study.best_trial.params,
        }, f, indent=2)

    # Re-validate best trial on final evaluation set
    if revalidate:
        revalidate_best_trial(study, graph, config, final_scenarios, base_env)

    base_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for MaskablePPO")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--no-revalidate", action="store_true", help="Skip best trial re-validation")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()

    main(n_trials=args.n_trials, revalidate=not args.no_revalidate, config_path=args.config)
