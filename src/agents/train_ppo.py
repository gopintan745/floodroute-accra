"""
Phase 5, step 2: train PPO (discrete action head) on the full study-area
graph. Only run this after DQN sanity checks on the small subgraph pass —
see train_dqn.py.

TODO:
- Load the full study-area graph
- Wrap AccraRoutingEnv; consider vectorizing with
  stable_baselines3.common.env_util.make_vec_env for config.training.ppo.n_envs
  parallel environments
- from stable_baselines3 import PPO
- model = PPO("MultiInputPolicy" or "MlpPolicy", env, ...) — depends on
  whether observation_space ended up a Dict or flattened Box
- Train for config.training.ppo.total_timesteps
- Log to results/logs/ (TensorBoard), save to models/ppo_full/
- Track training stability (reward variance, entropy) — PPO's clipping
  should keep this reasonably smooth; large oscillations suggest a reward
  scaling or state-design problem worth revisiting
"""

from stable_baselines3 import PPO  # noqa: F401


def main():
    raise NotImplementedError("TODO: wire up config loading, env, training loop")


if __name__ == "__main__":
    main()
