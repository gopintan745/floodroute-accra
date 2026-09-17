"""
Phase 5, step 1: train DQN on a SMALL subgraph first, as a cheap sanity
check of the reward design before committing to full PPO training.

If DQN can't learn anything sensible here, the problem is almost certainly
in the environment (reward scale, state leakage, broken termination logic)
rather than the algorithm — debug here, not in PPO on the full graph.

TODO:
- Load a small subgraph (e.g. one neighborhood, <50 nodes) via config
- Wrap AccraRoutingEnv around it
- from stable_baselines3 import DQN
- Train for config.training.dqn.total_timesteps
- Log to results/logs/, save to models/dqn_smallgraph/
- Sanity checks to run before trusting results:
    - does the agent do better than a random policy?
    - does it do better than static Dijkstra on this same subgraph?
    - does the training reward curve actually trend upward and stabilize?
"""

from stable_baselines3 import DQN  # noqa: F401


def main():
    raise NotImplementedError("TODO: wire up config loading, env, training loop")


if __name__ == "__main__":
    main()
