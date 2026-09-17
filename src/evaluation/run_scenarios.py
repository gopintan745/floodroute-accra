"""
Generate a fixed, seeded set of held-out (origin, destination, hazard
realization) scenarios used to evaluate the trained agent and both
baselines identically.

TODO:
- Sample config.evaluation.num_scenarios origin-destination pairs, held out
  from whatever distribution was used during training (don't evaluate on
  training scenarios)
- For each, draw a hazard realization from HazardSimulator using
  config.evaluation.seed for reproducibility
- Save scenario definitions to results/scenario_outputs/scenarios.json so
  every method (RL agent, static Dijkstra, dynamic Dijkstra) is evaluated
  on the exact same set
"""


def generate_scenarios(graph, hazard_simulator, num_scenarios: int, seed: int):
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    raise NotImplementedError("TODO: wire up CLI")
