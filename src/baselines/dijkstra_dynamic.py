"""
Baseline 2: shortest path re-weighted using currently-known hazard risk
scores (flood_risk, traffic_multiplier) — the "predict conditions, then
search" approach from our earlier discussion. This is the stronger,
fairer baseline: same information as the RL agent gets, no learning.

TODO:
- Combine flood_risk + traffic_multiplier + road_quality into a single
  edge cost using the SAME weighting logic the RL reward uses, so the
  comparison is apples-to-apples
- Optionally support re-planning: re-run Dijkstra from the current node
  whenever a hazard event is revealed mid-route, to give this baseline a
  fair shot at handling the "adaptive" scenarios too
"""


def dynamic_shortest_path(graph, origin, destination, current_risk_estimates):
    raise NotImplementedError("TODO")


def dynamic_shortest_path_with_replanning(env_or_graph, origin, destination):
    """Re-plans at each step using revealed information, for a fairer
    comparison against the RL agent's adaptive behavior."""
    raise NotImplementedError("TODO")
