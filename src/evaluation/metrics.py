"""
Metrics computed per scenario, per method (RL agent / static Dijkstra /
dynamic Dijkstra), as defined in docs/proposal.md Section 6.

TODO:
- travel_time(route, hazard_realization) -> float (minutes)
- hit_blocked_edge(route, hazard_realization) -> bool
- failure_rate(results: list) -> float  (fraction of scenarios that hit a
  blocked/flooded edge)
- summarize(results: list) -> dict  (mean/median/variance of travel time,
  failure rate, per method — this is what compare_results.py consumes)
"""


def travel_time(route, hazard_realization) -> float:
    raise NotImplementedError("TODO")


def hit_blocked_edge(route, hazard_realization) -> bool:
    raise NotImplementedError("TODO")


def summarize(results: list) -> dict:
    raise NotImplementedError("TODO")
