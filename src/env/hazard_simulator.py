"""
Generates synthetic/blended hazard dynamics used within an episode, so the
agent has to handle changing conditions rather than a static snapshot.

TODO:
- Given a graph with base flood_risk / traffic_multiplier / road_quality
  attributes (from build_graph_costs.py), sample an episode-specific
  realization: which edges are actually flooded/congested THIS episode,
  drawn probabilistically from the risk scores
- Support a "mid-episode event" mode: with some probability, an edge's
  condition changes partway through the episode (simulating a flash flood
  starting, or congestion building) — this is what actually tests adaptive
  re-routing rather than one-shot planning
- Keep this deterministic under a fixed seed (config.evaluation.seed) so
  evaluation scenarios are reproducible and comparable across agent/baseline
  runs
"""


class HazardSimulator:
    def __init__(self, graph, config: dict, seed: int | None = None):
        # TODO
        raise NotImplementedError("TODO")

    def sample_episode_realization(self):
        """Draw which edges are flooded/congested for this episode."""
        raise NotImplementedError("TODO")

    def maybe_trigger_event(self, step: int):
        """Possibly change conditions mid-episode. TODO."""
        raise NotImplementedError("TODO")

    def true_condition(self, edge) -> dict:
        """Ground-truth condition for an edge (used by env.step to reveal
        on arrival — NOT exposed to the agent before it commits to the edge)."""
        raise NotImplementedError("TODO")
