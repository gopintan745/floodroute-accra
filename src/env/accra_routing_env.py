"""
The core Gymnasium environment: a partially-observable sequential routing
task over the Accra study-area road graph.

This is the most important file in the project (see docs/proposal.md
Section 3 for the full MDP formulation). Get this right before training
anything.

State (per docs/proposal.md):
    - current node id
    - destination node id
    - locally observable conditions on edges adjacent to current node
      (traffic level, flood risk, road quality) — NOT the whole graph;
      this is what makes it partially observable
    - optionally: time-of-day / recent rainfall context

Action:
    - discrete choice among outgoing edges from the current node
      (variable action count per node — see note below)

Reward:
    - negative travel time for the edge taken
    - large penalty if the edge is revealed to be flooded/impassable
      on arrival
    - + bonus on reaching destination
    - small per-step penalty to discourage wandering

Episode:
    - one trip, sampled origin -> sampled destination within study area
    - conditions can change mid-episode via hazard_simulator.py, to test
      whether the agent adapts rather than just memorizing static weights

NOTE on variable action spaces: intersections have different numbers of
outgoing edges. Gymnasium's Discrete space assumes a fixed size, so you'll
need either (a) a fixed max-degree action space with invalid-action masking,
or (b) a different action encoding (e.g. selecting a target node from a
candidate list). Decide this early — it affects your SB3 policy choice.
Log the decision in docs/design_decisions.md.
"""

import gymnasium as gym
from gymnasium import spaces


class AccraRoutingEnv(gym.Env):
    """Sequential routing under partially-observed traffic/flood/road-quality
    conditions on the Accra study-area graph."""

    metadata = {"render_modes": []}

    def __init__(self, graph_path: str, config: dict):
        super().__init__()
        # TODO: load graph (networkx), store config (episode length, penalties,
        # reveal radius from config["env"])
        # TODO: define self.observation_space and self.action_space
        #   - observation_space: likely a Dict or flattened Box combining
        #     current node features, destination features, and local edge features
        #   - action_space: see variable-action-space note above
        self.observation_space = spaces.Dict({})  # TODO
        self.action_space = spaces.Discrete(1)  # TODO placeholder

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # TODO: sample origin/destination pair, reset hazard simulator state,
        # build initial observation
        raise NotImplementedError("TODO")

    def step(self, action):
        # TODO:
        #  1. map action -> chosen edge
        #  2. reveal that edge's true condition (may differ from what was
        #     "expected" — this is the partial-observability payoff)
        #  3. compute reward (travel time + flood penalty if applicable)
        #  4. advance hazard_simulator state (conditions can evolve)
        #  5. build next observation (local view from new current node)
        #  6. check termination (reached destination) / truncation (max steps)
        raise NotImplementedError("TODO")

    def render(self):
        # Optional: plot current route on the graph for debugging.
        raise NotImplementedError("TODO (optional)")
