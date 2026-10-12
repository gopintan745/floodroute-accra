# Phase 4 Spec: `graph_wrapper.py` + `accra_routing_env.py`

Handoff spec for the RL environment. Read alongside `docs/proposal.md`
(Section 3, MDP formulation), `docs/design_decisions.md`, and
`docs/phase4_hazard_simulator_spec.md` (this env depends directly on
`HazardSimulator`) before starting.

## Decided: action space

**`MaskablePPO` from `sb3-contrib`**, not vanilla `stable-baselines3` PPO or
DQN. Add `sb3-contrib>=2.3` to `requirements.txt`.

**Why:** each intersection has a different number of outgoing roads, which
doesn't map onto a fixed-size `Discrete` space as-is. The fix is
`Discrete(K)` where `K` = the largest out-degree anywhere in the graph,
with most actions invalid at any given lower-degree node. Proper masking
(not a penalty-based workaround) avoids wasting training signal on
impossible actions and is the standard tool for exactly this "variable
legal actions per state" situation.

**Consequence for `train_dqn.py`**: vanilla SB3 `DQN` has no equivalent
masking support, and hand-rolling one just for a debug step isn't worth
the complexity. Use `MaskablePPO` on a small subgraph for the "cheap sanity
check before full training" step too — same algorithm as the real run,
just faster to iterate on. This also means the masking logic itself gets
debugged early rather than deferred to the full run. Treat this as settled,
not open for reconsideration without a specific reason to revisit it.

## Why action index alignment matters

Action index `3` must mean roughly the same thing at every node, or the
policy can't generalize across intersections — it would effectively
memorize per-node behavior instead of learning transferable routing
sense. Fix: canonicalize each node's outgoing edges by **compass bearing**
(sorted, e.g. most-northward edge first), padded to `K` with an invalid
sentinel. Action `0` then consistently means "the most northward option
available here," across every node in the graph.

## `src/env/graph_wrapper.py`

```python
class GraphWrapper:
    def __init__(self, graph, max_degree: int | None = None):
        """
        max_degree: K, the fixed action-space size. Auto-computed from the
        graph (max out-degree) if not given — compute this from the REAL
        study-area graph once available, don't hardcode a guessed value.

        Precomputes, once at load time (NOT per step — this is called every
        env.step(), it must be fast):
          - per node: outgoing edges sorted by compass bearing, padded to
            max_degree with None
          - node feature lookup, edge feature lookup (for observation
            construction — see accra_routing_env.py below)
        """

    def action_to_edge(self, node, action_idx: int):
        """Returns (u, v, k) for the edge this action selects at `node`,
        or None if action_idx is invalid (padding) at this node."""

    def action_mask(self, node) -> list[bool]:
        """Boolean array of length max_degree — True where action_to_edge
        would return a real edge. Fed to MaskablePPO via
        accra_routing_env.action_masks()."""

    def neighbors_within_hops(self, node, hops: int) -> list:
        """Edges within `hops` of `node` — used to build the "locally
        observable" part of the state, and to know which edges
        HazardSimulator.is_flooded() may legally be queried for (see the
        corrected contract in phase4_hazard_simulator_spec.md — ground
        truth is queryable within reveal_radius_hops of the CURRENT
        position, not just the edge just traversed)."""
```

Compute bearing with the standard formula between two lon/lat points
(`atan2`-based initial bearing), using each edge's endpoint node
coordinates — no new dependency needed, plain math on `graph.nodes[n]['x'/'y']`.

## `src/env/accra_routing_env.py`

```python
class AccraRoutingEnv(gym.Env):
    def __init__(self, graph_path, traffic_profile_path, climatology_path, config, mode: str):
        # mode: "train" or "eval", passed straight through to HazardSimulator
        # load graph, wrap in GraphWrapper(max_degree=None)  # auto-computed
        # load traffic_profile.parquet -> lookup dict (highway_class, hour, is_weekend) -> multiplier
        # self.hazard_sim = HazardSimulator(graph, climatology, config, mode=mode, seed=...)
        self.action_space = spaces.Discrete(self.graph_wrapper.max_degree)
        self.observation_space = spaces.Dict({
            "current_node_features": spaces.Box(...),
            "destination_node_features": spaces.Box(...),
            "local_edge_features": spaces.Box(...),  # within reveal_radius_hops; padded to max_degree
            # local_edge_features per candidate action: predicted flood_susceptibility,
            # traffic_multiplier (known, not stochastic), road_quality_score, base_travel_time
        })

    def action_masks(self) -> list[bool]:
        # REQUIRED exact method name for sb3-contrib's MaskablePPO —
        # called during rollout collection. Easy to implement the masking
        # logic and forget to expose it through this specific name.
        return self.graph_wrapper.action_mask(self._current_node)

    def reset(self, *, seed=None, options=None):
        # sample origin/destination pair within study area
        # episode_context = self.hazard_sim.reset_episode()  # {month, hour, is_weekend, is_flood_day}
        # self._current_node = origin_node; self._step_count = 0
        # return self._build_observation(), info

    def step(self, action):
        # edge = self.graph_wrapper.action_to_edge(self._current_node, action)
        # assert edge is not None  # MaskablePPO should never select a masked action
        # newly_flooded = self.hazard_sim.maybe_trigger_event(self._step_count)
        # is_flooded = self.hazard_sim.is_flooded(*edge)  # ground truth reveal for the traversed edge
        #
        # traffic_multiplier = self.traffic_lookup[(highway_class, episode_hour, episode_is_weekend)]
        # realized_time = effective_travel_time(
        #     base_travel_time, traffic_multiplier,
        #     flood_susceptibility=0,   # realized outcome, not risk-weighted — see cost_model note below
        #     road_quality_score, cost_cfg.flood_weight, cost_cfg.quality_penalty_weight,
        # ) + (config.env.flood_penalty if is_flooded else 0)
        #
        # reward = -realized_time - config.env.step_penalty
        # self._current_node = edge's target node; self._step_count += 1
        # terminated = (self._current_node == destination_node)  # + destination bonus to reward
        # truncated = (self._step_count >= config.env.max_episode_steps)
        # return self._build_observation(), reward, terminated, truncated, info
```

**Cost model reminder** (from `docs/phase4_hazard_simulator_spec.md`):
`flood_susceptibility=0` in the realized-reward calculation is intentional,
not a bug — the multiplicative risk-weighting is only for *planning*
baselines that must commit blind. Once an edge is actually traversed, the
outcome is known, so the additive `flood_penalty` carries all of the flood
cost. Don't let these two get conflated when writing `step()`.

**Observation construction detail**: `local_edge_features` should use
`graph_wrapper.neighbors_within_hops(current_node, reveal_radius_hops)` for
which edges get *true* ground-truth values (via `hazard_sim.is_flooded()`)
versus which fall back to static predicted `flood_susceptibility` — this is
literally what makes the task partially observable rather than fully
observed. Get the boundary here right; it's the same boundary
`hazard_simulator`'s `is_flooded()` contract already encodes.

## Config additions (`config.example.yaml`)

```yaml
env:
  max_degree: null   # null = auto-compute from graph at env init; set explicitly only to override
```

(`max_episode_steps`, `flood_penalty`, `step_penalty`, `reveal_radius_hops`,
`train_flood_day_rate`, `mid_episode_event_base_rate` already exist from
earlier phases — no changes needed to those.)

## Testing checklist

- [ ] **`GraphWrapper.action_mask` correctness**: on a small hand-built
      graph with known node degrees, confirm the mask has exactly the
      right number of `True` entries per node, and that the `True`
      positions correspond to bearings in roughly the expected compass
      order (e.g. a 4-way intersection with edges N/E/S/W should mask in
      that bearing order consistently).
- [ ] **Action index consistency across nodes**: two different nodes that
      both have a "northward" outgoing edge should assign it the same
      action index — this is the test that actually validates the
      generalization rationale above, not just that masking runs.
- [ ] **`action_masks()` is called by name**: a quick smoke test
      instantiating `MaskablePPO` against the env for a handful of
      timesteps, confirming it runs without SB3 complaining about a
      missing/malformed mask method.
- [ ] **Partial observability boundary**: build a scenario where a flooded
      edge sits just outside `reveal_radius_hops` — confirm the
      observation shows only the static predicted risk for it, not the
      ground truth, until the agent gets within range or traverses it.
- [ ] **Reward reduces correctly on a "boring" episode**: `is_flood_day=False`,
      uniform traffic multiplier of 1.0, road quality 1.0 everywhere — the
      per-step reward should reduce to `-base_travel_time - step_penalty`
      with zero flood penalty ever applied, a good regression check against
      the cost-model reconciliation note above.
- [ ] **Termination vs. truncation**: confirm reaching the destination sets
      `terminated=True` (not `truncated`), and hitting `max_episode_steps`
      without arriving sets `truncated=True` (not `terminated`) — SB3
      treats these differently and mixing them up silently changes training
      dynamics.
