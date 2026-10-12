# Phase 4 Spec: `hazard_simulator.py`

Handoff spec — implements the flood-realization logic for the RL environment.
Read alongside `docs/proposal.md` (Section 3, MDP formulation) and
`docs/design_decisions.md` before starting.

## Scope (deliberately narrow)

`HazardSimulator` handles **flood realization only** — not traffic. Traffic
is deterministic given `(highway_class, hour, is_weekend)` (looked up from
`traffic_profile.parquet`), so it's identical whether "predicted" or
"realized" and needs no simulator involvement. Flooding is the only
genuinely stochastic, partially-observable hazard in this project.

## Critical design note: two different uses of `flood_susceptibility`

`effective_travel_time()` (from `src/common/cost_model.py`, built in Phase 3)
is used in two different ways depending on context — do not conflate them:

| Context | `flood_susceptibility` passed in | Flood penalty |
| --- | --- | --- |
| **Baseline planning** (`dijkstra_dynamic.py`) | Real static value (epistemic risk weight — baseline must commit to a route blind) | None (folded into the multiplier) |
| **Environment reward** (once an edge is *traversed*) | `0` (outcome is now known, not a probability) | Additive: `config.env.flood_penalty` if `hazard_simulator.is_flooded(edge)` is `True` this episode |

Both call the same shared function — only the arguments differ. This keeps
the baseline and the eventual RL reward using consistent cost logic
(needed for the Phase 6 comparison to mean anything).

## `HazardSimulator` API

```python
class HazardSimulator:
    def __init__(self, graph, climatology: dict, config: dict, mode: str, seed: int | None = None):
        """
        mode: "train" or "eval" — determines flood-day sampling distribution.
        Raise ValueError on any other mode value.
        climatology: parsed rainfall_climatology.json (from build_flood_layer.py).
        Stores static flood_susceptibility per edge from `graph`.
        self.rng = np.random.default_rng(seed)
        """

    def reset_episode(self) -> dict:
        """
        Call at the start of every episode.

        1. Sample month uniformly from 1-12 (see "Month sampling" note below).
        2. Sample hour (0-23) and is_weekend (bool) for the episode
           (used by env for traffic lookup — hazard_simulator just samples
           and returns these, doesn't use them itself).
        3. Determine is_flood_day:
             - mode == "train": Bernoulli(config.env.train_flood_day_rate)
             - mode == "eval":  Bernoulli(climatology["by_month"][month]["heavy_day_fraction"])
        4. If is_flood_day: draw initial realized-flooded edges via
           Bernoulli(flood_susceptibility[edge]) independently per edge.
           If not a flood day: zero edges start flooded.
        5. Reset internal state: step counter -> 0, flooded-edge set,
           "episode has been reset" flag (see is_flooded() below).

        Returns: {"month": int, "hour": int, "is_weekend": bool, "is_flood_day": bool}
        """

    def maybe_trigger_event(self, step: int) -> list:
        """
        Call once per environment step, BEFORE querying is_flooded() for the
        edge the agent just chose (a flood can newly appear right as the
        agent commits to a road — this is what tests real adaptivity).

        No-op (return []) if not an is_flood_day episode.

        For each not-yet-flooded edge:
            activation_prob = config.env.mid_episode_event_base_rate * flood_susceptibility[edge]
            if rng.random() < activation_prob: newly flooded

        Flooded edges accumulate — never un-flood within an episode.
        Returns list of newly-flooded edge ids (for logging/debugging).
        """

    def is_flooded(self, u, v, k) -> bool:
        """
        Ground truth query. Env calls this ONLY for the edge just traversed —
        NEVER for candidate edges under consideration (that would break
        partial observability, which is the whole point of this project).

        Raise a clear RuntimeError if called before reset_episode() has run
        for the current episode — do NOT silently return False. (Two prior
        bugs in this codebase were exactly this shape: a check silently
        passing when it should have failed loudly — the DEM/graph CRS
        mismatch and the .gitkeep false-positive in build_graph_costs.py.
        Don't make it three.)
        """
```

## Config additions (`config.example.yaml`)

```yaml
env:
  train_flood_day_rate: 0.35        # deliberate oversampling during training
  mid_episode_event_base_rate: 0.03 # per-step activation rate, scaled by edge's flood_susceptibility
  # evaluation always uses climatology's true heavy_day_fraction — no override needed
```

## Design decisions already made (log these in `docs/design_decisions.md`)

1. **Train/eval flood-day rate split**: training oversamples flood days
   (35%) so the agent gets enough exposure to learn flood-avoidance
   behavior; evaluation uses the true climatology rate for an honest
   performance estimate. **Consequence**: training-time reward and
   evaluation-time reward are NOT comparable numbers — only eval-mode
   results should ever be reported as project performance. Add a guard
   comment (or assertion) in the future `evaluation/run_scenarios.py`
   confirming it instantiates `HazardSimulator(mode="eval")`.

2. **Month sampled uniformly**, not weighted by climatology, in both
   modes. Rationale: in eval mode, seasonality already enters through the
   month-specific `heavy_day_fraction` used in the flood-day draw itself —
   weighting month selection on top of that would double-count
   seasonality. Revisit if eval results look off.

3. **Flooding is an additive penalty, not a blocking/impassable state.**
   The agent can still traverse a flooded edge (with the penalty applied);
   movement never fails outright. This avoids stuck-episode edge cases
   (e.g. only remaining route is flooded) while still creating a clear
   incentive to avoid flooded edges when possible.

## Testing checklist (make these real asserts, not skipped)

- [ ] **Mode separation**: same seed, `mode="train"` vs `mode="eval"`, many
      `reset_episode()` calls — confirm observed flood-day rate roughly
      matches `0.35` and the climatology's true rate respectively. This is
      the test that validates the entire point of this design.
- [ ] **Determinism**: same seed → identical sequence of `reset_episode()`
      outputs and `maybe_trigger_event()` activations across two separate
      runs.
- [ ] **Non-flood days stay clean**: `is_flood_day=False` → zero edges ever
      flooded, `maybe_trigger_event` always returns `[]`.
- [ ] **Monotonic flooding**: once an edge is flooded within an episode, it
      stays flooded for the rest of that episode — assert this explicitly,
      it's an easy accidental regression (re-rolling instead of
      accumulating).
- [ ] **Fails loudly, not quietly**: calling `is_flooded()` before
      `reset_episode()` raises, does not return `False`.
