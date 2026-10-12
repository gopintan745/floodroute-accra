# Phase 4b Spec: Risk-Assessment Redesign (Graded Flood Severity + Calibrated Beliefs)

Handoff spec. Read alongside `docs/proposal.md`, `docs/design_decisions.md`,
`docs/phase4_hazard_simulator_spec.md`, `docs/phase4_env_spec.md`, and
`docs/debug_brief_rl_training_failure.md`. This spec **supersedes parts of
the two Phase 4 specs** (see "Superseded items" below).

## Why this change

The project is about **assessing risk**, not avoiding floods. A flooded edge
may still be passable, and wading through (or waiting in traffic) can beat a
long detour. The current design cannot express that:

- Inside the reveal radius the env gives the agent **binary ground truth**
  (`flood_risk` is set to exactly `1.0` or `0.0` in `_edge_feature_vector`).
  Outside it, the agent sees the static susceptibility score, which is a
  terrain heuristic, not a calibrated probability of anything.
- Flooding is priced as a flat `flood_penalty * 60 = 3,000 s` toll, while a
  median full trip is about 380 s. At that price wading is never rational, so
  there is no tradeoff to learn. Evidence from the full-graph eval:
  `dynamic_replanning` avoided every flooded edge (0.0 rate) at nearly the
  same median time as static (380 s vs 381 s).

Risk assessment needs three things: (1) graded consequences, (2) uncertainty
the agent cannot fully resolve ahead of time, expressed as probabilities,
and (3) **calibration**: if the agent is told "70% shallow", about 70% of
such edges must actually be shallow.

## Decisions already made (log in `docs/design_decisions.md`)

1. **Three discrete severity levels**: `0 = dry`, `1 = shallow` (passable,
   slowed), `2 = deep` (stall / crawl, large penalty).
2. **Beliefs are Bayesian posteriors** over the three levels, generated from
   a noisy signal. Never a blend of prior and truth (see "Trap 1").
3. **Risk-neutral objective** (maximize expected return). The size of the
   deep-level penalty *is* the risk preference. Report tail metrics, but do
   not add a tail-sensitive training objective in v1.
4. **Latent levels are fixed at episode start** (same convention as before:
   conditions are per-episode realizations). The risk is therefore
   *epistemic*: uncertainty about a hidden level that sharpens on approach.
5. **The agent observes `is_flood_day` and day `intensity`.** A driver knows
   it is pouring. This also makes beliefs calibrated *conditional on day
   type*, so the train-time flood-day oversampling (35%) vs. eval-time true
   rate does not make beliefs miscalibrated; it only changes how often flood
   days occur.
6. **Traffic stays deterministic** (looked up by class/hour). Stochastic
   traffic is deferred.
7. **`mid_episode_event_base_rate` defaults to 0.0 in v1.** With events on,
   the agent's prior for unobserved edges drifts away from the truth over
   time and calibration is no longer exact. Keep the code path, document it
   as a limitation, and revisit with a time-dependent prior later.
8. Severity parameters are **synthetic assumptions** (no depth data exists).
   Say so in `docs/limitations.md`.

## Generative model

Levels `L = {0: dry, 1: shallow, 2: deep}`.

**Day intensity `I` in [0, 1].** Non-flood day: `I = 0` and every edge is dry.
Flood day: sample a daily precipitation `p` uniformly from the stored
heavy-day values, then
`I = clip((p - T) / (p_ref - T), 0, 1)` where `T = heavy_rain_threshold_mm`
and `p_ref` is the `intensity_reference_percentile` of heavy-day values.
If there are no heavy-day values or `p_ref <= T`, use
`default_intensity_if_no_heavy_days`.

This needs a small addition to `build_flood_layer.py`: store
`heavy_day_precip_mm` (the list of daily values at or above the threshold)
in `rainfall_climatology.json`.

**Prior per edge** (for edge susceptibility `s`, on a flood day):

```python
lambda   = clip(s * (1 - alpha + alpha * I), 0, 1)     # P(edge is wet at all)
f_deep   = deep_fraction_min + (deep_fraction_max - deep_fraction_min) * I
P(deep)    = lambda * f_deep
P(shallow) = lambda * (1 - f_deep)
P(dry)     = 1 - lambda
```

Non-flood day: prior is `[1, 0, 0]` for every edge.

**Signal model.** `C_h` is a 3x3 row-stochastic confusion matrix per hop
distance `h`, with **rows = true level, columns = reading**, so the
likelihood of reading `y` given level `l` is `C_h[l][y]`. Edge hop distance
`h(e) = min(dist(current, u), dist(current, v))` in the undirected view; an
edge is inside the reveal radius when `h <= reveal_radius_hops`.

**Posterior.** For an edge inside the radius with reading `y` at hop `h`:
`posterior ∝ prior * C_h[:, y]` (elementwise), normalized (guard the
denominator with a small epsilon). Outside the radius: posterior = prior.

**Reproducibility (critical).** Every method evaluated on the same seed must
face the *identical* world, regardless of the actions it takes.

- Use `np.random.SeedSequence(seed).spawn(2)` to get two independent streams:
  one for latent levels, one for signal readings.
- At reset, sample latent levels for all edges in a fixed (sorted) edge order
  from the latent stream. Then precompute `readings[edge][h]` for every
  `h` in `0..reveal_radius_hops` from the signal stream, also in fixed order.
  Readings are therefore fixed per (episode, edge, hop) and independent of
  visiting order.
- If any hashing is used anywhere for seeding, use a **stable** hash
  (`zlib.crc32` or `hashlib`), never Python's built-in `hash()`, which is
  randomized per process.

## Cost model (`src/common/cost_model.py`)

One source of truth, split into planning vs. realization:

```python
def base_edge_time(base_travel_time, traffic_multiplier, road_quality_score,
                   quality_penalty_weight) -> float:
    # base * traffic * (1 + (1 - quality) * quality_penalty_weight); no flood term

def level_cost(t, level, cfg) -> float:
    # t * multiplier[level] + penalty_s[level]
    # dry: (1.0, 0); shallow: (m_shallow, 0); deep: (m_deep, P_deep)

def realized_cost(t, true_level, cfg) -> float:   # used by the env and oracle
    return level_cost(t, true_level, cfg)

def expected_cost(t, belief, cfg) -> float:        # used by planning baselines
    return sum(belief[l] * level_cost(t, l, cfg) for l in range(3))
```

`level_cost` is linear in the belief, so `expected_cost` is exact (no
integration needed). The old flood-weight and additive `flood_penalty` are
removed from the flood path.

## `HazardSimulator` API (replaces `is_flooded`)

```python
class HazardSimulator:
    def reset_episode(self, scenario: dict | None = None) -> dict:
        """Returns {month, hour, is_weekend, is_flood_day, intensity}.
        Samples latent levels and precomputes readings (see Reproducibility).
        scenario may override: is_flood_day, intensity, edge_levels
        ({edge: level}), for tests."""

    def set_current_position(self, node) -> None:
        """Also computes and CACHES hop distances for the current position
        (one BFS per step, not one per query)."""

    def prior(self, u, v, k) -> np.ndarray:     # shape (3,)
    def belief(self, u, v, k) -> np.ndarray:    # posterior inside radius, prior outside
    def belief_map(self) -> dict:               # edge -> belief, for planning baselines

    def true_level(self, u, v, k) -> int:
        """GROUND TRUTH. Only for: (a) the env computing the realized cost of
        the edge just traversed, (b) the oracle baseline, (c) evaluation
        metrics. Must NEVER be used to build observations."""

    def maybe_trigger_event(self, step) -> list:
        """No-op unless mid_episode_event_base_rate > 0. If enabled, upgrades
        an edge's level by one step (dry->shallow, shallow->deep) with
        probability base_rate * susceptibility."""
```

A test-only option `signal_mode: "exact"` (identity confusion matrices) makes
unit tests deterministic.

## Environment changes (`accra_routing_env.py`)

**Observation.** Add a global feature vector and extend edge features.

- `global_features` Box(2): `[is_flood_day, intensity]`.
- `local_edge_features` per action slot, `edge_dim = 8`:
  `[P_dry, P_shallow, P_deep, traffic_multiplier, road_quality_score,
  base_travel_time, cost_to_go_via_action, cost_to_go_relative_to_best]`.
- `cost_to_go_via_action` = `expected_cost(edge, current belief)` +
  `prior_cost_to_go(v)`. Compute `prior_cost_to_go` once per episode with one
  reversed Dijkstra from the destination over prior-expected edge costs
  (including traffic and quality). Reason: with first-edge-only features the
  agent cannot weigh a risky edge against what the detour costs downstream,
  and that comparison is the whole tradeoff. This is recommended, and worth
  A/B testing against a run without it.
- **Information parity note.** The replanning baseline sees posteriors for
  all edges within the reveal radius; the agent's downstream term uses the
  prior beyond the first edge. Document this in `limitations.md`. If RL
  trails the replanner by roughly the value of those hop-1 readings, switch
  the agent's cost-to-go to use the full belief map each step (one reversed
  Dijkstra per step).

**Step.**

```python
edge = graph_wrapper.action_to_edge(current_node, action)
hazard_sim.maybe_trigger_event(step_count)
level = hazard_sim.true_level(*edge)
t = base_edge_time(...)
realized = realized_cost(t, level, cfg)
reward = -realized - step_penalty            # + terminal bonus / revisit logic as before
info.update(level=level, realized_travel_time=realized,
            belief_before=hazard_sim.belief(*edge), is_flooded=(level > 0))
```

Reward magnitudes are now bounded by roughly `m_deep * t + P_deep` per step
instead of a 3,000 s toll, which should also reduce the outlier-reward
instability seen in the earlier full-graph run.

**Do not** call `true_level` from `_build_observation`.

## Baselines

- `static_shortest_path`: unchanged (free-flow `travel_time` only).
- `dynamic_shortest_path`: edge weight = `expected_cost(t, belief_map()[e])`
  at the origin (posterior inside radius, prior elsewhere).
- `dynamic_replanning`: at every step rebuild `belief_map()` at the current
  position, replan on expected costs, take the first edge. Realized cost
  comes from the true level.
- **New: `oracle`**: knows true levels, edge weight = `realized_cost`. Used
  for regret and value-of-information analysis. The gap oracle -> baselines
  -> RL is the headline analysis.

## Scenarios and evaluation

- `scenarios.json`: store `is_flood_day`, `intensity`, and a sparse
  `edge_levels` (non-dry edges only) for audit. The seed remains the scenario.
- `run_scenarios.py` produces three sets: `natural`, `flood_stratified`, and
  **`decision_relevant`** (flood days where the static shortest path crosses
  at least one wet edge). The last is where risk tradeoffs actually exist.
- **All evaluators consume the scenario list.** The sanity-check evaluators
  in `training_utils.py` do their own ad hoc seeding and silently produced
  all-dry batches before. Fix this as part of this work.
- Metrics per method: mean / median / p90 / p95 / p99 realized time, CVaR(95)
  (mean of the worst 5%), completion rate, wet-edge traversals and deep
  traversals per episode, deep-traversal episode rate, `total_revisits`,
  and **regret vs. oracle** (mean and p95) on the same scenarios.
- **Decision-curve diagnostic.** At decision points where a wet edge competes
  with a detour, plot the agent's probability of choosing the wet edge against
  the break-even margin (`detour_extra_cost - expected_wading_extra_cost`).
  A risk-assessing agent should approximate a step at zero, as the
  expected-cost replanner does exactly.

## Traps to avoid

1. **Do not implement the belief as `(1-w)*prior + w*one_hot(truth)`.** The
   network can invert the blend and recover the truth. Use a noisy signal
   plus a true Bayes posterior.
2. **Confusion-matrix orientation.** Rows = true level, columns = reading;
   the likelihood vector for reading `y` is the *column* `C_h[:, y]`.
   Transposing it silently produces a miscalibrated posterior.
3. **Path-dependent worlds.** If readings or levels are drawn from a shared
   RNG on demand, different policies see different worlds on the same seed.
4. **Python `hash()` for seeding.** Not stable across processes.
5. **Leakage.** Beliefs outside the radius must equal the prior exactly.

## Config additions (`config.example.yaml`)

```yaml
flood_model:
  intensity_weight_alpha: 0.5
  deep_fraction_min: 0.1
  deep_fraction_max: 0.7
  intensity_reference_percentile: 95
  default_intensity_if_no_heavy_days: 0.5
  mid_episode_event_base_rate: 0.0
  signal_mode: noisy            # "exact" for deterministic unit tests
  level_costs:                  # assumptions, justify in design_decisions.md
    shallow: {multiplier: 2.0, penalty_s: 0}
    deep:    {multiplier: 3.0, penalty_s: 600}
  signal_confusion:             # rows = true level, cols = reading
    hop0: [[0.95, 0.05, 0.00], [0.05, 0.90, 0.05], [0.00, 0.10, 0.90]]
    hop1: [[0.85, 0.13, 0.02], [0.12, 0.76, 0.12], [0.03, 0.22, 0.75]]
```

Remove or deprecate `env.flood_penalty`. Keep `train_flood_day_rate`,
`step_penalty`, `max_episode_steps`, `reveal_radius_hops`.

## Superseded items from earlier specs

- `phase4_hazard_simulator_spec.md`: the boolean `is_flooded()` API, the
  per-edge Bernoulli flooding, and the "flooding is an additive penalty"
  decision are replaced by levels, beliefs, and `level_cost`. The
  train/eval flood-day-rate split, month sampling, and determinism
  requirements still stand.
- `phase4_env_spec.md`: the "pass `flood_susceptibility=0` in the env"
  reconciliation note is replaced by `realized_cost` (env) vs.
  `expected_cost` (planning). `action_masks()`, bearing-sorted actions, and
  the termination/truncation rules still stand.

## Milestones (each is a gate)

1. **Cost model + hazard simulator + calibration tests** (no RL). Do not
   proceed until every test below in Group A passes.
2. **Env integration**: observation, reward, info fields, Group B tests.
3. **Baselines + oracle + metrics + scenario sets**, including the three
   scenario sets and the shared-scenario evaluators.
4. **Informed revisit, redefined.** "New information" now means a belief
   change. Store, per visit, a snapshot of expected costs for the local edges;
   a revisit is forgiven only if some local edge's expected cost changed by
   more than an epsilon since the last visit to that node, and the snapshot
   is then refreshed. Until this lands, the uniform revisit rule applies to
   all methods, and `total_revisits` is reported so its effect on the
   replanning baseline is visible.
5. **Retrain**: small-graph sanity check on `flood_stratified` and
   `decision_relevant` scenarios first, then full graph. The Optuna gate in
   `phase5b_optuna_tuning_spec.md` still applies, plus: all Group A tests pass.

## Tests

### Group A (hazard simulator and cost model)

- [ ] **Prior calibration**: over a grid of `(s, I)`, sampled level
      frequencies match the prior formula within about 3 sigma (N >= 200k).
- [ ] **End-to-end posterior calibration through `HazardSimulator` itself**:
      run many resets on a synthetic graph spanning susceptibilities; group
      `(belief vector rounded, hop)` and compare the empirical frequency of
      each true level to the belief. Report expected calibration error.
      There are only 3 distinct posteriors per `(s, I, h)`, so this is cheap.
- [ ] **Path independence**: two different position/action sequences on the
      same seed see identical latent levels and identical readings per
      `(edge, hop)`; also identical across separate processes.
- [ ] **No leakage**: change the true level of an out-of-radius edge via a
      scenario override; its belief must be unchanged and equal to the prior.
- [ ] **Expected vs. realized consistency**: the Monte Carlo mean of
      `level_cost` over levels drawn from a belief matches `expected_cost`.
- [ ] **Degenerate case**: non-flood day gives all-dry levels, beliefs
      `[1, 0, 0]`, and a per-step reward of `-base_edge_time - step_penalty`.
- [ ] **Tradeoff sanity**: with default parameters, the break-even detour
      length for a typical edge at a moderate belief (for example
      `[0.6, 0.3, 0.1]`) lies between roughly 0.5x and 20x that edge's base
      time. Outside that range the config makes the problem trivial
      ("always avoid" or "never avoid").

### Group B (environment)

- [ ] `_build_observation` never calls `true_level` (monkeypatch it to raise).
- [ ] Observation shapes match `observation_space`; `action_masks()` still
      exposed by that exact name.
- [ ] Info fields `level`, `belief_before`, `realized_travel_time` present.
- [ ] Port the old tests: the loop-termination test should use an
      `edge_levels` override instead of `flooded_edges`.

### Group C (does it learn risk assessment): break-even toy benchmark

Replace the fixed rainy-day toy graph with a parametrized one: a risky
shortcut with a fixed mild belief versus a safe detour whose extra length is
randomized per episode. The expected-cost replanner should switch routes
exactly at the break-even length. Train the RL agent on randomized detour
lengths and confirm its choice flips near the same threshold. This is the
smallest test that distinguishes "learned risk assessment" from "learned to
avoid water".
