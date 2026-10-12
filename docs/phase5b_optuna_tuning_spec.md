# Phase 5b Spec: Hyperparameter Tuning with Optuna

Handoff spec for hyperparameter tuning. Read alongside
`docs/phase5_training_spec.md`. **Do not start this until the gate below is
cleared** — tuning now would optimize hyperparameters against an
environment/reward that's still being fixed, and risks the optimizer
learning to exploit whatever quirks remain rather than genuinely solving
the task.

## Gate: do not begin until all three are true

1. Environment and reward are frozen, including the informed-revisit
   change (see `docs/design_decisions.md`).
2. The small-graph sanity check passes specifically on flood-stratified
   scenarios (not just the natural/mostly-dry case — see the earlier
   debugging history in this project for why that distinction matters).
3. A full-graph run shows genuine learning signal: completion rate well
   above masked-random. Hyperparameter tuning cannot rescue a
   representation or reward-design problem — so far every meaningful
   improvement in this project has come from environment/observation
   fixes, not optimizer settings, and there's no reason to expect that to
   flip once tuning starts.

## What to tune

Standard `MaskablePPO` hyperparameters:

- `learning_rate` (log scale)
- `n_steps`
- `batch_size`
- `n_epochs`
- `gamma`
- `gae_lambda`
- `clip_range`
- `net_arch`
- `ent_coef` (log scale) — worth particular attention given the earlier
  looping/collapse pathology; entropy regularization directly affects how
  readily a policy converges to a degenerate deterministic behavior

## What NOT to tune

`flood_penalty`, `flood_weight`, `quality_penalty_weight`,
`revisit_penalty`, `step_penalty` — these define the task itself, and the
baselines (`static_shortest_path`, `dynamic_replanning`) are evaluated
against the same fixed cost model. Tuning these until RL wins would change
the question being asked rather than answer it, and would invalidate the
whole point of comparing against baselines. Fix these by reasoning, log
the choice in `design_decisions.md`, leave them untouched by the tuning
process entirely.

## Train/tune/test scenario separation

`run_scenarios.py` produces the scenario set used for final reported
comparison. If Optuna's objective function is evaluated on that same set,
hyperparameters are being selected against test data, and the final
reported result will be optimistically biased.

- Generate a **second, independent scenario set** (different seed) via the
  same `run_scenarios.py` machinery, used only as the tuning objective.
- Keep the original scenario set from Phase 6 completely untouched,
  reserved for the one final reported comparison.
- The tuning objective itself should mix flood-stratified and natural
  eval-mode episodes — never use training-mode reward as the objective,
  since training deliberately oversamples flood days at 35% (see
  `phase4_hazard_simulator_spec.md`) and doesn't reflect real performance.

## Performance check before committing to a tuning budget

Several things in the current `hazard_simulator.py` / `accra_routing_env.py`
look potentially expensive per-step or per-episode at full-graph scale:

- `_edge_is_revealed` builds an undirected graph view and runs a BFS on
  every single `is_flooded()` call
- `_build_observation` calls `is_flooded()` for up to `max_degree` edges,
  every step
- `reset()` computes `nx.descendants` for every node, every episode

**Measure steps/second on the full graph before scheduling trials.**
Optuna needs dozens of trials to be useful; if the environment is slow,
very few trials are affordable in reasonable time. If this turns out to be
a bottleneck, cache per-node reveal sets and graph reachability once at
`__init__` time rather than recomputing them repeatedly — this is very
likely worth doing regardless of tuning, given how much these same
functions are called during ordinary training too.

## Tuning design

- **Sampler/pruner**: TPE sampler with `MedianPruner`, generous
  `n_startup_trials` and `n_warmup_steps` — RL learning curves are noisy
  early on, and aggressive pruning risks killing configurations that are
  just slow starters, not genuinely bad.
- **Seeds per trial**: RL performance varies substantially by seed alone.
  Use 2-3 seeds per trial rather than 1, or at minimum re-run the top few
  candidate configurations with additional seeds before trusting a winner
  — a single lucky seed can look like the best hyperparameters and not be.
- **Budget**: 30-60 trials at roughly 20-25% of the full training budget
  each. If full-graph trials are too slow even after the performance check
  above, tune on a mid-size subgraph (larger than the 16-node sanity graph,
  smaller than the full graph) instead. `learning_rate`, `ent_coef`, and
  `clip_range` tend to transfer reasonably across graph sizes;
  horizon-related settings (`gamma`, `n_steps`) transfer less reliably —
  treat those as the riskiest to carry over from a subgraph tuning run.
- **Storage**: SQLite-backed Optuna study, so tuning can be stopped and
  resumed across sessions — fits the "no deadline, work on this
  intermittently" project constraint.
- **`VecNormalize` correctness**: each trial needs its own freshly
  constructed normalized env, and any mid-training evaluation callback
  must load/sync the correct normalization statistics rather than
  evaluating through a differently-normalized wrapper.
  `sb3_contrib`'s `MaskableEvalCallback` handles this. Given how many
  bugs in this project have traced back to `VecNormalize` mismatches
  specifically, write a dedicated test confirming the callback's
  evaluation env's normalization stats match the training env's at the
  moment of each evaluation — don't assume the off-the-shelf callback
  handles this correctly without checking.
- **Final step**: once the best trial is identified, retrain that exact
  configuration at full training budget across several seeds (not just
  the seed(s) used during tuning), and evaluate exactly once against the
  original, untouched Phase 6 scenario set for the number that actually
  gets reported.

## Fairness caveat — log this explicitly

The baselines (`static_shortest_path`, `dynamic_replanning`,
`masked_random`) receive no equivalent tuning. That's a normal and
defensible asymmetry (there's nothing to tune for a deterministic
graph-search baseline), but it should be stated plainly in
`docs/limitations.md` alongside the tuning budget spent, so the final
comparison doesn't read as more apples-to-apples than it actually is.

## Testing / sanity checklist

- [ ] **Gate conditions verified before starting** — don't take this
      spec as license to begin; confirm all three gate items above are
      actually true first.
- [ ] **Tuning objective uses the separate tuning scenario set**, never
      the reserved final-comparison set — verify this by checking the
      actual seed/file path used in the objective function, not just
      assuming it's correct.
- [ ] **Reward/cost-model parameters are absent from the search space** —
      a direct check of the Optuna search space definition, to catch an
      accidental inclusion before it wastes a tuning budget on the wrong
      question.
- [ ] **`VecNormalize` statistics verified in sync** during evaluation
      callbacks, per the dedicated test noted above.
- [ ] **Best trial re-validated across multiple seeds** before being
      treated as final — a single strong trial result is not sufficient
      evidence on its own, given known seed variance in RL performance.
