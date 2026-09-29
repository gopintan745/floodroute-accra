# Known Limitations

Honest, running list of gaps in data, validation, or scope. Per the
feasibility study, this list is part of the deliverable, not something to
minimize — it's what separates a real research contribution from an
overclaimed demo.

## Data

- [ ] Traffic proxy is (synthetic / sparse real samples / blend) — specify
      once decided, and note how far this is likely to diverge from ground
      truth
- [ ] Flood risk is a heuristic DEM+rainfall proxy, not a validated
      hydrological model
- [ ] Road quality labels are (manual spot-check / heuristic) — note
      coverage and confidence

## Validation

- [ ] No systematic ground-truth comparison exists; validation is limited to
      (news reports / personal knowledge / whatever is available) — be
      specific once you've actually attempted this

## Scope

- Study area is limited to [TODO: named sub-region], not city-wide
- No live deployment; this is a simulation study only
- Single-agent only — no multi-vehicle congestion interaction modeled

## Results caveats

(Fill in after Phase 6 — e.g. "RL agent outperforms dynamic Dijkstra on
failure rate but not on mean travel time" or similar honest findings.)

## Hyperparameter tuning asymmetry (fairness caveat)

The RL agent undergoes extensive hyperparameter tuning via Optuna
(~30–60 trials, each with 2–3 seeds, at ~25% of full training budget),
while the classical baselines (`static_shortest_path`, `dynamic_replanning`,
`masked_random`) receive **no equivalent tuning** — they are deterministic
graph-search algorithms with no learnable parameters to optimize.

This is a normal and defensible asymmetry: there is nothing to tune for a
deterministic baseline. However, it means the final comparison is **not**
a pure apples-to-apples "same compute budget" contest. The RL agent has
had its hyperparameters selected to maximize performance on a held-out
tuning scenario set, while baselines are evaluated out-of-the-box against
the same scenarios.

When reporting results, state plainly:

- RL agent: tuned on separate tuning scenario set (seed=12345), then
  re-validated on reserved final evaluation set (seed=42)
- Baselines: no tuning, evaluated directly on final evaluation set
- Tuning budget: N trials × M timesteps each (see `results/tuning/best_trial_revalidation.json`)

This caveat should appear alongside any quantitative comparison table
claiming "RL beats baselines."
