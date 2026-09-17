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
