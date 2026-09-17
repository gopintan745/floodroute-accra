# FloodRoute-Accra

Reinforcement learning for route recommendation under traffic, flooding, and
road-quality uncertainty in a bounded sub-region of Accra, Ghana.

See `docs/proposal.md` for the full problem statement, MDP formulation, and
feasibility study.

## Status

- [ ] Phase 1 — Road graph acquired and cleaned for study area
- [ ] Phase 2 — Flood-risk and traffic layers built
- [ ] Phase 3 — Baselines (static + dynamic Dijkstra) working
- [ ] Phase 4 — Custom Gymnasium environment validated
- [ ] Phase 5 — DQN sanity check on small subgraph
- [ ] Phase 5 — PPO trained on full study-area graph
- [ ] Phase 6 — Evaluation against baselines complete
- [ ] Phase 7 — Write-up

Update this checklist as you go — it's the fastest way to see where the
project actually stands without re-reading everything.

## Setup

```bash
conda env create -f environment.yml
conda activate floodroute-accra
# or: pip install -r requirements.txt
```

Copy `config/config.example.yaml` to `config/config.yaml` and fill in your
study-area bounding box and any API keys. `config/config.yaml` is gitignored
— never commit real API keys.

## Project layout

- `data/` — raw downloads (`raw/`) and reproducible derived data (`processed/`)
- `src/data_pipeline/` — scripts that turn `raw/` into `processed/`
- `src/env/` — the Gymnasium environment the RL agent trains against
- `src/baselines/` — classical shortest-path comparators
- `src/agents/` — SB3 training scripts
- `src/evaluation/` — scenario generation, metrics, comparison
- `notebooks/` — exploration only; nothing here is load-bearing
- `docs/` — proposal, design-decision log, known limitations

## Design principle

`data/processed/` should always be regeneratable by re-running the
`data_pipeline/` scripts against `data/raw/`. Don't hand-edit processed files.
