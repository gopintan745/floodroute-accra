# Design Decisions Log

Running log of non-obvious choices and why they were made. Add an entry
whenever you make a judgment call that isn't forced by the data or the
code — these are exactly the things you'll forget the reasoning for in
three months.

Template:

```text
## YYYY-MM-DD — Short title

**Decision:**

**Alternatives considered:**

**Why this one:**

**What would change my mind:**
```

---

## 2026-09-18 — Study area: Kaneshie market corridor

**Decision:** Use a bounding box centered on Kaneshie market and lorry
station (`(5.548, -0.245, 5.575, -0.210)`), covering Graphic Road, Ring
Road West, and surrounding streets.

**Alternatives considered:** Circle–Agbogbloshie–Odaw corridor, Dansoman.

**Why this one:** Kaneshie market was submerged and forced to shut down
during the 2015 Accra floods (well-documented, citable event), and the
area is also a major transport/congestion hub (trotro and regional bus
terminal) — giving both flooding and traffic relevance in one bounded,
solo-tractable region.

**What would change my mind:** If Phase 1 cleanup reveals the OSM graph
for this bbox is too sparse/disconnected to be usable, or if the area
turns out too small to have meaningfully different alternate routes for
the RL agent to choose between.

## 2026-09-18 — Flood risk computation from DEM + rainfall (heuristic proxy)

**Decision:** Compute flood risk as a weighted combination of:

1. **Topographic Wetness Index (TWI) proxy** (weight 0.6): A simplified TWI = ln(a/tan(β)) where specific catchment area `a` is approximated by inverse slope (flatter areas accumulate more water) and slope `β` is computed from DEM via Sobel filter. Combined with elevation percentile (lower = higher risk).
2. **Rainfall intensity** (weight 0.4): CHIRPS/GPM daily rainfall resampled to DEM grid, normalized via sigmoid around 50mm/day flood-triggering threshold.

Final risk per cell ∈ [0, 1], then sampled along road edges (mean of points every ~10m).

**Alternatives considered:**

- Full hydrological modeling (TAUDEM, PCRaster) — too complex for v1, requires drainage network.
- Simple elevation threshold only — misses flat low-lying areas that aren't absolute minima.
- Rainfall-only — misses persistent topographic risk.

**Why this one:**

- Topography is static and reliable (SRTM/Copernicus DEM), rainfall is noisier satellite proxy → weight topography higher.
- TWI proxy captures "where water accumulates" better than elevation alone.
- 50mm/day threshold aligns with Ghana Meteorological Agency "heavy rain" classification.
- Sigmoid normalization gives smooth gradient rather than hard threshold.
- Mean sampling along edges preserves spatial variation within long segments.

**What would change my mind:**

- If validation against known flood extents (e.g., 2015 Accra flood maps) shows systematic bias.
- If higher-resolution DEM (LiDAR) becomes available — could do proper flow accumulation.
- If local drainage infrastructure data (culverts, drains) becomes available — would fundamentally change the model.

## 2026-09-18 — DEM source: OpenTopography Copernicus GLO-30 (COP30)

**Decision:** Fetch 30m Copernicus GLO-30 DEM from OpenTopography's Global Datasets API (demtype=COP30) for the study area bbox.

**Alternatives considered:**

- SRTM 30m (via OpenTopography or USGS EarthExplorer) — older, void-filled, less accurate in urban areas
- NASADEM — improved SRTM but still 2000-era radar
- Local LiDAR — not available for Accra at open access
- ASTER GDEM — 30m but noisier, known artifacts

**Why this one:**

- COP30 (2020, 30m) is the best freely available global DEM — newer than SRTM, better vertical accuracy, fewer voids, better urban/forest penetration
- OpenTopography API provides easy programmatic access with bbox clipping (no manual tile stitching)
- Free personal API key, generous rate limits for research use
- Output is cloud-optimized GeoTIFF ready for rasterio

**What would change my mind:**

- If COP30 coverage gaps appear in the study area (unlikely for Accra)
- If a higher-resolution (e.g., 12m TanDEM-X, 5m LiDAR) becomes available for Accra
- If OpenTopography API becomes unreliable or rate-limited — would fall back to local SRTM mirror

## 2026-09-18 — DEM fetching pipeline

**Decision:** Separate DEM fetching (fetch_dem.py) from flood-risk computation (build_flood_layer.py). fetch_dem.py writes raw GeoTIFF to data/raw/dem/; build_flood_layer.py auto-detects and loads it.

**Alternatives considered:**

- Single script that fetches and processes in one run
- Manual download + place in data/raw/dem/

**Why this one:**

- Separation of concerns: data acquisition (needs network, API key) vs. processing (needs compute, no auth)
- Reproducibility: fetch_dem.py is idempotent (re-running just overwrites raw file); build_flood_layer.py is deterministic given the same raw input
- Flexibility: can swap DEM sources (COP30 vs SRTM vs local) without touching flood-risk logic
- Audit trail: raw/ holds the exact artifact used; processed/ holds derived products

**What would change my mind:**

- If we move to a workflow system (Airflow, Prefect) where separation is handled differently

## 2026-09-18 — Rainfall source: Open-Meteo Historical Weather API (point time series)

**Decision:** Use Open-Meteo's free Historical Weather (archive) API for daily precipitation at the study area centroid. No API key required. Returns ERA5-reanalysis daily time series (one point, not a raster).

**Alternatives considered:**

- CHIRPS (satellite, 0.05° grid, daily) — requires NetCDF download, more complex
- NASA GPM (satellite, 0.1° grid, half-hourly) — similar complexity
- Ghana Meteorological Agency ground stations — not programmatically accessible

**Why this one:**

- Free, no authentication, simple REST API
- Study area is small (~3km across) — a single centroid point is representative
- ERA5 reanalysis is high-quality for this region
- Returns daily time series directly; easy to aggregate (mean, max, percentile) for flood-risk heuristic
- Date range configurable in config.yaml

**Important constraint:** Open-Meteo REQUIRES both start_date AND end_date. If end_date is omitted, it returns "Bad Request". The script now defaults end_date to yesterday (most recent available archive day) when null in config.

**What would change my mind:**

- If spatial variation in rainfall across the bbox proves significant (e.g., orographic effects)
- If a gridded satellite product (CHIRPS/GPM) becomes easier to integrate programmatically
- If we need sub-daily rainfall resolution for flash-flood modeling

## 2026-09-18 — Open-Meteo rainfall processing: 95th percentile → uniform risk field

**Decision:** For the flood-risk heuristic, compute the 95th percentile of daily precipitation from the Open-Meteo time series, normalize via sigmoid around 50mm/day, and broadcast as a uniform field across the DEM grid.

**Alternatives considered:**

- Mean daily precipitation — too low, under-represents flood-triggering extremes
- Max daily precipitation — too noisy, single outlier dominates
- Multiple points across bbox — API doesn't support multi-point; could make N calls but study area is small
- Temporal aggregation by season — adds complexity, not needed for v1 heuristic

**Why this one:**

- 95th percentile is a standard "extreme event" metric in hydrology; captures heavy-rain tail without being a single outlier
- Sigmoid normalization (50mm/day threshold, 20mm scale) gives smooth 0-1 risk gradient
- Uniform field is intentional: for a ~3km bbox, spatial variation in daily rainfall is negligible compared to topographic variation
- Keeps the pipeline simple: point time series → single scalar → uniform raster → same sampling logic as gridded products

**What would change my mind:**

- If validation shows rainfall varies significantly within the bbox (e.g., coastal vs inland)
- If we switch to a gridded product (CHIRPS) — would then use the full spatial field
- If flood-risk model needs temporal dynamics (e.g., antecedent moisture) — would need time dimension in risk layer

## 2026-09-18 — Limitation: Open-Meteo point time series lacks spatial rainfall variation

**Decision:** Document as a known limitation that using a single centroid point from Open-Meteo ERA5 reanalysis means the flood-risk layer has no spatial variation in the rainfall component — it's a uniform field across the entire DEM grid.

**Why this matters:**

- Real rainfall varies spatially even at ~3km scales (coastal vs inland, orographic effects, convective cell patterns)
- The uniform assumption means flood risk differences across the study area come ONLY from topography (TWI + elevation)
- This could miss flood-prone areas that are topographically moderate but receive higher rainfall
- ERA5 reanalysis itself has ~30km native resolution — the point value is already interpolated from a coarse grid

**Mitigations in current design:**

- Topography weight (0.6) > rainfall weight (0.4) — persistent topographic risk dominates
- For flash floods in this region, topography (low-lying areas, drainage paths) is often the primary driver
- The study area is small (~3km); spatial rainfall gradients are typically secondary to topographic ones

**What would change my mind / future improvements:*s*

- If validation against historical flood extents shows systematic spatial bias
- Switch to gridded product (CHIRPS 0.05° ≈ 5.5km, or IMERG 0.1° ≈ 11km) — would provide spatial variation
- If we expand study area beyond ~5km radius — spatial rainfall variation becomes non-negligible
- Could implement multi-point querying (N calls to Open-Meteo) to build a coarse spatial field

## 2026-09-19 — Rainfall climatology: monthly heavy-rain-day frequency

**Decision:** Add a rainfall climatology output (`data/processed/rainfall_climatology.json`) computed from the Open-Meteo daily time series. The climatology provides a 12-month array of heavy-rain-day frequency (probability that a random day in that month exceeds 50mm), plus monthly mean/max precipitation and overall statistics.

**Alternatives considered:**

- Embed climatology computation inside the RL environment at runtime — adds dependency on raw data at training time, and recomputes the same thing repeatedly.
- Use a static seasonal multiplier (e.g., "rainy season = 2x flood risk") — too coarse, doesn't capture the actual frequency distribution from the historical record.
- No climatology; just use the single 95th-percentile scalar from the full record — loses the seasonal signal that's critical for Accra (major vs. minor rainy seasons).

**Why this one:**

- The Open-Meteo record spans 2015–present (~11 years), enough for a meaningful monthly climatology.
- Accra has a bimodal rainy season (major: April–July, minor: September–October); the monthly frequency captures this naturally (e.g., May/Jun/Jul/Oct show ~0.3% heavy-rain-day frequency vs. 0% in dry months).
- The output is a standalone JSON — easy for the RL environment to load at episode initialization to sample flood-event probabilities conditioned on month.
- Separates "climatology" (long-term frequency) from "weather" (realized flood event in an episode) — the RL agent learns to condition on the climatology prior, not the single historical realization.

**What would change my mind:**

- If the record length is insufficient for stable monthly estimates (current ~4278 days gives ~356 days/month, marginal but usable).
- If we switch to a gridded product (CHIRPS) with longer record (1981–present) — would recompute with that.
- If the RL environment needs sub-monthly resolution (e.g., weekly) — could extend the output format.

---

## 2026-09-19 — Flood risk layer now auto-generates climatology alongside raster

**Decision:** `build_flood_layer.py` now computes and saves `rainfall_climatology.json` by default during a full run (and offers `--climatology-only` for fast standalone generation).

**Alternatives considered:**

- Separate script for climatology — adds maintenance burden; the logic is small and shares the rainfall loading code.
- Compute climatology on-the-fly in the RL env — would require raw data access at training time.

**Why this one:**

- Single source of truth: the same rainfall loading logic (Open-Meteo JSON parsing, 50mm threshold) is used for both the flood-risk raster and the climatology.
- Reproducibility: climatology is a derived product of the exact same rainfall file used for flood risk.
- `--climatology-only` flag enables fast iteration on climatology without waiting for DEM processing.

**What would change my mind:**

- If climatology computation becomes complex enough to warrant its own module (e.g., multi-source blending, uncertainty quantification).
- If we adopt a workflow orchestrator that prefers separate tasks.

---

## 2026-09-19 — Config/schema drift between config.yaml and build_graph_costs.py — RESOLVED

**Decision:** Document and resolve several mismatches between the current `config.yaml` and what `build_graph_costs.py` / `build_flood_layer.py` actually expect.

### Issues identified

| Config key in YAML | Expected by code | Status |
| --- | --- | --- |
| `road_quality.highway_base_scores` | `road_quality.highway_fallback_scores` | **Mismatch** — code will KeyError |
| `road_quality.surface_multipliers` | `road_quality.surface_scores` | **Mismatch** — code will KeyError |
| `road_quality.default_highway_score` | `road_quality.default_score` | **Mismatch** — code will KeyError |
| `road_quality.smoothness_blend_weight` | `road_quality.smoothness_weight` | **Mismatch** — code will KeyError |
| `flood_risk.heavy_rain_threshold_mm: 20` | `FLOOD_THRESHOLD_MM = 50.0` (hardcoded in `build_flood_layer.py`) | **Inconsistent** — climatology uses 50mm, flood layer uses 20mm |
| `traffic_profile` section | Not used by `build_graph_costs.py` (uses `traffic_synthetic` instead) | **Dead config** — two parallel traffic configs exist |
| `traffic_synthetic` section | Used by `build_graph_costs.py` | **Missing from `config.example.yaml`** — tracked template incomplete |

### Root cause

The config evolved in two parallel branches:

1. `flood_risk` + `road_quality` + `traffic_profile` — designed for a different (never-built) pipeline
2. `traffic_synthetic` — designed for the actual `build_graph_costs.py` implementation

The example template (`config.example.yaml`) was never updated to match either branch.

### Resolution implemented

1. **Updated `road_quality_score()` in `build_graph_costs.py`** to read the YAML keys as they exist in config.yaml (`highway_base_scores`, `surface_multipliers`, `default_highway_score`, `smoothness_blend_weight`), with `.get()` fallbacks for safety.
2. **Made `build_flood_layer.py` read threshold from config** (`flood_risk.heavy_rain_threshold_mm`) instead of hardcoding 50mm. Default fallback remains 50mm if key missing.
3. **Updated `config.example.yaml`** to match the actual config.yaml — now includes `flood_risk`, `road_quality`, and `traffic_synthetic` sections.
4. **Left `traffic_profile` in config.yaml** (harmless dead config) but documented it as unused. Can be removed in a future cleanup pass.

### Verification

- `build_flood_layer.py --climatology-only` now uses 20mm threshold (from config) → 54 heavy rain days (1.3%) vs. 6 at 50mm.
- `build_graph_costs.py --synthetic-only` runs without KeyError, produces `road_graph_full.graphml` and `traffic_profile.parquet` (528 synthetic rows).

### What would change my mind

- If we decide to refactor the whole config schema at once (e.g., move to Pydantic models with validation) — then fix all at once rather than piecemeal.

---

## 2026-09-19 — Real traffic blending uses midpoint snap-to-graph (known limitation)

**Decision:** `_load_real_traffic_samples()` in `build_graph_costs.py` maps each traffic sample segment to a graph edge by snapping the segment's midpoint to the nearest edge via `ox.distance.nearest_edges`, then borrowing that edge's `highway_class`.

**Why this matters:**

- The 5 predefined segments in `sample_traffic.py` (Kaneshie First Light, Graphic Road, etc.) are arbitrary OD pairs, not OSM edges — they may cross multiple road classes.
- Midpoint snap assigns **one** highway class per segment, losing within-segment variation (e.g., a segment crossing from primary → residential).
- If the snapped edge is not representative of the segment's dominant character, the real multiplier gets attributed to the wrong class bucket.
- This is acceptable for v1 given small sample size (~5 segments × N days), but introduces label noise.

**Alternatives considered:**

- Sample multiple points along each segment and take majority class — more robust but adds complexity.
- Define segments as exact OSM edge sequences instead of arbitrary OD pairs — would require manual mapping effort.
- Use segment length-weighted average of classes along the route — requires route geometry from Mapbox, not just OD pairs.

**What would change my mind:**

- If real sample count grows large enough that misattribution noise becomes the limiting factor.
- If we switch to defining segments as explicit OSM edge sequences (e.g., from the cleaned graph itself).

---

## 2026-09-19 — Traffic profile blending: real overrides synthetic per (class, hour, weekend) bucket

**Decision:** `blend_traffic_profiles()` does a left-join merge where real samples **replace** the synthetic multiplier for matching (highway_class, hour, is_weekend) buckets; synthetic fills all other buckets. The `num_real_samples` column tracks evidence count (0 = synthetic-only).

**Alternatives considered:**

- Weighted average by sample count — smoother but dilutes real signal with synthetic prior.
- Bayesian update with synthetic as prior — statistically cleaner but adds complexity (need uncertainty estimates).
- Keep both separate, let RL env choose at runtime — more flexible but pushes complexity downstream.

**Why this one:**

- Simple, deterministic, auditable: "real data wins where it exists."
- `num_real_samples = 0` rows are explicitly flagged — downstream can apply confidence weighting if desired.
- Works even with N=1 real sample per bucket (no averaging needed).

**What would change my mind:**

- If real samples are very sparse (e.g., only 1–2 per bucket) and noisy — then a weighted blend or Bayesian shrink would be better.
- If we get enough real data to estimate per-bucket confidence intervals.

---

## 2026-09-19 — Shared cost model for RL and baselines (Phase 3)

**Decision:** Create `src/common/cost_model.py` with a single `effective_travel_time()` function used by both the RL environment (reward calculation) and classical baselines (Dijkstra/A* edge weights). Add `cost_model` section to config.yaml with `flood_weight` and `quality_penalty_weight`.

**Formula:**

```python
effective_travel_time = base_travel_time * traffic_multiplier * quality_multiplier * flood_multiplier
where
quality_multiplier = 1 + (1 - road_quality_score) * quality_penalty_weight
and
flood_multiplier   = 1 + flood_weight * flood_susceptibility
```

**Config defaults:** `flood_weight: 0.5`, `quality_penalty_weight: 1.0`

**Alternatives considered:**

- Separate cost functions for RL (reward) vs. baselines (edge weight) — allows tuning each independently but breaks comparability; "apples-to-apples" comparison requires identical cost model.
- Additive penalties instead of multiplicative — simpler but doesn't scale with base travel time (a 5-min penalty on a 1-min edge vs. 60-min edge has very different meaning).
- Exponential flood penalty (e.g., `exp(flood_weight * risk)`) — more aggressive on high-risk edges but harder to calibrate and explain.

**Why this one:**

- Multiplicative form naturally scales penalties with base travel time (a 10-min detour on a 60-min highway is proportional).
- Single shared function guarantees the RL agent is optimizing the exact same objective the baselines use — any performance difference is due to policy quality, not cost-model mismatch.
- Two interpretable weights (`flood_weight`, `quality_penalty_weight`) map directly to config; easy to sweep in sensitivity analysis.
- Pure, stateless function — trivial to test, serialize, and reason about.
- `CostModelConfig` dataclass with `from_yaml()` enables clean dependency injection in both RL env and baselines.

**What would change my mind:**

- If validation shows the multiplicative form creates pathological routing (e.g., completely avoiding all flood-prone edges even when detour is massive) — might need a saturating function like `1 + weight * risk / (1 + risk)`.
- If we add more hazard dimensions (e.g., landslide risk, security risk) — the function signature would grow; could refactor to a `HazardWeights` dict.

---

## 2026-09-19 — Cost model config: flood_weight=0.5, quality_penalty_weight=1.0

**Decision:** Set default weights to `flood_weight=0.5` and `quality_penalty_weight=1.0` in config.yaml.

**Interpretation:**

- `flood_weight=0.5`: A maximally flood-susceptible edge (score=1.0) incurs +50% travel time vs. zero-risk edge. This is a moderate penalty — flood risk increases cost but doesn't dominate routing unless risk is very high.
- `quality_penalty_weight=1.0`: A worst-quality edge (score=0.0) incurs 2x travel time vs. best quality (score=1.0). This is a strong penalty — road quality has a larger marginal effect than flood risk at these defaults.

**Rationale:** Road quality is a persistent, always-present signal (every edge has a score), while flood risk is episodic (only relevant during/after heavy rain). A stronger quality weight ensures the baseline routing prefers good roads even in dry conditions, while flood weight acts as a conditional amplifier during flood events.

**What would change my mind:**

- If real-world validation (e.g., driver surveys, GPS traces) shows different relative importance.
- If the flood risk scores turn out to be systematically over/under-estimated — would adjust weight to compensate.
- If we want to run ablation studies — the config structure makes it trivial to sweep weights.

---

## 2026-09-19 — Baseline 1: Static Dijkstra (free-flow only)

**Decision:** Implement `src/baselines/dijkstra_static.py` with `static_shortest_path()` using only the free-flow `travel_time` edge attribute as weight. This is the "naive" comparator — what a standard map app without live traffic/flood awareness would produce.

**Key properties:**

- Uses `ox.distance.nearest_nodes` to snap (lon, lat) to graph nodes
- Runs `nx.shortest_path(graph, orig, dest, weight="travel_time")`
- Returns path node list, total free-flow time, path length, edge count
- Handles disconnected graphs (returns `inf` travel time)
- Includes `evaluate_route()` for post-hoc evaluation against HazardSimulator's true episode conditions (flood penalties, actual travel time with traffic)

**Alternatives considered:**

- Use expected-traffic weights (synthetic profile average) instead of free-flow — would be a "slightly informed" baseline, but the proposal specifically calls for "naive" as Baseline 1.
- A* with haversine heuristic — faster on large graphs but this study area is small (~2300 nodes); Dijkstra is fast enough and simpler.
- Include flood_risk/road_quality in static weight — that would be Baseline 2 (dynamic Dijkstra), not the naive baseline.

**Why this one:**

- Matches the proposal's "static and dynamically-weighted shortest-path baselines" — this is the static one.
- Pure free-flow is the absolute minimum information baseline; any improvement over it demonstrates value of hazard awareness.
- Simple, well-understood, reproducible — serves as a solid anchor for comparison.

**What would change my mind:**

- If the study area expands significantly (city-wide) and Dijkstra becomes slow — would switch to A*.
- If the proposal's evaluation framework changes to require a different "naive" definition.

---

## 2026-09-19 — Static baseline evaluation against HazardSimulator

**Decision:** The `evaluate_route()` function in `dijkstra_static.py` evaluates a fixed pre-computed route against the episode's true conditions (from HazardSimulator), applying flood penalties and traffic multipliers post-hoc. The route does NOT adapt — that's the point.

**Metrics returned:**

- `actual_travel_time`: base * traffic_multiplier + flood penalties
- `flood_penalties`: total seconds added for flooded edges (default 50 min = 3000s per edge)
- `edges_hit_flooded`: count of flooded edges on the route
- `success`: True if zero flooded edges hit
- `base_travel_time`: the original free-flow time (for comparison)

**Why this matters:**

- The static baseline's "planned" time (free-flow) vs. "actual" time (with penalties) quantifies the cost of ignorance.
- RL agent and dynamic baselines can adapt mid-route; static cannot. This difference IS the research question.

---

## 2026-09-19 — Baseline 2: Dynamic Dijkstra (pre-planned with hazard-aware costs)

**Decision:** Implement `src/baselines/dijkstra_dynamic.py` with three functions:

1. `build_traffic_lookup(traffic_profile)` — converts DataFrame to O(1) dict
2. `annotate_dynamic_costs(graph, traffic_lookup, hour, is_weekend, cost_cfg, flood_event_edges)` — mutates graph edges with `dynamic_cost`
3. `dynamic_shortest_path(graph, traffic_profile, origin, destination, hour, is_weekend, cost_cfg, flood_event_edges)` — main entry point

**Key properties:**

- Uses shared `cost_model.effective_travel_time()` for apples-to-apples comparison with RL agent
- Precomputes `dynamic_cost` on all edges, then runs standard Dijkstra (avoids networkx callable pitfalls on MultiDiGraph)
- Returns both `total_dynamic_cost` (hazard-aware) AND `total_base_travel_time` (free-flow) — enables separating route choice quality from time estimation
- Supports episode-specific flood events via `flood_event_edges` set (overrides static flood_risk for those edges)
- Falls back to 1.0 traffic multiplier with logged warning if (highway_class, hour, weekend) combo missing from profile

**Alternatives considered:**

- Pass callable to `nx.shortest_path(weight=callable)` — networkx's MultiDiGraph signature `((u, v, d) where d is dict of parallel edges)` is error-prone and hard to debug
- Keep separate functions for "pre-planned" vs "replanning" — currently only pre-planned is implemented; replanning stub exists for Phase 4
- Include road quality in traffic profile — decided to keep quality as static per-edge attribute, only traffic varies by time

**Why this one:**

- Clean separation: `build_traffic_lookup` (data prep), `annotate_dynamic_costs` (graph mutation), `dynamic_shortest_path` (orchestration) — each testable independently
- Same cost model as RL reward → fair comparison
- Reports both dynamic and base costs → later evaluation can ask "did the route avoid hazards or just take a different path?"

**What would change my mind:**

- If graph mutation becomes problematic (e.g., concurrent runs) — would switch to copying graph or using edge attributes dict
- If traffic profile grows large enough that O(1) dict memory matters — unlikely for this scale

---

## 2026-09-19 — Dynamic Dijkstra returns both dynamic and base costs

**Decision:** `dynamic_shortest_path()` returns both `total_dynamic_cost` (hazard-aware estimate) and `total_base_travel_time` (free-flow sum along the chosen path).

**Why this matters:**

- The dynamic cost is what the planner *thinks* the trip will take (used for route selection)
- The base travel time is the actual distance/free-flow time of that route (useful for comparison)
- The ratio `dynamic_cost / base_travel_time` reveals how much the planner expects hazards to slow this specific route
- In evaluation, we can compare:
  - Static baseline's base vs. actual (cost of ignorance)
  - Dynamic baseline's predicted dynamic vs. actual (quality of prediction)
  - Dynamic baseline's chosen route vs. static's route (value of hazard-aware planning)

**What would change my mind:**

- If we need more granular breakdown (traffic component vs. flood component vs. quality component) — could add `cost_breakdown` dict to return value.

---

## 2026-09-20 - Flood realization and evaluation sampling

**Decision:** Train-mode episodes oversample flood days at 35%, while
eval-mode episodes use the month-specific climatology rate. Months are sampled
uniformly in both modes, and flooding is an additive penalty rather than an
impassable edge.

**Alternatives considered:** Use the climatology rate during training, weight
month selection by climatology, or block movement across flooded edges.

**Why this one:** Oversampling gives the agent enough flood exposure to learn
avoidance; using the true climatology in evaluation preserves an honest
performance estimate. Uniform month sampling avoids double-counting
seasonality, and additive penalties avoid stuck episodes when every remaining
route is flooded.

**Consequence:** Training and evaluation rewards are not directly comparable;
only eval-mode results are project performance. Future
`evaluation/run_scenarios.py` must instantiate `HazardSimulator(mode="eval")`.

**What would change my mind:** Evaluation distributions that remain badly
calibrated after validation, or evidence that additive penalties fail to
produce useful avoidance behavior.

---

## 2026-09-20 - Local flood observability radius

**Decision:** The simulator exposes realized flood state only for edges whose
source or destination node is within `env.reveal_radius_hops` undirected graph
hops of the driver's current node. The environment must call
`set_current_position()` after each movement before querying local hazards.

**Alternatives considered:** Reveal only the edge being traversed, expose the
whole realized flood map, or return `False` for edges outside the local view.

**Why this one:** Drivers can observe flooding on nearby streets before
committing to them, but distant edges remain uncertain. `is_flooded()` returns
`True` or `False` for visible edges and `None` for unrevealed edges, preserving
the distinction between a known dry edge and an edge whose realized condition
is still hidden. Undirected graph distance models local visibility across an
intersection regardless of travel direction.

**What would change my mind:** Evidence that a fixed hop radius poorly models
the observation distance, in which case a metric or road-class-aware radius
could replace it.

---

## TODO: next entries

Example candidates for your next few entries (fill in once decided):

- Fixed max-degree action space vs. candidate-node-list action encoding
- How road-quality labels were produced given no systematic dataset
- Reward weighting between travel time and flood penalty
