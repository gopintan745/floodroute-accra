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

## TODO: next entries

Example candidates for your next few entries (fill in once decided):

- Fixed max-degree action space vs. candidate-node-list action encoding
- How much of the graph is "locally observable" at each step (reveal radius)
- How road-quality labels were produced given no systematic dataset
- Reward weighting between travel time and flood penalty
