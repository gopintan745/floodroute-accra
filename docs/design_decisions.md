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

## TODO: next entries

Example candidates for your next few entries (fill in once decided):

- How flood risk is computed from DEM + rainfall (what threshold/heuristic)
- Fixed max-degree action space vs. candidate-node-list action encoding
- How much of the graph is "locally observable" at each step (reveal radius)
- How road-quality labels were produced given no systematic dataset
- Reward weighting between travel time and flood penalty
