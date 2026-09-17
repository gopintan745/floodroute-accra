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

## TODO: first entry

Example candidates for your first few entries (fill in once decided):

- How flood risk is computed from DEM + rainfall (what threshold/heuristic)
- Fixed max-degree action space vs. candidate-node-list action encoding
- How much of the graph is "locally observable" at each step (reveal radius)
- How road-quality labels were produced given no systematic dataset
- Reward weighting between travel time and flood penalty
