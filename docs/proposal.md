# FloodRoute-Accra: Adaptive Route Learning Under Traffic, Flooding, and Road-Quality Uncertainty

A solo research project proposal and feasibility study

---

## 1. Problem Statement

Accra's road network experiences highly variable travel times due to three compounding, poorly-instrumented hazards: dense traffic congestion, seasonal flash flooding that intermittently closes or slows road segments, and inconsistent road surface quality. Existing navigation tools (e.g., Google Maps) model congestion reasonably well in data-rich cities, but they do not explicitly model flood risk propagation or road-quality degradation, and Accra itself has very limited official sensor or historical traffic data to draw on.

This creates a genuine sequential decision-making problem, not just a static shortest-path problem: a driver (or agent) choosing a route must act under **partial observability** — it doesn't know in advance which segments will be flooded, congested, or degraded at the moment of travel — and may need to revise its route mid-journey as new information arrives (a road turns out to be flooded, traffic builds up unexpectedly). This is precisely the setting where a learned, adaptive policy can outperform a "predict conditions once, then run static shortest-path" pipeline.

**Research question:** Can a reinforcement learning agent, trained on a graph representation of Accra's road network with proxy signals for traffic, flood risk, and road quality, learn a routing policy that is more robust (fewer failed/blocked routes, lower expected travel time under uncertainty) than classical shortest-path baselines (Dijkstra/A*) using the same information?

---

## 2. Project Objectives

**Primary objective:** Build and evaluate an RL-based routing agent for a bounded sub-region of Accra that outperforms static and dynamically-weighted shortest-path baselines under simulated flooding and congestion scenarios.

**Secondary objectives:**

- Produce a reusable pipeline for turning open geospatial data (OpenStreetMap, elevation, rainfall) into a routable graph with hazard-aware edge costs — useful beyond this one project, for other data-scarce cities.
- Quantify how much a partial-observability / sequential-decision framing actually buys you over a simpler "predict weights, then search" baseline. This is a real open question, not a foregone conclusion — the honest answer might be "not much, for this scope," and that's a valid, useful finding.

**Explicitly out of scope (for a solo, no-deadline project):**

- Live production deployment or a real-time mobile app.
- City-wide coverage of all of Accra — this is a scope-killer for one person.
- Building original sensor infrastructure. Everything must use existing/open/proxy data.

---

## 3. Proposed MDP Formulation

This is the most important design decision in the project, so it's worth being explicit about it up front rather than discovering it three months in.

| Element | Definition |
| --- | --- |
| **State** | Current node (intersection), destination node, locally observable conditions on edges adjacent to the current node (traffic level, flood risk, road quality), and optionally time-of-day / recent rainfall as context |
| **Action** | Choose one outgoing edge (road segment) from the current node |
| **Reward** | Negative travel time for the edge taken; large penalty if the edge turns out to be flooded/impassable (revealed only on arrival — this is what makes it partially observable); bonus on reaching the destination; small step penalty to discourage wandering |
| **Episode** | One trip from a sampled origin to a sampled destination within the study area |
| **Transition dynamics** | Conditions on unvisited edges are stochastic and partially hidden; conditions can also *change* within an episode (e.g., a flash flood event triggered mid-episode) to simulate real-world non-stationarity |

This framing is what justifies using RL (and specifically actor-critic methods with a discrete action head, as discussed earlier) rather than just Dijkstra on predicted weights — the agent has to learn to hedge against uncertainty and adapt, not just optimize against a fixed prediction.

---

## 4. Feasibility Study

### 4.1 Data feasibility — the central risk

This is the part most likely to make or break the project, so it gets the most scrutiny.

| Data need | Source | Feasibility |
| --- | --- | --- |
| Road network graph | OpenStreetMap via OSMnx | **High**, but expect quality gaps — Accra's OSM coverage is decent in central areas, weaker in peripheral/informal areas. Budget time for manual graph cleaning. |
| Elevation / flood susceptibility | SRTM 30m DEM (free, USGS EarthExplorer or Copernicus DEM) | **High.** Precedent exists — earlier DEM-based flood modeling has been done specifically for Accra. |
| Rainfall / flood trigger events | CHIRPS or NASA GPM satellite rainfall (free, global, daily) | **High** for satellite proxies; official Ghana Meteorological Agency data would be better but harder to access as an individual. |
| Traffic conditions | Google Maps Directions API (sampled duration_in_traffic over time) | **Medium.** Feasible at hobby/research scale within free-tier API limits if sampled sparingly (e.g., a few times a day over weeks) for a bounded sub-region. This mirrors the approach used in a comparable Kampala, Uganda study. |
| Road quality / surface condition | No systematic dataset. Would need manual labeling from satellite imagery, Mapillary street-level images, or your own judgment for a sub-region | **Low-Medium.** This is the weakest link — plan to approximate this with a coarse manual or heuristic layer rather than a learned one, at least initially. |
| Ground-truth validation | Sparse. News reports of flooded roads, personal/local knowledge, anecdotal reports | **Low.** Validation will necessarily be partial. Be upfront in any write-up that this is a simulation study, not a validated deployment. |

**Bottom line:** full real-time, validated data is not feasible for one person. A **hybrid real-and-synthetic environment** is the realistic path: real road graph and real elevation/rainfall-derived flood risk, combined with a synthetic/simulated traffic and flood-event generator calibrated loosely against whatever real samples you can gather. This is a legitimate and common approach in this exact research niche — it's what related developing-city studies have done given the same data constraints.

### 4.2 Technical feasibility

- **Skills needed:** Python RL (Gymnasium + Stable-Baselines3 for PPO), graph tooling (NetworkX, OSMnx), basic GIS (rasterio/QGIS for DEM work), and enough Actor-Critic / DQN understanding to debug training — all of which you're already building context on in this conversation.
- **Compute:** Modest. A sub-region graph (a few hundred to low-thousands of nodes) with a discrete PPO agent trains fine on a laptop or free-tier Colab GPU. This is not a distributed-training-scale project.
- **Verdict: feasible**, with the main cost being *learning time* across RL, GIS, and simulation-design skills simultaneously — which is exactly why "no deadline" matters here.

### 4.3 Scope feasibility for a solo, no-deadline contributor

Realistic, **not** feasible: city-wide Accra, real-time deployment, validated against real driver outcomes.

Realistic, **feasible**: a bounded sub-region (e.g., one district, or a corridor known for both flooding and congestion — Circle, Kaneshie, or a section of the Odaw river floodplain would be thematically strong choices given documented flood impact), evaluated in simulation against classical baselines, with a clearly scoped and honestly-labeled set of limitations.

### 4.4 Risk summary

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Data scarcity undermines realism | High | Lean into hybrid real+synthetic design from the start; treat this as a stated limitation, not a flaw to hide |
| RL agent doesn't beat baselines meaningfully | Medium | This is a legitimate possible finding, not a failure — frame the project around answering the question, not proving a预 predetermined outcome |
| Scope creep (wanting all of Accra) | High for a solo project | Commit early to a bounded sub-region; expand only after the MVP works |
| Getting stuck between GIS work and RL work | Medium | Sequence the phases (below) so each is a complete, usable checkpoint before moving to the next |

---

## 5. Phased Roadmap (no fixed dates — sequenced checkpoints)

1. **Environment groundwork:** Pull the OSM road graph for one bounded sub-region via OSMnx; clean and verify it manually against satellite imagery.
2. **Hazard layers:** Build a flood-risk layer from DEM + rainfall data; build a synthetic/sampled traffic layer (start synthetic, add real Google API samples opportunistically).
3. **Baseline first:** Implement Dijkstra/A* with (a) static weights and (b) dynamically-updated weights, *before* touching RL. This gives you a working, demoable artifact early and a concrete comparison target.
4. **Custom Gymnasium environment:** Encode the MDP from Section 3. Validate it with a trivial random-action agent and a greedy-heuristic agent before training anything real.
5. **Agent training, smallest-first:** Start with tabular Q-learning or DQN on a small subgraph to validate the reward design cheaply, then move to PPO (discrete action head) on the full sub-region graph, per the reasoning from our earlier discussion on discrete vs. continuous actor-critic methods.
6. **Evaluation:** Compare RL agent vs. both baselines across held-out origin-destination pairs and simulated flood-event scenarios. Metrics: expected travel time, failure rate (routes that hit a blocked/flooded edge), and variance across scenarios.
7. **Write-up and optional extension:** Document findings honestly, including where RL did *not* help. Natural extensions if you want to keep going: multi-agent congestion effects, or a simple interactive map demo of the trained policy.

---

## 6. Success Criteria

The project succeeds if it produces a working, evaluated comparison — regardless of which method wins. Concretely:

- A clean, documented pipeline from open data → routable hazard-aware graph.
- A working custom RL environment and at least one trained agent.
- A fair, quantified comparison against classical baselines on the same data.
- An honest account of where the approach helps, where it doesn't, and why — this is what makes it a genuine research contribution rather than a demo.
