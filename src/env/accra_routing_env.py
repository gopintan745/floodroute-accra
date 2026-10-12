"""
The core Gymnasium environment: a partially-observable sequential routing
task over the Accra study-area road graph.

This is the most important file in the project (see docs/proposal.md
Section 3 for the full MDP formulation). Get this right before training
anything.

State (per docs/proposal.md):
    - current node id
    - destination node id
    - locally observable conditions on edges adjacent to current node
      (traffic level, flood risk, road quality) — NOT the whole graph;
      this is what makes it partially observable
    - optionally: time-of-day / recent rainfall context

Action:
    - discrete choice among outgoing edges from the current node
      (variable action count per node — see note below)

Reward:
    - negative travel time for the edge taken
    - large penalty if the edge is revealed to be flooded/impassable
      on arrival
    - + bonus on reaching destination
    - small per-step penalty to discourage wandering

Episode:
    - one trip, sampled origin -> sampled destination within study area
    - conditions can change mid-episode via hazard_simulator.py, to test
      whether the agent adapts rather than just memorizing static weights

NOTE on variable action spaces: intersections have different numbers of
outgoing edges. Gymnasium's Discrete space assumes a fixed size, so you'll
need either (a) a fixed max-degree action space with invalid-action masking,
or (b) a different action encoding (e.g. selecting a target node from a
candidate list). Decide this early — it affects your SB3 policy choice.
Log the decision in docs/design_decisions.md.
"""

import math
import gymnasium as gym
import networkx as nx
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.common.cost_model import (
    CostModelConfig,
    base_edge_time,
    expected_cost,
    realized_cost,
)
from src.env.graph_wrapper import GraphWrapper
from src.env.hazard_simulator import HazardSimulator
import json
from pathlib import Path;


class AccraRoutingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        graph_path,
        traffic_profile_path: str,
        climatology_path: str,
        config: dict,
        mode: str,
    ):
        super().__init__()
        self.config = config or {}
        self.mode = mode
        if isinstance(graph_path, nx.Graph):
            self.graph = graph_path.copy()
        else:
            self.graph = nx.read_graphml(graph_path)
        self._reachable_destinations = {}
        env_cfg = self.config.get("env", {})
        self.graph_wrapper = GraphWrapper(self.graph, max_degree=env_cfg.get("max_degree"))
        self.max_degree = self.graph_wrapper.max_degree
        self.action_space = spaces.Discrete(self.max_degree)
        self._current_node = None
        self._destination_node = None
        self._origin_node = None
        self._step_count = 0
        self.visited_nodes: set = set()
        self.visit_counts: dict = {}  # node -> (visit_count, expected-cost snapshot)
        self.total_revisits = 0
        self.revisit_mode = env_cfg.get("revisit_mode", "terminate")
        if self.revisit_mode not in {"terminate", "penalty"}:
            raise ValueError(
                "env.revisit_mode must be 'terminate' or 'penalty', "
                f"got {self.revisit_mode!r}"
            )
        self.revisit_penalty = float(env_cfg.get("revisit_penalty", 50.0))
        self.revisit_penalty_growth = float(
            env_cfg.get("revisit_penalty_growth", 2.0)
        )
        if self.revisit_penalty < 0:
            raise ValueError("env.revisit_penalty must be non-negative")
        if self.revisit_penalty_growth < 1.0:
            raise ValueError("env.revisit_penalty_growth must be at least 1.0")

        self.traffic_profile = self._load_traffic_profile(traffic_profile_path)
        self.traffic_lookup = self._build_traffic_lookup(self.traffic_profile)
        self.climatology = self._load_climatology(climatology_path)
        self.cost_cfg = CostModelConfig.from_config(self.config)
        self.hazard_sim = HazardSimulator(
            self.graph,
            self.climatology,
            self.config,
            mode=mode,
            seed=0,
        )

        node_dim = 2
        goal_relative_dim = 3  # distance_m, sin(bearing), cos(bearing)
        edge_dim = 8
        self.observation_space = spaces.Dict(
            {
                "current_node_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(node_dim,),
                    dtype=np.float32,
                ),
                "destination_node_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(node_dim,),
                    dtype=np.float32,
                ),
                "goal_relative_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(goal_relative_dim,),
                    dtype=np.float32,
                ),
                "global_features": spaces.Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array([1.0, 1.0], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                ),
                "local_edge_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_degree, edge_dim),
                    dtype=np.float32,
                ),
            }
        )

    def _load_climatology(self, climatology_path: str) -> dict:
        path = Path(climatology_path)
        if not path.exists():
            return {"by_month": {m: {"heavy_day_fraction": 0.05} for m in range(1, 13)}}
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_traffic_profile(self, traffic_profile_path: str):
        path = Path(traffic_profile_path)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                return pd.DataFrame([])
        return pd.DataFrame([])

    @staticmethod
    def _build_traffic_lookup(profile: pd.DataFrame) -> dict:
        lookup: dict[tuple[str, int, bool], float] = {}
        if profile is None or profile.empty:
            return lookup
        for row in profile.to_dict(orient="records"):
            highway = str(row.get("highway_class", "unclassified"))
            hour = int(row.get("hour", 0))
            is_weekend = bool(row.get("is_weekend", False))
            multiplier = float(row.get("multiplier", 1.0))
            lookup[(highway, hour, is_weekend)] = multiplier
        return lookup

    def _lookup_traffic_multiplier(self, highway_class: str, hour: int, is_weekend: bool) -> float:
        if not self.traffic_lookup:
            return 1.0
        key = (str(highway_class), int(hour), bool(is_weekend))
        return float(self.traffic_lookup.get(key, self.traffic_lookup.get((str(highway_class), int(hour), False), 1.0)))

    def action_masks(self) -> list[bool]:
        """Exact method name expected by sb3-contrib MaskablePPO."""
        if self._current_node is None:
            return [False] * self.max_degree
        mask = self.graph_wrapper.action_mask(self._current_node)
        # MaskablePPO cannot sample from an all-false mask. A sink is handled
        # as a terminal failure in step(), so expose a dummy choice only there.
        return mask if any(mask) else [True] * self.max_degree

    def _expected_edge_cost(self, edge) -> float:
        u, v, k = edge
        data = self.graph.edges[u, v, k]
        traffic = self._lookup_traffic_multiplier(
            str(data.get("highway_class", data.get("highway", "unclassified"))),
            self.episode_context["hour"],
            self.episode_context["is_weekend"],
        )
        edge_time = base_edge_time(
            data.get("travel_time", 0.0),
            traffic,
            data.get("road_quality_score", 1.0),
            self.cost_cfg.quality_penalty_weight,
        )
        return expected_cost(edge_time, self.hazard_sim.belief(*edge), self.cost_cfg)

    def _local_cost_snapshot(self, node) -> dict:
        edges = self.graph_wrapper.neighbors_within_hops(
            node, self.hazard_sim.reveal_radius_hops
        )
        return {edge: self._expected_edge_cost(edge) for edge in edges}

    def _record_visit(self, node) -> tuple[int, dict]:
        """Record visit count and local expected-cost beliefs for revisit checks."""
        prior_count, _ = self.visit_counts.get(node, (0, {}))
        visit_count = prior_count + 1
        snapshot = self._local_cost_snapshot(node)
        self.visit_counts[node] = (visit_count, snapshot)
        self.visited_nodes.add(node)
        return visit_count, snapshot

    def _revisit_penalty_for(self, revisit_number: int) -> float:
        """Return the escalating penalty for the episode-wide revisit number."""
        return self.revisit_penalty * (
            self.revisit_penalty_growth ** (revisit_number - 1)
        )

    def _node_features(self, node) -> np.ndarray:
        node_data = self.graph.nodes[node]
        lon = float(node_data.get("x", 0.0))
        lat = float(node_data.get("y", 0.0))
        return np.array([lon, lat], dtype=np.float32)

    def _goal_relative_features(self) -> np.ndarray:
        """Return distance and continuous bearing from current node to goal."""
        current = self.graph.nodes[self._current_node]
        destination = self.graph.nodes[self._destination_node]
        current_lon = math.radians(float(current.get("x", 0.0)))
        current_lat = math.radians(float(current.get("y", 0.0)))
        destination_lon = math.radians(float(destination.get("x", 0.0)))
        destination_lat = math.radians(float(destination.get("y", 0.0)))

        delta_lat = destination_lat - current_lat
        delta_lon = destination_lon - current_lon
        haversine_a = (
            math.sin(delta_lat / 2.0) ** 2
            + math.cos(current_lat)
            * math.cos(destination_lat)
            * math.sin(delta_lon / 2.0) ** 2
        )
        distance_m = 6_371_000.0 * 2.0 * math.asin(
            min(1.0, math.sqrt(haversine_a))
        )
        bearing = math.atan2(
            math.sin(delta_lon) * math.cos(destination_lat),
            math.cos(current_lat) * math.sin(destination_lat)
            - math.sin(current_lat)
            * math.cos(destination_lat)
            * math.cos(delta_lon),
        )
        return np.array(
            [distance_m, math.sin(bearing), math.cos(bearing)],
            dtype=np.float32,
        )

    def _prior_cost_to_go(self) -> dict:
        """Compute prior-expected downstream costs once for this episode."""
        reverse_costs = nx.DiGraph()
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            traffic = self._lookup_traffic_multiplier(
                str(data.get("highway_class", data.get("highway", "unclassified"))),
                self.episode_context["hour"],
                self.episode_context["is_weekend"],
            )
            edge_time = base_edge_time(
                data.get("travel_time", 0.0),
                traffic,
                data.get("road_quality_score", 1.0),
                self.cost_cfg.quality_penalty_weight,
            )
            belief = self.hazard_sim.prior(u, v, key)
            cost = expected_cost(edge_time, belief, self.cost_cfg)
            if not reverse_costs.has_edge(v, u) or cost < reverse_costs[v][u]["weight"]:
                reverse_costs.add_edge(v, u, weight=cost)
        try:
            return nx.single_source_dijkstra_path_length(
                reverse_costs, self._destination_node, weight="weight"
            )
        except nx.NodeNotFound:
            return {}

    def _build_observation(self):
        current_features = self._node_features(self._current_node)
        destination_features = self._node_features(self._destination_node)
        edge_rows = np.zeros((self.max_degree, 8), dtype=np.float32)
        cost_to_go = []
        prior_cost_to_go = getattr(self, "_prior_cost_to_go_cache", {})
        for idx in range(self.max_degree):
            edge = self.graph_wrapper.action_to_edge(self._current_node, idx)
            if edge is None:
                cost_to_go.append(None)
                continue
            u, v, k = edge
            data = self.graph.edges[u, v, k]
            belief = self.hazard_sim.belief(*edge)
            traffic = self._lookup_traffic_multiplier(
                str(data.get("highway_class", data.get("highway", "unclassified"))),
                self.episode_context["hour"],
                self.episode_context["is_weekend"],
            )
            quality = float(data.get("road_quality_score", 1.0))
            base_time = float(data.get("travel_time", 0.0))
            adjusted_base = base_edge_time(
                base_time, traffic, quality, self.cost_cfg.quality_penalty_weight
            )
            via_cost = expected_cost(adjusted_base, belief, self.cost_cfg)
            downstream_cost = float(prior_cost_to_go.get(v, float("inf")))
            total_cost = via_cost + downstream_cost
            edge_rows[idx] = [
                *belief.tolist(), traffic, quality, base_time, total_cost, 0.0
            ]
            cost_to_go.append(total_cost)
        finite_costs = [
            value for value in cost_to_go
            if value is not None and math.isfinite(value)
        ]
        best_cost = min(finite_costs, default=0.0)
        worst_finite_cost = max(finite_costs, default=best_cost)
        for idx, value in enumerate(cost_to_go):
            if value is not None:
                safe_value = value if math.isfinite(value) else worst_finite_cost + 1.0
                edge_rows[idx, 6] = safe_value
                edge_rows[idx, 7] = max(0.0, safe_value - best_cost)
        return {
            "current_node_features": current_features,
            "destination_node_features": destination_features,
            "goal_relative_features": self._goal_relative_features(),
            "global_features": np.asarray(
                [float(self.episode_context["is_flood_day"]), self.episode_context["intensity"]],
                dtype=np.float32,
            ),
            "local_edge_features": edge_rows,
        }, {
            "origin": self._origin_node,
            "destination": self._destination_node,
            "month": self.episode_context["month"],
            "hour": self.episode_context["hour"],
            "is_weekend": self.episode_context["is_weekend"],
            "is_flood_day": self.episode_context["is_flood_day"],
            "intensity": self.episode_context["intensity"],
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.hazard_sim.rng = np.random.default_rng(seed)
        options = options or {}
        scenario = options.get("scenario", options)
        nodes = list(self.graph.nodes)
        if len(nodes) < 2:
            raise ValueError("At least two nodes are required for a routing episode")
        self._reachable_destinations = {
            node: tuple(destination for destination in nx.descendants(self.graph, node) if destination != node)
            for node in nodes
        }
        valid_origins = [node for node in nodes if self._reachable_destinations[node]]
        if not valid_origins:
            raise ValueError("Graph has no directed origin-destination pair")
        self._origin_node = scenario.get(
            "origin", valid_origins[int(rng.integers(len(valid_origins)))]
        )
        destinations = self._reachable_destinations.get(self._origin_node, ())
        if not destinations:
            raise ValueError(f"Scenario origin {self._origin_node!r} has no reachable destination")
        self._destination_node = scenario.get(
            "destination", destinations[int(rng.integers(len(destinations)))]
        )
        if self._destination_node not in destinations:
            raise ValueError(
                f"Scenario destination {self._destination_node!r} is not reachable "
                f"from origin {self._origin_node!r}"
            )

        self.hazard_sim.set_seed(scenario.get("seed", seed))
        self.episode_context = self.hazard_sim.reset_episode(scenario)
        self.hazard_sim.set_current_position(self._origin_node)
        self._current_node = self._origin_node
        self._prior_cost_to_go_cache = self._prior_cost_to_go()
        self._step_count = 0
        self.visited_nodes = set()
        self.visit_counts = {}
        self.total_revisits = 0
        self._record_visit(self._origin_node)
        obs, info = self._build_observation()
        info["origin"] = self._origin_node
        info["destination"] = self._destination_node
        info["month"] = self.episode_context["month"]
        return obs, info

    def step(self, action):
        if not isinstance(action, (int, np.integer)):
            raise TypeError(f"action must be an integer, got {type(action)!r}")
        if not 0 <= int(action) < self.action_space.n:
            raise ValueError(f"action {action} is out of range for action_space.n={self.action_space.n}")
        action = int(action)
        edge = self.graph_wrapper.action_to_edge(self._current_node, action)
        if edge is None:
            if not any(self.graph_wrapper.action_mask(self._current_node)):
                self._step_count += 1
                reward = -float(
                    self.config.get("env", {}).get("dead_end_penalty", 50.0)
                ) - float(self.config.get("env", {}).get("step_penalty", 0.1))
                obs, info = self._build_observation()
                info["terminated"] = True
                info["truncated"] = False
                info["dead_end"] = True
                info["success"] = False
                info["reward"] = reward
                return obs, reward, True, False, info
            raise ValueError(f"masked invalid action {action} chosen at node {self._current_node!r}")

        self.hazard_sim.maybe_trigger_event(self._step_count)
        u, v, k = edge
        belief_before = self.hazard_sim.belief(u, v, k).copy()
        level = self.hazard_sim.true_level(u, v, k)

        edge_data = self.graph.get_edge_data(u, v, k)
        base_travel_time = float(edge_data.get("travel_time", 0.0))
        road_quality_score = float(edge_data.get("road_quality_score", 1.0))
        traffic_multiplier = self._lookup_traffic_multiplier(
            str(edge_data.get("highway_class", edge_data.get("highway", "unclassified"))),
            self.episode_context["hour"],
            self.episode_context["is_weekend"],
        )
        adjusted_base_time = base_edge_time(
            base_travel_time,
            traffic_multiplier,
            road_quality_score,
            self.cost_cfg.quality_penalty_weight,
        )
        realized_time = realized_cost(adjusted_base_time, level, self.cost_cfg)
        is_flooded = level > 0

        reward = -realized_time - float(self.config.get("env", {}).get("step_penalty", 0.1))

        self._current_node = v
        self.hazard_sim.set_current_position(self._current_node)
        self._step_count += 1

        terminated = self._current_node == self._destination_node
        prior_count, prior_snapshot = self.visit_counts.get(self._current_node, (0, {}))
        is_revisit = prior_count > 0
        revisit_penalty = 0.0
        revisit_failure = False
        visit_count, current_snapshot = self._record_visit(self._current_node)

        if is_revisit:
            informed_revisit = any(
                abs(current_snapshot.get(edge, 0.0) - old_cost) > 1e-6
                for edge, old_cost in prior_snapshot.items()
            )
            if informed_revisit:
                # Rational backtrack: new flood information discovered since last visit
                # No penalty, no termination — treat as fresh step
                pass
            else:
                # Blind loop: no new local information
                self.total_revisits += 1
                revisit_penalty = self._revisit_penalty_for(self.total_revisits)
                reward -= revisit_penalty
                if self.revisit_mode == "terminate":
                    terminated = True
                    revisit_failure = True
        if terminated:
            if not revisit_failure:
                reward += 100.0
        truncated = (not terminated) and (self._step_count >= int(self.config.get("env", {}).get("max_episode_steps", 200)))

        obs, info = self._build_observation()
        info["terminated"] = terminated
        info["truncated"] = truncated
        info["dead_end"] = False
        info["success"] = terminated and not revisit_failure
        info["revisit"] = is_revisit
        info["revisit_failure"] = revisit_failure
        info["revisit_penalty"] = float(revisit_penalty)
        info["visit_count"] = visit_count
        info["total_revisits"] = self.total_revisits
        info["reward"] = reward
        info["realized_travel_time"] = float(realized_time)
        info["is_flooded"] = bool(is_flooded)
        info["level"] = int(level)
        info["belief_before"] = belief_before.tolist()
        info["edge"] = [u, v, k]
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return None
