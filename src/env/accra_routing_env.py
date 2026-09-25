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

import gymnasium as gym
import math
import networkx as nx
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.common.cost_model import CostModelConfig, effective_travel_time
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
        self.visit_counts: dict = {}
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
        edge_dim = 4
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

    def _record_visit(self, node) -> int:
        """Record a node visit and return its one-based visit count."""
        visit_count = self.visit_counts.get(node, 0) + 1
        self.visit_counts[node] = visit_count
        self.visited_nodes.add(node)
        return visit_count

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

    def _edge_feature_vector(self, edge, hour: int, is_weekend: bool) -> np.ndarray:
        if edge is None:
            return np.zeros(4, dtype=np.float32)
        u, v, k = edge
        edge_data = self.graph.get_edge_data(u, v, k)
        if edge_data is None:
            return np.zeros(4, dtype=np.float32)
        flood_risk = float(edge_data.get("flood_risk", edge_data.get("flood_susceptibility", 0.0)))
        if self.hazard_sim._episode_reset and self.hazard_sim.current_position is not None:
            try:
                if self.hazard_sim.is_flooded(u, v, k):
                    flood_risk = 1.0
                else:
                    flood_risk = 0.0
            except Exception:
                pass
        traffic_multiplier = self._lookup_traffic_multiplier(
            str(edge_data.get("highway_class", edge_data.get("highway", "unclassified"))),
            hour,
            is_weekend,
        )
        road_quality = float(edge_data.get("road_quality_score", 1.0))
        base_travel_time = float(edge_data.get("travel_time", 0.0))
        return np.array(
            [flood_risk, traffic_multiplier, road_quality, base_travel_time],
            dtype=np.float32,
        )

    def _build_observation(self):
        current_features = self._node_features(self._current_node)
        destination_features = self._node_features(self._destination_node)
        edge_rows = []
        for idx in range(self.max_degree):
            edge = self.graph_wrapper.action_to_edge(self._current_node, idx)
            edge_rows.append(self._edge_feature_vector(edge, self.episode_context["hour"], self.episode_context["is_weekend"]))
        local_edge_features = np.stack(edge_rows, axis=0)
        return {
            "current_node_features": current_features,
            "destination_node_features": destination_features,
            "goal_relative_features": self._goal_relative_features(),
            "local_edge_features": local_edge_features,
        }, {
            "origin": self._origin_node,
            "destination": self._destination_node,
            "month": self.episode_context["month"],
            "hour": self.episode_context["hour"],
            "is_weekend": self.episode_context["is_weekend"],
            "is_flood_day": self.episode_context["is_flood_day"],
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

        self.episode_context = self.hazard_sim.reset_episode()
        for key in ("month", "hour", "is_weekend", "is_flood_day"):
            if key in scenario:
                self.episode_context[key] = scenario[key]
        if "flooded_edges" in scenario:
            self.hazard_sim.is_flood_day = bool(self.episode_context["is_flood_day"])
            self.hazard_sim.flooded_edges = {
                tuple(edge) for edge in scenario["flooded_edges"]
            }
        if "flood_events" in scenario:
            self.hazard_sim.event_schedule = {
                int(event["step"]): [tuple(edge) for edge in event["edges"]]
                for event in scenario["flood_events"]
            }
        self.hazard_sim.set_current_position(self._origin_node)
        self._current_node = self._origin_node
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
        is_flooded = False
        try:
            is_flooded = self.hazard_sim.is_flooded(u, v, k)
        except Exception:
            is_flooded = False

        edge_data = self.graph.get_edge_data(u, v, k)
        base_travel_time = float(edge_data.get("travel_time", 0.0))
        road_quality_score = float(edge_data.get("road_quality_score", 1.0))
        traffic_multiplier = self._lookup_traffic_multiplier(
            str(edge_data.get("highway_class", edge_data.get("highway", "unclassified"))),
            self.episode_context["hour"],
            self.episode_context["is_weekend"],
        )
        realized_time = effective_travel_time(
            base_travel_time=base_travel_time,
            traffic_multiplier=traffic_multiplier,
            flood_susceptibility=0.0,
            road_quality_score=road_quality_score,
            flood_weight=self.cost_cfg.flood_weight,
            quality_penalty_weight=self.cost_cfg.quality_penalty_weight,
        )
        if is_flooded:
            flood_penalty_minutes = float(
                self.config.get("env", {}).get("flood_penalty", 50.0)
            )
            realized_time += flood_penalty_minutes * 60.0

        reward = -realized_time - float(self.config.get("env", {}).get("step_penalty", 0.1))

        self._current_node = v
        self.hazard_sim.set_current_position(self._current_node)
        self._step_count += 1

        terminated = self._current_node == self._destination_node
        prior_visits = self.visit_counts.get(self._current_node, 0)
        is_revisit = prior_visits > 0
        revisit_penalty = 0.0
        revisit_failure = False
        visit_count = self._record_visit(self._current_node)
        if is_revisit:
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
        info["edge"] = [u, v, k]
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return None
