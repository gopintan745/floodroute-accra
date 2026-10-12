"""Calibrated graded flood levels and local Bayesian beliefs."""

from __future__ import annotations

import ast
import math
from typing import Hashable

import networkx as nx
import numpy as np


EdgeId = tuple[Hashable, Hashable, Hashable]


class HazardSimulator:
    """Generate path-independent flood levels and noisy local observations."""

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        climatology: dict,
        config: dict,
        mode: str,
        seed: int | None = None,
    ):
        if mode not in {"train", "eval"}:
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
        self.graph = graph
        self.climatology = climatology or {}
        self.config = config or {}
        self.mode = mode
        self.reveal_radius_hops = self._read_nonnegative_int(
            self.config.get("env", {}).get("reveal_radius_hops", 1),
            "env.reveal_radius_hops",
        )
        self.edges: tuple[EdgeId, ...] = tuple(
            sorted(graph.edges(keys=True), key=self._stable_edge_key)
        )
        self.susceptibility = {
            edge: self._read_susceptibility(graph.edges[edge]) for edge in self.edges
        }
        self.current_position = None
        self._hop_distances: dict = {}
        self.true_levels: dict[EdgeId, int] = {}
        self.readings: dict[EdgeId, dict[int, int]] = {}
        self.event_schedule: dict[int, list[tuple[EdgeId, int]]] = {}
        self.step = 0
        self.is_flood_day = False
        self.intensity = 0.0
        self.month = 1
        self._episode_reset = False
        self.set_seed(seed)

    @staticmethod
    def _stable_edge_key(edge: EdgeId) -> str:
        return repr(tuple(str(part) for part in edge))

    @staticmethod
    def _read_nonnegative_int(value, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        return int(value)

    @staticmethod
    def _read_susceptibility(data: dict) -> float:
        value = data.get("flood_susceptibility", data.get("flood_risk", 0.0))
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        if not math.isfinite(value):
            value = 0.0
        return float(np.clip(value, 0.0, 1.0))

    def set_seed(self, seed: int | None) -> None:
        """Reset the episode stream sequence, typically from Env.reset(seed=...)."""
        self._seed_sequence = np.random.SeedSequence(seed)

    def _flood_config(self) -> dict:
        return self.config.get(
            "flood_model",
            self.config.get("flood_risk", {}).get("flood_model", {}),
        )

    def _env_config(self) -> dict:
        return self.config.get("env", {})

    def _confusion_matrix(self, hop: int) -> np.ndarray:
        flood_model = self._flood_config()
        if flood_model.get("signal_mode", "noisy") == "exact":
            return np.eye(3, dtype=np.float64)
        matrices = flood_model.get("signal_confusion", {})
        matrix = matrices.get(f"hop{hop}")
        if matrix is None:
            available = sorted(
                (int(name[3:]), value)
                for name, value in matrices.items()
                if name.startswith("hop") and name[3:].isdigit()
            )
            matrix = available[-1][1] if available else np.eye(3)
        result = np.asarray(matrix, dtype=np.float64)
        if result.shape != (3, 3) or np.any(result < 0):
            raise ValueError(f"signal_confusion.hop{hop} must be a 3x3 probability matrix")
        if not np.allclose(result.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError(f"signal_confusion.hop{hop} rows must sum to 1")
        return result

    def _monthly_flood_fraction(self, month: int) -> float:
        by_month = self.climatology.get("by_month")
        if by_month is not None:
            record = by_month.get(month, by_month.get(str(month)))
            if record is None:
                record = by_month.get(month - 1, by_month.get(str(month - 1)))
            if record is None:
                raise ValueError(f"climatology has no entry for month {month}")
            fraction = record["heavy_day_fraction"]
        else:
            frequencies = self.climatology.get("monthly_heavy_rain_frequency")
            if frequencies is None or len(frequencies) != 12:
                raise ValueError("climatology must contain by_month or 12 monthly frequencies")
            fraction = frequencies[month - 1]
        fraction = float(fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"heavy-day fraction must be in [0, 1], got {fraction}")
        return fraction

    def _sample_intensity(self, flood_day: bool, rng) -> float:
        if not flood_day:
            return 0.0
        flood_model = self._flood_config()
        heavy_days = np.asarray(self.climatology.get("heavy_day_precip_mm", []), dtype=float)
        threshold = float(
            flood_model.get(
                "heavy_rain_threshold_mm",
                self.climatology.get("flood_threshold_mm", 20.0),
            )
        )
        if heavy_days.size == 0:
            return float(np.clip(flood_model.get("default_intensity_if_no_heavy_days", 0.5), 0, 1))
        reference_percentile = float(flood_model.get("intensity_reference_percentile", 95.0))
        reference = float(np.percentile(heavy_days, reference_percentile))
        if reference <= threshold:
            return float(np.clip(flood_model.get("default_intensity_if_no_heavy_days", 0.5), 0, 1))
        precipitation = float(rng.choice(heavy_days))
        return float(np.clip((precipitation - threshold) / (reference - threshold), 0.0, 1.0))

    def prior(self, u, v, k) -> np.ndarray:
        """Return prior probabilities in the order dry, shallow, deep."""
        edge = (u, v, k)
        if edge not in self.susceptibility:
            raise KeyError(f"unknown graph edge {edge!r}")
        if not self.is_flood_day:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        params = self._flood_config()
        alpha = float(params.get("intensity_weight_alpha", 0.5))
        wet_probability = float(
            np.clip(self.susceptibility[edge] * (1.0 - alpha + alpha * self.intensity), 0, 1)
        )
        deep_min = float(params.get("deep_fraction_min", 0.1))
        deep_max = float(params.get("deep_fraction_max", 0.7))
        deep_fraction = deep_min + (deep_max - deep_min) * self.intensity
        deep_probability = wet_probability * deep_fraction
        shallow_probability = wet_probability * (1.0 - deep_fraction)
        return np.array(
            [1.0 - wet_probability, shallow_probability, deep_probability],
            dtype=np.float64,
        )

    def _normalize_edge_override(self, key) -> EdgeId:
        if isinstance(key, str):
            try:
                key = ast.literal_eval(key)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"invalid edge_levels key {key!r}") from exc
        if isinstance(key, list):
            key = tuple(key)
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError(f"edge_levels keys must identify (u, v, key), got {key!r}")
        # JSON turns graphml node IDs and edge keys into strings.
        candidate = tuple(str(part) for part in key)
        if candidate in self.susceptibility:
            return candidate
        if key in self.susceptibility:
            return key
        for edge in self.edges:
            if tuple(str(part) for part in edge) == candidate:
                return edge
        raise ValueError(f"scenario overrides unknown edge {key!r}")

    def reset_episode(self, scenario: dict | None = None) -> dict:
        """Sample a reproducible world and precompute signals independent of path."""
        scenario = scenario or {}
        latent_seed, signal_seed = self._seed_sequence.spawn(2)
        latent_rng = np.random.default_rng(latent_seed)
        signal_rng = np.random.default_rng(signal_seed)
        month = int(latent_rng.integers(1, 13))
        hour = int(latent_rng.integers(0, 24))
        is_weekend = bool(latent_rng.integers(0, 2))
        if self.mode == "train":
            flood_rate = float(self._env_config().get("train_flood_day_rate", 0.35))
        else:
            flood_rate = self._monthly_flood_fraction(month)
        if not 0.0 <= flood_rate <= 1.0:
            raise ValueError(f"flood-day rate must be in [0, 1], got {flood_rate}")
        is_flood_day = bool(scenario.get("is_flood_day", latent_rng.random() < flood_rate))
        intensity = float(
            scenario.get("intensity", self._sample_intensity(is_flood_day, latent_rng))
        )
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"scenario intensity must be in [0, 1], got {intensity}")
        if not is_flood_day:
            intensity = 0.0

        self.month = int(scenario.get("month", month))
        hour = int(scenario.get("hour", hour))
        is_weekend = bool(scenario.get("is_weekend", is_weekend))
        self.is_flood_day = is_flood_day
        self.intensity = intensity
        self.true_levels = {}
        for edge in self.edges:
            probabilities = self.prior(*edge)
            self.true_levels[edge] = int(latent_rng.choice(3, p=probabilities))

        self._signal_rng = signal_rng
        overrides = scenario.get("edge_levels")
        if overrides is None and "flooded_edges" in scenario:
            flooded = scenario["flooded_edges"]
            overrides = {tuple(e): 2 for e in flooded} if isinstance(flooded, list) else {}
        overrides = overrides or {}
        if isinstance(overrides, list):
            overrides = {tuple(record["edge"]): record["level"] for record in overrides}
        for edge_key, level in overrides.items():
            edge = self._normalize_edge_override(edge_key)
            if isinstance(level, bool) or int(level) not in (0, 1, 2):
                raise ValueError(f"edge level must be 0, 1, or 2, got {level!r}")
            self.true_levels[edge] = int(level)

        self.readings = {}
        for edge in self.edges:
            true_level = self.true_levels[edge]
            self.readings[edge] = {}
            for hop in range(self.reveal_radius_hops + 1):
                confusion = self._confusion_matrix(hop)
                self.readings[edge][hop] = int(signal_rng.choice(3, p=confusion[true_level]))

        self.event_schedule = {}
        self.step = 0
        self.current_position = None
        self._hop_distances = {}
        self._episode_reset = True
        event_rate = float(
            self._flood_config().get(
                "mid_episode_event_base_rate",
                self._env_config().get("mid_episode_event_base_rate", 0.0),
            )
        )
        if not 0.0 <= event_rate <= 1.0:
            raise ValueError("mid_episode_event_base_rate must be in [0, 1]")
        if is_flood_day and event_rate > 0:
            max_steps = int(self._env_config().get("max_episode_steps", 64))
            simulated_levels = dict(self.true_levels)
            for step in range(max_steps):
                changes = []
                for edge in self.edges:
                    if simulated_levels[edge] < 2 and latent_rng.random() < event_rate * self.susceptibility[edge]:
                        simulated_levels[edge] += 1
                        changes.append((edge, simulated_levels[edge]))
                self.event_schedule[step] = changes

        return {
            "month": self.month,
            "hour": hour,
            "is_weekend": is_weekend,
            "is_flood_day": self.is_flood_day,
            "intensity": self.intensity,
        }

    def set_current_position(self, node) -> None:
        """Set driver position and cache one undirected BFS for local beliefs."""
        if not self._episode_reset:
            raise RuntimeError("reset_episode() must be called before set_current_position()")
        if node not in self.graph:
            raise ValueError(f"current position {node!r} is not a graph node")
        self.current_position = node
        self._hop_distances = nx.single_source_shortest_path_length(
            self.graph.to_undirected(as_view=True), node
        )

    def _edge_hop(self, edge: EdgeId) -> int:
        if self.current_position is None:
            raise RuntimeError("set_current_position() must be called before belief()")
        u, v, _ = edge
        return min(
            self._hop_distances.get(u, math.inf),
            self._hop_distances.get(v, math.inf),
        )

    def belief(self, u, v, k) -> np.ndarray:
        """Return a Bayesian posterior locally and the unchanged prior elsewhere."""
        edge = (u, v, k)
        prior = self.prior(*edge)
        hop = self._edge_hop(edge)
        if hop > self.reveal_radius_hops:
            return prior
        hop = int(hop)
        reading = self.readings[edge][hop]
        likelihood = self._confusion_matrix(hop)[:, reading]
        posterior = prior * likelihood
        denominator = float(posterior.sum())
        if denominator <= np.finfo(np.float64).eps:
            posterior = np.zeros(3, dtype=np.float64)
            posterior[reading] = 1.0
            return posterior
        return posterior / denominator

    def belief_map(self) -> dict[EdgeId, np.ndarray]:
        """Return beliefs for all edges using the current cached position."""
        if self.current_position is None:
            raise RuntimeError("set_current_position() must be called before belief_map()")
        return {edge: self.belief(*edge) for edge in self.edges}

    def true_level(self, u, v, k) -> int:
        """Ground-truth level. Restrict callers to traversal, oracle, or metrics."""
        if not self._episode_reset:
            raise RuntimeError("reset_episode() must be called before true_level()")
        edge = (u, v, k)
        try:
            return int(self.true_levels[edge])
        except KeyError as exc:
            raise KeyError(f"unknown graph edge {edge!r}") from exc

    def set_edge_level(self, u, v, k, level: int) -> None:
        """Update an edge's ground truth level and its sensor readings accordingly."""
        if not self._episode_reset:
            raise RuntimeError("reset_episode() must be called before set_edge_level()")
        edge = self._normalize_edge_override((u, v, k))
        level = int(level)
        if level not in (0, 1, 2):
            raise ValueError(f"edge level must be 0, 1, or 2, got {level}")
        self.true_levels[edge] = level
        signal_rng = getattr(self, "_signal_rng", np.random.default_rng(0))
        for hop in range(self.reveal_radius_hops + 1):
            if self._flood_config().get("signal_mode", "noisy") == "exact":
                self.readings[edge][hop] = level
            else:
                confusion = self._confusion_matrix(hop)
                self.readings[edge][hop] = int(signal_rng.choice(3, p=confusion[level]))

    @property
    def flooded_edges(self) -> set[EdgeId]:
        """Backward-compatible view of non-dry edges."""
        return {edge for edge, level in self.true_levels.items() if level > 0}

    @flooded_edges.setter
    def flooded_edges(self, edges) -> None:
        """Backward-compatible setter for flooded edges."""
        edge_set = {tuple(e) for e in (edges or ())}
        for edge in self.edges:
            self.set_edge_level(*edge, 2 if edge in edge_set else 0)

    def maybe_trigger_event(self, step: int) -> list[tuple[EdgeId, int]]:
        """Apply pre-sampled monotone one-level upgrades for this episode step."""
        if not self._episode_reset:
            raise RuntimeError("reset_episode() must be called before maybe_trigger_event()")
        self.step = int(step)
        changes = self.event_schedule.get(self.step, [])
        applied = []
        for edge, scheduled_level in changes:
            if scheduled_level > self.true_levels[edge]:
                self.set_edge_level(*edge, scheduled_level)
                applied.append((edge, scheduled_level))
        return applied
