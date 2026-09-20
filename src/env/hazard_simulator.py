"""Episode-level flood realization for the routing environment."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np


class HazardSimulator:
    """Sample and expose ground-truth flooding for one routing episode."""

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
        self.climatology = climatology
        self.config = config
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.flood_susceptibility = {
            (u, v, key): self._edge_susceptibility(data)
            for u, v, key, data in graph.edges(keys=True, data=True)
        }
        self.flooded_edges: set[tuple] = set()
        self.step = 0
        self.is_flood_day = False
        self._episode_reset = False

    @staticmethod
    def _edge_susceptibility(edge_data: dict) -> float:
        value = edge_data.get("flood_susceptibility", edge_data.get("flood_risk", 0.0))
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        if not math.isfinite(value):
            value = 0.0
        return float(np.clip(value, 0.0, 1.0))

    def _env_value(self, key: str, default=None):
        return self.config.get("env", {}).get(key, default)

    def _monthly_flood_fraction(self, month: int) -> float:
        if "by_month" in self.climatology:
            by_month = self.climatology["by_month"]
            record = by_month.get(month, by_month.get(str(month)))
            if record is None:
                record = by_month.get(month - 1, by_month.get(str(month - 1)))
            if record is None:
                raise ValueError(f"climatology has no entry for month {month}")
            fraction = record["heavy_day_fraction"]
        else:
            frequencies = self.climatology.get("monthly_heavy_rain_frequency")
            if frequencies is None or len(frequencies) != 12:
                raise ValueError(
                    "climatology must contain by_month or 12 monthly frequencies"
                )
            fraction = frequencies[month - 1]
        try:
            fraction = float(fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid flood-day fraction for month {month}: {fraction!r}"
            ) from exc
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"flood-day fraction must be in [0, 1], got {fraction}")
        return fraction

    def reset_episode(self) -> dict:
        """Sample the episode calendar and its initial flooded edges."""
        month = int(self.rng.integers(1, 13))
        hour = int(self.rng.integers(0, 24))
        is_weekend = bool(self.rng.integers(0, 2))
        flood_rate = (
            float(self._env_value("train_flood_day_rate", 0.35))
            if self.mode == "train"
            else self._monthly_flood_fraction(month)
        )
        if not 0.0 <= flood_rate <= 1.0:
            raise ValueError(f"flood-day rate must be in [0, 1], got {flood_rate}")
        self.is_flood_day = bool(self.rng.random() < flood_rate)
        self.flooded_edges = set()
        self.step = 0
        self._episode_reset = True
        if self.is_flood_day:
            self.flooded_edges = {
                edge_id
                for edge_id, susceptibility in self.flood_susceptibility.items()
                if self.rng.random() < susceptibility
            }
        return {
            "month": month,
            "hour": hour,
            "is_weekend": is_weekend,
            "is_flood_day": self.is_flood_day,
        }

    def maybe_trigger_event(self, step: int) -> list:
        """Activate additional flooding before the edge chosen at ``step``."""
        if not self._episode_reset:
            raise RuntimeError(
                "reset_episode() must be called before maybe_trigger_event()"
            )
        self.step = step
        if not self.is_flood_day:
            return []
        base_rate = float(self._env_value("mid_episode_event_base_rate", 0.03))
        if not 0.0 <= base_rate <= 1.0:
            raise ValueError(
                f"mid-episode event rate must be in [0, 1], got {base_rate}"
            )
        newly_flooded = []
        for edge_id, susceptibility in self.flood_susceptibility.items():
            if edge_id not in self.flooded_edges and self.rng.random() < base_rate * susceptibility:
                self.flooded_edges.add(edge_id)
                newly_flooded.append(edge_id)
        return newly_flooded

    def is_flooded(self, u, v, k) -> bool:
        """Return ground truth for one edge after the episode has been reset."""
        if not self._episode_reset:
            raise RuntimeError("reset_episode() must be called before is_flooded()")
        return (u, v, k) in self.flooded_edges
