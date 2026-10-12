"""Common metrics for scenario-level routing results."""

from __future__ import annotations

import ast
from statistics import mean, median, pvariance

import numpy as np


def _edge_key(edge):
    """Normalize tuple, list, and serialized tuple edge identifiers."""
    if isinstance(edge, str):
        try:
            edge = ast.literal_eval(edge)
        except (SyntaxError, ValueError):
            return edge
    if isinstance(edge, (list, tuple)) and len(edge) >= 3:
        return tuple(edge[:3])
    return edge


def _edge_records(hazard_realization):
    if hazard_realization is None:
        return {}
    if isinstance(hazard_realization, dict):
        records = hazard_realization.get("edges", hazard_realization)
    else:
        records = {edge: {} for edge in hazard_realization}
    return {_edge_key(edge): record for edge, record in records.items()}


def _result_value(result, *names, default=None):
    if isinstance(result, dict):
        for name in names:
            if name in result:
                return result[name]
    return default


def travel_time(route, hazard_realization) -> float:
    """Return realized route time from edge records, in the records' units.

    Edge records may provide ``travel_time`` directly, or ``base_travel_time``
    plus an optional ``traffic_multiplier`` and ``flood_penalty``.
    """
    if isinstance(route, dict):
        direct = _result_value(
            route, "realized_travel_time", "actual_travel_time", "travel_time"
        )
        if direct is not None:
            return float(direct)
        route = route.get("edges_traversed", route.get("path", []))
    records = _edge_records(hazard_realization)
    total = 0.0
    for edge in route:
        edge_id = _edge_key(edge)
        record = records.get(edge_id, {})
        if isinstance(record, (int, float)):
            total += float(record)
            continue
        base = float(record.get("travel_time", record.get("base_travel_time", 0.0)))
        total += base * float(record.get("traffic_multiplier", 1.0))
        level = int(record.get("level", 1 if record.get("is_flooded", False) else 0))
        if level == 1:
            total += base * max(0.0, float(record.get("shallow_multiplier", 2.0)) - 1.0)
        elif level == 2:
            total += base * max(0.0, float(record.get("deep_multiplier", 3.0)) - 1.0)
            total += float(record.get("deep_penalty_s", 600.0))
    return float(total)


def hit_blocked_edge(route, hazard_realization) -> bool:
    """Return whether a route traverses an edge marked flooded or blocked."""
    if isinstance(route, dict):
        flooded_count = _result_value(route, "edges_hit_flooded", "flooded_edges", "blocked_edges")
        if flooded_count is not None:
            return bool(flooded_count)
        route = route.get("edges_traversed", route.get("path", []))
    records = _edge_records(hazard_realization)
    return any(
            int(records.get(_edge_key(edge), {}).get("level", 0)) > 0
            or bool(records.get(_edge_key(edge), {}).get("is_flooded", False))
        or bool(records.get(_edge_key(edge), {}).get("blocked", False))
        for edge in route
        if isinstance(records.get(_edge_key(edge), {}), dict)
    )


def summarize(results: list) -> dict:
    """Summarize per-scenario results for one routing method."""
    if not results:
        return {
            "episodes": 0,
            "mean_travel_time": 0.0,
            "median_travel_time": 0.0,
            "variance_travel_time": 0.0,
            "p90_travel_time": 0.0,
            "p95_travel_time": 0.0,
            "p99_travel_time": 0.0,
            "cvar95_travel_time": 0.0,
            "completion_rate": 0.0,
            "route_failure_rate": 0.0,
            "flooded_edge_rate": 0.0,
            "blocked_edge_rate": 0.0,
            "mean_reward": 0.0,
            "mean_wet_edge_traversals": 0.0,
            "mean_deep_edge_traversals": 0.0,
            "deep_traversal_episode_rate": 0.0,
            "mean_total_revisits": 0.0,
            "mean_regret_vs_oracle": 0.0,
            "p95_regret_vs_oracle": 0.0,
        }
    times = [
        travel_time(result, result.get("hazard_realization")) if isinstance(result, dict) else float(result)
        for result in results
    ]
    records = [result for result in results if isinstance(result, dict)]
    successes = [bool(result.get("success", False)) for result in records]
    route_failures = [
        bool(result.get("failure", False))
        or bool(result.get("dead_end", False))
        or bool(result.get("truncated", False))
        or not bool(result.get("success", False))
        for result in records
    ]
    flooded = [
        bool(result.get(
            "wet_edge_traversals",
            result.get("flooded_edges", result.get("edges_hit_flooded", 0)),
        ))
        for result in records
    ]
    blocked = [
        bool(result.get("blocked_edges", result.get("blocked", False)))
        for result in records
    ]
    rewards = [float(result["reward"]) for result in records if "reward" in result]
    regrets = [
        float(result["regret_vs_oracle"])
        for result in records if result.get("regret_vs_oracle") is not None
    ]
    sorted_times = np.sort(np.asarray(times, dtype=np.float64))
    cvar_start = max(0, int(np.floor(0.95 * len(sorted_times))))
    cvar_values = sorted_times[cvar_start:]
    wet_counts = [int(result.get("wet_edge_traversals", result.get("flooded_edges", 0)) or 0) for result in records]
    deep_counts = [int(result.get("deep_edge_traversals", 0) or 0) for result in records]
    revisits = [int(result.get("total_revisits", 0) or 0) for result in records]
    return {
        "episodes": len(results),
        "mean_travel_time": float(mean(times)),
        "median_travel_time": float(median(times)),
        "variance_travel_time": float(pvariance(times)),
        "p90_travel_time": float(np.percentile(times, 90)),
        "p95_travel_time": float(np.percentile(times, 95)),
        "p99_travel_time": float(np.percentile(times, 99)),
        "cvar95_travel_time": float(np.mean(cvar_values)),
        "completion_rate": float(mean(successes)) if records else 0.0,
        "route_failure_rate": float(mean(route_failures)) if records else 0.0,
        "flooded_edge_rate": float(mean(flooded)) if records else 0.0,
        "blocked_edge_rate": float(mean(blocked)) if records else 0.0,
        "mean_reward": float(mean(rewards)) if rewards else 0.0,
        "mean_wet_edge_traversals": float(mean(wet_counts)) if wet_counts else 0.0,
        "mean_deep_edge_traversals": float(mean(deep_counts)) if deep_counts else 0.0,
        "deep_traversal_episode_rate": float(mean([count > 0 for count in deep_counts])) if deep_counts else 0.0,
        "mean_total_revisits": float(mean(revisits)) if revisits else 0.0,
        "mean_regret_vs_oracle": float(mean(regrets)) if regrets else 0.0,
        "p95_regret_vs_oracle": float(np.percentile(regrets, 95)) if regrets else 0.0,
    }
