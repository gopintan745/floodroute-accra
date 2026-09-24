"""Common metrics for scenario-level routing results."""

from __future__ import annotations

import ast
from statistics import mean, median, pvariance


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
        if record.get("is_flooded", record.get("flooded", False)):
            total += float(record.get("flood_penalty", 0.0))
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
        bool(records.get(_edge_key(edge), {}).get("is_flooded", False))
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
            "completion_rate": 0.0,
            "route_failure_rate": 0.0,
            "flooded_edge_rate": 0.0,
            "blocked_edge_rate": 0.0,
            "mean_reward": 0.0,
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
        bool(result.get("flooded_edges", result.get("edges_hit_flooded", 0)))
        for result in records
    ]
    blocked = [
        bool(result.get("blocked_edges", result.get("blocked", False)))
        for result in records
    ]
    rewards = [float(result["reward"]) for result in records if "reward" in result]
    return {
        "episodes": len(results),
        "mean_travel_time": float(mean(times)),
        "median_travel_time": float(median(times)),
        "variance_travel_time": float(pvariance(times)),
        "completion_rate": float(mean(successes)) if records else 0.0,
        "route_failure_rate": float(mean(route_failures)) if records else 0.0,
        "flooded_edge_rate": float(mean(flooded)) if records else 0.0,
        "blocked_edge_rate": float(mean(blocked)) if records else 0.0,
        "mean_reward": float(mean(rewards)) if rewards else 0.0,
    }
