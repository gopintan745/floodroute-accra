"""Common metrics for scenario-level routing results."""

from __future__ import annotations

from statistics import mean, median, pvariance


def _edge_records(hazard_realization):
    if hazard_realization is None:
        return {}
    if isinstance(hazard_realization, dict):
        return hazard_realization.get("edges", hazard_realization)
    return {tuple(edge[:3]) if isinstance(edge, (list, tuple)) else edge: {} for edge in hazard_realization}


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
        direct = _result_value(route, "actual_travel_time", "travel_time")
        if direct is not None:
            return float(direct)
        reward = _result_value(route, "reward")
        if reward is not None:
            # Environment rewards are negative realized travel cost plus any
            # terminal bonus, so this is only a compatibility fallback.
            return float(-reward)
        route = route.get("edges_traversed", route.get("path", []))
    records = _edge_records(hazard_realization)
    total = 0.0
    for edge in route:
        edge_id = tuple(edge[:3]) if isinstance(edge, (list, tuple)) else edge
        record = records.get(edge_id, records.get(str(edge_id), {}))
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
        flooded_count = _result_value(route, "edges_hit_flooded", "blocked_edges")
        if flooded_count is not None:
            return bool(flooded_count)
        route = route.get("edges_traversed", route.get("path", []))
    records = _edge_records(hazard_realization)
    return any(
        bool(records.get(edge, records.get(str(edge), {})).get("is_flooded", False))
        or bool(records.get(edge, records.get(str(edge), {})).get("blocked", False))
        for edge in route
        if isinstance(records.get(edge, records.get(str(edge), {})), dict)
    )


def summarize(results: list) -> dict:
    """Summarize per-scenario results for one routing method."""
    if not results:
        return {
            "episodes": 0,
            "mean_travel_time": 0.0,
            "median_travel_time": 0.0,
            "variance_travel_time": 0.0,
            "failure_rate": 0.0,
        }
    times = [
        travel_time(result, result.get("hazard_realization")) if isinstance(result, dict) else float(result)
        for result in results
    ]
    failures = [
        bool(result.get("failure", result.get("blocked", False)))
        or bool(result.get("edges_hit_flooded", 0))
        or (result.get("success") is False if isinstance(result, dict) else False)
        for result in results
    ]
    return {
        "episodes": len(results),
        "mean_travel_time": float(mean(times)),
        "median_travel_time": float(median(times)),
        "variance_travel_time": float(pvariance(times)),
        "failure_rate": float(mean(failures)),
    }
