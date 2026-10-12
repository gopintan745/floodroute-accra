"""Expected-cost and oracle routing baselines for graded flood severity."""

from __future__ import annotations

import logging

import networkx as nx
import osmnx as ox
import pandas as pd

from src.common.cost_model import (
    CostModelConfig,
    base_edge_time,
    expected_cost,
    realized_cost,
)

logger = logging.getLogger(__name__)


def build_traffic_lookup(traffic_profile: pd.DataFrame) -> dict:
    """Convert traffic profile rows to (class, hour, weekend) lookup values."""
    if traffic_profile is None or traffic_profile.empty:
        return {}
    return {
        (row["highway_class"], int(row["hour"]), bool(row["is_weekend"])): float(row["multiplier"])
        for _, row in traffic_profile.iterrows()
    }


def _traffic(traffic_lookup, data, hour, is_weekend):
    road_class = str(data.get("highway_class", data.get("highway", "unclassified")))
    key = (road_class, int(hour), bool(is_weekend))
    if key not in traffic_lookup:
        logger.warning("Missing traffic profile for %s; falling back to 1.0", key)
    return float(traffic_lookup.get(key, 1.0))


def _base_time(data, traffic_lookup, hour, is_weekend, cost_cfg):
    return base_edge_time(
        float(data.get("travel_time", 0.0)),
        _traffic(traffic_lookup, data, hour, is_weekend),
        float(data.get("road_quality_score", 1.0)),
        cost_cfg.quality_penalty_weight,
    )


def _edge_belief(belief_map, edge):
    belief = belief_map.get(edge, (1.0, 0.0, 0.0))
    return belief


def annotate_dynamic_costs(
    graph,
    traffic_lookup: dict,
    hour: int,
    is_weekend: bool,
    cost_cfg: CostModelConfig,
    belief_map: dict | None = None,
) -> None:
    """Set each edge's Dijkstra weight to its expected graded-level cost."""
    belief_map = belief_map or {}
    for u, v, key, data in graph.edges(keys=True, data=True):
        edge = (u, v, key)
        edge_time = _base_time(data, traffic_lookup, hour, is_weekend, cost_cfg)
        data["dynamic_cost"] = expected_cost(
            edge_time, _edge_belief(belief_map, edge), cost_cfg
        )


def _nearest_node(graph, point):
    try:
        return ox.distance.nearest_nodes(graph, X=point[0], Y=point[1])
    except (KeyError, TypeError, ValueError):
        return min(
            graph.nodes,
            key=lambda node: (
                (float(graph.nodes[node].get("x", 0.0)) - float(point[0])) ** 2
                + (float(graph.nodes[node].get("y", 0.0)) - float(point[1])) ** 2
            ),
        )


def _best_edge(graph, u, v):
    edge_data = graph.get_edge_data(u, v)
    if not edge_data:
        raise nx.NetworkXNoPath(f"No edge exists between {u!r} and {v!r}")
    return min(edge_data.items(), key=lambda item: float(item[1].get("dynamic_cost", float("inf"))))


def _edges_touching_radius(graph, node, radius):
    undirected = graph.to_undirected(as_view=True)
    visible = nx.single_source_shortest_path_length(undirected, node, cutoff=radius)
    return [
        (u, v, key)
        for u, v, key in graph.edges(keys=True)
        if u in visible or v in visible
    ]


def _route_edge_totals(graph, path):
    edges = []
    for u, v in zip(path[:-1], path[1:]):
        key, data = _best_edge(graph, u, v)
        edges.append((u, v, key))
    return edges


def dynamic_shortest_path(
    graph,
    traffic_profile: pd.DataFrame,
    origin: tuple,
    destination: tuple,
    hour: int,
    is_weekend: bool,
    cost_cfg: CostModelConfig,
    belief_map: dict | None = None,
    hazard_sim=None,
) -> dict:
    """Plan using posterior beliefs at the origin and priors elsewhere."""
    traffic_lookup = build_traffic_lookup(traffic_profile)
    origin_node = _nearest_node(graph, origin)
    destination_node = _nearest_node(graph, destination)
    if hazard_sim is not None:
        hazard_sim.set_current_position(origin_node)
        belief_map = hazard_sim.belief_map()
    annotate_dynamic_costs(
        graph, traffic_lookup, hour, is_weekend, cost_cfg, belief_map
    )
    if origin_node == destination_node:
        return {
            "path": [origin_node], "total_expected_cost": 0.0,
            "origin_node": origin_node, "destination_node": destination_node,
            "edges_traversed": [], "num_edges": 0,
        }
    try:
        path = nx.shortest_path(graph, origin_node, destination_node, weight="dynamic_cost")
    except nx.NetworkXNoPath:
        return {
            "path": [], "total_expected_cost": float("inf"),
            "origin_node": origin_node, "destination_node": destination_node,
            "edges_traversed": [], "num_edges": 0,
            "error": "No path exists between origin and destination",
        }
    edges = _route_edge_totals(graph, path)
    return {
        "path": path,
        "total_expected_cost": sum(float(graph.edges[edge]["dynamic_cost"]) for edge in edges),
        "edges_traversed": edges,
        "origin_node": origin_node,
        "destination_node": destination_node,
        "num_edges": len(edges),
    }


def dynamic_shortest_path_with_replanning(
    graph,
    hazard_sim,
    traffic_lookup,
    origin,
    destination,
    hour,
    is_weekend,
    cost_cfg,
    reveal_radius_hops=None,
    max_steps=None,
) -> dict:
    """Replan at every node using the current posterior map and realize one hop."""
    current = _nearest_node(graph, origin)
    destination_node = _nearest_node(graph, destination)
    step_limit = int(hazard_sim.config.get("env", {}).get("max_episode_steps", max_steps or 64))
    path, traversed, total_time, total_revisits = [current], [], 0.0, 0
    wet_traversals = deep_traversals = 0
    revisit_penalty_total = 0.0
    visit_snapshots = {}

    def local_expected_costs(node):
        hazard_sim.set_current_position(node)
        local = {}
        for edge in _edges_touching_radius(graph, node, hazard_sim.reveal_radius_hops):
            data = graph.edges[edge]
            base = _base_time(data, traffic_lookup, hour, is_weekend, cost_cfg)
            local[edge] = expected_cost(base, hazard_sim.belief(*edge), cost_cfg)
        return local

    visit_snapshots[current] = local_expected_costs(current)
    while current != destination_node and len(traversed) < step_limit:
        hazard_sim.set_current_position(current)
        hazard_sim.maybe_trigger_event(len(traversed))
        annotate_dynamic_costs(
            graph, traffic_lookup, hour, is_weekend, cost_cfg,
            hazard_sim.belief_map(),
        )
        try:
            planned_path = nx.shortest_path(
                graph, current, destination_node, weight="dynamic_cost"
            )
        except nx.NetworkXNoPath:
            break
        next_node = planned_path[1]
        key, data = _best_edge(graph, current, next_node)
        edge = (current, next_node, key)
        level = hazard_sim.true_level(*edge)
        wet_traversals += int(level > 0)
        deep_traversals += int(level == 2)
        edge_time = _base_time(data, traffic_lookup, hour, is_weekend, cost_cfg)
        realized = realized_cost(edge_time, level, cost_cfg)
        traversed.append(edge)
        total_time += realized
        current = next_node
        path.append(current)
        current_snapshot = local_expected_costs(current)
        prior_snapshot = visit_snapshots.get(current)
        if prior_snapshot is not None:
            informed = any(
                abs(current_snapshot.get(edge, 0.0) - prior_cost) > 1e-6
                for edge, prior_cost in prior_snapshot.items()
            )
            if not informed:
                total_revisits += 1
                if hazard_sim.config.get("env", {}).get("revisit_mode", "terminate") == "terminate":
                    break
                revisit_penalty_total += float(
                    hazard_sim.config.get("env", {}).get("revisit_penalty", 50.0)
                ) * float(
                    hazard_sim.config.get("env", {}).get("revisit_penalty_growth", 2.0)
                ) ** (total_revisits - 1)
        visit_snapshots[current] = current_snapshot
    return {
        "path": path,
        "edges_traversed": traversed,
        "total_realized_time": total_time,
        "reached_destination": current == destination_node,
        "steps": len(traversed),
        "total_revisits": total_revisits,
        "revisit_penalty": revisit_penalty_total,
        "wet_edge_traversals": wet_traversals,
        "deep_edge_traversals": deep_traversals,
        "revisit_failure": current != destination_node and total_revisits > 0,
    }


def oracle_shortest_path(
    graph,
    hazard_sim,
    traffic_lookup,
    origin,
    destination,
    hour,
    is_weekend,
    cost_cfg,
) -> dict:
    """Return the full-information shortest path knowing episode levels."""
    origin_node = _nearest_node(graph, origin)
    destination_node = _nearest_node(graph, destination)
    for u, v, key, data in graph.edges(keys=True, data=True):
        edge_time = _base_time(data, traffic_lookup, hour, is_weekend, cost_cfg)
        data["dynamic_cost"] = realized_cost(
            edge_time, hazard_sim.true_level(u, v, key), cost_cfg
        )
    try:
        path = nx.shortest_path(graph, origin_node, destination_node, weight="dynamic_cost")
    except nx.NetworkXNoPath:
        return {"path": [], "total_realized_time": float("inf"), "reached_destination": False}
    edges = _route_edge_totals(graph, path)
    return {
        "path": path,
        "edges_traversed": edges,
        "total_realized_time": sum(float(graph.edges[edge]["dynamic_cost"]) for edge in edges),
        "reached_destination": True,
        "steps": len(edges),
        "wet_edge_traversals": sum(level > 0 for level in (
            hazard_sim.true_level(*edge) for edge in edges
        )),
        "deep_edge_traversals": sum(level == 2 for level in (
            hazard_sim.true_level(*edge) for edge in edges
        )),
    }
