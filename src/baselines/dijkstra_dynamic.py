"""
Baseline 2: shortest path re-weighted using currently-known hazard risk
scores (flood_risk, traffic_multiplier) — the "predict conditions, then
search" approach. This is the stronger, fairer baseline: same information
as the RL agent gets, no learning.

Three-function structure:
1. build_traffic_lookup() — converts traffic_profile DataFrame to O(1) dict
2. annotate_dynamic_costs() — mutates graph edges with dynamic_cost
3. dynamic_shortest_path() — main entry point, returns both dynamic and base costs

The replanning variant uses the same HazardSimulator realization as the
environment and replans from the driver's locally revealed flood state.
"""

import logging
import networkx as nx
import osmnx as ox
import pandas as pd
from src.common.cost_model import CostModelConfig, effective_travel_time

logger = logging.getLogger(__name__)


def build_traffic_lookup(traffic_profile: pd.DataFrame) -> dict:
    """
    Convert traffic_profile DataFrame to O(1) lookup dict.

    Args:
        traffic_profile: DataFrame with columns ['highway_class', 'hour', 'is_weekend', 'multiplier']

    Returns:
        dict mapping (highway_class, hour, is_weekend) -> multiplier
    """
    return {
        (row["highway_class"], row["hour"], row["is_weekend"]): row["multiplier"]
        for _, row in traffic_profile.iterrows()
    }


def annotate_dynamic_costs(
    graph,
    traffic_lookup: dict,
    hour: int,
    is_weekend: bool,
    cost_cfg: CostModelConfig,
    flood_event_edges: set = None,
    reveal_overrides: dict = None,
) -> None:
    """
    Mutate graph in place: set edge['dynamic_cost'] = effective_travel_time(...) for every edge.

    Args:
        graph: NetworkX MultiDiGraph with edge attributes:
            - 'travel_time': free-flow travel time (seconds)
            - 'flood_risk': flood susceptibility [0, 1]
            - 'road_quality_score': quality [0, 1]
            - 'highway_class': for traffic profile lookup
        traffic_lookup: dict from build_traffic_lookup()
        hour: int (0-23)
        is_weekend: bool
        cost_cfg: CostModelConfig with flood_weight and quality_penalty_weight
        flood_event_edges: optional set of (u, v, key) tuples that are flooded THIS episode
        reveal_overrides: optional mapping of (u, v, key) to a revealed boolean.
            ``True`` and ``False`` replace the static susceptibility for that
            edge. ``None`` means the edge is not revealed and keeps its static
            susceptibility.
    """
    if flood_event_edges is None:
        flood_event_edges = set()
    if reveal_overrides is None:
        reveal_overrides = {}

    for u, v, key, data in graph.edges(keys=True, data=True):
        try:
            base_tt = float(data.get("travel_time", 0.0))
        except (TypeError, ValueError):
            base_tt = 0.0
        highway_class = data.get("highway_class", "unclassified")

        # Traffic multiplier from lookup, with fallback and warning
        lookup_key = (highway_class, hour, is_weekend)
        traffic_mult = float(traffic_lookup.get(lookup_key, 1.0))
        if lookup_key not in traffic_lookup:
            logger.warning(
                f"Missing traffic profile for {lookup_key} (edge {u}-{v}-{key}); "
                f"falling back to 1.0"
            )

        edge_id = (u, v, key)
        # Revealed ground truth replaces the prior only for this planning step.
        if edge_id in reveal_overrides and reveal_overrides[edge_id] is not None:
            flood_susc = 1.0 if reveal_overrides[edge_id] else 0.0
        elif edge_id in flood_event_edges:
            flood_susc = 1.0
        else:
            flood_susc = data.get("flood_risk", 0.0)
            # Handle NaN and string values
            try:
                flood_susc = float(flood_susc)
                import math
                if math.isnan(flood_susc):
                    flood_susc = 0.0
            except (ValueError, TypeError):
                flood_susc = 0.0

        # Road quality
        rq_score = data.get("road_quality_score", 1.0)
        # Handle NaN and string values
        try:
            rq_score = float(rq_score)
            import math
            if math.isnan(rq_score):
                rq_score = 1.0
        except (ValueError, TypeError):
            rq_score = 1.0

        # Compute effective travel time using shared cost model
        data["dynamic_cost"] = effective_travel_time(
            base_travel_time=base_tt,
            traffic_multiplier=traffic_mult,
            flood_susceptibility=flood_susc,
            road_quality_score=rq_score,
            flood_weight=cost_cfg.flood_weight,
            quality_penalty_weight=cost_cfg.quality_penalty_weight,
        )


def dynamic_shortest_path(
    graph,
    traffic_profile: pd.DataFrame,
    origin: tuple,
    destination: tuple,
    hour: int,
    is_weekend: bool,
    cost_cfg: CostModelConfig,
    flood_event_edges: set = None,
) -> dict:
    """
    Compute shortest path using dynamically-weighted edge costs.

    Args:
        graph: NetworkX MultiDiGraph with required edge attributes.
        traffic_profile: DataFrame from traffic_profile.parquet.
        origin: (longitude, latitude) tuple.
        destination: (longitude, latitude) tuple.
        hour: int (0-23) for traffic profile lookup.
        is_weekend: bool for traffic profile lookup.
        cost_cfg: CostModelConfig with flood_weight and quality_penalty_weight.
        flood_event_edges: optional set of (u, v, key) tuples flooded this episode.

    Returns:
        dict with keys:
            - "path": list of node IDs along the shortest path
            - "total_dynamic_cost": sum of dynamic costs along path (seconds)
            - "total_base_travel_time": sum of free-flow travel_time along path (seconds)
            - "origin_node": snapped origin node ID
            - "destination_node": snapped destination node ID
            - "path_length_m": sum of 'length' along path
            - "num_edges": number of edges in path
            - "edges_traversed": list of (u, v, key) tuples for each step
    """
    # Build O(1) traffic lookup
    traffic_lookup = build_traffic_lookup(traffic_profile)

    # Annotate dynamic costs on graph (mutates in place)
    annotate_dynamic_costs(graph, traffic_lookup, hour, is_weekend, cost_cfg, flood_event_edges)

    # Snap origin/destination to nearest graph nodes
    orig_node = ox.distance.nearest_nodes(graph, X=origin[0], Y=origin[1])
    dest_node = ox.distance.nearest_nodes(graph, X=destination[0], Y=destination[1])

    if orig_node == dest_node:
        return {
            "path": [orig_node],
            "total_dynamic_cost": 0.0,
            "total_base_travel_time": 0.0,
            "origin_node": orig_node,
            "destination_node": dest_node,
            "path_length_m": 0.0,
            "num_edges": 0,
            "edges_traversed": [],
        }

    # Run Dijkstra on the precomputed dynamic_cost attribute
    try:
        path = nx.shortest_path(graph, orig_node, dest_node, weight="dynamic_cost")
    except nx.NetworkXNoPath:
        return {
            "path": [],
            "total_dynamic_cost": float("inf"),
            "total_base_travel_time": float("inf"),
            "origin_node": orig_node,
            "destination_node": dest_node,
            "path_length_m": float("inf"),
            "num_edges": 0,
            "edges_traversed": [],
            "error": "No path exists between origin and destination",
        }

    # Compute totals and collect traversed edge keys
    total_dynamic_cost = 0.0
    total_base_travel_time = 0.0
    total_length = 0.0
    edges_traversed = []

    for u, v in zip(path[:-1], path[1:]):
        edge_data = graph.get_edge_data(u, v)
        if not edge_data:
            continue
        # Select the parallel edge with minimum dynamic_cost (the one Dijkstra would use)
        best_key, best_edge = min(edge_data.items(), key=lambda kv: kv[1].get("dynamic_cost", float("inf")))
        total_dynamic_cost += best_edge.get("dynamic_cost", 0.0)
        total_base_travel_time += best_edge.get("travel_time", 0.0)
        total_length += best_edge.get("length", 0.0)
        edges_traversed.append((u, v, best_key))

    return {
        "path": path,
        "total_dynamic_cost": total_dynamic_cost,
        "total_base_travel_time": total_base_travel_time,
        "origin_node": orig_node,
        "destination_node": dest_node,
        "path_length_m": total_length,
        "num_edges": len(path) - 1,
        "edges_traversed": edges_traversed,
    }


# --------------------------------------------------------------------------
# Replanning baseline
# --------------------------------------------------------------------------

def _nearest_node(graph, point):
    try:
        return ox.distance.nearest_nodes(graph, X=point[0], Y=point[1])
    except (KeyError, TypeError, ValueError):
        # OSMnx assumes integer node IDs when converting its nearest-node
        # result. Evaluation fixtures and some GraphML files use strings.
        return min(
            graph.nodes,
            key=lambda node: (
                (float(graph.nodes[node].get("x", 0.0)) - float(point[0])) ** 2
                + (float(graph.nodes[node].get("y", 0.0)) - float(point[1])) ** 2
            ),
        )


def _edges_within_hops(graph, node, radius: int) -> set[tuple]:
    """Return all directed edge IDs touching nodes in the local view."""
    undirected = graph.to_undirected(as_view=True)
    visible_nodes = nx.single_source_shortest_path_length(
        undirected, node, cutoff=radius
    )
    return {
        (u, v, key)
        for u, v, key in graph.edges(keys=True)
        if u in visible_nodes or v in visible_nodes
    }


def _traffic_multiplier(traffic_lookup, data, hour, is_weekend):
    highway_class = data.get("highway_class", "unclassified")
    lookup_key = (highway_class, hour, is_weekend)
    multiplier = traffic_lookup.get(lookup_key, 1.0)
    if lookup_key not in traffic_lookup:
        logger.warning(
            "Missing traffic profile for %s; falling back to 1.0", lookup_key
        )
    return multiplier


def _best_edge(graph, u, v):
    edge_data = graph.get_edge_data(u, v)
    if not edge_data:
        raise nx.NetworkXNoPath(f"No edge exists between {u!r} and {v!r}")
    return min(
        edge_data.items(),
        key=lambda item: item[1].get("dynamic_cost", float("inf")),
    )

def dynamic_shortest_path_with_replanning(
    graph, hazard_sim, traffic_lookup, origin, destination,
    hour, is_weekend, cost_cfg, reveal_radius_hops, max_steps,
) -> dict:
    """Replan one hop at a time using locally revealed flood realizations.

    ``max_steps`` is retained for API compatibility, but the authoritative
    cap is ``hazard_sim.config['env']['max_episode_steps']`` so this baseline
    follows the same episode limit as the environment.
    """
    current = _nearest_node(graph, origin)
    dest_node = _nearest_node(graph, destination)
    configured_max_steps = hazard_sim.config.get("env", {}).get(
        "max_episode_steps", max_steps
    )
    if configured_max_steps is None:
        configured_max_steps = max_steps
    if configured_max_steps is None or configured_max_steps <= 0:
        raise ValueError("env.max_episode_steps must be a positive integer")
    max_steps = int(configured_max_steps)
    path, total_realized_time, step = [current], 0.0, 0

    while current != dest_node and step < max_steps:
        hazard_sim.set_current_position(current)
        hazard_sim.maybe_trigger_event(step)

        # Re-plan from scratch each step using best current knowledge:
        # true ground truth for edges within reveal_radius_hops of `current`,
        # static predicted flood_susceptibility everywhere else.
        # Radius zero means no new flood information before commitment.
        nearby_edges = (
            _edges_within_hops(graph, current, reveal_radius_hops)
            if reveal_radius_hops > 0
            else set()
        )
        reveal_overrides = {}
        for edge_id in nearby_edges:
            reveal_overrides[edge_id] = hazard_sim.is_flooded(*edge_id)
        annotate_dynamic_costs(
            graph, traffic_lookup, hour, is_weekend, cost_cfg,
            reveal_overrides=reveal_overrides,
        )
        try:
            next_hop_path = nx.shortest_path(
                graph, current, dest_node, weight="dynamic_cost"
            )
        except nx.NetworkXNoPath:
            break
        next_node = next_hop_path[1]
        best_key, best_edge = _best_edge(graph, current, next_node)

        # Move first, then query the traversed edge from the new position. This
        # also makes radius zero a true no-lookahead mode.
        hazard_sim.set_current_position(next_node)
        is_flooded = hazard_sim.is_flooded(current, next_node, best_key)
        traffic_mult = _traffic_multiplier(
            traffic_lookup, best_edge, hour, is_weekend
        )
        base_travel_time = float(best_edge.get("travel_time", 0.0))
        road_quality_score = float(best_edge.get("road_quality_score", 1.0))
        flood_penalty_seconds = float(
            hazard_sim.config.get("env", {}).get("flood_penalty", 50)
        ) * 60.0
        realized_time = effective_travel_time(
            base_travel_time=base_travel_time,
            traffic_multiplier=traffic_mult,
            flood_susceptibility=0.0,
            road_quality_score=road_quality_score,
            flood_weight=cost_cfg.flood_weight,
            quality_penalty_weight=cost_cfg.quality_penalty_weight,
        ) + (flood_penalty_seconds if is_flooded else 0.0)

        total_realized_time += realized_time
        current, path, step = next_node, path + [next_node], step + 1

    return {
        "path": path, "total_realized_time": total_realized_time,
        "reached_destination": current == dest_node, "steps": step,
    }


if __name__ == "__main__":
    # Quick smoke test with the processed graph
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import osmnx as ox

    graph = ox.load_graphml(Path("data/processed/road_graph_full.graphml"))
    traffic_profile = pd.read_parquet(Path("data/processed/traffic_profile.parquet"))
    config = CostModelConfig.from_yaml(Path("config/config.yaml"))

    origin = (-0.23, 5.56)
    destination = (-0.22, 5.57)

    # Test at 8am on a weekday
    result = dynamic_shortest_path(
        graph=graph,
        traffic_profile=traffic_profile,
        origin=origin,
        destination=destination,
        hour=8,
        is_weekend=False,
        cost_cfg=config,
    )
    print(f"Path: {len(result['path'])} nodes, {result['num_edges']} edges")
    print(f"Total dynamic cost: {result['total_dynamic_cost']:.1f}s ({result['total_dynamic_cost']/60:.1f} min)")
    print(f"Total base travel time: {result['total_base_travel_time']:.1f}s ({result['total_base_travel_time']/60:.1f} min)")
    print(f"Path length: {result['path_length_m']:.0f}m")

    # Test at 2pm (off-peak)
    result2 = dynamic_shortest_path(
        graph=graph,
        traffic_profile=traffic_profile,
        origin=origin,
        destination=destination,
        hour=14,
        is_weekend=False,
        cost_cfg=config,
    )
    print(f"\nOff-peak (2pm):")
    print(f"Total dynamic cost: {result2['total_dynamic_cost']:.1f}s ({result2['total_dynamic_cost']/60:.1f} min)")
    print(f"Total base travel time: {result2['total_base_travel_time']:.1f}s ({result2['total_base_travel_time']/60:.1f} min)")
