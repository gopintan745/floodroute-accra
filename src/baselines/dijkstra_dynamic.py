"""
Baseline 2: shortest path re-weighted using currently-known hazard risk
scores (flood_risk, traffic_multiplier) — the "predict conditions, then
search" approach. This is the stronger, fairer baseline: same information
as the RL agent gets, no learning.

Three-function structure:
1. build_traffic_lookup() — converts traffic_profile DataFrame to O(1) dict
2. annotate_dynamic_costs() — mutates graph edges with dynamic_cost
3. dynamic_shortest_path() — main entry point, returns both dynamic and base costs

The replanning variant (`dynamic_shortest_path_with_replanning`) is
NOT implemented here — it depends on HazardSimulator (Phase 4) which
defines what "revealed" means. Will be added in Phase 4.
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
    """
    if flood_event_edges is None:
        flood_event_edges = set()

    for u, v, key, data in graph.edges(keys=True, data=True):
        base_tt = data.get("travel_time", 0.0)
        highway_class = data.get("highway_class", "unclassified")

        # Traffic multiplier from lookup, with fallback and warning
        lookup_key = (highway_class, hour, is_weekend)
        traffic_mult = traffic_lookup.get(lookup_key, 1.0)
        if lookup_key not in traffic_lookup:
            logger.warning(
                f"Missing traffic profile for {lookup_key} (edge {u}-{v}-{key}); "
                f"falling back to 1.0"
            )

        # Flood susceptibility: episode-specific flood overrides static flood_risk
        if (u, v, key) in flood_event_edges:
            flood_susc = 1.0  # fully flooded this episode
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
# Replanning stub — to be implemented in Phase 4 when HazardSimulator exists
# --------------------------------------------------------------------------

def dynamic_shortest_path_with_replanning(env_or_graph, origin, destination):
    """
    Re-plans at each step using revealed information, for a fairer
    comparison against the RL agent's adaptive behavior.

    NOT YET IMPLEMENTED — requires HazardSimulator (Phase 4) to define:
    - What "revealed" conditions look like at each step
    - The observation/reveal radius mechanism
    - Episode termination on flood encounter

    This stub exists as a placeholder and raises NotImplementedError.
    """
    raise NotImplementedError(
        "dynamic_shortest_path_with_replanning requires HazardSimulator (Phase 4). "
        "Will be implemented when the environment's observation/reveal model is defined."
    )


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
