"""
Baseline 1: shortest path using fixed (base/expected) edge weights — no
knowledge of the current episode's actual hazard realization.

This is the "naive" comparator: what a driver following a normal map app
with no live flood/traffic awareness would do.

Pure free-flow baseline — no flood, traffic, or quality awareness.
This is "a normal map app with no live conditions."

Usage:
    origin/destination are (lon, lat) tuples
    snap to nearest graph nodes via ox.distance.nearest_nodes
    run nx.shortest_path(graph, orig_node, dest_node, weight="travel_time")
    return {"path": [node_ids], "total_travel_time": seconds}

The returned path is evaluated against the HazardSimulator's true
conditions for that episode (the route doesn't adapt if it hits a
flooded edge — that's the point, it's the naive baseline).
"""

import networkx as nx
import osmnx as ox


def static_shortest_path(graph, origin: tuple, destination: tuple) -> dict:
    """
    Compute shortest path using free-flow travel_time as edge weight.

    Args:
        graph: NetworkX MultiDiGraph with 'travel_time' edge attribute (seconds).
        origin: (longitude, latitude) tuple.
        destination: (longitude, latitude) tuple.

    Returns:
        dict with keys:
            - "path": list of node IDs along the shortest path
            - "total_travel_time": sum of 'travel_time' along path (seconds)
            - "origin_node": snapped origin node ID
            - "destination_node": snapped destination node ID
            - "path_length_m": sum of 'length' along path (meters, if available)
            - "num_edges": number of edges in path
    """
    # Snap origin/destination to nearest graph nodes
    orig_node = ox.distance.nearest_nodes(graph, X=origin[0], Y=origin[1])
    dest_node = ox.distance.nearest_nodes(graph, X=destination[0], Y=destination[1])

    if orig_node == dest_node:
        return {
            "path": [orig_node],
            "total_travel_time": 0.0,
            "origin_node": orig_node,
            "destination_node": dest_node,
            "path_length_m": 0.0,
            "num_edges": 0,
        }

    # Compute shortest path using free-flow travel_time as weight
    try:
        path = nx.shortest_path(graph, orig_node, dest_node, weight="travel_time")
    except nx.NetworkXNoPath:
        return {
            "path": [],
            "total_travel_time": float("inf"),
            "origin_node": orig_node,
            "destination_node": dest_node,
            "path_length_m": float("inf"),
            "num_edges": 0,
            "error": "No path exists between origin and destination",
        }

    # Compute total travel time and length along the path
    total_travel_time = 0.0
    total_length = 0.0
    for u, v in zip(path[:-1], path[1:]):
        # Get the first edge (MultiDiGraph may have parallel edges)
        edge_data = graph.get_edge_data(u, v)
        if edge_data:
            # Take the edge with minimum travel_time (fastest parallel edge)
            best_edge = min(edge_data.values(), key=lambda d: d.get("travel_time", float("inf")))
            total_travel_time += best_edge.get("travel_time", 0.0)
            total_length += best_edge.get("length", 0.0)

    return {
        "path": path,
        "total_travel_time": total_travel_time,
        "origin_node": orig_node,
        "destination_node": dest_node,
        "path_length_m": total_length,
        "num_edges": len(path) - 1,
    }


def evaluate_route(graph, path: list, hazard_simulator) -> dict:
    """
    Evaluate a pre-computed route against the HazardSimulator's true conditions.

    This is the "naive" evaluation — the route is fixed in advance and does not
    adapt to hazards encountered along the way.

    Args:
        graph: NetworkX MultiDiGraph with edge attributes.
        path: List of node IDs from static_shortest_path().
        hazard_simulator: HazardSimulator instance with true edge conditions.

    Returns:
        dict with evaluation metrics including actual travel time, penalties, etc.
    """
    if not path or len(path) < 2:
        return {
            "actual_travel_time": float("inf"),
            "flood_penalties": 0,
            "edges_hit_flooded": 0,
            "success": False,
        }

    actual_travel_time = 0.0
    flood_penalties = 0
    edges_hit_flooded = 0

    for u, v in zip(path[:-1], path[1:]):
        edge_data = graph.get_edge_data(u, v)
        if not edge_data:
            continue
        best_edge = min(edge_data.values(), key=lambda d: d.get("travel_time", float("inf")))

        # Get true conditions from hazard simulator
        conditions = hazard_simulator.get_edge_conditions(u, v, best_edge)

        # Base travel time + traffic multiplier
        base_tt = best_edge.get("travel_time", 0.0)
        traffic_mult = conditions.get("traffic_multiplier", 1.0)
        actual_travel_time += base_tt * traffic_mult

        # Flood penalty if edge is flooded in this episode
        if conditions.get("is_flooded", False):
            edges_hit_flooded += 1
            # Apply flood penalty from environment config (default 50 min = 3000 sec)
            flood_penalties += 3000  # 50 minutes in seconds
            actual_travel_time += 3000

    success = edges_hit_flooded == 0

    return {
        "actual_travel_time": actual_travel_time,
        "flood_penalties": flood_penalties,
        "edges_hit_flooded": edges_hit_flooded,
        "success": success,
        "base_travel_time": sum(
            min(graph.get_edge_data(u, v).values(), key=lambda d: d.get("travel_time", float("inf"))).get("travel_time", 0.0)
            for u, v in zip(path[:-1], path[1:])
        ) if len(path) >= 2 else 0.0,
    }


if __name__ == "__main__":
    # Quick smoke test with the processed graph
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data_pipeline.fetch_osm import load_config
    import osmnx as ox

    config = load_config()
    graph = ox.load_graphml(config["study_area"]["bbox"])

    # Test with two points in the study area
    origin = (-0.23, 5.56)  # Kaneshie area
    destination = (-0.22, 5.57)

    result = static_shortest_path(graph, origin, destination)
    print(f"Path: {len(result['path'])} nodes, {result['num_edges']} edges")
    print(f"Total travel time: {result['total_travel_time']:.1f}s ({result['total_travel_time']/60:.1f} min)")
    print(f"Path length: {result['path_length_m']:.0f}m")
