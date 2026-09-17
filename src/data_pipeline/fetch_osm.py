"""
Download and clean the drivable road graph for the study area.

Input:  config.yaml study_area.bbox
Output: data/processed/road_graph.graphml

TODO:
- Pull the graph with osmnx.graph_from_bbox(..., network_type="drive")
- Manually cross-check against satellite imagery for missing/incorrect roads
  (Accra's OSM coverage is uneven outside central areas — budget real time
  for this step, don't assume the raw pull is trustworthy)
- Simplify the graph (osmnx.simplify_graph) and ensure it's strongly connected;
  drop or flag disconnected components
- Attach base edge attributes: length, highway type, speed limit if tagged
- Save both a human-inspectable GeoJSON (for QGIS sanity checks) and the
  GraphML used downstream
"""

import networkx as nx  # noqa: F401
import osmnx as ox  # noqa: F401


def fetch_graph(bbox: tuple[float, float, float, float], network_type: str = "drive") -> "nx.MultiDiGraph":
    """Fetch the raw OSM drivable graph for a (south, west, north, east) bbox."""
    raise NotImplementedError("TODO: wrap osmnx.graph_from_bbox")


def clean_graph(graph: "nx.MultiDiGraph") -> "nx.MultiDiGraph":
    """Simplify, ensure connectivity, and attach base attributes."""
    raise NotImplementedError("TODO")


def main() -> None:
    # TODO: load config, run fetch_graph -> clean_graph, save to
    # data/processed/road_graph.graphml
    raise NotImplementedError("TODO: wire up CLI / config loading")


if __name__ == "__main__":
    main()
