"""
Download and clean the drivable road graph for the Kaneshie study area.

Input:  config.yaml study_area.bbox
Output: data/processed/road_graph.graphml
        data/processed/road_graph_edges.geojson  (for QGIS / satellite sanity-check)

Usage:
    python -m src.data_pipeline.fetch_osm
    (run from the repo root, with config/config.yaml already created from
    config/config.example.yaml)

IMPORTANT: this script calls out to OSM's Overpass API over the network.
It cannot be run in a sandboxed environment without that access — run it
on your own machine.

Manual step after running this: open road_graph_edges.geojson in QGIS
alongside a satellite basemap and visually check the Kaneshie area for
missing or misaligned roads. Accra's OSM coverage is decent in central
areas but uneven at the edges — don't trust the raw pull without a look.
"""

import argparse
from pathlib import Path

import networkx as nx
import osmnx as ox
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
OUTPUT_GRAPHML = REPO_ROOT / "data" / "processed" / "road_graph.graphml"
OUTPUT_GEOJSON = REPO_ROOT / "data" / "processed" / "road_graph_edges.geojson"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. Copy config/config.example.yaml to "
            "config/config.yaml first (it already has the Kaneshie bbox filled in)."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_graph(bbox: tuple[float, float, float, float], network_type: str = "drive") -> "nx.MultiDiGraph":
    """Fetch the raw OSM drivable graph for a (south, west, north, east) bbox.

    Handles both osmnx >=2.0 (bbox tuple as (west, south, east, north)) and
    the older <2.0 positional (north, south, east, west) signature.
    """
    south, west, north, east = bbox
    try:
        return ox.graph_from_bbox(bbox=(west, south, east, north), network_type=network_type)
    except TypeError:
        return ox.graph_from_bbox(north, south, east, west, network_type=network_type)


def clean_graph(graph: "nx.MultiDiGraph") -> "nx.MultiDiGraph":
    """Simplify, keep only the largest strongly-connected component (a
    routing environment doesn't work if some nodes are unreachable), and
    make sure length/travel-time base attributes are present."""
    if not graph.graph.get("simplified", False):
        graph = ox.simplify_graph(graph)

    if not nx.is_strongly_connected(graph):
        components = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
        largest = components[0]
        dropped = sum(len(c) for c in components[1:])
        print(
            f"Dropping {dropped} node(s) across {len(components) - 1} disconnected "
            f"component(s); keeping the largest component ({len(largest)} nodes). "
            "If this drops a meaningful chunk of the study area, the bbox or "
            "OSM coverage likely needs adjusting — check the GeoJSON output."
        )
        graph = graph.subgraph(largest).copy()

    # Add free-flow base travel time (minutes) from length + speed, as a
    # starting point later layers (flood risk, traffic) will multiply against.
    graph = ox.add_edge_speeds(graph)   # fills missing maxspeed via OSM highway-type defaults
    graph = ox.add_edge_travel_times(graph)  # adds 'travel_time' in seconds from length/speed

    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and clean the Kaneshie study-area road graph.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config = load_config(args.config)
    bbox = tuple(config["study_area"]["bbox"])
    network_type = config["data_sources"]["osm"].get("network_type", "drive")
    area_name = config["study_area"].get("name", "study area")

    print(f"Fetching '{area_name}' graph — bbox={bbox}, network_type={network_type} ...")
    graph = fetch_graph(bbox, network_type=network_type)
    print(f"Raw graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    graph = clean_graph(graph)
    print(f"Cleaned graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    OUTPUT_GRAPHML.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, OUTPUT_GRAPHML)
    print(f"Saved GraphML -> {OUTPUT_GRAPHML}")

    edges_gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    edges_gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print(f"Saved edge GeoJSON -> {OUTPUT_GEOJSON}")
    print("Next: open the GeoJSON in QGIS over a satellite basemap and check "
          "for missing/misaligned roads before moving to Phase 2.")


if __name__ == "__main__":
    main()

