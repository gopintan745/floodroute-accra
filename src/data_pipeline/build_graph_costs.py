"""
Merge road graph + flood-risk layer + traffic profile + road-quality labels
into a single graph with all per-edge attributes the environment needs.

Input:  data/processed/road_graph.graphml
        data/processed/flood_risk.tif (via build_flood_layer.sample_risk_along_edges)
        data/processed/traffic_profile.parquet (or synthetic generator)
        data/processed/road_quality.geojson
Output: data/processed/road_graph_full.graphml
        (every edge has: length, base_travel_time, flood_risk, traffic_multiplier,
         road_quality_score)

TODO:
- If real traffic samples are sparse, generate a synthetic time-of-day
  congestion multiplier per road class (e.g. arterial roads get heavier
  rush-hour multipliers than residential streets) and blend with any real
  samples you do have
- Road quality: since no systematic dataset exists, start with a manual or
  heuristic label (e.g. from satellite/Mapillary spot checks) rather than
  trying to learn this from scratch — log the labeling method used
- Keep raw components (flood_risk, traffic_multiplier, road_quality_score)
  as SEPARATE edge attributes rather than collapsing into one cost up front —
  the env needs them separable to build partial-observability logic later
"""


def merge_layers(graph, flood_raster_path, traffic_profile_path, road_quality_path):
    """Attach all derived attributes to graph edges. TODO."""
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    raise NotImplementedError("TODO: wire up CLI")
