"""
Build a per-edge (or per-cell, then sampled per-edge) flood-risk layer from
elevation and rainfall data.

Input:  data/raw/dem/*, data/raw/rainfall/*
Output: data/processed/flood_risk.tif  (raster)
        + a per-edge flood-risk column merged in build_graph_costs.py

TODO:
- Load DEM tiles (rasterio), compute a simple flow-accumulation / low-lying-area
  proxy (full hydrological modeling is out of scope for a first pass — a
  coarse "low elevation + near known drainage channel" heuristic is a
  reasonable v1)
- Load rainfall data (CHIRPS/GPM) and define a threshold or intensity curve
  for "flood-triggering" rainfall
- Combine into a flood-risk score per raster cell, 0-1
- Sample the raster along each road edge's geometry to get an edge-level
  flood-risk score
- Document every threshold/heuristic choice in docs/design_decisions.md —
  these are exactly the judgment calls that need to be defensible later
"""

import rasterio  # noqa: F401


def compute_flood_risk_raster(dem_path: str, rainfall_path: str) -> None:
    """Combine DEM + rainfall into a flood-risk raster. TODO."""
    raise NotImplementedError("TODO")


def sample_risk_along_edges(graph, risk_raster_path: str):
    """Attach a flood_risk attribute to each graph edge. TODO."""
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    raise NotImplementedError("TODO: wire up CLI")
