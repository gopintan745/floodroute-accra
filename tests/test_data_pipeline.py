"""
Sanity checks on the processed data layers — cheap to run, catches broken
runs before they propagate into env/training.

TODO:
- test_graph_is_connected: road_graph.graphml has no (or minimal, documented)
  disconnected components
- test_flood_risk_in_range: flood_risk values on edges are in [0, 1]
- test_all_edges_have_required_attrs: every edge has base_travel_time,
  flood_risk, traffic_multiplier, road_quality_score before it reaches env/
"""

import pytest  # noqa: F401


def test_graph_is_connected():
    raise NotImplementedError("TODO")


def test_flood_risk_in_range():
    raise NotImplementedError("TODO")


def test_all_edges_have_required_attrs():
    raise NotImplementedError("TODO")
