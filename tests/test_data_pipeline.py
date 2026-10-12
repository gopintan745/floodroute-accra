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

from pathlib import Path

import networkx as nx
import pytest

from src.agents.training_utils import GRAPH_PATH

GRAPH_FILE = Path(GRAPH_PATH)


@pytest.mark.skipif(not GRAPH_FILE.exists(), reason="Processed road_graph_full.graphml not found")
def test_graph_is_connected():
    graph = nx.read_graphml(GRAPH_FILE)
    assert len(graph) > 0, "Graph is empty"
    assert nx.is_strongly_connected(graph), "Road graph is not strongly connected"


@pytest.mark.skipif(not GRAPH_FILE.exists(), reason="Processed road_graph_full.graphml not found")
def test_flood_risk_in_range():
    import math
    graph = nx.read_graphml(GRAPH_FILE)
    for u, v, k, data in graph.edges(keys=True, data=True):
        risk = float(data.get("flood_risk", data.get("flood_susceptibility", 0.0)))
        if not math.isnan(risk):
            assert 0.0 <= risk <= 1.0, f"Edge ({u}, {v}, {k}) has out-of-range risk: {risk}"


@pytest.mark.skipif(not GRAPH_FILE.exists(), reason="Processed road_graph_full.graphml not found")
def test_all_edges_have_required_attrs():
    graph = nx.read_graphml(GRAPH_FILE)
    required_attrs = {"travel_time", "flood_risk", "road_quality_score", "highway_class"}
    for u, v, k, data in graph.edges(keys=True, data=True):
        missing = required_attrs - set(data.keys())
        assert not missing, f"Edge ({u}, {v}, {k}) missing required attributes: {missing}"
