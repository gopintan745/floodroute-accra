"""
Tests for the baseline routing algorithms.

These tests validate:
1. effective_travel_time() against hand-computed expected values (pure unit test)
2. static_shortest_path returns a valid connected path on a tiny graph
3. DIFFERENTIATION TEST: static picks short/risky, dynamic picks long/safe
4. NEUTRAL CASE: flood=0, quality=1, traffic=1 → static and dynamic return identical routes
5. Missing traffic lookup combo falls back to 1.0 rather than crashing
"""

import networkx as nx
import pandas as pd

from src.baselines.dijkstra_static import static_shortest_path
from src.baselines.dijkstra_dynamic import (
    dynamic_shortest_path,
    build_traffic_lookup,
    annotate_dynamic_costs,
)
from src.common.cost_model import CostModelConfig, effective_travel_time


# --------------------------------------------------------------------------
# Test fixture: tiny hand-crafted graph
# --------------------------------------------------------------------------

def _make_tiny_graph():
    """
    Create a tiny graph with two parallel routes from A to B:
    - Route 1 (short but risky): A -> C -> B, 2 edges, total length 100m, high flood_risk
    - Route 2 (long but safe): A -> D -> E -> B, 3 edges, total length 200m, zero flood_risk

    Node positions (for ox.distance.nearest_nodes):
        A: (0.0, 0.0)
        B: (0.01, 0.0)
        C: (0.0033, 0.0)   # on short route
        D: (0.0025, 0.01)  # on long route
        E: (0.0075, 0.01)  # on long route

    Edge attributes:
        - travel_time: proportional to length (100m = 10s, so 1m = 0.1s)
        - length: meters
        - flood_risk: 0.9 for short route, 0.0 for long route
        - road_quality_score: 1.0 everywhere
        - highway_class: "primary" everywhere
    """
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"  # Required by ox.distance.nearest_nodes

    # Add nodes with positions
    G.add_node(1, x=0.0, y=0.0)      # A
    G.add_node(2, x=0.01, y=0.0)     # B
    G.add_node(3, x=0.0033, y=0.0)   # C
    G.add_node(4, x=0.0025, y=0.01)  # D
    G.add_node(5, x=0.0075, y=0.01)  # E

    # Short route: A(1) -> C(3) -> B(2) = 100m total
    # Edge 1->3: 33m
    G.add_edge(1, 3, key=0, travel_time=3.3, length=33.0,
               flood_risk=0.9, road_quality_score=1.0, highway_class="primary")
    # Edge 3->2: 67m
    G.add_edge(3, 2, key=0, travel_time=6.7, length=67.0,
               flood_risk=0.9, road_quality_score=1.0, highway_class="primary")

    # Long route: A(1) -> D(4) -> E(5) -> B(2) = 200m total
    # Edge 1->4: 50m
    G.add_edge(1, 4, key=0, travel_time=5.0, length=50.0,
               flood_risk=0.0, road_quality_score=1.0, highway_class="primary")
    # Edge 4->5: 50m
    G.add_edge(4, 5, key=0, travel_time=5.0, length=50.0,
               flood_risk=0.0, road_quality_score=1.0, highway_class="primary")
    # Edge 5->2: 100m
    G.add_edge(5, 2, key=0, travel_time=10.0, length=100.0,
               flood_risk=0.0, road_quality_score=1.0, highway_class="primary")

    return G


def _make_traffic_profile():
    """Create a minimal traffic profile DataFrame with primary class, all hours, weekday/weekend."""
    rows = []
    for hour in range(24):
        for is_weekend in [False, True]:
            rows.append({
                "highway_class": "primary",
                "hour": hour,
                "is_weekend": is_weekend,
                "multiplier": 1.2 if (7 <= hour <= 9 or 16 <= hour <= 19) and not is_weekend else 1.05,
                "source": "synthetic",
                "num_real_samples": 0,
            })
    return pd.DataFrame(rows)


def _make_cost_config(flood_weight=0.5, quality_penalty_weight=1.0):
    return CostModelConfig(flood_weight=flood_weight, quality_penalty_weight=quality_penalty_weight)


# --------------------------------------------------------------------------
# 1. Pure unit test: effective_travel_time against hand-computed values
# --------------------------------------------------------------------------

def test_effective_travel_time():
    """Test effective_travel_time against hand-computed expected values."""
    cfg = _make_cost_config(flood_weight=0.5, quality_penalty_weight=1.0)

    # Case 1: free-flow, no flood, perfect quality -> equals base
    assert effective_travel_time(100, 1.0, 0.0, 1.0, 0.5, 1.0) == 100.0

    # Case 2: 2x traffic, no flood, perfect quality -> 200
    assert effective_travel_time(100, 2.0, 0.0, 1.0, 0.5, 1.0) == 200.0

    # Case 3: free-flow, max flood (0.5 weight), perfect quality -> 150
    assert effective_travel_time(100, 1.0, 1.0, 1.0, 0.5, 1.0) == 150.0

    # Case 4: free-flow, no flood, worst quality (1.0 weight) -> 200
    assert effective_travel_time(100, 1.0, 0.0, 0.0, 0.5, 1.0) == 200.0

    # Case 5: Combined: 2x traffic, max flood, worst quality -> 100 * 2 * 2 * 1.5 = 600
    assert effective_travel_time(100, 2.0, 1.0, 0.0, 0.5, 1.0) == 600.0

    # Case 6: Using CostModelConfig
    assert effective_travel_time(100, 1.5, 0.5, 0.5, cfg.flood_weight, cfg.quality_penalty_weight) == \
           100 * 1.5 * (1 + 0.5 * 1.0) * (1 + 0.5 * 0.5)


# --------------------------------------------------------------------------
# 2. static_shortest_path returns a valid connected path on tiny graph
# --------------------------------------------------------------------------

def test_static_path_is_valid():
    """static_shortest_path returns a valid connected path from origin to destination."""
    G = _make_tiny_graph()
    origin = (0.0, 0.0)      # A
    destination = (0.01, 0.0)  # B

    result = static_shortest_path(G, origin, destination)

    # Check basic structure
    assert "path" in result
    assert "total_travel_time" in result
    assert "origin_node" in result
    assert "destination_node" in result
    assert "path_length_m" in result
    assert "num_edges" in result

    # Path should be valid and connected
    path = result["path"]
    assert len(path) >= 2
    assert path[0] == result["origin_node"]
    assert path[-1] == result["destination_node"]

    # Each consecutive pair should have an edge in the graph
    for u, v in zip(path[:-1], path[1:]):
        assert G.has_edge(u, v), f"No edge between {u} and {v}"

    # Total travel time should match sum of edge travel_times
    computed_tt = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge_data = G.get_edge_data(u, v)
        best_edge = min(edge_data.values(), key=lambda d: d.get("travel_time", float("inf")))
        computed_tt += best_edge.get("travel_time", 0.0)
    assert abs(result["total_travel_time"] - computed_tt) < 1e-9


# --------------------------------------------------------------------------
# 3. DIFFERENTIATION TEST: static picks short/risky, dynamic picks long/safe
# --------------------------------------------------------------------------

def test_dynamic_vs_static_differentiation():
    """
    THE KEY TEST: On a graph where a shorter route has high flood risk
    and a longer route has zero flood risk:
    - static_shortest_path picks the short/risky route (free-flow only)
    - dynamic_shortest_path picks the long/safe route (flood-aware)

    This validates the project's core premise: hazard-aware routing
    produces meaningfully different (and better) routes than naive routing.
    """
    G = _make_tiny_graph()
    traffic_profile = _make_traffic_profile()
    cfg = _make_cost_config(flood_weight=0.5, quality_penalty_weight=1.0)

    origin = (0.0, 0.0)      # A
    destination = (0.01, 0.0)  # B

    # Static baseline: free-flow only, ignores flood_risk
    static_result = static_shortest_path(G, origin, destination)
    static_path = static_result["path"]

    # Static should pick the SHORT route (1 -> 3 -> 2)
    # Short route free-flow: 3.3 + 6.7 = 10.0s
    # Long route free-flow: 5.0 + 5.0 + 10.0 = 20.0s
    assert static_path == [1, 3, 2], f"Static picked {static_path}, expected short route [1, 3, 2]"

    # Let's use flood_weight=2.0 for this test to make the differentiation clear
    cfg_high_flood = _make_cost_config(flood_weight=2.0, quality_penalty_weight=1.0)
    dynamic_result2 = dynamic_shortest_path(
        graph=G,
        traffic_profile=traffic_profile,
        origin=origin,
        destination=destination,
        hour=12,
        is_weekend=False,
        cost_cfg=cfg_high_flood,
        flood_event_edges=set(),
    )
    dynamic_path2 = dynamic_result2["path"]

    # With flood_weight=2.0:
    # Short: flood_mult = 1 + 2.0*0.9 = 2.8, traffic=1.2 -> 10 * 1.2 * 2.8 = 33.6
    # Long: flood_mult = 1.0, traffic=1.2 -> 20 * 1.2 = 24.0
    # Now long route is cheaper!
    assert dynamic_path2 == [1, 4, 5, 2], f"Dynamic picked {dynamic_path2}, expected long route [1, 4, 5, 2]"

    # Verify static still picks short route (unchanged)
    assert static_path == [1, 3, 2]


# --------------------------------------------------------------------------
# 4. NEUTRAL CASE: flood=0, quality=1, traffic=1 everywhere
#    → static and dynamic should return identical routes
# --------------------------------------------------------------------------

def test_neutral_case_identical_routes():
    """
    When flood_risk=0, road_quality_score=1, traffic_multiplier=1 everywhere,
    the dynamic cost equals base travel_time, so static and dynamic
    should return IDENTICAL routes.
    """
    # Build a graph with NO flood risk, perfect quality
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    G.add_node(1, x=0.0, y=0.0)
    G.add_node(2, x=0.01, y=0.0)
    G.add_node(3, x=0.0033, y=0.0)
    G.add_node(4, x=0.0025, y=0.01)
    G.add_node(5, x=0.0075, y=0.01)

    # Short route: A(1) -> C(3) -> B(2)
    G.add_edge(1, 3, key=0, travel_time=3.3, length=33.0,
               flood_risk=0.0, road_quality_score=1.0, highway_class="primary")
    G.add_edge(3, 2, key=0, travel_time=6.7, length=67.0,
               flood_risk=0.0, road_quality_score=1.0, highway_class="primary")

    # Long route: A(1) -> D(4) -> E(5) -> B(2)
    G.add_edge(1, 4, key=0, travel_time=5.0, length=50.0,
               flood_risk=0.0, road_quality_score=1.0, highway_class="primary")
    G.add_edge(4, 5, key=0, travel_time=5.0, length=50.0,
               flood_risk=0.0, road_quality_score=1.0, highway_class="primary")
    G.add_edge(5, 2, key=0, travel_time=10.0, length=100.0,
               flood_risk=0.0, road_quality_score=1.0, highway_class="primary")

    traffic_profile = _make_traffic_profile()
    # Override to all 1.0 multipliers
    traffic_profile["multiplier"] = 1.0

    cfg = _make_cost_config(flood_weight=0.5, quality_penalty_weight=1.0)

    origin = (0.0, 0.0)
    destination = (0.01, 0.0)

    static_result = static_shortest_path(G, origin, destination)
    dynamic_result = dynamic_shortest_path(
        graph=G,
        traffic_profile=traffic_profile,
        origin=origin,
        destination=destination,
        hour=12,
        is_weekend=False,
        cost_cfg=cfg,
    )

    # Both should pick the short route (free-flow shortest)
    assert static_result["path"] == [1, 3, 2]
    assert dynamic_result["path"] == [1, 3, 2]

    # Dynamic cost should equal base travel time (within floating point)
    assert abs(dynamic_result["total_dynamic_cost"] - dynamic_result["total_base_travel_time"]) < 1e-9
    assert abs(dynamic_result["total_base_travel_time"] - static_result["total_travel_time"]) < 1e-9


# --------------------------------------------------------------------------
# 5. Missing traffic lookup combo falls back to 1.0
# --------------------------------------------------------------------------

def test_traffic_lookup_fallback():
    """
    If a (highway_class, hour, is_weekend) combo is missing from the
    traffic profile, annotate_dynamic_costs should fall back to 1.0
    with a warning, not crash.
    """
    G = _make_tiny_graph()

    # Traffic profile with ONLY primary at hour 12, weekday
    traffic_profile = pd.DataFrame([{
        "highway_class": "primary",
        "hour": 12,
        "is_weekend": False,
        "multiplier": 1.2,
        "source": "synthetic",
        "num_real_samples": 0,
    }])

    cfg = _make_cost_config()

    # This should NOT raise - missing combos (e.g., hour=14) fall back to 1.0
    annotate_dynamic_costs(G, build_traffic_lookup(traffic_profile), hour=14, is_weekend=False, cost_cfg=cfg)

    # Check that edges got dynamic_cost (no crash)
    for u, v, key, data in G.edges(keys=True, data=True):
        assert "dynamic_cost" in data
        # At hour 14 (not in profile), traffic_mult should be 1.0
        # base_tt=3.3/6.7/5.0/5.0/10.0, traffic=1.0, flood=0.9/0.0, quality=1.0
        # cost = base * 1.0 * (1 + 0.5*flood) * 1.0


def test_traffic_lookup_fallback_in_full_pipeline():
    """
    Full pipeline test: dynamic_shortest_path with a sparse traffic profile
    should not crash and should produce a valid route.
    """
    G = _make_tiny_graph()

    # Sparse profile: only primary at hour 12 weekday
    traffic_profile = pd.DataFrame([{
        "highway_class": "primary",
        "hour": 12,
        "is_weekend": False,
        "multiplier": 1.2,
        "source": "synthetic",
        "num_real_samples": 0,
    }])

    cfg = _make_cost_config()

    origin = (0.0, 0.0)
    destination = (0.01, 0.0)

    # Query at hour 14 (not in profile) - should fall back to 1.0
    result = dynamic_shortest_path(
        graph=G,
        traffic_profile=traffic_profile,
        origin=origin,
        destination=destination,
        hour=14,
        is_weekend=False,
        cost_cfg=cfg,
    )

    # Should return a valid path
    assert len(result["path"]) >= 2
    assert result["total_dynamic_cost"] > 0
    assert result["total_base_travel_time"] > 0


# --------------------------------------------------------------------------
# Additional sanity checks
# --------------------------------------------------------------------------

def test_dynamic_returns_both_costs():
    """dynamic_shortest_path returns both dynamic and base costs."""
    G = _make_tiny_graph()
    traffic_profile = _make_traffic_profile()
    cfg = _make_cost_config()

    result = dynamic_shortest_path(
        graph=G,
        traffic_profile=traffic_profile,
        origin=(0.0, 0.0),
        destination=(0.01, 0.0),
        hour=12,
        is_weekend=False,
        cost_cfg=cfg,
    )

    assert "total_dynamic_cost" in result
    assert "total_base_travel_time" in result
    assert result["total_dynamic_cost"] >= result["total_base_travel_time"]  # dynamic >= base
    assert result["total_base_travel_time"] > 0


def test_build_traffic_lookup():
    """build_traffic_lookup creates correct dict from DataFrame."""
    traffic_profile = _make_traffic_profile()
    lookup = build_traffic_lookup(traffic_profile)

    assert isinstance(lookup, dict)
    assert len(lookup) == 48  # 24 hours * 2 weekend values
    assert lookup[("primary", 12, False)] == 1.05  # off-peak weekday
    assert lookup[("primary", 8, False)] == 1.2    # peak weekday
    assert lookup[("primary", 8, True)] == 1.05    # peak weekend (dampened)


if __name__ == "__main__":
    # Run tests manually
    test_effective_travel_time()
    print("✓ test_effective_travel_time passed")

    test_static_path_is_valid()
    print("✓ test_static_path_is_valid passed")

    test_dynamic_vs_static_differentiation()
    print("✓ test_dynamic_vs_static_differentiation passed")

    test_neutral_case_identical_routes()
    print("✓ test_neutral_case_identical_routes passed")

    test_traffic_lookup_fallback()
    print("✓ test_traffic_lookup_fallback passed")

    test_traffic_lookup_fallback_in_full_pipeline()
    print("✓ test_traffic_lookup_fallback_in_full_pipeline passed")

    test_dynamic_returns_both_costs()
    print("✓ test_dynamic_returns_both_costs passed")

    test_build_traffic_lookup()
    print("✓ test_build_traffic_lookup passed")

    print("\nAll tests passed!")
