import json

import networkx as nx

from src.evaluation.compare_results import compare_results, write_summary
from src.evaluation.metrics import hit_blocked_edge, summarize, travel_time
from src.evaluation.run_scenarios import generate_scenarios, save_scenarios


def _graph():
    graph = nx.MultiDiGraph()
    graph.add_node("a", x=0.0, y=0.0)
    graph.add_node("b", x=1.0, y=0.0)
    graph.add_node("c", x=2.0, y=0.0)
    graph.add_edge("a", "b", key=0, travel_time=2.0, flood_risk=0.0, road_quality_score=1.0)
    graph.add_edge("b", "c", key=0, travel_time=3.0, flood_risk=1.0, road_quality_score=1.0)
    return graph


def _config():
    return {
        "env": {
            "max_episode_steps": 10,
            "reveal_radius_hops": 1,
            "train_flood_day_rate": 1.0,
            "mid_episode_event_base_rate": 0.0,
        },
        "cost_model": {"flood_weight": 0.5, "quality_penalty_weight": 1.0},
    }


def test_generate_scenarios_returns_reachable_natural_and_flood_sets(tmp_path):
    climatology = tmp_path / "climatology.json"
    climatology.write_text(
        json.dumps({"monthly_heavy_rain_frequency": [1.0] * 12}),
        encoding="utf-8",
    )
    scenarios = generate_scenarios(
        _graph(),
        ("missing-traffic.parquet", str(climatology)),
        _config(),
        num_scenarios=3,
        num_flood_scenarios=2,
        seed=11,
    )

    assert len(scenarios["natural"]) == 3
    assert len(scenarios["flood_stratified"]) == 2
    assert all(record["is_flood_day"] for record in scenarios["flood_stratified"])
    assert all(nx.has_path(_graph(), record["origin"], record["destination"]) for record in scenarios["natural"])
    output = save_scenarios(scenarios, tmp_path / "scenarios.json")
    assert json.loads(output.read_text(encoding="utf-8")) == scenarios


def test_metrics_and_comparison_summarize_route_results(tmp_path):
    realization = {
        "edges": {
            "('a', 'b', 0)": {"travel_time": 2.0, "traffic_multiplier": 1.5},
            "('b', 'c', 0)": {"travel_time": 3.0, "is_flooded": True, "flood_penalty": 10.0},
        }
    }
    route = [("a", "b", 0), ("b", "c", 0)]

    assert travel_time(route, realization) == 16.0
    assert hit_blocked_edge(route, realization)

    results = [
        {"actual_travel_time": 4.0, "edges_hit_flooded": 0, "success": True},
        {"actual_travel_time": 8.0, "edges_hit_flooded": 1, "success": False},
    ]
    summary = summarize(results)
    assert summary["mean_travel_time"] == 6.0
    assert summary["failure_rate"] == 0.5

    comparison = compare_results({"rl_agent": results})
    assert comparison["methods"]["rl_agent"]["episodes"] == 2
    summary_path = write_summary(comparison, tmp_path / "summary.md")
    assert "rl_agent" in summary_path.read_text(encoding="utf-8")