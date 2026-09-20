import networkx as nx
import pytest

from src.env.hazard_simulator import HazardSimulator


def _graph():
    graph = nx.MultiDiGraph()
    graph.add_edge("a", "b", key=0, flood_risk=0.0)
    graph.add_edge("b", "c", key=0, flood_risk=1.0)
    graph.add_edge("c", "d", key=0, flood_susceptibility=0.5)
    return graph


def _config():
    return {
        "env": {
            "train_flood_day_rate": 0.35,
            "mid_episode_event_base_rate": 1.0,
            "reveal_radius_hops": 1,
        }
    }


def _climatology():
    return {"by_month": {month: {"heavy_day_fraction": month / 12} for month in range(1, 13)}}


def test_mode_separation_uses_train_and_eval_rates():
    train = HazardSimulator(_graph(), _climatology(), _config(), "train", seed=7)
    eval_sim = HazardSimulator(_graph(), _climatology(), _config(), "eval", seed=7)
    train_days = [train.reset_episode()["is_flood_day"] for _ in range(2000)]
    eval_days = [eval_sim.reset_episode()["is_flood_day"] for _ in range(2000)]

    assert abs(sum(train_days) / len(train_days) - 0.35) < 0.04
    assert abs(sum(eval_days) / len(eval_days) - 0.5417) < 0.04


def test_same_seed_reproduces_resets_and_events():
    first = HazardSimulator(_graph(), _climatology(), _config(), "train", seed=11)
    second = HazardSimulator(_graph(), _climatology(), _config(), "train", seed=11)

    for step in range(20):
        assert first.reset_episode() == second.reset_episode()
        assert first.maybe_trigger_event(step) == second.maybe_trigger_event(step)


def test_non_flood_day_stays_clean():
    simulator = HazardSimulator(
        _graph(),
        _climatology(),
        {"env": {"train_flood_day_rate": 0.0, "reveal_radius_hops": 1}},
        "train",
        seed=1,
    )
    result = simulator.reset_episode()

    assert result["is_flood_day"] is False
    assert simulator.maybe_trigger_event(1) == []
    assert not simulator.flooded_edges


def test_flooding_is_monotonic():
    simulator = HazardSimulator(_graph(), _climatology(), _config(), "train", seed=2)
    simulator.reset_episode()
    simulator.is_flood_day = True
    simulator.flooded_edges = {("a", "b", 0)}

    simulator.maybe_trigger_event(1)
    flooded_after_first_event = set(simulator.flooded_edges)
    simulator.maybe_trigger_event(2)

    assert flooded_after_first_event <= simulator.flooded_edges
    simulator.set_current_position("a")
    assert simulator.is_flooded("a", "b", 0)


def test_is_flooded_requires_reset():
    simulator = HazardSimulator(_graph(), _climatology(), _config(), "train", seed=3)

    with pytest.raises(RuntimeError, match="reset_episode"):
        simulator.is_flooded("a", "b", 0)


def test_reveal_radius_exposes_nearby_edges_only():
    simulator = HazardSimulator(_graph(), _climatology(), _config(), "train", seed=4)
    simulator.reset_episode()
    simulator.is_flood_day = True
    simulator.flooded_edges = {("b", "c", 0), ("c", "d", 0)}
    simulator.set_current_position("a")

    assert simulator.is_flooded("a", "b", 0) is False
    assert simulator.is_flooded("b", "c", 0) is True
    assert simulator.is_flooded("c", "d", 0) is None


def test_position_must_be_set_before_local_query():
    simulator = HazardSimulator(_graph(), _climatology(), _config(), "train", seed=5)
    simulator.reset_episode()

    with pytest.raises(RuntimeError, match="set_current_position"):
        simulator.is_flooded("a", "b", 0)
