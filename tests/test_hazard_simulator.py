"""Group A calibration and reproducibility checks for graded flood beliefs."""

import networkx as nx
import numpy as np

from src.common.cost_model import CostModelConfig, expected_cost, level_cost
from src.env.hazard_simulator import HazardSimulator


def _graph():
    graph = nx.MultiDiGraph()
    graph.add_edges_from([
        ("a", "b", 0, {"flood_susceptibility": 0.2}),
        ("b", "c", 0, {"flood_susceptibility": 0.6}),
        ("c", "d", 0, {"flood_susceptibility": 0.9}),
    ])
    return graph


def _config(signal_mode="noisy"):
    return {
        "env": {"train_flood_day_rate": 1.0, "reveal_radius_hops": 1, "max_episode_steps": 8},
        "flood_model": {
            "intensity_weight_alpha": 0.5,
            "deep_fraction_min": 0.1,
            "deep_fraction_max": 0.7,
            "mid_episode_event_base_rate": 0.0,
            "signal_mode": signal_mode,
            "signal_confusion": {
                "hop0": [[0.95, 0.05, 0.0], [0.05, 0.9, 0.05], [0.0, 0.1, 0.9]],
                "hop1": [[0.85, 0.13, 0.02], [0.12, 0.76, 0.12], [0.03, 0.22, 0.75]],
            },
        },
    }


def _sim(seed=123, signal_mode="noisy"):
    return HazardSimulator(_graph(), {"monthly_heavy_rain_frequency": [1.0] * 12}, _config(signal_mode), "train", seed)


def test_prior_calibration_over_susceptibility_and_intensity_grid():
    simulator = _sim()
    rng = np.random.default_rng(19)
    sample_count = 200_000
    for susceptibility in (0.2, 0.6, 0.9):
        for intensity in (0.0, 0.5, 1.0):
            simulator.is_flood_day = True
            simulator.intensity = intensity
            edge = next(edge for edge, value in simulator.susceptibility.items() if value == susceptibility)
            probabilities = simulator.prior(*edge)
            samples = rng.choice(3, size=sample_count, p=probabilities)
            observed = np.bincount(samples, minlength=3) / sample_count
            standard_error = np.sqrt(probabilities * (1.0 - probabilities) / sample_count)
            assert np.all(np.abs(observed - probabilities) <= 4.0 * standard_error + 1e-4)


def test_non_flood_day_is_dry_and_belief_is_degenerate():
    simulator = _sim()
    context = simulator.reset_episode({"is_flood_day": False, "intensity": 0.0})
    simulator.set_current_position("a")

    assert context["intensity"] == 0.0
    assert all(level == 0 for level in simulator.true_levels.values())
    for edge in simulator.edges:
        np.testing.assert_array_equal(simulator.prior(*edge), [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(simulator.belief(*edge), [1.0, 0.0, 0.0])


def test_exact_signal_returns_latent_level_one_hot():
    simulator = _sim(signal_mode="exact")
    simulator.reset_episode({"is_flood_day": True, "intensity": 0.7})
    simulator.set_current_position("a")
    for edge, level in simulator.true_levels.items():
        if simulator._edge_hop(edge) <= simulator.reveal_radius_hops:
            belief = simulator.belief(*edge)
            assert belief[level] == 1.0
            assert np.count_nonzero(belief) == 1


def test_bayesian_posterior_calibration_from_simulator_signals():
    graph = nx.MultiDiGraph()
    graph.add_edge("a", "b", key=0, flood_susceptibility=0.65)
    simulator = HazardSimulator(graph, {}, _config(), "train", seed=101)
    grouped = {}
    for _ in range(12_000):
        simulator.reset_episode({"is_flood_day": True, "intensity": 0.65})
        simulator.set_current_position("a")
        edge = ("a", "b", 0)
        belief = simulator.belief(*edge)
        key = tuple(np.round(belief, 3))
        grouped.setdefault(key, []).append(simulator.true_level(*edge))

    calibration_errors = []
    for belief_key, levels in grouped.items():
        if len(levels) < 25:
            continue
        empirical = np.bincount(levels, minlength=3) / len(levels)
        calibration_errors.append(float(np.mean(np.abs(empirical - belief_key))))
    assert calibration_errors
    assert float(np.mean(calibration_errors)) < 0.07


def test_path_independence_and_seed_reproducibility():
    first = _sim(seed=55)
    second = _sim(seed=55)
    first.reset_episode({"is_flood_day": True, "intensity": 0.6})
    second.reset_episode({"is_flood_day": True, "intensity": 0.6})

    assert first.true_levels == second.true_levels
    assert first.readings == second.readings
    first.set_current_position("a")
    first_a = {edge: first.belief(*edge).tolist() for edge in first.edges}
    first.set_current_position("c")
    first.belief_map()
    first.set_current_position("a")
    assert first_a == {edge: first.belief(*edge).tolist() for edge in first.edges}


def test_out_of_radius_belief_equals_prior_even_if_scenario_truth_changes():
    first, second = _sim(seed=88), _sim(seed=88)
    edge = ("c", "d", 0)
    scenario = {"is_flood_day": True, "intensity": 0.5}
    first.reset_episode(scenario)
    second.reset_episode({**scenario, "edge_levels": {edge: 2}})
    first.set_current_position("a")
    second.set_current_position("a")

    assert first._edge_hop(edge) > first.reveal_radius_hops
    np.testing.assert_array_equal(first.belief(*edge), first.prior(*edge))
    np.testing.assert_array_equal(second.belief(*edge), second.prior(*edge))
    np.testing.assert_array_equal(first.belief(*edge), second.belief(*edge))


def test_expected_cost_matches_monte_carlo_realized_cost():
    config = CostModelConfig()
    belief = np.array([0.6, 0.3, 0.1])
    rng = np.random.default_rng(44)
    levels = rng.choice(3, size=200_000, p=belief)
    sampled_mean = np.mean([level_cost(120.0, int(level), config) for level in levels])
    assert abs(sampled_mean - expected_cost(120.0, belief, config)) < 4.0


def test_reveal_distance_is_cached_per_position():
    simulator = _sim(signal_mode="exact")
    simulator.reset_episode({"is_flood_day": True, "intensity": 0.5})
    simulator.set_current_position("a")
    assert simulator._hop_distances["c"] == 2
    simulator.set_current_position("b")
    assert simulator._hop_distances["c"] == 1


def test_scenario_override_and_intensity_are_validated():
    simulator = _sim(signal_mode="exact")
    simulator.reset_episode({
        "is_flood_day": True,
        "intensity": 0.8,
        "edge_levels": {("a", "b", 0): 2},
    })
    assert simulator.true_level("a", "b", 0) == 2
    assert simulator.intensity == 0.8
