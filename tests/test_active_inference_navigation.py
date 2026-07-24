import numpy as np

from rssi_localization.active_inference import (
    NavigationAgentConfig,
    run_navigation_episode,
)
from rssi_localization.active_inference.animation import README_SCENARIOS, simulate_scenarios


def test_continuous_shallow_navigation_approaches_one_cell_from_goal():
    result = run_navigation_episode(
        config=NavigationAgentConfig(random_seed=7),
        planning_windows=20,
    )

    assert result.actions.shape == (20, 2)
    assert np.all(np.isfinite(result.distances))
    assert np.all(np.isfinite(result.positions))
    assert result.distances.min() <= 25.0
    assert result.distances[-1] < 0.15 * result.distances[0]


def test_continuous_deep_navigation_approaches_goal_at_coarse_resolution():
    result = run_navigation_episode(
        config=NavigationAgentConfig(
            model_size=20,
            goal_resolution=2,
            temporal_horizon=3,
            message_passing_iterations=8,
            policy_samples=300,
            random_seed=7,
        ),
        planning_windows=8,
    )

    assert result.actions.shape == (16, 2)
    assert np.all(np.isfinite(result.distances))
    assert np.all(np.isfinite(result.positions))
    assert result.distances.min() < 0.5 * result.distances[0]
    assert result.distances[-1] <= result.distances.min() + 25.0


def test_readme_animation_scenarios_reach_their_sources():
    results = simulate_scenarios()

    assert len({scenario.start for scenario in README_SCENARIOS}) == len(README_SCENARIOS)
    assert len({scenario.source for scenario in README_SCENARIOS}) == len(README_SCENARIOS)
    assert all(result.reached_goal for result in results)
    assert all(result.distances[-1] <= 18.0 for result in results)
