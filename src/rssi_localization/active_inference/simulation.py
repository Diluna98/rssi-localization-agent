"""Episode runner for the continuous-observation navigation agent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agent import NavigationAgentConfig, build_navigation_agent
from .environment import GridNavigationEnvironment


@dataclass(frozen=True)
class NavigationEpisodeResult:
    distances: np.ndarray
    positions: np.ndarray
    actions: np.ndarray
    reached_goal: bool


def run_navigation_episode(
    *,
    config: NavigationAgentConfig | None = None,
    environment: GridNavigationEnvironment | None = None,
    planning_windows: int = 8,
) -> NavigationEpisodeResult:
    """Run a deterministic continuous-observation navigation episode."""

    if config is None:
        config = NavigationAgentConfig()
    if planning_windows < 1:
        raise ValueError("planning_windows must be positive.")
    if environment is None:
        environment = GridNavigationEnvironment(
            model_size=config.model_size,
            random_seed=config.random_seed,
        )

    agent = build_navigation_agent(config)
    observation = environment.reset()
    distances = [environment.distance_to_goal()]
    positions = [environment.position.copy()]
    actions = []
    reached_goal = False
    agent.reset()

    for window in range(planning_windows):
        if config.temporal_horizon > 1:
            agent.reset()
            time_steps = range(config.temporal_horizon)
        else:
            time_steps = (window,)

        for time_step in time_steps:
            agent.observe(observation, time_step=time_step)
            agent.infer_states()
            agent.infer_policies()
            action = agent.select_action()
            if action is None:
                continue

            navigation_action = np.asarray(action[:2], dtype=int)
            observation, reached_goal = environment.step(navigation_action)
            actions.append(navigation_action)
            distances.append(environment.distance_to_goal())
            positions.append(environment.position.copy())
            if reached_goal:
                break
        if reached_goal:
            break

    return NavigationEpisodeResult(
        distances=np.asarray(distances, dtype=float),
        positions=np.asarray(positions, dtype=float),
        actions=np.asarray(actions, dtype=int),
        reached_goal=reached_goal,
    )
