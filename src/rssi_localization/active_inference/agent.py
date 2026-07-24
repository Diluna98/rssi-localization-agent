"""Construction of a PyAIF continuous-observation navigation agent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyAIF import (
    ActiveInfAgent,
    ContinuousLikelihood,
    DeepTemporalInference,
    GenerativeModel,
    ShallowInference,
    utils,
)

from .likelihoods import RssiNavigationLikelihood


def _object_array(*arrays) -> np.ndarray:
    result = np.empty(len(arrays), dtype=object)
    for index, array in enumerate(arrays):
        result[index] = np.asarray(array, dtype=float)
    return result


def _transition_model(
    states_dim: tuple[int, int, int],
) -> np.ndarray:
    transitions = []
    for factor, state_count in enumerate(states_dim):
        action_count = 3 if factor < 2 else 1
        transition = np.zeros((state_count, state_count, action_count))
        transition[:, :, 0] = np.eye(state_count)
        if factor < 2:
            for state in range(state_count):
                transition[max(0, state - 1), state, 1] = 1.0
                transition[min(state_count - 1, state + 1), state, 2] = 1.0
        transitions.append(transition)
    return _object_array(*transitions)


def _cardinal_policies(
    states_dim: tuple[int, int, int],
    controls_dim: tuple[int, int, int],
    horizon: int,
) -> list[np.ndarray]:
    policies = utils.construct_policies(
        states_dim,
        controls_dim,
        horizon - 1,
        [0, 1],
    )
    return [
        policy
        for policy in policies
        if not np.any((policy[:, 0] != 0) & (policy[:, 1] != 0)) and np.all(policy[:, 2:] == 0)
    ]


@dataclass(frozen=True)
class NavigationAgentConfig:
    model_size: int = 20
    goal_resolution: int = 10
    temporal_horizon: int = 1
    message_passing_iterations: int = 5
    policy_samples: int = 200
    exact_state_limit: int = 100
    random_seed: int = 0
    policy_workers: int = 1


def build_navigation_agent(
    config: NavigationAgentConfig | None = None,
) -> ActiveInfAgent:
    """Build the navigation agent using the PyAIF component API."""

    if config is None:
        config = NavigationAgentConfig()
    states_dim = (
        config.model_size,
        config.model_size,
        config.goal_resolution**2,
    )
    controls_dim = (3, 3, 1)
    horizon = config.temporal_horizon
    policies = _cardinal_policies(states_dim, controls_dim, horizon) if horizon > 1 else None
    model = GenerativeModel(
        B=_transition_model(states_dim),
        D=_object_array(*(np.ones(state_count) for state_count in states_dim)),
        controls_dim=controls_dim,
        controllable_factors=[0, 1],
        policies=policies,
    )

    domain_likelihood = RssiNavigationLikelihood(states_dim)
    likelihood = ContinuousLikelihood.from_model(
        domain_likelihood,
        modality_dependencies=[[0], [1], [0, 1, 2]],
        grid_size=domain_likelihood.grid_size,
        policy_samples=config.policy_samples,
        exact_state_limit=config.exact_state_limit,
        random_seed=config.random_seed,
    )
    if horizon > 1:
        inference = DeepTemporalInference(
            horizon=horizon,
            message_passing_iterations=config.message_passing_iterations,
            policy_workers=config.policy_workers,
        )
    else:
        inference = ShallowInference(
            message_passing_iterations=config.message_passing_iterations,
            policy_workers=config.policy_workers,
        )

    return ActiveInfAgent(
        model=model,
        likelihood=likelihood,
        inference=inference,
        action_selection="deterministic",
    )
