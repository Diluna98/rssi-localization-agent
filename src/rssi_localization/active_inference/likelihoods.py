"""Domain likelihoods for continuous RSSI navigation observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RssiNavigationLikelihood:
    """Continuous position and RSSI likelihood over discrete spatial states.

    Hidden-state factors are current x cell, current y cell, and a flattened
    square grid of possible transmitter/goal cells.
    """

    states_dim: Sequence[int]
    workspace_size: float = 500.0
    maximum_rssi: float = 30.0
    signal_decay: float = 0.01
    position_sigma: float = 1.0
    signal_sigma: float = 2.0
    grid_size: int = 100
    log_preferences: dict = field(init=False)

    def __post_init__(self) -> None:
        self.states_dim = tuple(int(size) for size in self.states_dim)
        if len(self.states_dim) != 3:
            raise ValueError("states_dim must contain x, y, and flattened goal-grid factors.")
        goal_resolution = int(np.sqrt(self.states_dim[2]))
        if goal_resolution**2 != self.states_dim[2]:
            raise ValueError("The goal-state dimension must be a perfect square.")
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least two.")
        if min(self.position_sigma, self.signal_sigma) <= 0:
            raise ValueError("Likelihood standard deviations must be positive.")

        self.goal_resolution = goal_resolution
        self.x_centers = self._cell_centers(self.states_dim[0])
        self.y_centers = self._cell_centers(self.states_dim[1])
        goal_x = self._cell_centers(goal_resolution)
        goal_y = self._cell_centers(goal_resolution)

        current_x, current_y, transmitter_x, transmitter_y = np.meshgrid(
            self.x_centers,
            self.y_centers,
            goal_x,
            goal_y,
            indexing="ij",
        )
        distance = np.sqrt((current_x - transmitter_x) ** 2 + (current_y - transmitter_y) ** 2)
        signal_mean = self.maximum_rssi * np.exp(-self.signal_decay * distance)
        self.signal_mean = np.transpose(signal_mean, (0, 1, 3, 2)).reshape(self.states_dim)
        self.log_preferences = self._build_log_preferences()

    def _cell_centers(self, resolution: int) -> np.ndarray:
        cell_size = self.workspace_size / resolution
        return (np.arange(resolution, dtype=float) + 0.5) * cell_size

    def _build_log_preferences(self) -> dict:
        position_grid = self.get_o_grid(0)
        joint_position = np.full(
            (len(position_grid), len(position_grid)),
            1.0 / len(position_grid) ** 2,
        )

        signal_grid = self.get_o_grid(2)
        utility = 1.0 / (1.0 + np.exp(-0.25 * (signal_grid - 10.0)))
        signal_probability = np.exp(utility - utility.max())
        signal_probability /= signal_probability.sum()
        return {
            (0, 1): np.log(joint_position),
            2: np.log(signal_probability),
        }

    def get_o_grid(self, modality: int, N_grid: int | None = None) -> np.ndarray:
        size = self.grid_size if N_grid is None else int(N_grid)
        if modality in (0, 1):
            return np.linspace(0.0, self.workspace_size, size)
        if modality == 2:
            return np.linspace(0.0, self.maximum_rssi, size)
        raise ValueError(f"Unknown observation modality: {modality}")

    def likelihoods(self, observation: float, modality: int) -> np.ndarray:
        if modality == 0:
            mean = self.x_centers
            sigma = self.position_sigma
        elif modality == 1:
            mean = self.y_centers
            sigma = self.position_sigma
        elif modality == 2:
            mean = self.signal_mean
            sigma = self.signal_sigma
        else:
            raise ValueError(f"Unknown observation modality: {modality}")

        standardized = (float(observation) - mean) / sigma
        return np.exp(-0.5 * standardized**2) / (sigma * np.sqrt(2.0 * np.pi))

    def likelihoods_grid_vec(
        self,
        observation_grid: np.ndarray,
        modality: int,
        state_samples,
    ) -> np.ndarray:
        observation_grid = np.asarray(observation_grid, dtype=float)
        if modality == 0:
            mean = self.x_centers[np.asarray(state_samples, dtype=int)]
            sigma = self.position_sigma
        elif modality == 1:
            mean = self.y_centers[np.asarray(state_samples, dtype=int)]
            sigma = self.position_sigma
        elif modality == 2:
            x_state, y_state, goal_state = state_samples
            mean = self.signal_mean[
                np.asarray(x_state, dtype=int),
                np.asarray(y_state, dtype=int),
                np.asarray(goal_state, dtype=int),
            ]
            sigma = self.signal_sigma
        else:
            raise ValueError(f"Unknown observation modality: {modality}")

        standardized = (observation_grid[None, :] - mean[:, None]) / sigma
        return np.exp(-0.5 * standardized**2) / (sigma * np.sqrt(2.0 * np.pi))
