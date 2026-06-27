from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IndoorEnvironment:
    """Rectangular 2D localization environment with fixed anchor positions."""

    width_m: float
    height_m: float
    anchors: np.ndarray

    def __post_init__(self) -> None:
        anchors = np.asarray(self.anchors, dtype=np.float32)
        if anchors.ndim != 2 or anchors.shape[1] != 2:
            raise ValueError("anchors must have shape (n_anchors, 2)")
        if len(anchors) < 3:
            raise ValueError("at least three anchors are required for 2D localization")
        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError("environment dimensions must be positive")
        object.__setattr__(self, "anchors", anchors)

    def sample_positions(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        x = rng.uniform(0.0, self.width_m, size=n_samples)
        y = rng.uniform(0.0, self.height_m, size=n_samples)
        return np.column_stack([x, y]).astype(np.float32)
