from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogDistanceRssiModel:
    """Log-distance RSSI model with additive Gaussian noise."""

    reference_rssi_dbm: float = -40.0
    reference_distance_m: float = 1.0
    path_loss_exponent: float = 2.2
    noise_std_db: float = 3.0
    min_distance_m: float = 0.25

    def rssi_from_distance(
        self,
        distances_m: np.ndarray,
        rng: np.random.Generator | None = None,
        noisy: bool = True,
    ) -> np.ndarray:
        distances = np.maximum(np.asarray(distances_m, dtype=np.float32), self.min_distance_m)
        rssi = self.reference_rssi_dbm - (
            10.0 * self.path_loss_exponent * np.log10(distances / self.reference_distance_m)
        )
        if noisy:
            if rng is None:
                raise ValueError("rng is required when noisy=True")
            rssi = rssi + rng.normal(0.0, self.noise_std_db, size=distances.shape)
        return rssi.astype(np.float32)

    def distance_from_rssi(self, rssi_dbm: np.ndarray) -> np.ndarray:
        rssi = np.asarray(rssi_dbm, dtype=np.float32)
        exponent = (self.reference_rssi_dbm - rssi) / (10.0 * self.path_loss_exponent)
        return (self.reference_distance_m * np.power(10.0, exponent)).astype(np.float32)


def pairwise_distances(positions: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.float32)
    deltas = positions[:, None, :] - anchors[None, :, :]
    return np.linalg.norm(deltas, axis=2).astype(np.float32)


def simulate_rssi(
    positions: np.ndarray,
    anchors: np.ndarray,
    model: LogDistanceRssiModel,
    rng: np.random.Generator,
) -> np.ndarray:
    distances = pairwise_distances(positions, anchors)
    return model.rssi_from_distance(distances, rng=rng, noisy=True)
