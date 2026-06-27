from __future__ import annotations

import numpy as np


def trilaterate_least_squares(anchors: np.ndarray, distances_m: np.ndarray) -> np.ndarray:
    """Estimate 2D positions from anchor coordinates and distances.

    Uses the standard linearized least-squares form relative to the first anchor.
    """
    anchors = np.asarray(anchors, dtype=np.float32)
    distances = np.asarray(distances_m, dtype=np.float32)

    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError("anchors must have shape (n_anchors, 2)")
    if distances.ndim == 1:
        distances = distances[None, :]
    if distances.shape[1] != anchors.shape[0]:
        raise ValueError("distance count must match anchor count")

    anchor_0 = anchors[0]
    other_anchors = anchors[1:]
    matrix_a = 2.0 * (other_anchors - anchor_0)

    ata_00 = float(np.sum(matrix_a[:, 0] * matrix_a[:, 0]))
    ata_01 = float(np.sum(matrix_a[:, 0] * matrix_a[:, 1]))
    ata_11 = float(np.sum(matrix_a[:, 1] * matrix_a[:, 1]))
    determinant = ata_00 * ata_11 - ata_01 * ata_01
    if abs(determinant) < 1e-8:
        raise ValueError("anchor geometry is degenerate for 2D trilateration")

    estimates = []
    for row in distances:
        rhs = (
            row[0] ** 2
            - row[1:] ** 2
            - np.sum(anchor_0**2)
            + np.sum(other_anchors**2, axis=1)
        )
        atb_0 = float(np.sum(matrix_a[:, 0] * rhs))
        atb_1 = float(np.sum(matrix_a[:, 1] * rhs))
        x = (ata_11 * atb_0 - ata_01 * atb_1) / determinant
        y = (-ata_01 * atb_0 + ata_00 * atb_1) / determinant
        estimates.append([x, y])

    return np.asarray(estimates, dtype=np.float32)
