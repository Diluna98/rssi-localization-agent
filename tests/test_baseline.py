import numpy as np

from rssi_localization.models.baseline import trilaterate_least_squares


def test_trilateration_recovers_position_from_exact_distances() -> None:
    anchors = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )
    position = np.asarray([[3.0, 4.0]], dtype=np.float32)
    distances = np.linalg.norm(position[:, None, :] - anchors[None, :, :], axis=2)

    estimate = trilaterate_least_squares(anchors, distances)

    np.testing.assert_allclose(estimate, position, atol=1e-4)
