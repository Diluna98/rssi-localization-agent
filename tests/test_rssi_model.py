import numpy as np

from rssi_localization.simulation.rssi_model import LogDistanceRssiModel


def test_distance_rssi_round_trip_without_noise() -> None:
    model = LogDistanceRssiModel(noise_std_db=0.0)
    distances = np.asarray([1.0, 2.0, 5.0], dtype=np.float32)

    rssi = model.rssi_from_distance(distances, noisy=False)
    recovered = model.distance_from_rssi(rssi)

    np.testing.assert_allclose(recovered, distances, rtol=1e-5)
