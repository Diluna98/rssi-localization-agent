from __future__ import annotations

from pathlib import Path

import pandas as pd

from rssi_localization.simulation.environment import IndoorEnvironment
from rssi_localization.simulation.rssi_model import LogDistanceRssiModel, simulate_rssi


def generate_split(
    environment: IndoorEnvironment,
    rssi_model: LogDistanceRssiModel,
    n_samples: int,
    rng,
) -> pd.DataFrame:
    positions = environment.sample_positions(n_samples, rng)
    rssi = simulate_rssi(positions, environment.anchors, rssi_model, rng)

    data = {f"rssi_anchor_{idx}": rssi[:, idx] for idx in range(rssi.shape[1])}
    data["x_m"] = positions[:, 0]
    data["y_m"] = positions[:, 1]
    return pd.DataFrame(data)


def write_splits(splits: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for split_name, frame in splits.items():
        frame.to_csv(path / f"{split_name}.csv", index=False)
