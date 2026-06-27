from __future__ import annotations

import argparse

import numpy as np

from rssi_localization.config import load_config
from rssi_localization.simulation.dataset_generator import generate_split, write_splits
from rssi_localization.simulation.environment import IndoorEnvironment
from rssi_localization.simulation.rssi_model import LogDistanceRssiModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    rng = np.random.default_rng(config["seed"])

    environment = IndoorEnvironment(
        width_m=config["environment"]["width_m"],
        height_m=config["environment"]["height_m"],
        anchors=np.asarray(config["environment"]["anchors"], dtype=np.float32),
    )
    rssi_model = LogDistanceRssiModel(**config["rssi_model"])

    data_config = config["data"]
    splits = {
        "train": generate_split(environment, rssi_model, data_config["train_samples"], rng),
        "validation": generate_split(environment, rssi_model, data_config["validation_samples"], rng),
        "test": generate_split(environment, rssi_model, data_config["test_samples"], rng),
    }
    write_splits(splits, data_config["output_dir"])
    print(f"Wrote simulated data to {data_config['output_dir']}")


if __name__ == "__main__":
    main()
