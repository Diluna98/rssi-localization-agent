from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rssi_localization.config import load_config
from rssi_localization.data.dataset import RSSI_PREFIX, RssiLocalizationDataset, StandardScaler
from rssi_localization.models.mlp import LocalizationMLP
from rssi_localization.training.train import train_model
from rssi_localization.tracking.mlflow_tracking import log_artifacts, log_history, mlflow_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(config["data"]["output_dir"])
    train_frame = pd.read_csv(data_dir / "train.csv")
    validation_frame = pd.read_csv(data_dir / "validation.csv")

    rssi_columns = sorted([column for column in train_frame.columns if column.startswith(RSSI_PREFIX)])
    scaler = StandardScaler.fit(train_frame[rssi_columns].to_numpy(dtype="float32"))

    train_dataset = RssiLocalizationDataset(train_frame, scaler=scaler)
    validation_dataset = RssiLocalizationDataset(validation_frame, scaler=scaler)

    training_config = config["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
    )

    model = LocalizationMLP(
        input_dim=len(rssi_columns),
        hidden_dims=training_config["hidden_dims"],
        dropout=training_config["dropout"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with mlflow_run(config, "train_run_name"):
        history = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=training_config["epochs"],
            learning_rate=training_config["learning_rate"],
            device=device,
        )

        model_path = Path(training_config["model_path"])
        scaler_path = Path(training_config["scaler_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "history": history}, model_path)
        scaler.save(str(scaler_path))
        log_history(history)
        log_artifacts([model_path, scaler_path])

    print(f"Saved model to {model_path}")
    print(f"Final validation loss: {history['validation_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
