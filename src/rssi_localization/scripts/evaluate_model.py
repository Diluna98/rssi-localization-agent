from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rssi_localization.config import load_config
from rssi_localization.data.dataset import RSSI_PREFIX
from rssi_localization.models.baseline import trilaterate_least_squares
from rssi_localization.simulation.rssi_model import LogDistanceRssiModel
from rssi_localization.training.evaluate import (
    localization_errors,
    summarize_errors,
    write_experiment_report,
    write_metrics,
)
from rssi_localization.tracking.mlflow_tracking import log_artifacts, log_metrics, mlflow_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    test_frame = pd.read_csv(Path(config["data"]["output_dir"]) / "test.csv")
    rssi_columns = sorted([column for column in test_frame.columns if column.startswith(RSSI_PREFIX)])

    features = test_frame[rssi_columns].to_numpy(dtype=np.float32)
    targets = test_frame[["x_m", "y_m"]].to_numpy(dtype=np.float32)

    prediction_input_path = Path(config["evaluation"]["prediction_plot_path"]).with_name(
        "prediction_input.npz"
    )
    prediction_output_path = Path(config["evaluation"]["prediction_plot_path"]).with_name(
        "model_predictions.npz"
    )
    prediction_input_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(prediction_input_path, features=features)

    training_config = config["training"]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rssi_localization.agents.predict_cli",
            "--input",
            str(prediction_input_path),
            "--output",
            str(prediction_output_path),
            "--model-path",
            training_config["model_path"],
            "--scaler-path",
            training_config["scaler_path"],
            "--input-dim",
            str(len(rssi_columns)),
            "--hidden-dims",
            ",".join(str(value) for value in training_config["hidden_dims"]),
            "--dropout",
            str(training_config["dropout"]),
        ],
        check=True,
    )
    model_predictions = np.load(prediction_output_path)["predictions"]
    model_errors = localization_errors(model_predictions, targets)

    rssi_model = LogDistanceRssiModel(**config["rssi_model"])
    anchors = np.asarray(config["environment"]["anchors"], dtype=np.float32)
    baseline_distances = rssi_model.distance_from_rssi(features)
    baseline_predictions = trilaterate_least_squares(anchors, baseline_distances)
    baseline_errors = localization_errors(baseline_predictions, targets)

    metrics = {
        "neural_network": summarize_errors(model_errors),
        "trilateration_baseline": summarize_errors(baseline_errors),
    }

    with mlflow_run(config, "evaluation_run_name"):
        write_metrics(metrics, config["evaluation"]["metrics_path"])

        plot_input_path = Path(config["evaluation"]["prediction_plot_path"]).with_suffix(".npz")
        plot_input_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            plot_input_path,
            targets=targets,
            predictions=model_predictions,
            anchors=anchors,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "rssi_localization.visualization.plot_predictions_cli",
                "--input",
                str(plot_input_path),
                "--output",
                config["evaluation"]["prediction_plot_path"],
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "rssi_localization.visualization.plot_error_heatmap_cli",
                "--input",
                str(plot_input_path),
                "--output",
                config["evaluation"]["error_heatmap_path"],
            ],
            check=True,
        )
        write_experiment_report(
            metrics=metrics,
            path=config["evaluation"]["report_path"],
            config_path=args.config,
            prediction_plot_path=config["evaluation"]["prediction_plot_path"],
            error_heatmap_path=config["evaluation"]["error_heatmap_path"],
        )
        log_metrics(metrics)
        log_artifacts(
            [
                config["evaluation"]["metrics_path"],
                config["evaluation"]["prediction_plot_path"],
                config["evaluation"]["error_heatmap_path"],
                config["evaluation"]["report_path"],
            ]
        )
    print(metrics)


if __name__ == "__main__":
    main()
