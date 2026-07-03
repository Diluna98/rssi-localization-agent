from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def localization_errors(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    return np.linalg.norm(predictions - targets, axis=1)


def summarize_errors(errors: np.ndarray) -> dict[str, float]:
    return {
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "p90_error_m": float(np.percentile(errors, 90)),
        "p95_error_m": float(np.percentile(errors, 95)),
    }


def write_metrics(metrics: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def write_experiment_report(
    metrics: dict,
    path: str | Path,
    config_path: str,
    prediction_plot_path: str,
    error_heatmap_path: str,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path.with_name("metrics.json").as_posix()
    prediction_plot = Path(prediction_plot_path).as_posix()
    error_heatmap = Path(error_heatmap_path).as_posix()

    neural = metrics["neural_network"]
    baseline = metrics["trilateration_baseline"]
    report = f"""# RSSI Localization Experiment Report

## Configuration

- Config: `{config_path}`

## Metrics

| Method | Mean error | Median error | P90 error | P95 error |
| --- | ---: | ---: | ---: | ---: |
| Neural network | {neural["mean_error_m"]:.3f} m | {neural["median_error_m"]:.3f} m | {neural["p90_error_m"]:.3f} m | {neural["p95_error_m"]:.3f} m |
| Trilateration baseline | {baseline["mean_error_m"]:.3f} m | {baseline["median_error_m"]:.3f} m | {baseline["p90_error_m"]:.3f} m | {baseline["p95_error_m"]:.3f} m |

## Artifacts

- Metrics JSON: `{metrics_path}`
- Prediction plot: `{prediction_plot}`
- Error heatmap: `{error_heatmap}`

## Interpretation

The prediction plot compares true receiver positions against neural-network estimates.
The error heatmap shows where localization error is spatially concentrated.
"""
    output_path.write_text(report, encoding="utf-8")
