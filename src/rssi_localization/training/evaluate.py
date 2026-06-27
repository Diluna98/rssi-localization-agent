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
