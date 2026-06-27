from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(
    targets: np.ndarray,
    predictions: np.ndarray,
    anchors: np.ndarray,
    output_path: str | Path,
    max_points: int = 500,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = min(max_points, len(targets))
    targets = targets[:count]
    predictions = predictions[:count]

    plt.figure(figsize=(8, 6))
    plt.scatter(targets[:, 0], targets[:, 1], s=12, alpha=0.45, label="true")
    plt.scatter(predictions[:, 0], predictions[:, 1], s=12, alpha=0.45, label="predicted")
    plt.scatter(anchors[:, 0], anchors[:, 1], s=90, marker="^", label="anchors")
    for true, predicted in zip(targets, predictions):
        plt.plot([true[0], predicted[0]], [true[1], predicted[1]], color="gray", alpha=0.12)
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.title("RSSI localization predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
