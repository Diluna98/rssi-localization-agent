from __future__ import annotations

import argparse

import numpy as np

from rssi_localization.visualization.plots import plot_error_heatmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    arrays = np.load(args.input)
    plot_error_heatmap(
        targets=arrays["targets"],
        predictions=arrays["predictions"],
        anchors=arrays["anchors"],
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
