from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rssi_localization.agents.localization_agent import LocalizationAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--scaler-path", required=True)
    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--hidden-dims", required=True)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    hidden_dims = [int(value) for value in args.hidden_dims.split(",") if value]
    features = np.load(args.input)["features"]
    agent = LocalizationAgent.from_checkpoint(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        input_dim=args.input_dim,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    )
    predictions = agent.predict(features)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, predictions=predictions)


if __name__ == "__main__":
    main()
