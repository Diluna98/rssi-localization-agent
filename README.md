# RSSI Localization Agent

Portfolio-grade RSSI indoor localization project using simulation, classical baselines, and neural network regression.

The system estimates a receiver's 2D position from RSSI measurements collected from fixed anchors.

```text
RSSI readings from N anchors -> localization model -> estimated (x, y)
```

## Why This Project Exists

RSSI localization is a useful applied machine learning problem because the data is noisy, physics-informed, and spatial. This repository demonstrates an end-to-end workflow:

- synthetic data generation using a log-distance path loss model
- baseline localization with RSSI-derived distances and trilateration
- neural network regression with PyTorch
- evaluation with localization error in meters
- reusable agent wrapper for inference

## Project Structure

```text
rssi-localization-agent/
  configs/
    default.yaml
  scripts/
    simulate_data.py
    train_model.py
    evaluate_model.py
  src/
    rssi_localization/
      agents/
      data/
      models/
      simulation/
      training/
      visualization/
  tests/
  pyproject.toml
  README.md
```

## Method

The simulator places anchors at known coordinates and samples receiver locations inside a rectangular environment. RSSI is generated with the log-distance path loss model:

```text
RSSI(d) = P0 - 10 * n * log10(d / d0) + noise
```

Where:

- `P0` is RSSI at reference distance `d0`
- `n` is the path loss exponent
- `noise` models measurement uncertainty and multipath effects

The supervised learning task is:

```text
input:  RSSI vector [rssi_anchor_1, ..., rssi_anchor_n]
output: receiver position [x, y]
```

## Quickstart

Create a virtual environment, then install the project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Generate data:

```powershell
python scripts/simulate_data.py --config configs/default.yaml
```

Train the neural network:

```powershell
python scripts/train_model.py --config configs/default.yaml
```

Evaluate the trained model and the trilateration baseline:

```powershell
python scripts/evaluate_model.py --config configs/default.yaml
```

Run tests:

```powershell
pytest
```

## Current Metrics

After running evaluation, metrics are written to:

```text
artifacts/metrics.json
```

The main metric is localization error in meters:

- mean error
- median error
- 90th percentile error
- 95th percentile error

Initial synthetic run:

| Method | Mean error | Median error | P90 error | P95 error |
| --- | ---: | ---: | ---: | ---: |
| Neural network | 1.38 m | 1.20 m | 2.56 m | 3.13 m |
| Trilateration baseline | 4.23 m | 3.14 m | 8.52 m | 11.74 m |

## Roadmap

- Add real RSSI measurement ingestion
- Add experiment tracking with MLflow or Weights & Biases
- Add FastAPI inference endpoint
- Add Docker image
- Add richer indoor effects such as walls, shadowing, and anchor dropout
- Add uncertainty estimation
