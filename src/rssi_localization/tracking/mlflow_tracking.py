from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator


def is_tracking_enabled(config: dict) -> bool:
    return bool(config.get("tracking", {}).get("enabled", False))


def flatten_config(config: dict, prefix: str = "") -> dict[str, str | int | float | bool]:
    flattened: dict[str, str | int | float | bool] = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_config(value, name))
        elif isinstance(value, list):
            flattened[name] = str(value)
        elif isinstance(value, str | int | float | bool):
            flattened[name] = value
        elif value is None:
            flattened[name] = "None"
        else:
            flattened[name] = str(value)
    return flattened


def _import_mlflow():
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow tracking is enabled, but mlflow is not installed. "
            "Install the project dependencies with `python -m pip install -e .`."
        ) from error
    return mlflow


@contextmanager
def mlflow_run(config: dict, run_name_key: str) -> Iterator[Any | None]:
    tracking_config = config.get("tracking", {})
    if not tracking_config.get("enabled", False):
        with nullcontext(None) as run:
            yield run
        return

    mlflow = _import_mlflow()
    tracking_uri = tracking_config.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(str(tracking_uri))

    experiment_name = str(tracking_config["experiment_name"])
    artifact_location = tracking_config.get("artifact_location")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None and artifact_location:
        artifact_uri = Path(str(artifact_location)).resolve().as_uri()
        client.create_experiment(experiment_name, artifact_location=artifact_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=tracking_config.get(run_name_key)) as run:
        mlflow.log_params(flatten_config(config))
        yield run


def log_metrics(metrics: dict[str, Any], prefix: str = "") -> None:
    mlflow = _import_mlflow()
    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            log_metrics(value, name)
        elif isinstance(value, int | float):
            mlflow.log_metric(name, float(value))


def log_history(history: dict[str, list[float]]) -> None:
    mlflow = _import_mlflow()
    for metric_name, values in history.items():
        for step, value in enumerate(values, start=1):
            mlflow.log_metric(metric_name, float(value), step=step)


def log_artifacts(paths: list[str | Path]) -> None:
    mlflow = _import_mlflow()
    for path in paths:
        artifact_path = Path(path)
        if artifact_path.exists():
            mlflow.log_artifact(str(artifact_path))
