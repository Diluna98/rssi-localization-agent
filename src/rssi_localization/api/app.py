from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from fastapi import FastAPI, HTTPException

from rssi_localization.agents.localization_agent import LocalizationAgent
from rssi_localization.api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from rssi_localization.config import load_config


class Predictor(Protocol):
    def predict(self, rssi_dbm: list[float] | np.ndarray) -> np.ndarray:
        """Predict one or more 2D positions from RSSI values."""


@dataclass(frozen=True)
class ApiSettings:
    config_path: str = "configs/default.yaml"
    model_path: str | None = None
    scaler_path: str | None = None
    device: str = "cpu"


def settings_from_environment() -> ApiSettings:
    return ApiSettings(
        config_path=os.getenv("RSSI_CONFIG_PATH", "configs/default.yaml"),
        model_path=os.getenv("RSSI_MODEL_PATH"),
        scaler_path=os.getenv("RSSI_SCALER_PATH"),
        device=os.getenv("RSSI_DEVICE", "cpu"),
    )


def load_agent(settings: ApiSettings) -> LocalizationAgent | None:
    config = load_config(settings.config_path)
    training_config = config["training"]
    model_path = Path(settings.model_path or training_config["model_path"])
    scaler_path = Path(settings.scaler_path or training_config["scaler_path"])

    if not model_path.exists() or not scaler_path.exists():
        return None

    input_dim = len(config["environment"]["anchors"])
    return LocalizationAgent.from_checkpoint(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        input_dim=input_dim,
        hidden_dims=training_config["hidden_dims"],
        dropout=training_config["dropout"],
        device=settings.device,
    )


def create_app(agent: Predictor | None = None, settings: ApiSettings | None = None) -> FastAPI:
    app_settings = settings or settings_from_environment()
    loaded_agent = agent if agent is not None else load_agent(app_settings)
    app = FastAPI(
        title="RSSI Localization Agent",
        version="0.1.0",
        description="Estimate 2D position from RSSI readings.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=loaded_agent is not None)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        if loaded_agent is None:
            raise HTTPException(
                status_code=503,
                detail="Model artifacts are not available. Train a model before serving predictions.",
            )

        predictions = loaded_agent.predict(request.rssi_dbm)
        first_prediction = predictions[0]
        return PredictionResponse(x_m=float(first_prediction[0]), y_m=float(first_prediction[1]))

    return app


app = create_app()
