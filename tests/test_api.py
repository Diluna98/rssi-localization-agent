from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException

from rssi_localization.api.app import ApiSettings, create_app
from rssi_localization.api.schemas import PredictionRequest


class FakeAgent:
    def predict(self, rssi_dbm):
        return np.asarray([[1.25, 2.5]], dtype=np.float32)


def test_health_reports_model_loaded_when_agent_is_injected() -> None:
    app = create_app(agent=FakeAgent())
    health = next(route.endpoint for route in app.routes if route.path == "/health")

    response = health()

    assert response.model_dump() == {"status": "ok", "model_loaded": True}


def test_predict_returns_position() -> None:
    app = create_app(agent=FakeAgent())
    predict = next(route.endpoint for route in app.routes if route.path == "/predict")

    response = predict(PredictionRequest(rssi_dbm=[-60.0, -70.0, -65.0]))

    assert response.model_dump() == {"x_m": 1.25, "y_m": 2.5}


def test_predict_returns_unavailable_when_model_is_missing() -> None:
    settings = ApiSettings(
        config_path="configs/ci.yaml",
        model_path="missing/model.pt",
        scaler_path="missing/scaler.npz",
    )
    app = create_app(agent=None, settings=settings)
    predict = next(route.endpoint for route in app.routes if route.path == "/predict")

    with pytest.raises(HTTPException) as error:
        predict(PredictionRequest(rssi_dbm=[-60.0, -70.0, -65.0]))
    assert error.value.status_code == 503
