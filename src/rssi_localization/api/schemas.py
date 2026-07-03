from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictionRequest(BaseModel):
    rssi_dbm: list[float] = Field(..., min_length=1)

    @field_validator("rssi_dbm")
    @classmethod
    def validate_rssi_values(cls, values: list[float]) -> list[float]:
        if any(value > 0 for value in values):
            raise ValueError("RSSI values are expected to be non-positive dBm values")
        return values


class PredictionResponse(BaseModel):
    x_m: float
    y_m: float
