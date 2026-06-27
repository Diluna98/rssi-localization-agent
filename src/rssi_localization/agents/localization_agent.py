from __future__ import annotations

import numpy as np
import torch

from rssi_localization.data.dataset import StandardScaler
from rssi_localization.models.mlp import LocalizationMLP


class LocalizationAgent:
    """Inference wrapper around a trained RSSI localization model."""

    def __init__(self, model: LocalizationMLP, scaler: StandardScaler, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.scaler = scaler

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str,
        scaler_path: str,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
        device: str = "cpu",
    ) -> "LocalizationAgent":
        model = LocalizationMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        scaler = StandardScaler.load(scaler_path)
        return cls(model=model, scaler=scaler, device=device)

    @torch.no_grad()
    def predict(self, rssi_dbm: list[float] | np.ndarray) -> np.ndarray:
        features = np.asarray(rssi_dbm, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        features = self.scaler.transform(features)
        tensor = torch.from_numpy(features).to(self.device)
        return self.model(tensor).cpu().numpy()
