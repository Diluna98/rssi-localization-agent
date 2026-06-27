from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


RSSI_PREFIX = "rssi_anchor_"


@dataclass(frozen=True)
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler":
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "StandardScaler":
        data = np.load(path)
        return cls(mean=data["mean"].astype(np.float32), std=data["std"].astype(np.float32))


class RssiLocalizationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, scaler: StandardScaler | None = None) -> None:
        rssi_columns = sorted([column for column in frame.columns if column.startswith(RSSI_PREFIX)])
        if not rssi_columns:
            raise ValueError("dataset must contain RSSI columns")

        features = frame[rssi_columns].to_numpy(dtype=np.float32)
        targets = frame[["x_m", "y_m"]].to_numpy(dtype=np.float32)

        self.scaler = scaler
        if scaler is not None:
            features = scaler.transform(features)

        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def load_frame(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
