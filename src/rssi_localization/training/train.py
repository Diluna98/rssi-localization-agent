from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, list[float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    history = {"train_loss": [], "validation_loss": []}

    for _ in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

            batch_size = len(features)
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size

        validation_loss = evaluate_loss(model, validation_loader, loss_fn, device)
        history["train_loss"].append(train_loss_sum / train_count)
        history["validation_loss"].append(validation_loss)

    return history


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    loss_sum = 0.0
    count = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        loss = loss_fn(model(features), targets)
        batch_size = len(features)
        loss_sum += loss.item() * batch_size
        count += batch_size
    return loss_sum / count
