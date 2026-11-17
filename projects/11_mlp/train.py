"""Device selection and the generic train/eval loop shared by every MLP
configuration -- this is where autograd actually gets exercised:
`loss.backward()` walks the graph PyTorch built during the forward pass
and fills in `.grad` on every `nn.Parameter`, and `optimizer.step()`
applies the update.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    """check if MPS (Apple Silicon GPU) is available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    loss_fn = nn.MSELoss(reduction="sum")
    is_training = optimizer is not None
    model.train(is_training)

    total_loss, n_examples = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        with torch.set_grad_enabled(is_training):
            preds = model(X)  # forward pass
            loss = loss_fn(preds, y)

        if is_training:
            optimizer.zero_grad()  # clear .grad from the previous step (gradients accumulate otherwise)
            loss.backward()  # backpropagation: walk the forward pass's computation graph in reverse, filling in .grad on every parameter
            optimizer.step()  # use those gradients to update every parameter in place

        total_loss += loss.item()
        n_examples += X.shape[0]

    return total_loss / n_examples


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-2,
) -> dict[str, list[float]]:

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for _ in range(epochs):
        history["train_loss"].append(_run_epoch(model, train_loader, device, optimizer))
        history["val_loss"].append(_run_epoch(model, val_loader, device, None))
    return history


@torch.no_grad()
def predict(
    model: nn.Module, 
    loader: DataLoader, 
    device: torch.device
) -> tuple[np.ndarray, np.ndarray]:

    model.eval()
    preds, targets = [], []
    for X, y in loader:
        preds.append(model(X.to(device)).cpu())
        targets.append(y)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def rmse_r2(preds: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res = np.sum((preds - targets) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    return rmse, r2
