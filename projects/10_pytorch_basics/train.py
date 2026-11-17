"""Device selection and evaluation helpers shared by both the
from-scratch and the idiomatic `nn.Linear` model. The training loops
themselves live next to each model (`scratch_linear_regression.py`,
`torch_linear_regression.py`) since the whole point of this project is
that those two loops look different -- one calls `.backward()` and
updates tensors by hand, the other calls `optimizer.step()`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(
    model, 
    loader: DataLoader, 
    device: torch.device
) -> tuple[np.ndarray, np.ndarray]:

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
