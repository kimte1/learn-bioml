"""
- Load the Delaney ESOL dataset 
- turn it into standardized train/val/test tensors, and `Dataset`/`DataLoader`

- Features are the six descriptors 
- "ESOL predicted" is a published linear regression over similar descriptors,
    - it's excluded as a feature 
    - but carried through as a baseline to compare the models against
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ESOL_CSV = REPO_ROOT / "data" / "delaney-esol.csv"

FEATURE_COLUMNS = [
    "Minimum Degree",
    "Molecular Weight",
    "Number of H-Bond Donors",
    "Number of Rings",
    "Number of Rotatable Bonds",
    "Polar Surface Area",
]
TARGET_COLUMN = "measured log solubility in mols per litre"
BASELINE_COLUMN = "ESOL predicted log solubility in mols per litre"


@dataclass
class SplitData:

    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    y_mean: float
    y_std: float
    baseline_test: np.ndarray  # ESOL-predicted column
    feature_names: list[str]

    def inverse_y(self, y_standardized: np.ndarray) -> np.ndarray:
        return y_standardized * self.y_std + self.y_mean


def load_dataset(
    path: Path = ESOL_CSV,
    seed: int = 0,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> SplitData:

    df = pd.read_csv(path)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float64)
    baseline = df[BASELINE_COLUMN].to_numpy(dtype=np.float64)

    n = len(df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    # Standardize using TRAINING SET stats only
    # to prevent leakage
    X_mean =  X[train_idx].mean(axis=0)
    X_std = X[train_idx].std(axis=0)
    y_mean = y[train_idx].mean()
    y_std = y[train_idx].std()

    def scale_X(idx: np.ndarray) -> torch.Tensor:
        return torch.tensor((X[idx] - X_mean) / X_std, dtype=torch.float32)

    def scale_y(idx: np.ndarray) -> torch.Tensor:
        return torch.tensor((y[idx] - y_mean) / y_std, dtype=torch.float32)

    return SplitData(
        X_train=scale_X(train_idx),
        y_train=scale_y(train_idx),
        X_val=scale_X(val_idx),
        y_val=scale_y(val_idx),
        X_test=scale_X(test_idx),
        y_test=scale_y(test_idx),
        y_mean=float(y_mean),
        y_std=float(y_std),
        baseline_test=baseline[test_idx],
        feature_names=list(FEATURE_COLUMNS),
    )


def make_loaders(
    data: SplitData, 
    batch_size: int = 32
) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    train_loader = DataLoader(
        TensorDataset(data.X_train, data.y_train), 
        batch_size=batch_size, 
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(data.X_val, data.y_val), 
        batch_size=batch_size, 
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(data.X_test, data.y_test), 
        batch_size=batch_size, 
        shuffle=False,
    )
    return train_loader, val_loader, test_loader
