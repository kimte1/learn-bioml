"""
Load the Delaney ESOL dataset and turn it into standardized train/val/test tensors for MLP

Features are the six numeric descriptors already computed in the CSV

("ESOL predicted" column is itself the output of a
published linear regression over similar descriptors, so it is excluded as a feature, but carried through to compare the MLP's test).

REUSABLE: unchanged from `projects/12_mlp_training/data_load_prep.py` --
data loading has nothing to do with experiment tracking, so none of this
file needed to change to add W&B.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    baseline_test: np.ndarray  # ESOL-predicted column, test rows, original scale
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

    # First split off the test set, then split the remainder into
    # train/val, so the three fractions are of the full dataset rather
    # than of each other.
    test_frac = 1.0 - train_frac - val_frac
    X_train_val, X_test, y_train_val, y_test, _, baseline_test = train_test_split(
        X, y, baseline, test_size=test_frac, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_frac / (train_frac + val_frac), random_state=seed
    )

    # Standardize using train-set statistics only -- fit the scaler on
    # train, then transform everything with it. Val/test must never leak
    # their own mean/std into the scaling, or the reported generalization
    # error is optimistic.
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

    def scale_X(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(x_scaler.transform(arr), dtype=torch.float32)

    def scale_y(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(y_scaler.transform(arr.reshape(-1, 1)).squeeze(-1), dtype=torch.float32)

    return SplitData(
        X_train=scale_X(X_train),
        y_train=scale_y(y_train),
        X_val=scale_X(X_val),
        y_val=scale_y(y_val),
        X_test=scale_X(X_test),
        y_test=scale_y(y_test),
        y_mean=float(y_scaler.mean_[0]),
        y_std=float(y_scaler.scale_[0]),
        baseline_test=baseline_test,
        feature_names=list(FEATURE_COLUMNS),
    )


def make_loaders(data: SplitData, batch_size: int = 32) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        TensorDataset(data.X_train, data.y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(data.X_val, data.y_val), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(data.X_test, data.y_test), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, test_loader
