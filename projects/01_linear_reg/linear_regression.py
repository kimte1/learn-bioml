"""
Univariate linear regression via batch gradient descent

Dataset: delaney-esol.csv
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


from plotting import (
    plot_loss,
    plot_fit,
    plot_fit_evolution,
    plot_lr_sweep,
    plot_standardization,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
FIG_DIR = Path(__file__).resolve().parent / "fig"

# SHEET_NAME = "RNF4 BioE3"
X_COL = "Molecular Weight"
Y_COL = "measured log solubility in mols per litre"

LEARNING_RATES = np.logspace(-4, np.log10(1.5), 40)
DIVERGED_CAP = 1.0e4  # sentinel ceiling so diverging runs are still visible on a log-scale plot



def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        help="Gradient descent learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of gradient descent epochs",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=20,
        help="Record the fit line every N epochs for the evolution plot",
    )
    return parser.parse_args()



def load_xy() -> tuple[np.ndarray, np.ndarray]:
    # the excel file has multiple sheets
    # xlsx_files = sorted(glob.glob(str(DATA_DIR / "barroso-gomila2023.xlsx")))

    df = pd.read_csv(DATA_DIR / "delaney-esol.csv") # load the first sheet
    x = df[X_COL].to_numpy(dtype=float)
    y = df[Y_COL].to_numpy(dtype=float)
    return x, y



def standardize(a: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean, std = a.mean(), a.std()
    return (a - mean) / std, mean, std



def linear_reg(
    x: np.ndarray,
    y: np.ndarray,
    lr: float,
    n_epochs: int,
    snapshot_interval: int = 20
) -> tuple[float, float, list[float], list[tuple[int, float, float]]]:
    """
    Fit y = mx + b by minimizing MSE with full-batch gradient descent.

    Also records (epoch, intercept, slope) every `snapshot_interval` epochs, plus the
    initial and final weights, so the fit's evolution can be visualized afterward.
    """
    n = x.size
    intercept = 0.0
    slope = 0.0
    loss_history = []
    snapshots = [(0, intercept, slope)]

    for epoch in range(1, n_epochs + 1):
        y_pred = intercept + slope * x
        error = y_pred - y
        loss_history.append(float(np.mean(error**2)))

        grad_intercept = (2.0 / n) * np.sum(error)
        grad_slope = (2.0 / n) * np.sum(error * x)
        intercept -= lr * grad_intercept
        slope -= lr * grad_slope
        if epoch % snapshot_interval == 0 or epoch == n_epochs:
            snapshots.append((epoch, intercept, slope))
    return intercept, slope, loss_history, snapshots




def sweep_final_loss(
    x_std: np.ndarray,
    y: np.ndarray,
    learning_rates: np.ndarray,
    n_epochs: int
) -> np.ndarray:

    final_losses = []
    for lr in learning_rates:
        with np.errstate(over="ignore", invalid="ignore"):
            _, _, loss_history, _ = linear_reg(
                x_std,
                y,
                lr,
                n_epochs,
                snapshot_interval=n_epochs
            )
        final_loss = loss_history[-1]
        if not np.isfinite(final_loss):
            final_loss = DIVERGED_CAP
        final_losses.append(min(final_loss, DIVERGED_CAP))
    return np.array(final_losses)



def main() -> None:
    args = build_parser()

    FIG_DIR.mkdir(exist_ok=True)

    x, y = load_xy()

    x_std, x_mean, x_scale = standardize(x)

    b, slope, loss_history, snapshots = linear_reg(
        x_std,
        y,
        args.lr,
        args.epochs,
        args.snapshot_interval
    )

    # Undo standardization: y = intercept_std + slope_std * (x - x_mean) / x_scale
    gd_slope = slope / x_scale
    gd_intercept = b - slope * x_mean / x_scale

    # y_pred_gd = gd_intercept + gd_slope * x

    # re-run linear_reg with different learning rates
    # keep track of final losses for each learning rate
    final_losses = sweep_final_loss(x_std, y, LEARNING_RATES, args.epochs)


    plot_loss(loss_history, FIG_DIR / "loss_vs_epoch.png")
    plot_fit(x, y, gd_intercept, gd_slope,FIG_DIR / "y_vs_x_fit.png")
    plot_fit_evolution(x, y, snapshots, x_mean, x_scale, FIG_DIR / "fit_evolution.png")
    plot_lr_sweep(LEARNING_RATES, final_losses, FIG_DIR / "lr_sweep.png")
    plot_standardization(x, x_std, X_COL, FIG_DIR / "standardization.png")


if __name__ == "__main__":
    main()
