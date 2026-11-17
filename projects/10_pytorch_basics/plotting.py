"""Visualization functions for the from-scratch vs. idiomatic PyTorch
linear regression demo on the Delaney ESOL solubility dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRATCH_COLOR = "steelblue"
TORCH_COLOR = "darkorange"
BASELINE_COLOR = "seagreen"


def plot_loss_curves(
    scratch_history: dict[str, list[float]],
    torch_history: dict[str, list[float]],
    path: Path,
) -> None:
    """Train/val loss curves for the raw-tensor model (scratch_linear_regression.py)
    and the nn.Linear model (torch_linear_regression.py), trained
    independently (each from its own zero-init, not weight-copied) on
    the same data with the same learning rate and loss reduction. Since
    both are minimizing the exact same convex MSE objective, they should
    converge to the same loss regardless of which one writes the update
    rule out by hand.
    """
    plt.figure(figsize=(8, 5.5))
    epochs = np.arange(1, len(scratch_history["train_loss"]) + 1)
    plt.plot(epochs, scratch_history["train_loss"], color=SCRATCH_COLOR, label="scratch (train)")
    plt.plot(epochs, scratch_history["val_loss"], color=SCRATCH_COLOR, linestyle="--", label="scratch (val)")
    plt.plot(epochs, torch_history["train_loss"], color=TORCH_COLOR, label="nn.Linear (train)")
    plt.plot(epochs, torch_history["val_loss"], color=TORCH_COLOR, linestyle="--", label="nn.Linear (val)")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log scale)")
    plt.title("From-scratch tensors + autograd vs. nn.Linear + optim.SGD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_weight_comparison(
    scratch_weights: np.ndarray,
    torch_weights: np.ndarray,
    feature_names: list[str],
    path: Path,
) -> None:
    """Final learned weight (on standardized features) per model, side by
    side. Both models started from zero-init and were trained
    independently -- if the scratch update rule is really doing the same
    math as optimizer.step(), the two bars for each feature should land
    at essentially the same height.
    """
    x = np.arange(len(feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, scratch_weights, width, color=SCRATCH_COLOR, label="scratch")
    ax.bar(x + width / 2, torch_weights, width, color=TORCH_COLOR, label="nn.Linear")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=30, ha="right")
    ax.set_ylabel("Learned weight (standardized features)")
    ax.set_title("Learned weights: hand-written update rule vs. optim.SGD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_predictions(
    scratch_preds: np.ndarray,
    torch_preds: np.ndarray,
    baseline_preds: np.ndarray,
    measured: np.ndarray,
    scratch_rmse: float,
    torch_rmse: float,
    baseline_rmse: float,
    path: Path,
) -> None:
    """Predicted vs. measured log solubility on the held-out test set:
    the from-scratch model, the nn.Linear model, and the dataset's own
    published linear baseline, all on the same compounds. Points on the
    diagonal are exact predictions.
    """
    series = [
        ("Scratch (tensors + autograd)", scratch_preds, scratch_rmse, SCRATCH_COLOR),
        ("nn.Linear + optim.SGD", torch_preds, torch_rmse, TORCH_COLOR),
        ("ESOL published baseline", baseline_preds, baseline_rmse, BASELINE_COLOR),
    ]
    lo = min(measured.min(), *(preds.min() for _, preds, _, _ in series))
    hi = max(measured.max(), *(preds.max() for _, preds, _, _ in series))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5), sharex=True, sharey=True)
    for ax, (title, preds, rmse, color) in zip(axes, series):
        ax.scatter(measured, preds, s=14, alpha=0.6, color=color)
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{title}\nRMSE = {rmse:.3f}")
        ax.set_xlabel("Measured log solubility")
    axes[0].set_ylabel("Predicted log solubility")

    fig.suptitle("Test-set predictions: scratch vs. nn.Linear vs. published linear baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
