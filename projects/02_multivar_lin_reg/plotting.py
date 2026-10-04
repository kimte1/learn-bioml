"""Visualization functions for multivariable linear regression."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection


def plot_loss(
    loss_history: list[float],
    path: Path,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (standardized features)")
    plt.title("Batch gradient descent convergence")
    # plt.yscale("log")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_fit_3d(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    intercept: float,
    w: np.ndarray,
    feature_names: list[str],
    target_name: str,
    path: Path,
) -> None:
    """3D scatter of (x1, x2, y) with the fitted regression plane overlaid.

    This is the direct 3-variable generalization of 01_linear_reg's 2D
    "y vs x with a fit line" plot: with 2 features (x1, x2) and 1 target (y),
    the fitted model y = b + w1*x1 + w2*x2 is a plane, not a line.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x1, x2, y, s=12, alpha=0.5, color="steelblue", label=f"Data (n={x1.size})")

    x1_grid, x2_grid = np.meshgrid(
        np.linspace(x1.min(), x1.max(), 20),
        np.linspace(x2.min(), x2.max(), 20),
    )
    y_grid = intercept + w[0] * x1_grid + w[1] * x2_grid
    ax.plot_surface(x1_grid, x2_grid, y_grid, color="crimson", alpha=0.3)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(target_name)
    ax.set_title("Multivariable linear regression: fitted plane in 3D")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_feat_corr(X: np.ndarray, feature_names: list[str], path: Path) -> None:
    """scatterplot"""

    sns.scatterplot(x=X[:, 0], y=X[:, 2])
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[2])
    plt.title("Feature correlation")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """Predicted vs. actual scatter -- the standard diagnostic once you have more
    than one feature and can no longer just plot y against a single x."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=12, alpha=0.5, color="steelblue")

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, color="black", linestyle="--", label="Perfect prediction")

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_weight_trajectories(
    weight_history: np.ndarray,
    feature_names: list[str],
    path: Path,
) -> None:
    """Line plot of every weight's value vs. training epoch (one line per feature).

    weight_history has shape (n_epochs + 1, n_features) -- row 0 is the
    zero-initialized weights before any gradient step.
    """
    epochs = np.arange(weight_history.shape[0])

    plt.figure(figsize=(7, 5))
    for j, name in enumerate(feature_names):
        plt.plot(epochs, weight_history[:, j], label=name)

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Weight value (standardized features)")
    plt.title("How gradient descent weights change over training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_weight_split(
    weight_history: np.ndarray,
    idx_a: int,
    idx_b: int,
    name_a: str,
    name_b: str,
    path: Path,
) -> None:
    """The limitation, visualized with gradient descent alone (no OLS/closed form):

    Top panel -- the two near-duplicate features' *individual* weights over
    epoch: because the two columns carry almost the same information, the
    loss surface is nearly flat along the direction that trades weight
    between them, so gradient descent's progress along that direction is very
    slow -- these two lines keep drifting long after the rest of the model has
    settled.

    Bottom panel -- their *sum* over epoch: this is the direction the loss
    surface is steep along (it controls the actual prediction), so it
    converges almost immediately and then stays flat. The gap between "sum
    converges instantly" and "individual split keeps drifting" *is* the
    limitation -- the model has no way to decide how to split credit between
    near-duplicate features, even though it has clearly decided their total.
    """
    epochs = np.arange(weight_history.shape[0])
    w_a = weight_history[:, idx_a]
    w_b = weight_history[:, idx_b]

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    axes[0].plot(epochs, w_a, color="steelblue", label=name_a)
    axes[0].plot(epochs, w_b, color="crimson", label=name_b)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Individual weight")
    axes[0].set_title("Individual weights: still drifting")
    axes[0].legend()

    axes[1].plot(epochs, w_a + w_b, color="darkgreen")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Sum of the two weights")
    axes[1].set_title("Sum of the near-duplicate pair: converges almost immediately")

    fig.suptitle("The limitation: near-duplicate features have an undetermined split")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
