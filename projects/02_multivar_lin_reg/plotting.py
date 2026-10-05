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
    """
    3D scatter of (x1, x2, y) with the fitted regression plane overlaid
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        x1,
        x2,
        y,
        s=12,
        alpha=0.5,
        color="steelblue",
        label=f"Data (n={x1.size})",
    )

    x1_grid, x2_grid = np.meshgrid(
        np.linspace(x1.min(), x1.max(), 20),
        np.linspace(x2.min(), x2.max(), 20),
    )
    y_grid = intercept + w[0] * x1_grid + w[1] * x2_grid
    ax.plot_surface(
        x1_grid,
        x2_grid,
        y_grid,
        color="crimson",
        alpha=0.3,
    )

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(target_name)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_feat_corr(
    X: np.ndarray, 
    feature_names: list[str], 
    path: Path
) -> None:
    """scatterplot"""

    sns.scatterplot(x=X[:, 0], y=X[:, 2])
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[2])
    plt.title("Feature correlation")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
) -> None:
    
    plt.figure(figsize=(6, 6))
    plt.scatter(
        y_true,
        y_pred,
        s=12,
        alpha=0.5,
        color="steelblue",
    )

    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]

    plt.plot(
        lims,
        lims,
        color="black",
        linestyle="--",
        label="Perfect prediction",
    )

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
    """
    Line plot of every weight's value vs. training epoch
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


def plot_null_space_solutions(
    feature_names: list[str],
    w1: np.ndarray,
    w2: np.ndarray,
    path: Path,
) -> None:
    """
    Grouped bar chart comparing two weight vectors that fit the (tiny,
    p > n) training set equally well. 

    The point of this plot is that the bars for w1 and w2 are different per-feature, even though both vectors produce identical predictions on the training data 
    """
    x = np.arange(len(feature_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(
        x - width / 2, 
        w1, 
        width, 
        label="Solution 1 (min-norm lstsq)", 
        color="steelblue"
    )
    plt.bar(
        x + width / 2, 
        w2, 
        width, 
        label="Solution 2 (+ null-space shift)", 
        color="crimson"
    )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, feature_names, rotation=30, ha="right")
    plt.ylabel("Weight value (standardized features)")
    plt.title("p > n: different weight vectors fit the training data identically")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_overfitting_curve(
    noise_counts: np.ndarray,
    train_mse: np.ndarray,
    test_mse: np.ndarray,
    n_train: int,
    path: Path,
) -> None:
    
    plt.figure(figsize=(7, 5))
    plt.plot(
        noise_counts, 
        train_mse,
        marker="o",
        color="steelblue",
        label="Train MSE",
    )
    plt.plot(
        noise_counts,
        test_mse,
        marker="o",
        color="crimson",
        label="Test MSE",
    )
    plt.xlabel("Number of extra random noise features added")
    plt.ylabel("MSE")
    # plt.title("No complexity penalty -> OLS overfits as useless features pile up")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_noise_feature_weights(
    feature_names: list[str],
    weights: np.ndarray,
    n_real_features: int,
    path: Path,
) -> None:
    """
    Bar chart of fitted coefficients, colored by real vs. pure-noise feature.
    """
    colors = ["steelblue"] * n_real_features + ["crimson"] * (len(feature_names) - n_real_features)

    plt.figure(figsize=(10, 5))
    plt.bar(
        np.arange(len(feature_names)),
        weights,
        color=colors,
    )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks([])
    plt.xlabel(
        f"Features (blue = real [{n_real_features}], red = pure noise [{len(feature_names) - n_real_features}])"
    )
    plt.ylabel("Weight value (standardized features)")
    plt.title("No feature selection: noise features still get nonzero weight")
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
