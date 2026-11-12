"""Visualization functions for the XGBoost-vs-LightGBM boosting demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

XGB_COLOR = "steelblue"
LGBM_COLOR = "darkorange"


def plot_loss_curves(
    rounds: np.ndarray,
    xgb_train: np.ndarray,
    xgb_test: np.ndarray,
    lgbm_train: np.ndarray,
    lgbm_test: np.ndarray,
    path: Path,
) -> None:
    """RMSE vs. boosting round -- the "loss function change in time" that
    is the whole point of boosting: each new tree is fit to the current
    residual, so loss should fall additively, round after round, with
    train loss dropping faster/further than test loss once the ensemble
    starts overfitting.
    """
    plt.figure(figsize=(8, 5.5))
    plt.plot(rounds, xgb_train, color=XGB_COLOR, linestyle="--", linewidth=1.8, label="XGBoost train")
    plt.plot(rounds, xgb_test, color=XGB_COLOR, linewidth=2.2, label="XGBoost test")
    plt.plot(rounds, lgbm_train, color=LGBM_COLOR, linestyle="--", linewidth=1.8, label="LightGBM train")
    plt.plot(rounds, lgbm_test, color=LGBM_COLOR, linewidth=2.2, label="LightGBM test")
    plt.xlabel("Boosting round (number of trees)")
    plt.ylabel("RMSE")
    plt.title("Loss vs. boosting round: train/test, XGBoost vs. LightGBM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pred_vs_actual(
    y_test: np.ndarray,
    xgb_pred: np.ndarray,
    lgbm_pred: np.ndarray,
    path: Path,
) -> None:
    """Predicted vs. actual melting temperature on held-out proteins --
    points hugging the diagonal indicate accurate regression."""
    lo = min(y_test.min(), xgb_pred.min(), lgbm_pred.min())
    hi = max(y_test.max(), xgb_pred.max(), lgbm_pred.max())

    plt.figure(figsize=(6.5, 6.5))
    plt.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="Perfect prediction")
    plt.scatter(y_test, xgb_pred, color=XGB_COLOR, alpha=0.6, s=25, label="XGBoost")
    plt.scatter(y_test, lgbm_pred, color=LGBM_COLOR, alpha=0.6, s=25, label="LightGBM")
    plt.xlabel("Actual melting temperature (°C)")
    plt.ylabel("Predicted melting temperature (°C)")
    plt.title("Predicted vs. actual Tm (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_feature_importance(
    feature_names: list[str],
    xgb_importances: np.ndarray,
    lgbm_importances: np.ndarray,
    path: Path,
    top_k: int = 12,
) -> None:
    """Top-k features by combined importance, XGBoost vs. LightGBM --
    both boosters should converge on similar biologically-driven signal
    features even though they grow trees differently."""
    combined_rank = xgb_importances + lgbm_importances
    top_idx = np.argsort(combined_rank)[::-1][:top_k]
    names = [feature_names[i] for i in top_idx]
    y_pos = np.arange(len(names))
    height = 0.35

    plt.figure(figsize=(8, 6))
    plt.barh(y_pos + height / 2, xgb_importances[top_idx], height, label="XGBoost", color=XGB_COLOR)
    plt.barh(y_pos - height / 2, lgbm_importances[top_idx], height, label="LightGBM", color=LGBM_COLOR)
    plt.yticks(y_pos, names)
    plt.gca().invert_yaxis()
    plt.xlabel("Feature importance (normalized total gain)")
    plt.title(f"Top {top_k} feature importances: XGBoost vs. LightGBM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_model_comparison(
    labels: list[str],
    rmse_values: list[float],
    r2_values: list[float],
    path: Path,
) -> None:
    """Final test RMSE and R^2 across models -- the accuracy cross-check
    used in every prior week's project, extended to two regression
    metrics side by side."""
    colors = [XGB_COLOR, LGBM_COLOR, "gray"][: len(labels)]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].bar(x, rmse_values, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Test RMSE (°C)")
    axes[0].set_title("Test RMSE")
    for xi, v in zip(x, rmse_values):
        axes[0].text(xi, v + 0.01 * max(rmse_values), f"{v:.2f}", ha="center")

    axes[1].bar(x, r2_values, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("Test $R^2$")
    axes[1].set_title("Test $R^2$")
    axes[1].set_ylim(0, 1)
    for xi, v in zip(x, r2_values):
        axes[1].text(xi, v + 0.01, f"{v:.3f}", ha="center")

    fig.suptitle("Test accuracy: XGBoost vs. LightGBM vs. scikit-learn baseline")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_tree_growth_strategy(
    xgb_leaves: np.ndarray,
    xgb_depth: np.ndarray,
    lgbm_leaves: np.ndarray,
    lgbm_depth: np.ndarray,
    path: Path,
) -> None:
    """A unique-to-boosting architectural contrast: XGBoost grows each
    tree level-wise (breadth-first to max_depth), so leaf count tracks
    2^depth fairly tightly, while LightGBM grows leaf-wise (always
    splitting the single highest-gain leaf, bounded by num_leaves), which
    can reach the same leaf budget via a much deeper, lopsided tree.
    """
    rounds = np.arange(1, len(xgb_leaves) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True)

    axes[0].plot(rounds, xgb_leaves, color=XGB_COLOR, linewidth=1.5, label="XGBoost (level-wise)")
    axes[0].plot(rounds, lgbm_leaves, color=LGBM_COLOR, linewidth=1.5, label="LightGBM (leaf-wise)")
    axes[0].set_xlabel("Boosting round")
    axes[0].set_ylabel("Leaves in that round's tree")
    axes[0].set_title("Leaves per tree")
    axes[0].legend()

    axes[1].plot(rounds, xgb_depth, color=XGB_COLOR, linewidth=1.5, label="XGBoost (level-wise)")
    axes[1].plot(rounds, lgbm_depth, color=LGBM_COLOR, linewidth=1.5, label="LightGBM (leaf-wise)")
    axes[1].set_xlabel("Boosting round")
    axes[1].set_ylabel("Max depth reached in that round's tree")
    axes[1].set_title("Depth per tree")
    axes[1].legend()

    fig.suptitle("Level-wise vs. leaf-wise tree growth: same leaf budget, different shape")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_goss_effect(
    rounds: np.ndarray,
    no_goss_train: np.ndarray,
    goss_train: np.ndarray,
    used_fraction: np.ndarray,
    path: Path,
) -> None:
    """Gradient-based One-Side Sampling (GOSS) is LightGBM's other unique
    trick: train loss with GOSS enabled should track full-data training
    loss closely, despite each round only touching a fraction of the
    training rows (top-gradient instances kept, the rest subsampled and
    reweighted).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, height_ratios=[3, 1])

    ax1.plot(rounds, no_goss_train, color="gray", linewidth=2, label="Full data (no GOSS)")
    ax1.plot(rounds, goss_train, color=LGBM_COLOR, linewidth=2, label="GOSS-sampled")
    ax1.set_ylabel("Train RMSE")
    ax1.set_title("GOSS: comparable convergence from a fraction of the data")
    ax1.legend()

    ax2.plot(rounds, used_fraction, color=LGBM_COLOR, linewidth=1.5)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Boosting round")
    ax2.set_ylabel("Fraction of rows\nused per round")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
