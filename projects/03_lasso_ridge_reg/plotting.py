from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_coef_comparison(
    feature_names: list[str],
    scratch_coef: np.ndarray,
    sklearn_coef: np.ndarray,
    method_name: str,
    path: Path,
) -> None:
    """Grouped bar chart"""
    n = len(feature_names)
    x = np.arange(n)
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(
        x - width / 2, 
        scratch_coef, 
        width, 
        label="From scratch", 
        color="steelblue"
    )
    plt.bar(
        x + width / 2, 
        sklearn_coef, 
        width, 
        label="scikit-learn", 
        color="darkorange"
    )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, feature_names, rotation=45, ha="right")
    plt.ylabel("Coefficient (standardized features)")
    plt.title(f"{method_name}: from-scratch vs. scikit-learn coefficients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    


def plot_loss_history(obj_history: list[float], path: Path) -> None:

    plt.figure(figsize=(6, 4))
    plt.plot(obj_history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

 

def plot_regularization_path(
    alphas_ridge: np.ndarray,
    coefs_ridge: np.ndarray,
    alphas_lasso: np.ndarray,
    coefs_lasso: np.ndarray,
    feature_names: list[str],
    path: Path,
) -> None:
    """
    weights vs. alpha for Ridge and Lasso.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    cmap = plt.get_cmap("tab10")

    for j, name in enumerate(feature_names):
        axes[0].plot(alphas_ridge, coefs_ridge[:, j], color=cmap(j % 10), label=name)
        axes[1].plot(alphas_lasso, coefs_lasso[:, j], color=cmap(j % 10), label=name)

    for ax, title, alphas in (
        (axes[0], "Ridge (L2): smooth shrinkage", alphas_ridge),
        (axes[1], "Lasso (L1): shrinkage to exact zero", alphas_lasso),
    ):
        ax.set_xscale("log")
        ax.set_xlabel("alpha")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlim(alphas.max(), alphas.min())  # decreasing alpha -> left to right

    axes[0].set_ylabel("Coefficient (standardized features)")
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Regularization path: coefficients vs. penalty strength")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def plot_cv_curve(
    alphas_ridge: np.ndarray,
    cv_mse_ridge: np.ndarray,
    best_alpha_ridge: float,
    alphas_lasso: np.ndarray,
    cv_mse_lasso: np.ndarray,
    best_alpha_lasso: float,
    path: Path,
) -> None:
    """k-fold cross-validation MSE vs. alpha, for both methods, with the chosen alpha marked."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, alphas, cv_mse, best_alpha, title in (
        (axes[0], alphas_ridge, cv_mse_ridge, best_alpha_ridge, "Ridge"),
        (axes[1], alphas_lasso, cv_mse_lasso, best_alpha_lasso, "Lasso"),
    ):
        ax.plot(alphas, cv_mse, "o-", color="steelblue", markersize=4)
        ax.axvline(best_alpha, color="crimson", linestyle="--", label=f"best alpha={best_alpha:.4g}")
        ax.set_xscale("log")
        ax.set_xlabel("alpha")
        ax.set_ylabel("k-fold CV mean squared error")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Bias-variance tradeoff: cross-validated error vs. regularization strength")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



def plot_coefficient_stability(
    feature_names: list[str],
    coefs_ols: np.ndarray,
    coefs_ridge: np.ndarray,
    coefs_lasso: np.ndarray,
    best_alpha_ridge: float,
    best_alpha_lasso: float,
    path: Path,
) -> None:
    """Box plots of bootstrap-resampled coefficients per feature, for plain least
    squares (OLS, alpha=0) vs. Ridge vs. Lasso at their CV-optimal alphas.

    A wide box for a feature under OLS but a narrow box under Ridge/Lasso is the
    direct visual meaning of "unstable coefficients": resampling the same
    underlying data repeatedly gives wildly different OLS coefficients, because
    multiple weight combinations fit the training data almost equally well under
    multicollinearity -- regularization narrows that ambiguity.
    """
    n_features = len(feature_names)
    positions_ols = np.arange(n_features) * 4.0
    positions_ridge = positions_ols + 1.0
    positions_lasso = positions_ols + 2.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: all three methods, full range -- shows how much OLS dwarfs the rest.
    ax = axes[0]
    bp_ols = ax.boxplot(coefs_ols, positions=positions_ols, widths=0.8, patch_artist=True, showfliers=False)
    bp_ridge = ax.boxplot(coefs_ridge, positions=positions_ridge, widths=0.8, patch_artist=True, showfliers=False)
    bp_lasso = ax.boxplot(coefs_lasso, positions=positions_lasso, widths=0.8, patch_artist=True, showfliers=False)
    for box in bp_ols["boxes"]:
        box.set_facecolor("lightcoral")
    for box in bp_ridge["boxes"]:
        box.set_facecolor("steelblue")
    for box in bp_lasso["boxes"]:
        box.set_facecolor("gold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions_ols + 1.0)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Bootstrap coefficient distribution (standardized features)")
    ax.set_title("Full range: OLS dwarfs the regularized methods")
    ax.legend(
        [bp_ols["boxes"][0], bp_ridge["boxes"][0], bp_lasso["boxes"][0]],
        ["OLS (alpha=0)", "Ridge", "Lasso"],
        loc="upper right",
        fontsize=8,
    )

    # Right: Ridge vs. Lasso only, zoomed in -- OLS's scale would otherwise hide
    # any visible difference between the two regularized methods.
    ax = axes[1]
    positions_ridge_z = np.arange(n_features) * 3.0
    positions_lasso_z = positions_ridge_z + 1.0
    bp_ridge_z = ax.boxplot(coefs_ridge, positions=positions_ridge_z, widths=0.8, patch_artist=True, showfliers=False)
    bp_lasso_z = ax.boxplot(coefs_lasso, positions=positions_lasso_z, widths=0.8, patch_artist=True, showfliers=False)
    for box in bp_ridge_z["boxes"]:
        box.set_facecolor("steelblue")
    for box in bp_lasso_z["boxes"]:
        box.set_facecolor("gold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions_ridge_z + 0.5)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_title("Zoomed in: Ridge vs. Lasso")
    ax.legend(
        [bp_ridge_z["boxes"][0], bp_lasso_z["boxes"][0]],
        ["Ridge", "Lasso"],
        loc="upper right",
        fontsize=8,
    )

    fig.suptitle(
        f"Coefficient stability under resampling: OLS (alpha=0) vs. Ridge (alpha={best_alpha_ridge:.3g}) "
        f"vs. Lasso (alpha={best_alpha_lasso:.3g})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
