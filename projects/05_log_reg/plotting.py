"""Visualization functions for the logistic regression demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_sigmoid(z: np.ndarray, s: np.ndarray, path: Path) -> None:
    """The sigmoid squashing function that turns a raw score z = w.x + b into
    a probability in (0, 1). z=0 always maps to p=0.5 -- that's the decision
    boundary at the default classification threshold.
    """
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(z, s, color="steelblue", linewidth=2)
    plt.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    plt.axvline(0, color="gray", linestyle=":", linewidth=1)
    plt.scatter([0], [0.5], color="crimson", zorder=5, label="z=0 -> p=0.5 (decision boundary)")
    plt.xlabel("z = w . x + b")
    plt.ylabel("sigmoid(z) = 1 / (1 + e^-z)")
    plt.title("The sigmoid function")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_loss_curve(loss_history: list[float], path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, color="steelblue")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Batch gradient descent convergence")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_weight_evolution(
    epochs: np.ndarray,
    weight_history: np.ndarray,
    bias_history: np.ndarray,
    feature_names: list[str],
    path: Path,
) -> None:
    """
    How each weight (for top 10 features only) moves epoch by epoch. 
    """
    
    final_weights = weight_history[-1] 
    top_k = 10

    # index for the positions in `final_weights` that have highest values
    top_idx = np.argsort(np.abs(final_weights))[::-1][:top_k]
    
    plt.figure(figsize=(8, 5.5))
    cmap = plt.get_cmap("tab10")
    # plot the weight trajectories for the top 10 final weights
    for color_i, j in enumerate(top_idx):  
        plt.plot(
            epochs,
            weight_history[:, j],
            color=cmap(color_i % 10),
            label=feature_names[j],
        )
    plt.plot(
        epochs, 
        bias_history, 
        color="black", 
        linestyle="--", 
        linewidth=1.5, 
        label="bias"
    )

    # plt.axhline(0, color="gray", linewidth=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Weight (standardized features)")
    plt.title(f"Weight trajectories during gradient descent (top {top_k} by final |weight|)")
    plt.legend(
        fontsize=8, 
        loc="center left", 
        bbox_to_anchor=(1.0, 0.5),
        title='features'
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()



def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str, path: Path) -> None:
    """cm is [[TN, FP], [FN, TP]]."""
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_roc_comparison(
    fpr_scratch: np.ndarray,
    tpr_scratch: np.ndarray,
    auc_scratch: float,
    fpr_sklearn: np.ndarray,
    tpr_sklearn: np.ndarray,
    auc_sklearn: float,
    auc_sklearn_builtin: float,
    path: Path,
) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_scratch, tpr_scratch, color="steelblue", linewidth=2, label=f"From scratch (AUC={auc_scratch:.3f})")
    plt.plot(
        fpr_sklearn,
        tpr_sklearn,
        color="darkorange",
        linewidth=2,
        linestyle="--",
        label=f"scikit-learn (AUC={auc_sklearn:.3f}, roc_auc_score={auc_sklearn_builtin:.3f})",
    )
    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random guessing (AUC=0.5)")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve: from-scratch vs. scikit-learn")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_precision_recall_f1_vs_threshold(
    thresholds: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    path: Path,
) -> None:
    """Precision, recall, and F1 as the classification threshold varies,
    instead of just their values at the default threshold=0.5.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precision, color="steelblue", linewidth=2, label="Precision")
    plt.plot(thresholds, recall, color="darkorange", linewidth=2, label="Recall")
    plt.plot(thresholds, f1, color="seagreen", linewidth=2, linestyle="--", label="F1")
    plt.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="Default threshold (0.5)")
    plt.xlabel("Classification threshold")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 vs. classification threshold")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_coef_comparison(
    feature_names: list[str],
    scratch_coef: np.ndarray,
    sklearn_coef: np.ndarray,
    path: Path,
    top_k: int = 12,
) -> None:
    combined_rank = np.abs(scratch_coef) + np.abs(sklearn_coef)
    top_idx = np.argsort(combined_rank)[::-1][:top_k]

    names = [feature_names[i] for i in top_idx]
    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, scratch_coef[top_idx], width, label="From scratch", color="steelblue")
    plt.bar(x + width / 2, sklearn_coef[top_idx], width, label="scikit-learn", color="darkorange")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Coefficient (standardized features)")
    plt.title(f"Top {top_k} coefficients by magnitude: from-scratch vs. scikit-learn")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


