"""Visualization functions for the decision tree / random forest demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_overfitting_diagnostic(
    max_depths: np.ndarray,
    train_acc: np.ndarray,
    test_acc: np.ndarray,
    path: Path,
) -> None:
    """A single tree's train vs. test accuracy as max_depth grows --
    shallow trees underfit (both low), deep trees overfit (train accuracy
    keeps climbing toward 1.0 while test accuracy plateaus or falls).
    """
    plt.figure(figsize=(7, 5))
    plt.plot(max_depths, train_acc, color="steelblue", linewidth=2, marker="o", label="Train accuracy")
    plt.plot(max_depths, test_acc, color="darkorange", linewidth=2, marker="o", label="Test accuracy")
    plt.xlabel("Max depth")
    plt.ylabel("Accuracy")
    plt.title("Single decision tree: overfitting vs. max depth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_feature_importance(
    feature_names: list[str],
    tree_importances: np.ndarray,
    forest_importances: np.ndarray,
    path: Path,
    top_k: int = 12,
) -> None:
    """Top-k features by combined importance, single tree vs. forest."""
    combined_rank = tree_importances + forest_importances
    top_idx = np.argsort(combined_rank)[::-1][:top_k]
    names = [feature_names[i] for i in top_idx]
    y_pos = np.arange(len(names))
    height = 0.35

    plt.figure(figsize=(8, 6))
    plt.barh(y_pos + height / 2, tree_importances[top_idx], height, label="Single tree", color="steelblue")
    plt.barh(y_pos - height / 2, forest_importances[top_idx], height, label="Random forest", color="darkorange")
    plt.yticks(y_pos, names)
    plt.gca().invert_yaxis()
    plt.xlabel("Feature importance (normalized impurity decrease)")
    plt.title(f"Top {top_k} feature importances: single tree vs. random forest")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_bagging_variance_reduction(
    n_estimators_list: np.ndarray,
    forest_test_error: np.ndarray,
    single_tree_errors: np.ndarray,
    path: Path,
) -> None:
    """Forest test error vs. number of trees, against the spread of many
    independently bootstrap-resampled single trees' test error -- shows
    bagging's averaging driving error/variance down as n_estimators grows,
    well below any individual tree's typical error.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(
        n_estimators_list,
        forest_test_error,
        color="steelblue",
        linewidth=2,
        marker="o",
        label="Random forest test error",
    )
    plt.axhline(
        single_tree_errors.mean(),
        color="darkorange",
        linestyle="--",
        label=f"Single bootstrapped tree, mean test error (n={len(single_tree_errors)})",
    )
    plt.axhspan(
        single_tree_errors.mean() - single_tree_errors.std(),
        single_tree_errors.mean() + single_tree_errors.std(),
        color="darkorange",
        alpha=0.15,
        label="Single tree, ±1 std across resamples",
    )
    plt.xlabel("Number of trees in forest")
    plt.ylabel("Test error")
    plt.title("Bagging reduces variance: forest error vs. single-tree spread")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_model_accuracy_comparison(
    labels: list[str],
    accuracies: list[float],
    path: Path,
) -> None:
    """Test accuracy across from-scratch and scikit-learn tree/forest
    models -- the cross-check against scikit-learn used in every prior
    week's project.
    """
    plt.figure(figsize=(7, 5))
    x = np.arange(len(labels))
    plt.bar(x, accuracies, color=["steelblue", "steelblue", "darkorange", "darkorange"][: len(labels)])
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Test accuracy")
    plt.ylim(0, 1)
    plt.title("Test accuracy: from-scratch vs. scikit-learn")
    for xi, acc in zip(x, accuracies):
        plt.text(xi, acc + 0.01, f"{acc:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
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
