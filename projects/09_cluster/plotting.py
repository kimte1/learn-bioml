"""Visualization functions for the k-means clustering demo on DeepLoc
subcellular localization data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

KMEANS_COLOR = "darkorange"


def plot_kmeans_elbow(k_values: np.ndarray, inertias: np.ndarray, chosen_k: int, path: Path) -> None:
    """Inertia (total squared distance to assigned centroid) vs. k. Inertia
    always falls as k grows -- more clusters can only fit the data at
    least as well -- so there's no single "correct" k from this curve
    alone. What you look for is the elbow: the point past which adding
    clusters buys shrinking returns, a visual proxy for diminishing
    marginal explanatory value.
    """
    plt.figure(figsize=(7.5, 5.5))
    plt.plot(k_values, inertias, color=KMEANS_COLOR, marker="o")
    plt.axvline(chosen_k, color="gray", linestyle="--", linewidth=1, label=f"k = {chosen_k} used below")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Inertia (total squared distance to centroid)")
    plt.title("K-means elbow plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_kmeans_convergence(inertia_history: list[float], path: Path) -> None:
    """Inertia at every Lloyd's-algorithm iteration of the best run. Each
    assign/update cycle can only decrease or hold inertia steady -- this
    curve should be monotonically non-increasing, which is the concrete,
    visible version of k-means' convergence guarantee.
    """
    iterations = np.arange(1, len(inertia_history) + 1)
    plt.figure(figsize=(7.5, 5.5))
    plt.plot(iterations, inertia_history, color=KMEANS_COLOR, marker="o", markersize=4)
    plt.xlabel("Lloyd's algorithm iteration")
    plt.ylabel("Inertia")
    plt.title("K-means convergence: inertia never increases")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_cluster_vs_label(true_labels: np.ndarray, cluster_labels: np.ndarray, path: Path) -> None:
    """Contingency heatmap: how the unsupervised k-means clusters (columns)
    distribute across the true, held-out localization labels (rows). A
    cluster that lines up almost entirely with one row means k-means
    rediscovered a biologically real grouping purely from sequence-derived
    features, with no label ever shown to it.
    """
    classes = sorted(set(true_labels))
    k = len(set(cluster_labels))
    table = np.zeros((len(classes), k))
    for i, c in enumerate(classes):
        mask = true_labels == c
        for cluster_id in range(k):
            table[i, cluster_id] = np.sum(cluster_labels[mask] == cluster_id)

    row_normalized = table / table.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(1.0 * k + 3, 0.5 * len(classes) + 2))
    im = ax.imshow(row_normalized, cmap="viridis", aspect="auto")
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"cluster {i}" for i in range(k)], rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(k):
            ax.text(j, i, int(table[i, j]), ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, label="Fraction of true-label row in this cluster")
    ax.set_title("True localization vs. k-means cluster assignment")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
