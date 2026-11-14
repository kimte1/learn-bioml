"""Visualization functions for the PCA demo on DeepLoc subcellular
localization data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PCA_COLOR = "steelblue"


def _label_colors(labels: np.ndarray, cmap_name: str = "tab10") -> tuple[list, dict]:
    classes = sorted(set(labels))
    cmap = plt.colormaps[cmap_name]
    color_of = {c: cmap(i % cmap.N) for i, c in enumerate(classes)}
    return classes, color_of


def plot_scree(explained_variance_ratio_full: np.ndarray, path: Path, n_show: int = 20) -> None:
    """Individual and cumulative explained variance per component --
    this is how you decide how many PCA components actually carry
    signal before the rest is mostly noise. A sharp "elbow" in the bars
    (or the cumulative line leveling off) marks that cutoff.
    """
    n_show = min(n_show, len(explained_variance_ratio_full))
    ratios = explained_variance_ratio_full[:n_show]
    cumulative = np.cumsum(ratios)
    components = np.arange(1, n_show + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.bar(components, ratios, color=PCA_COLOR, label="Per-component variance")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio", color=PCA_COLOR)
    ax1.set_xticks(components)

    ax2 = ax1.twinx()
    ax2.plot(components, cumulative, color="black", marker="o", markersize=4, label="Cumulative")
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.9, color="gray", linestyle="--", linewidth=1)

    fig.suptitle("PCA scree plot: how much variance each component explains")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pca_scatter(X_pca_2d: np.ndarray, labels: np.ndarray, explained_variance_ratio: np.ndarray, path: Path) -> None:
    """The first two PCA axes, colored by ground-truth localization.
    PCA preserves global variance, so it keeps true distances honest but
    often can't separate classes that differ in a nonlinear combination
    of features -- that's the contrast UMAP (see 09_cluster) is built to
    address.
    """
    classes, color_of = _label_colors(labels)

    plt.figure(figsize=(8, 7))
    for c in classes:
        mask = labels == c
        plt.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], s=10, alpha=0.6, color=color_of[c], label=c)
    plt.xlabel(f"PC1 ({explained_variance_ratio[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({explained_variance_ratio[1]:.1%} variance)")
    plt.title("PCA (first two components), colored by true localization")
    plt.legend(markerscale=2, fontsize=8, loc="best", ncol=1)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
