"""Visualizations for the Week 10 attention demos.

Attention weight is a magnitude (always >= 0, each row sums to 1), not a
signed or categorical quantity, so heatmaps use a single-hue sequential
colormap rather than a rainbow. The scaled-vs-unscaled comparison reuses
this repo's existing two-condition color convention (steelblue / darkorange,
see `11_mlp/plotting.py`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCALED_COLOR = "steelblue"
UNSCALED_COLOR = "darkorange"
HEATMAP_CMAP = "Blues"


def plot_attention_heatmap(
    attn_weights: np.ndarray,
    path: Path,
    title: str,
    query_labels: list[str] | None = None,
    key_labels: list[str] | None = None,
) -> None:
    """`attn_weights`: (L_q, L_k) single attention matrix -- one example,
    not a batch. Every row sums to 1; that's the softmax property to check
    by eye here (each row's shading should look like one distribution).
    """
    fig, ax = plt.subplots(figsize=(0.4 * attn_weights.shape[1] + 2, 0.4 * attn_weights.shape[0] + 2))
    im = ax.imshow(attn_weights, cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    fig.colorbar(im, ax=ax, label="Attention weight")

    if key_labels is not None:
        ax.set_xticks(range(len(key_labels)))
        ax.set_xticklabels(key_labels)
    if query_labels is not None:
        ax.set_yticks(range(len(query_labels)))
        ax.set_yticklabels(query_labels)

    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_entropy_vs_dk(
    d_k_values: list[int],
    scaled_entropy: list[float],
    unscaled_entropy: list[float],
    path: Path,
) -> None:
    """Mean softmax entropy of the attention distribution vs. d_k, with and
    without the 1/sqrt(d_k) scaling factor. Low entropy means the softmax
    has saturated onto one or a few keys; the unscaled curve should fall
    toward zero as d_k grows while the scaled curve stays roughly flat.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(d_k_values, scaled_entropy, marker="o", markersize=7, linewidth=2, color=SCALED_COLOR, label="Scaled (1/sqrt(d_k))")
    plt.plot(d_k_values, unscaled_entropy, marker="o", markersize=7, linewidth=2, color=UNSCALED_COLOR, label="Unscaled")
    plt.xlabel("d_k")
    plt.ylabel("Mean softmax entropy (nats)")
    plt.title("Scaling keeps attention entropy stable as d_k grows")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_mask_comparison(
    causal_attn: np.ndarray,
    padding_attn: np.ndarray,
    path: Path,
) -> None:
    """Side-by-side heatmaps: a causal mask's lower-triangular pattern next
    to a padding mask's zeroed-out columns.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im = None
    for ax, attn, title in zip(axes, [causal_attn, padding_attn], ["Causal mask", "Padding mask"]):
        im = ax.imshow(attn, cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Key position")
    axes[0].set_ylabel("Query position")
    fig.colorbar(im, ax=axes, label="Attention weight", shrink=0.8)
    fig.savefig(path, dpi=150)
    plt.close(fig)
