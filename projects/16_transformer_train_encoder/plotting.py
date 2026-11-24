"""Visualizations for Week 16: training convergence, classification
quality, and the length x batch-size profiling sweep.

Confusion-matrix counts are a magnitude (>= 0) -> single-hue sequential
`Blues`, same convention as the heatmaps in `14_attention`/
`15_transformer_encoder`. Batch size in the profiling plots is an ordered
quantity (not an unordered category) -> a single sequential colormap
(`viridis`), the same choice `15_transformer_encoder` made for sequence
position.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TRAIN_COLOR = "steelblue"
VAL_COLOR = "darkorange"
HEATMAP_CMAP = "Blues"
BATCH_CMAP = "viridis"


def plot_convergence(history: dict[str, list[float]], path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(epochs, history["train_loss"], color=TRAIN_COLOR, marker="o", label="Train")
    axes[0].plot(epochs, history["val_loss"], color=VAL_COLOR, marker="o", label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], color=TRAIN_COLOR, marker="o", label="Train")
    axes[1].plot(epochs, history["val_acc"], color=VAL_COLOR, marker="o", label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.suptitle("Training convergence")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    """`cm`: (num_classes, num_classes) raw counts, rows = true label,
    columns = predicted label.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap=HEATMAP_CMAP, vmin=0, vmax=cm.max(), aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_profiling(results: list[dict], path: Path) -> None:
    """`results`: list of {seq_len, batch_size, seconds, bytes} dicts from
    `profiling.sweep`. Two panels: time vs. seq_len, memory vs. seq_len,
    one line per batch size. Attention's O(L^2) cost should show up as
    super-linear growth in the left panel.
    """
    batch_sizes = sorted(set(r["batch_size"] for r in results))
    cmap = plt.colormaps[BATCH_CMAP]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for i, B in enumerate(batch_sizes):
        rows = sorted((r for r in results if r["batch_size"] == B), key=lambda r: r["seq_len"])
        seq_lens = [r["seq_len"] for r in rows]
        color = cmap(i / max(len(batch_sizes) - 1, 1))
        axes[0].plot(seq_lens, [r["seconds"] * 1000 for r in rows], marker="o", color=color, label=f"B={B}")
        axes[1].plot(seq_lens, [r["bytes"] / 1e6 for r in rows], marker="o", color=color, label=f"B={B}")

    axes[0].set_xlabel("Sequence length (L)")
    axes[0].set_ylabel("Time per step (ms)")
    axes[0].set_title("Forward + backward time vs. L")
    axes[0].legend(title="Batch size")

    axes[1].set_xlabel("Sequence length (L)")
    axes[1].set_ylabel("Peak memory (MB)")
    axes[1].set_title("Memory vs. L")
    axes[1].legend(title="Batch size")

    fig.suptitle("Attention's O(L^2) cost: time/memory vs. sequence length and batch size")
    fig.savefig(path, dpi=150)
    plt.close(fig)
