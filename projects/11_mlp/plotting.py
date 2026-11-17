"""Visualization for the idiomatic-PyTorch MLP demo on the Delaney ESOL
solubility dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TRAIN_COLOR = "steelblue"
VAL_COLOR = "darkorange"


def plot_init_comparison(histories: dict[str, dict[str, list[float]]], path: Path) -> None:
    """Training loss vs. epoch, one line per weight-init scheme, same
    architecture and data otherwise. This is the direct, visual version
    of the symmetry-breaking / vanishing-variance arguments behind each
    scheme: "zeros" should barely move at all (every hidden unit computes
    the same thing and gets the same gradient, so there's nothing to
    break the symmetry), while "xavier" and "he" should converge
    smoothly since they're scaled to keep pre-activation variance roughly
    constant layer to layer.
    """
    plt.figure(figsize=(8, 5.5))
    cmap = plt.colormaps["tab10"]
    for i, (scheme, history) in enumerate(histories.items()):
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        plt.plot(epochs, history["train_loss"], color=cmap(i % cmap.N), label=scheme)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE (log scale)")
    plt.title("Effect of weight initialization on convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_predictions(
    preds_by_scheme: dict[str, np.ndarray],
    measured: np.ndarray,
    baseline_preds: np.ndarray,
    rmse_by_scheme: dict[str, float],
    baseline_rmse: float,
    path: Path,
) -> None:
    """Predicted vs. measured log solubility on the held-out test set --
    one panel per weight-init scheme, plus one for the published ESOL
    linear baseline, all on the same compounds. Points on the diagonal
    are exact predictions. A model that never broke symmetry (e.g.
    "zeros" init) should collapse toward predicting the same value for
    every compound, visible here as a flat horizontal band rather than a
    cloud of points tracking the diagonal.
    """
    all_preds = list(preds_by_scheme.values()) + [baseline_preds]
    lo = min(measured.min(), min(p.min() for p in all_preds))
    hi = max(measured.max(), max(p.max() for p in all_preds))

    # one panel per init scheme, plus one for the baseline, laid out 2
    # rows x 3 cols regardless of how many schemes there are -- any
    # unused cells (e.g. a 5th panel in a 6-cell grid) are hidden.
    panels = [(f"MLP (init={scheme})", preds, rmse_by_scheme[scheme], TRAIN_COLOR) for scheme, preds in preds_by_scheme.items()]
    panels.append(("ESOL linear baseline", baseline_preds, baseline_rmse, VAL_COLOR))

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.8 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (title, preds, rmse, color) in zip(axes, panels):
        ax.scatter(measured, preds, s=14, alpha=0.6, color=color)
        ax.set_title(f"{title}\nRMSE = {rmse:.3f}")
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Measured log solubility")

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    for row in range(n_rows):
        axes[row * n_cols].set_ylabel("Predicted log solubility")

    fig.suptitle("Test-set predictions by weight init, vs. published linear QSPR baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
