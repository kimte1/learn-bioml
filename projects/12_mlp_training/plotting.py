"""Visualization for the MLP training-mechanics ablations: optimizer
choice, weight decay, dropout x batch norm, learning-rate schedules,
gradient clipping, and the final tuned model's test-set predictions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TRAIN_COLOR = "steelblue"
VAL_COLOR = "darkorange"


def _loss_curve_comparison(
    histories: dict[str, dict[str, list[float]]],
    title: str,
    path: Path,
    loss_key: str = "train_loss",
    ylabel: str = "Train MSE (log scale)",
) -> None:
    plt.figure(figsize=(8, 5.5))
    cmap = plt.colormaps["tab10"]
    for i, (name, history) in enumerate(histories.items()):
        epochs = np.arange(1, len(history[loss_key]) + 1)
        plt.plot(epochs, history[loss_key], color=cmap(i % cmap.N), label=name)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_optimizer_comparison(histories: dict[str, dict[str, list[float]]], path: Path) -> None:
    """Train loss vs. epoch, one line per optimizer (SGD / SGD+momentum /
    Adam / AdamW), same architecture, data, and learning rate otherwise.
    Plain SGD should be the slowest; momentum should visibly speed it up
    along consistent gradient directions; Adam/AdamW's per-parameter
    adaptive steps typically converge fastest of all on a small, well-
    conditioned problem like this one. Adam and AdamW should be
    *identical* here (their curves fully overlap) -- AdamW only differs
    from Adam in how it applies weight decay, and this ablation runs
    with weight_decay=0, where that difference has nothing to act on.
    """
    _loss_curve_comparison(histories, "Effect of optimizer choice on convergence", path)


def plot_weight_decay_sweep(histories: dict[str, dict[str, list[float]]], path: Path) -> None:
    """Train vs. val loss for each weight-decay value, one panel per
    value. Weight decay penalizes large weights, so as it increases, the
    train/val gap should narrow (more regularization) up to a point --
    too much and both train and val loss rise together (the model is now
    too constrained to fit the data at all).
    """
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5), sharey=True, squeeze=False)
    for ax, (name, history) in zip(axes[0], histories.items()):
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], color=TRAIN_COLOR, label="train")
        ax.plot(epochs, history["val_loss"], color=VAL_COLOR, label="val")
        ax.set_yscale("log")
        ax.set_title(f"weight_decay={name}")
        ax.set_xlabel("Epoch")
    axes[0][0].set_ylabel("MSE (log scale)")
    axes[0][0].legend()
    fig.suptitle("Effect of weight decay on train/val loss")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_regularization_ablation(histories: dict[str, dict[str, list[float]]], path: Path) -> None:
    """Train vs. val loss curves for the 2x2 dropout x batch-norm grid.
    Batch norm should mainly show up as faster/smoother train-loss
    descent (it re-centers each layer's inputs every step, so the
    optimizer isn't fighting a shifting target). Dropout should mainly
    show up as a narrower train/val gap (it's a regularizer -- it should
    cost some train-loss headroom in exchange for val loss that tracks
    train loss more closely instead of diverging from it).
    """
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5), sharey=True, squeeze=False)
    for ax, (name, history) in zip(axes[0], histories.items()):
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], color=TRAIN_COLOR, label="train")
        ax.plot(epochs, history["val_loss"], color=VAL_COLOR, label="val")
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
    axes[0][0].set_ylabel("MSE (log scale)")
    axes[0][0].legend()
    fig.suptitle("Dropout x batch-norm ablation: train vs. val loss")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lr_schedule_comparison(histories: dict[str, dict[str, list[float]]], path: Path) -> None:
    """Two panels: val loss per epoch (top) and the actual learning rate
    per epoch (bottom), one line per schedule. The bottom panel is what
    makes each schedule's shape legible -- step's staircase, cosine's
    smooth decay, plateau's irregular drops timed to when val loss
    stalls, and a flat line for the constant-lr baseline.
    """
    cmap = plt.colormaps["tab10"]
    fig, (ax_loss, ax_lr) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for i, (name, history) in enumerate(histories.items()):
        epochs = np.arange(1, len(history["val_loss"]) + 1)
        color = cmap(i % cmap.N)
        ax_loss.plot(epochs, history["val_loss"], color=color, label=name)
        ax_lr.plot(epochs, history["lr"], color=color, label=name)

    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("Val MSE (log scale)")
    ax_loss.set_title("Effect of learning-rate schedule on convergence")
    ax_loss.legend()

    ax_lr.set_yscale("log")
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning rate (log scale)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grad_clip_comparison(histories: dict[str, dict[str, list[float]]], path: Path) -> None:
    """Train loss vs. epoch, with vs. without gradient clipping, run at a
    learning rate deliberately high enough that momentum's accumulated
    velocity sends an early update far too large. Unclipped, loss should
    spike to an enormous value within the first epoch or two -- so large
    (~1e15) that the weights never find their way back to a useful
    region even once the raw loss becomes finite again. Clipping caps
    how far a single large gradient can move the weights in one step, so
    the clipped run should train smoothly through the same learning rate
    that wrecks the unclipped one.
    """
    _loss_curve_comparison(
        histories,
        "Effect of gradient clipping at a high learning rate",
        path,
        ylabel="Train MSE (log scale)",
    )


def plot_final_predictions(
    mlp_preds: np.ndarray,
    measured: np.ndarray,
    baseline_preds: np.ndarray,
    mlp_rmse: float,
    baseline_rmse: float,
    path: Path,
) -> None:
    """Predicted vs. measured log solubility on the held-out test set,
    for the final tuned MLP (best optimizer, weight decay, regularization,
    and LR schedule from the ablations above, trained with early stopping
    and checkpointing) and for the published ESOL linear baseline.
    """
    lo = min(measured.min(), mlp_preds.min(), baseline_preds.min())
    hi = max(measured.max(), mlp_preds.max(), baseline_preds.max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)
    axes[0].scatter(measured, mlp_preds, s=14, alpha=0.6, color=TRAIN_COLOR)
    axes[0].set_title(f"MLP (tuned)\nRMSE = {mlp_rmse:.3f}")
    axes[1].scatter(measured, baseline_preds, s=14, alpha=0.6, color=VAL_COLOR)
    axes[1].set_title(f"ESOL linear baseline\nRMSE = {baseline_rmse:.3f}")

    for ax in axes:
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Measured log solubility")
    axes[0].set_ylabel("Predicted log solubility")

    fig.suptitle("Test-set predictions: tuned MLP vs. published linear QSPR baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
