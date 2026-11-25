"""
Visualizations comparing the 4 fine-tuning strategies.

`tab10` for the 4 strategies (a fixed, small set of distinct named
things, same convention as `18_esm2`'s globin genes); `Blues`-style single
hue isn't used here since nothing plotted is a pure magnitude -- MSE loss,
Spearman correlation, and trainable-parameter counts are each their own
kind of quantity, so each gets its own chart type instead of being
crammed into one heatmap.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CATEGORICAL_CMAP = "tab10"
STRATEGY_ORDER = [
    "linear_probe",
    "frozen_backbone",
    "lora",
    "full_finetune",
]


def _strategy_colors() -> dict[str, tuple]:
    cmap = plt.colormaps[CATEGORICAL_CMAP]
    return {
        name: cmap(i) for i, name in enumerate(STRATEGY_ORDER)
    }


def plot_convergence(
    histories_by_strategy: dict[str, dict[str, list[float]]],
    path: Path,
) -> None:
    """One line per strategy, train MSE and val Spearman correlation side
    by side -- loss going down and Spearman going up are two different
    ways of watching the same fine-tuning process improve.
    """
    colors = _strategy_colors()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        constrained_layout=True,
    )

    for strategy in STRATEGY_ORDER:
        history = histories_by_strategy[strategy]
        epochs = np.arange(
            1,
            len(history["train_loss"]) + 1,
        )
        axes[0].plot(
            epochs,
            history["train_loss"],
            color=colors[strategy],
            marker="o",
            label=strategy,
        )
        axes[1].plot(
            epochs,
            history["val_spearman"],
            color=colors[strategy],
            marker="o",
            label=strategy,
        )

    axes[0].set_xlabel(
        "Epoch",
    )
    axes[0].set_ylabel(
        "Train MSE",
    )
    axes[0].set_title(
        "Training loss",
    )
    axes[0].legend()

    axes[1].set_xlabel(
        "Epoch",
    )
    axes[1].set_ylabel(
        "Validation Spearman r",
    )
    axes[1].set_title(
        "Validation fitness-ranking correlation",
    )
    axes[1].legend()

    fig.suptitle(
        "Convergence by fine-tuning strategy",
    )
    fig.savefig(
        path,
        dpi=150,
    )
    plt.close(
        fig,
    )


def plot_trainable_params(
    counts_by_strategy: dict[str, int],
    path: Path,
) -> None:
    """Log-scale bar chart: the plot that makes "parameter-efficient"
    concrete rather than just asserted. LoRA's bar should sit much closer
    to linear-probe's/frozen-backbone's than to full-fine-tune's, even
    though LoRA (unlike those two) does adapt the backbone.
    """
    colors = _strategy_colors()
    counts = [
        counts_by_strategy[s] for s in STRATEGY_ORDER
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5),
    )
    bars = ax.bar(
        STRATEGY_ORDER,
        counts,
        color=[colors[s] for s in STRATEGY_ORDER],
    )
    ax.set_yscale(
        "log",
    )
    ax.set_ylabel(
        "Trainable parameters (log scale)",
    )
    ax.set_title(
        "How many parameters does each strategy actually train?",
    )

    for bar, count in zip(bars, counts):
        ax.annotate(
            f"{count:,}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=150,
    )
    plt.close(
        fig,
    )


def plot_before_after_finetuning(
    true_scores: list[float],
    preds_before_by_strategy: dict[str, np.ndarray],
    preds_after_by_strategy: dict[str, np.ndarray],
    spearman_before_by_strategy: dict[str, float],
    spearman_after_by_strategy: dict[str, float],
    path: Path,
) -> None:
    """4x2 grid: one row per strategy, "before fine-tuning" (freshly
    initialized head, untrained) vs. "after fine-tuning" (same model,
    post-training), both evaluated on the same held-out test set.

    "Before" should look like a random, uncorrelated cloud -- the head's
    weights are random, so its predictions have no relationship to the
    true fitness scores yet, regardless of strategy, since none of the 4
    strategies pretrain the head itself. "After" should show some
    visible positive trend for the strategies that learned something.

    No shared-scale diagonal reference line here, deliberately: Spearman
    correlation only cares about RANK order, not matching the true
    score's absolute scale, and this regressor never learns to -- its
    predictions stay in a narrow band (observed: ~0.007 wide) while true
    scores span ~6.3 units. Forcing the y-axis to match the x-axis's wide
    range (which a `[lo, hi]` diagonal line would do, since its own
    endpoints get included in the axis autoscaling) would squash that
    real, rank-correlated variation down to what looks like a flat line
    -- the exact same "shared scale washes out real variation" mistake
    already caught and fixed in `14_attention`/`18_esm2`. Each panel gets
    its own independent, auto-scaled y-axis instead.
    """
    true_scores = np.array(
        true_scores,
    )

    fig, axes = plt.subplots(
        len(STRATEGY_ORDER),
        2,
        figsize=(8, 3.2 * len(STRATEGY_ORDER)),
        constrained_layout=True,
    )

    for row, strategy in enumerate(STRATEGY_ORDER):
        for col, (preds_by_strategy, spearman_by_strategy, title) in enumerate(
            [
                (preds_before_by_strategy, spearman_before_by_strategy, "before fine-tuning"),
                (preds_after_by_strategy, spearman_after_by_strategy, "after fine-tuning"),
            ],
        ):
            ax = axes[row, col]
            predictions = preds_by_strategy[strategy]
            spearman_r = spearman_by_strategy[strategy]

            ax.scatter(
                true_scores,
                predictions,
                s=10,
                alpha=0.5,
            )
            ax.set_title(
                f"{strategy}, {title}\nSpearman r = {spearman_r:.2f}",
            )
            ax.set_xlabel(
                "True DMS_score",
            )
            ax.set_ylabel(
                "Predicted",
            )

    fig.suptitle(
        "Predicted vs. true fitness, before and after fine-tuning (test set)",
    )
    fig.savefig(
        path,
        dpi=150,
    )
    plt.close(
        fig,
    )
