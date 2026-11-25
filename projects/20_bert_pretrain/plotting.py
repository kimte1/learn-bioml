"""
Visualizations for the masked-language-model pretraining run.

Same conventions as earlier projects: `tab10` for a small set of named
categories, `Blues` for non-negative magnitudes, independently-scaled
panels whenever two plotted quantities live on very different numeric
ranges -- loss/perplexity (can be anywhere from ~1 to ~20+) and accuracy
(always in [0, 1]) must NOT share an axis, the same "shared scale washes
out real variation" mistake already caught and fixed repeatedly in
`14_attention`/`18_esm2`/`19_esm2_finetune`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_load_prep import AMINO_ACIDS
from data_load_prep import MASK_IDX
from masking import IGNORE_INDEX
from train import tokenize_batch

INDEX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}


def plot_pretraining_curves(
    history: dict[str, list[float]],
    path: Path,
) -> None:
    """3 independently-scaled panels: loss, perplexity, masked-token
    accuracy -- each its own kind of quantity, so each gets its own
    y-axis rather than being crammed onto a shared one.
    """
    epochs = np.arange(
        1,
        len(history["train_loss"]) + 1,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
        constrained_layout=True,
    )

    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Masked-token cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_ppl"], marker="o", label="train")
    axes[1].plot(epochs, history["val_ppl"], marker="o", label="val")
    axes[1].axhline(
        len(AMINO_ACIDS),
        color="gray",
        linestyle="--",
        label="random guess (20 residues)",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity (exp(loss))")
    axes[1].set_title("Perplexity")
    axes[1].legend()

    axes[2].plot(epochs, history["val_masked_acc"], marker="o", color="tab:green")
    axes[2].axhline(
        1.0 / len(AMINO_ACIDS),
        color="gray",
        linestyle="--",
        label="random guess (1/20)",
    )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Top-1 accuracy at masked positions")
    axes[2].set_title("Validation masked-token accuracy")
    axes[2].set_ylim(0, 1)
    axes[2].legend()

    fig.suptitle("MLM pretraining: loss, perplexity, and masked-token accuracy")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_masking_sanity_check(
    original_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    labels: torch.Tensor,
    path: Path,
) -> None:
    """A direct visual unit test of `mask_batch`: among the positions it
    actually selected for corruption, what fraction really ended up
    <mask>-replaced vs. random-replaced vs. left unchanged? Should land
    close to the intended 80/10/10 split.
    """
    selected = labels != IGNORE_INDEX

    is_mask_token = selected & (corrupted_ids == MASK_IDX)
    is_unchanged = selected & (corrupted_ids == original_ids)
    is_random = selected & ~is_mask_token & ~is_unchanged

    n_selected = selected.sum().item()
    counts = {
        "<mask>": is_mask_token.sum().item() / n_selected,
        "random residue": is_random.sum().item() / n_selected,
        "unchanged": is_unchanged.sum().item() / n_selected,
    }
    targets = {
        "<mask>": 0.8,
        "random residue": 0.1,
        "unchanged": 0.1,
    }

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(counts))
    width = 0.35

    ax.bar(
        x - width / 2,
        list(counts.values()),
        width,
        label="realized",
        color="tab:blue",
    )
    ax.bar(
        x + width / 2,
        list(targets.values()),
        width,
        label="intended",
        color="lightgray",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(list(counts.keys()))
    ax.set_ylabel("Fraction of corrupted positions")
    ax.set_title(f"Masking sanity check (n={n_selected} corrupted positions)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fill_in_the_blank(
    sequences: list[str],
    model: torch.nn.Module,
    device: torch.device,
    path: Path,
    n_sequences: int = 6,
    n_per_sequence: int = 2,
    seed: int = 0,
) -> None:
    """The qualitative answer to "did pretraining teach the model
    anything about protein sequences": for a handful of validation
    sequences, hide a few real positions and show the model's single
    best guess next to the true residue.
    """
    model.eval()
    sample_sequences = sequences[:n_sequences]

    ids, key_padding_mask = tokenize_batch(sample_sequences)

    # hand-pick exactly `n_per_sequence` real positions per sequence to
    # hide -- `mask_batch`'s own random ~15% selection isn't used here
    # since a sparse, fixed number of blanks per sequence is easier to
    # read in a small table than a full random pass would be
    rows = []
    rng = torch.Generator()
    rng.manual_seed(seed)
    for seq_idx, seq in enumerate(sample_sequences):
        positions = torch.randperm(len(seq), generator=rng)[:n_per_sequence]
        for pos in positions.tolist():
            rows.append((seq_idx, pos, seq[pos]))

    with torch.no_grad():
        single_ids = ids.clone()
        for seq_idx, pos, _true_aa in rows:
            single_ids[seq_idx, pos] = MASK_IDX
        single_ids = single_ids.to(device)
        key_padding_mask_device = key_padding_mask.to(device)
        logits = model(single_ids, key_padding_mask_device)  # (B, L, vocab)
        predicted_ids = logits.argmax(dim=-1).cpu()

    table_rows = []
    for seq_idx, pos, true_aa in rows:
        predicted_aa = INDEX_TO_AA.get(predicted_ids[seq_idx, pos].item(), "?")
        correct = predicted_aa == true_aa
        table_rows.append([f"seq {seq_idx}", str(pos), true_aa, predicted_aa, "yes" if correct else "no"])

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(table_rows) + 1))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=["sequence", "position", "true residue", "predicted", "correct?"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for row_idx, row in enumerate(table_rows, start=1):
        cell = table[row_idx, 4]
        cell.set_facecolor("#d9ead3" if row[4] == "yes" else "#f4cccc")

    ax.set_title("Fill-in-the-blank: true vs. predicted residue at masked positions")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
