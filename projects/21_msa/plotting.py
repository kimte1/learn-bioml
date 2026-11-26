"""
Visualizations for the MSA conservation/covariation/attention pipeline.
Same conventions as earlier projects: `Blues` for non-negative
magnitudes, a diverging colormap for signed quantities (MI_APC can be
negative), independently-scaled panels whenever two plotted quantities
live on very different numeric ranges.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_gap_occupancy(gap_frac: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(gap_frac)), gap_frac, color="tab:blue", width=1.0)
    ax.axhline(0.5, color="gray", linestyle="--", label="50% gap filter threshold")
    ax.set_xlabel("Alignment column (post-filtering)")
    ax.set_ylabel("Fraction of sequences with a gap")
    ax.set_title("Gap occupancy of the retained match-state columns")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_conservation_track(
    column_to_uniprot_position: dict[int, int],
    entropy: np.ndarray,
    path: Path,
) -> None:
    positions = sorted(column_to_uniprot_position.values())
    cols_by_position = {pos: col for col, pos in column_to_uniprot_position.items()}
    values = [entropy[cols_by_position[pos]] for pos in positions]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(positions, values, color="tab:blue")
    ax.set_xlabel("TEM-1 residue position (UniProt P62593 numbering)")
    ax.set_ylabel("Column entropy (bits)")
    ax.set_title("Conservation along the TEM-1 beta-lactamase sequence (low = conserved)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_covariation_heatmap(mi_apc: np.ndarray, path: Path) -> None:
    limit = np.abs(mi_apc).max()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mi_apc, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xlabel("Alignment column")
    ax.set_ylabel("Alignment column")
    ax.set_title("Column covariation (APC-corrected mutual information)")
    fig.colorbar(im, ax=ax, label="MI_APC (bits)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_conservation_vs_dms(
    column_to_uniprot_position: dict[int, int],
    entropy: np.ndarray,
    dms_scores_by_position: dict[int, list[float]],
    path: Path,
) -> None:
    """The real biological check: do more-conserved MSA columns
    correspond to positions where measured point mutations (ProteinGym's
    BLAT_ECOLX_Jacquier_2013 assay, `19_esm2_finetune`'s dataset) are on
    average more damaging?
    """
    xs, ys = [], []
    for col, position in column_to_uniprot_position.items():
        scores = dms_scores_by_position.get(position)
        if scores is None:
            continue
        xs.append(entropy[col])
        ys.append(-np.mean(scores))  # flip sign: more damaging (more negative DMS_score) -> larger value

    xs, ys = np.array(xs), np.array(ys)
    r = np.corrcoef(xs, ys)[0, 1]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(xs, ys, alpha=0.6, color="tab:blue")
    ax.set_xlabel("Column entropy (bits) -- higher = less conserved")
    ax.set_ylabel("Mean mutational damage at that position\n(-mean DMS_score)")
    ax.set_title(f"Conservation vs. measured mutational damage\n(r = {r:.2f}, n = {len(xs)})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_attention_maps(
    row_weights: np.ndarray,
    col_weights: np.ndarray,
    conserved_col: int,
    variable_col: int,
    path: Path,
) -> None:
    """row_weights: (nhead, num_cols, num_cols), untrained tied row
    attention. col_weights: (num_cols, nhead, num_seqs, num_seqs),
    untrained column attention -- shown at one conserved and one
    variable column for contrast. All from random weights: this is a
    shapes/operation demo, not a claim that these particular maps mean
    anything biologically (no training happened).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(row_weights[0], cmap="Blues")
    axes[0].set_title("Tied row attention (head 0)")
    axes[0].set_xlabel("Column (key)")
    axes[0].set_ylabel("Column (query)")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(col_weights[conserved_col, 0], cmap="Blues")
    axes[1].set_title(f"Column attention, head 0\n(conserved column {conserved_col})")
    axes[1].set_xlabel("Sequence (key)")
    axes[1].set_ylabel("Sequence (query)")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(col_weights[variable_col, 0], cmap="Blues")
    axes[2].set_title(f"Column attention, head 0\n(variable column {variable_col})")
    axes[2].set_xlabel("Sequence (key)")
    axes[2].set_ylabel("Sequence (query)")
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle("MSA Transformer-style attention, untrained (random weights)")
    fig.savefig(path, dpi=150)
    plt.close(fig)
