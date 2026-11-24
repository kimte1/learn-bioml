from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

HEATMAP_CMAP = "Blues"
CATEGORICAL_CMAP = "tab10"

STANDARD_AMINO_ACIDS = set(
    "ACDEFGHIKLMNPQRSTVWY",
)
SPECIAL_TOKEN_COLORS = {
    "<cls>": "#d62728",
    "<eos>": "#9467bd",
    "<pad>": "#7f7f7f",
    "<mask>": "#ff7f0e",
    "<unk>": "#7f7f7f",
}


def plot_tokenization(
    token_strings: list[str],
    path: Path,
) -> None:
    
    n = len(token_strings)
    tokens_per_row = 30
    n_rows = -(-n // tokens_per_row)  # ceil division

    fig, ax = plt.subplots(
        figsize=(0.4 * tokens_per_row + 2, 0.5 * n_rows + 1),
    )

    for i, token in enumerate(token_strings):
        row = i // tokens_per_row
        col = i % tokens_per_row
        y = n_rows - 1 - row  # row 0 (sequence start) drawn at the top

        color = SPECIAL_TOKEN_COLORS.get(
            token,
            "#4a90d9" if token in STANDARD_AMINO_ACIDS else "#2ca02c",
        )
        ax.add_patch(
            plt.Rectangle(
                (col, y),
                1,
                1,
                facecolor=color,
                edgecolor="white",
            ),
        )
        ax.text(
            col + 0.5,
            y + 0.5,
            token,
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )

    ax.set_xlim(0, tokens_per_row)
    ax.set_ylim(0, n_rows)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(
        "Tokenized sequence (standard residues in blue, special tokens in their own color)",
    )
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=150,
    )
    plt.close(
        fig,
    )


def plot_contextual_vs_noncontextual(
    contextual_vecs: np.ndarray,
    noncontextual_vecs: np.ndarray,
    occurrence_labels: list[str],
    target_residue: str,
    path: Path,
) -> None:
    """# KEY SKILL: contextual embeddings.

    `contextual_vecs`/`noncontextual_vecs`: (n_occurrences, hidden_size),
    every occurrence of `target_residue` in one sequence. The
    non-contextual panel should be a flat wall of 1.0 (same lookup-table
    row every time); the contextual panel should show real variation --
    this is the falsifiable proof that "contextual" means something.
    """
    noncontextual_sim = cosine_similarity(
        noncontextual_vecs,
    )
    contextual_sim = cosine_similarity(
        contextual_vecs,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4.5),
        constrained_layout=True,
    )

    for ax, sim, title in zip(
        axes,
        [noncontextual_sim, contextual_sim],
        ["Non-contextual (input embedding table)", "Contextual (after the encoder)"],
    ):
        im = ax.imshow(
            sim,
            cmap=HEATMAP_CMAP,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        ax.set_xticks(
            range(len(occurrence_labels)),
        )
        ax.set_xticklabels(
            occurrence_labels,
            rotation=45,
            ha="right",
        )
        ax.set_yticks(
            range(len(occurrence_labels)),
        )
        ax.set_yticklabels(
            occurrence_labels,
        )
        ax.set_title(
            title,
        )

    fig.colorbar(
        im,
        ax=list(axes),
        label="Cosine similarity",
        shrink=0.8,
    )
    fig.suptitle(
        f"Every occurrence of residue '{target_residue}' in one sequence: same vector, or not?",
    )
    fig.savefig(
        path,
        dpi=150,
    )
    plt.close(
        fig,
    )


def plot_family_representations(
    gene_names: list[str],
    sequence_vecs: np.ndarray,
    path: Path,
) -> None:
    """# KEY SKILL: representation probing.

    `sequence_vecs`: (n_proteins, hidden_size), one pooled embedding per
    family member. Left panel: PCA(2D) scatter, labeled by gene name.
    Right panel: full pairwise cosine-similarity heatmap -- reuses the
    Week 8 PCA skill on a pretrained protein-LM embedding instead of
    hand-engineered composition features.
    """
    n = len(gene_names)
    cmap = plt.colormaps[CATEGORICAL_CMAP]
    colors = [cmap(i % cmap.N) for i in range(n)]

    coords = PCA(
        n_components=2,
    ).fit_transform(
        sequence_vecs,
    )
    similarity = cosine_similarity(
        sequence_vecs,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.5),
        constrained_layout=True,
    )

    axes[0].scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=80,
    )
    for i, gene in enumerate(gene_names):
        axes[0].annotate(
            gene,
            coords[i],
            xytext=(5, 5),
            textcoords="offset points",
        )
    axes[0].set_xlabel(
        "PC1",
    )
    axes[0].set_ylabel(
        "PC2",
    )
    axes[0].set_title(
        "PCA of sequence-level embeddings",
    )

    # mean-pooled sequence embeddings from a protein LM are all broadly
    # similar to each other (general "protein-ness" dominates over
    # family-specific differences) -- a fixed [0, 1] scale compresses the
    # real variation (here: 0.946-1.0) into the top of the colormap and
    # washes it out to a near-solid wall. Use the data's own range instead.
    im = axes[1].imshow(
        similarity,
        cmap=HEATMAP_CMAP,
        vmin=similarity.min(),
        vmax=1.0,
        aspect="auto",
    )
    axes[1].set_xticks(
        range(n),
    )
    axes[1].set_xticklabels(
        gene_names,
        rotation=45,
        ha="right",
    )
    axes[1].set_yticks(
        range(n),
    )
    axes[1].set_yticklabels(
        gene_names,
    )
    axes[1].set_title(
        "Pairwise cosine similarity",
    )
    fig.colorbar(
        im,
        ax=axes[1],
        label="Cosine similarity",
        shrink=0.8,
    )

    fig.suptitle(
        "Human globin family: does ESM-2's representation space group these sensibly?",
    )
    fig.savefig(
        path,
        dpi=150,
    )
    plt.close(
        fig,
    )
