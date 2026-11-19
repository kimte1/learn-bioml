from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCALED_COLOR = "steelblue"
UNSCALED_COLOR = "darkorange"
HEATMAP_CMAP = "Blues"
DIVERGING_CMAP = "RdBu_r"


def _heatmap_panel(
    ax,
    data: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str,
    xlabel: str,
    ylabel: str,
    xticklabels: list[str],
    yticklabels: list[str],
):

    im = ax.imshow(
        data, 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax, 
        aspect="auto"
    )
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im


def plot_qkv_weights(
    X: np.ndarray,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    sequence: str,
    path: Path,
) -> None:

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    residues = list(sequence)
    proj_dims = [str(i) for i in range(W_Q.shape[1])]

    w_vmax = max(np.abs(W_Q).max(), np.abs(W_K).max(), np.abs(W_V).max())

    fig, axes = plt.subplots(1, 4, figsize=(18, 0.35 * len(sequence) + 2), constrained_layout=True)

    im_x = _heatmap_panel(
        axes[0], 
        X, 
        0.0, 
        1.0, 
        HEATMAP_CMAP, 
        "Amino acid", 
        "Sequence position", 
        amino_acids, 
        residues
    )
    axes[0].set_title("Input X\n(one-hot encoding)")

    im_w = None

    for ax, name, W in zip(
        axes[1:], 
        ["W_Q", "W_K", "W_V"], 
        [W_Q, W_K, W_V]
    ):
        im_w = _heatmap_panel(
            ax, 
            W, 
            -w_vmax, 
            w_vmax, 
            DIVERGING_CMAP, 
            "Projection dimension", 
            "Amino acid", proj_dims, amino_acids)
        ax.set_title(name)

    fig.colorbar(
        im_x, 
        ax=axes[0], 
        label="Value", 
        shrink=0.8
    )
    fig.colorbar(
        im_w, 
        ax=list(axes[1:]), 
        label="Weight", 
        shrink=0.8
    )
    fig.suptitle("W_Q, W_K, and W_V")
    fig.savefig(path, dpi=150)
    plt.close(fig)



def plot_qkv_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    output: np.ndarray,
    sequence: str,
    path: Path,
) -> None:

    residues = list(sequence)
    proj_dims_k = [str(i) for i in range(Q.shape[1])]
    proj_dims_v = [str(i) for i in range(V.shape[1])]

    qkv_vmax = max(
        np.abs(Q).max(), 
        np.abs(K).max(), 
        np.abs(V).max()
    )

    scores_vmax = np.abs(scores).max()
    weights_vmax = weights.max()
    output_vmax = np.abs(output).max()

    fig, axes = plt.subplots(
        1, 
        6, 
        figsize=(26, 0.35 * len(sequence) + 2),
        constrained_layout=True
    )

    im_qkv = None
    panels = [("Q = X @ W_Q", Q, proj_dims_k), ("K = X @ W_K", K, proj_dims_k), ("V = X @ W_V", V, proj_dims_v)]

    for ax, (name, M, dims) in zip(axes[:3], panels):
        im_qkv = _heatmap_panel(
            ax, 
            M, 
            -qkv_vmax, 
            qkv_vmax, 
            DIVERGING_CMAP, 
            "Projection dimension", 
            "Sequence position", 
            dims, 
            residues
        )
        ax.set_title(name)

    im_scores = _heatmap_panel(
        axes[3], 
        scores, 
        -scores_vmax, 
        scores_vmax, 
        DIVERGING_CMAP, 
        "Key position", 
        "Query position", 
        residues, 
        residues
    )
    axes[3].set_title("Scores\n(QK^T / sqrt(d_k))")

    im_weights = _heatmap_panel(
        axes[4], 
        weights, 
        0.0, 
        weights_vmax, 
        HEATMAP_CMAP, 
        "Key position", 
        "Query position", 
        residues, 
        residues
    )
    axes[4].set_title("Weights\n(softmax(scores))")

    im_out = _heatmap_panel(
        axes[5], 
        output, 
        -output_vmax, 
        output_vmax, 
        DIVERGING_CMAP, 
        "Projection dimension", 
        "Sequence position", 
        proj_dims_v, 
        residues
    )
    axes[5].set_title("Output\n(weights @ V)")

    fig.colorbar(
        im_qkv, 
        ax=list(axes[:3]), 
        label="Value", 
        shrink=0.8
    )
    fig.colorbar(
        im_scores, 
        ax=axes[3], 
        label="Score", 
        shrink=0.8
    )
    fig.colorbar(
        im_weights, 
        ax=axes[4], 
        label="Attention weight", 
        shrink=0.8
    )
    fig.colorbar(
        im_out, 
        ax=axes[5], 
        label="Value", 
        shrink=0.8
    )
    fig.suptitle(f"Attention from projected Q, K, V: {sequence}")
    fig.savefig(path, dpi=150)
    plt.close(fig)



def plot_attention_pipeline(
    token_in: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    token_out: np.ndarray,
    sequence: str,
    path: Path,
) -> None:
    """One real sequence's trip through `softmax(QK^T/sqrt(d_k))V`, stage by
    stage, left to right:

    1. Input: (L, 20) one-hot amino-acid encoding.
    2. Scores: (L, L) raw QK^T/sqrt(d_k), before softmax -- unbounded above,
       so it gets its own color scale. Because Q=K is built so same-residue
       dot products are large and different-residue ones are exactly 0,
       this panel is already near-binary (bright on matches, white
       elsewhere) -- softmax's job here isn't to create that contrast.
    3. Weights: (L, L) softmax(scores) -- fixed [0, 1] scale. What softmax
       actually adds, visible by comparing to panel 2: every row now sums
       to 1 (a valid probability distribution, not just raw scores), and
       positions whose residue repeats elsewhere in the sequence split
       their weight evenly across every match -- visible as lighter blue,
       not solid navy, on rows with 2+ repeats of the same amino acid.
    4. Output: (L, 20) weights @ V -- same [0, 1] scale as the input panel,
       so "input" and "output" are directly comparable: each row should
       look like the input row, blended slightly toward other positions
       sharing that amino acid.
    """
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    residues = list(sequence)

    fig, axes = plt.subplots(1, 4, figsize=(22, 0.35 * len(sequence) + 2), constrained_layout=True)

    im_in = _heatmap_panel(axes[0], token_in, 0.0, 1.0, HEATMAP_CMAP, "Amino acid", "Sequence position", amino_acids, residues)
    axes[0].set_title("1. Input\n(one-hot encoding)")

    im_scores = _heatmap_panel(axes[1], scores, 0.0, scores.max(), HEATMAP_CMAP, "Key position", "Query position", residues, residues)
    axes[1].set_title("2. Scores\n(QK^T / sqrt(d_k))")

    im_weights = _heatmap_panel(axes[2], weights, 0.0, 1.0, HEATMAP_CMAP, "Key position", "Query position", residues, residues)
    axes[2].set_title("3. Weights\n(softmax(scores))")

    _heatmap_panel(axes[3], token_out, 0.0, 1.0, HEATMAP_CMAP, "Amino acid", "Sequence position", amino_acids, residues)
    axes[3].set_title("4. Output\n(weights @ V)")

    fig.colorbar(im_in, ax=[axes[0], axes[3]], label="Value (input/output)", shrink=0.8)
    fig.colorbar(im_scores, ax=axes[1], label="Score", shrink=0.8)
    fig.colorbar(im_weights, ax=axes[2], label="Attention weight", shrink=0.8)
    fig.suptitle(f"One sequence through softmax(QK^T/sqrt(d_k))V: {sequence}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    plt.plot(
        d_k_values, 
        scaled_entropy, 
        marker="o", 
        markersize=7, 
        linewidth=2, 
        color=SCALED_COLOR, 
        label="Scaled (1/sqrt(d_k))"
    )
    plt.plot(
        d_k_values, 
        unscaled_entropy, 
        marker="o", 
        markersize=7, 
        linewidth=2, 
        color=UNSCALED_COLOR, 
        label="Unscaled"
    )
    plt.xlabel("d_k")
    plt.ylabel("Mean softmax entropy (nats)")
    plt.title("Scaling keeps attention entropy stable as d_k grows")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()



def plot_mask(
    attn: np.ndarray,
    title: str,
    path: Path,
) -> None:
    # the data's own max, not a fixed 1.0 -- with Q=K=V unscaled one-hot
    # vectors, the score gap between matching/non-matching residues is
    # small, so softmax comes out close to uniform (~1/L per weight), far
    # below 1.0. A fixed [0, 1] scale would wash that real variation out.
    vmax = attn.max()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(
        attn,
        cmap=HEATMAP_CMAP,
        vmin=0.0,
        vmax=vmax,
        aspect="auto"
    )
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    fig.colorbar(im, ax=ax, label="Attention weight", shrink=0.8)
    fig.savefig(path, dpi=150)
    plt.close(fig)
