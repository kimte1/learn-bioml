"""
Visualizations for the Week 11 Transformer encoder.

Two kinds of heatmap appear here, and they get different colormaps on
purpose:
- Attention weights are a magnitude (>= 0, each row sums to 1) -> the same
  single-hue sequential `Blues` colormap used in `14_attention/plotting.py`.
- Embeddings and hidden representations are signed, with no natural zero
  point other than "no signal" -> a diverging colormap (`RdBu_r`) centered
  at 0, per the sequential-vs-diverging distinction (one hue for magnitude,
  two hues + a neutral midpoint for polarity).

Sequence positions are an ordered quantity (not an unordered category), so
the norm/PCA plots color them with a single sequential colormap (`viridis`)
rather than a fixed categorical palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

HEATMAP_CMAP = "Blues"
DIVERGING_CMAP = "RdBu_r"
POSITION_CMAP = "viridis"


def _label_axes(ax, sequence: list[str]) -> None:
    ax.set_yticks(range(len(sequence)))
    ax.set_yticklabels(sequence)


def plot_embedding_stages(
    token_emb: np.ndarray,
    pos_enc: np.ndarray,
    combined: np.ndarray,
    sequence: str,
    path: Path,
) -> None:
    """Three (L, d_model) heatmaps side by side: the raw token embedding,
    the positional encoding alone (the sinusoid stripe pattern should be
    visible here regardless of which sequence this is -- it doesn't depend
    on content), and their sum, which is what actually enters the encoder.

    The token embedding is scaled by sqrt(d_model) (sec 3.4) and so is
    roughly an order of magnitude larger than the positional encoding,
    which is bounded in [-1, 1] by construction -- sharing one color scale
    across all three would wash the positional-encoding panel out to
    near-white. So "token embedding" and "token + positional" (directly
    comparable magnitudes) share a scale, while "positional encoding" gets
    its own, fixed to [-1, 1].
    """
    shared_vmax = max(np.abs(token_emb).max(), np.abs(combined).max())

    fig, axes = plt.subplots(1, 3, figsize=(13, 0.4 * len(sequence) + 2), constrained_layout=True)
    panels = [
        ("Token embedding", token_emb, shared_vmax),
        ("Positional encoding", pos_enc, 1.0),
        ("Token + positional", combined, shared_vmax),
    ]
    images = []
    for ax, (title, data, vmax) in zip(axes, panels):
        images.append(ax.imshow(data, cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax, aspect="auto"))
        ax.set_title(title)
        ax.set_xlabel("d_model dimension")
        _label_axes(ax, list(sequence))

    fig.colorbar(images[0], ax=[axes[0], axes[2]], label="Value", shrink=0.8)
    fig.colorbar(images[1], ax=axes[1], label="Value", shrink=0.8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_attention_heads(attn_weights: np.ndarray, sequence: str, path: Path, title: str) -> None:
    """`attn_weights`: (num_heads, L, L). One heatmap per head, so heads
    that learned to attend differently are visibly different side by side.

    `vmax` is the data's own max, not a fixed 1.0: a fixed scale already
    washed out real-but-modest attention variation twice in
    `14_attention` (`qkv_attention.png`, `padded_mask.png`) -- this
    project's current weights happen to have decent contrast, but a fixed
    scale is fragile the same way.
    """
    num_heads = attn_weights.shape[0]
    vmax = attn_weights.max()
    fig, axes = plt.subplots(num_heads, 1, figsize=(4, 10), constrained_layout=True)
    if num_heads == 1:
        axes = [axes]

    im = None
    for h, ax in enumerate(axes):
        im = ax.imshow(attn_weights[h], cmap=HEATMAP_CMAP, vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_title(f"Head {h}")
        ax.set_xticks(range(len(sequence)))
        ax.set_xticklabels(list(sequence))
        ax.set_xlabel("Key position")
        _label_axes(ax, list(sequence))

    fig.suptitle(title)
    fig.colorbar(im, ax=axes, label="Attention weight", shrink=0.8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_attention_scores_heads(scores: np.ndarray, sequence: str, path: Path, title: str) -> None:
    """`scores`: (num_heads, L, L), pre-softmax QK^T/sqrt(d_k). Mirrors
    `plot_attention_heads` exactly, but for the stage *before* softmax --
    seeing both side by side is the actual multi-head story: each head
    diverges from the others starting here, not just in the final weights.
    Signed (DIVERGING_CMAP), since Q and K come from a real (learned)
    linear projection rather than a non-negative one-hot vector.
    """
    num_heads = scores.shape[0]
    vmax = np.abs(scores).max()
    fig, axes = plt.subplots(num_heads, 1 , figsize=(4, 10), constrained_layout=True)
    if num_heads == 1:
        axes = [axes]

    im = None
    for h, ax in enumerate(axes):
        im = ax.imshow(scores[h], cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"Head {h}")
        ax.set_xticks(range(len(sequence)))
        ax.set_xticklabels(list(sequence))
        ax.set_xlabel("Key position")
        _label_axes(ax, list(sequence))

    fig.suptitle(title)
    fig.colorbar(im, ax=axes, label="Score", shrink=0.8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_representation_norms(
    stage_names: list[str],
    norms_per_position: np.ndarray,
    d_model: int,
    path: Path,
) -> None:
    """`norms_per_position`: (num_stages, L). One line per sequence
    position, L2 norm vs. stage. A dashed reference line at sqrt(d_model)
    marks the norm a LayerNorm'd vector settles near (LayerNorm fixes each
    position's per-feature variance to ~1, so its L2 norm over d_model
    features is ~sqrt(d_model)) -- the sawtooth-then-reset pattern across
    stages is attention/the feed-forward network growing the norm, with
    each LayerNorm pulling it back down.
    """
    num_stages, L = norms_per_position.shape
    cmap = plt.colormaps[POSITION_CMAP]

    plt.figure(figsize=(9, 5.5))
    for pos in range(L):
        plt.plot(range(num_stages), norms_per_position[:, pos], color=cmap(pos / max(L - 1, 1)), linewidth=1.5, marker="o", markersize=4)

    plt.axhline(d_model**0.5, color="gray", linestyle="--", linewidth=1, label="sqrt(d_model)")
    plt.xticks(range(num_stages), stage_names, rotation=45, ha="right")
    plt.ylabel("L2 norm of position's representation")
    plt.title("Representation norm by sequence position, across encoder stages")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, L - 1))
    plt.colorbar(sm, ax=plt.gca(), label="Sequence position")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()



def plot_head_detail(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    context: np.ndarray,
    sequence: str,
    path: Path,
    title: str,
) -> None:
    """Where one head's Q, K, V, and output ("context", the per-head
    result before `W_o` merges every head back together) come from --
    all (L, d_k), all signed (DIVERGING_CMAP), sharing one scale so their
    magnitudes are directly comparable. This is head 0 only: seeing all 4
    heads' projections side by side here wouldn't add much (they look
    similarly distributed) -- the actual multi-head story is in
    `plot_attention_scores_heads` / `plot_attention_heads` instead, where
    the heads visibly diverge.
    """
    residues = list(sequence)
    dims = [str(i) for i in range(Q.shape[1])]
    vmax = max(np.abs(Q).max(), np.abs(K).max(), np.abs(V).max(), np.abs(context).max())

    fig, axes = plt.subplots(4, 1, figsize=(5, 10), constrained_layout=True)
    panels = [("Q (head 0)", Q), ("K (head 0)", K), ("V (head 0)", V), ("context (head 0)", context)]
    im = None
    for ax, (name, M) in zip(axes, panels):
        im = ax.imshow(M, cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels(dims)
        ax.set_xlabel("d_k dimension")
        _label_axes(ax, residues)
        ax.set_title(name)

    fig.colorbar(im, ax=list(axes), label="Value", shrink=0.8)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ffn_pipeline(
    ffn_in: np.ndarray,
    hidden: np.ndarray,
    ffn_out: np.ndarray,
    sequence: str,
    path: Path,
    title: str,
) -> None:
    """The position-wise feed-forward sublayer: input (L, d_model) ->
    hidden (L, d_ff), after Linear+ReLU -> output (L, d_model), after the
    second Linear. Input/output are signed (DIVERGING_CMAP, shared
    scale -- directly comparable, same dimensionality); the hidden
    activation is `>= 0` because of the ReLU, so it gets the sequential
    HEATMAP_CMAP instead -- a magnitude, not a signed quantity.
    """
    residues = list(sequence)
    io_vmax = max(np.abs(ffn_in).max(), np.abs(ffn_out).max())
    hidden_vmax = hidden.max()

    fig, axes = plt.subplots(
        3, 1, figsize=(5, 8), constrained_layout=True
    )

    im_in = axes[0].imshow(ffn_in, cmap=DIVERGING_CMAP, vmin=-io_vmax, vmax=io_vmax, aspect="auto")
    axes[0].set_title("Input")
    axes[0].set_xlabel("d_model dimension")
    _label_axes(axes[0], residues)

    im_hidden = axes[1].imshow(hidden, cmap=HEATMAP_CMAP, vmin=0.0, vmax=hidden_vmax, aspect="auto")
    axes[1].set_title("Hidden (after Linear1 + ReLU)")
    axes[1].set_xlabel("d_ff dimension")
    _label_axes(axes[1], residues)

    im_out = axes[2].imshow(ffn_out, cmap=DIVERGING_CMAP, vmin=-io_vmax, vmax=io_vmax, aspect="auto")
    axes[2].set_title("Output (after Linear2)")
    axes[2].set_xlabel("d_model dimension")
    _label_axes(axes[2], residues)

    fig.colorbar(im_in, ax=[axes[0], axes[2]], label="Value", shrink=0.8)
    fig.colorbar(im_hidden, ax=axes[1], label="Value (>= 0, post-ReLU)", shrink=0.8)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residual_effect(
    x: np.ndarray,
    sublayer_out: np.ndarray,
    summed: np.ndarray,
    normed: np.ndarray,
    sequence: str,
    path: Path,
    title: str,
) -> None:
    """The residual add and the normalization that follows it, each as its
    own visible panel: the sublayer's input `x`, the sublayer's raw output,
    their sum `x + sublayer_out` (pre-norm), and `LayerNorm(x +
    sublayer_out)`.

    Three separate scales, not one shared across all four: `x` and
    `x + Sublayer(x)` share a scale (the point is to directly compare
    "before" vs. "after adding the correction"), but `Sublayer(x)` gets
    its own -- its natural magnitude is typically smaller than `x`'s (the
    residual connection lets the sublayer act as a modest correction, not
    replace `x` outright), and sharing `x`'s scale would crush its own
    internal contrast down to near-invisible, the same mistake a fixed
    scale made in `14_attention`'s `qkv_attention.png`/`padded_mask.png`.
    `LayerNorm(...)`'s output lands in yet another, much smaller range by
    construction, so it gets its own scale too.
    """
    residues = list(sequence)
    dims = [str(i) for i in range(x.shape[1])]
    xsum_vmax = max(np.abs(x).max(), np.abs(summed).max())
    sublayer_vmax = np.abs(sublayer_out).max()
    normed_vmax = np.abs(normed).max()

    fig, axes = plt.subplots(4, 1, figsize=(5, 10), constrained_layout=True)
    panels = [
        ("x", x, xsum_vmax),
        ("Sublayer(x)", sublayer_out, sublayer_vmax),
        ("x + Sublayer(x)", summed, xsum_vmax),
        ("LayerNorm(x + Sublayer(x))", normed, normed_vmax),
    ]
    images = []
    for ax, (name, M, vmax) in zip(axes, panels):
        images.append(ax.imshow(M, cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax, aspect="auto"))
        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels(dims)
        ax.set_xlabel("d_model dimension")
        _label_axes(ax, residues)
        ax.set_title(name)

    fig.colorbar(images[0], ax=[axes[0], axes[2]], label="Value (x / x + Sublayer(x))", shrink=0.8)
    fig.colorbar(images[1], ax=axes[1], label="Value (Sublayer(x))", shrink=0.8)
    fig.colorbar(images[3], ax=axes[3], label="Value (normed)", shrink=0.8)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)
