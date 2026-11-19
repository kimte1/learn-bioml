"""
Scaled dot-product attention from scratch.

Forward-pass only, no prediction
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attention import (
    attention_entropy,
    causal_mask,
    init_qkv_weights,
    padding_mask,
    scaled_dot_product_attention
)

from data_load_prep import (
    load_sequences,
    load_variable_length_sequences,
    one_hot_encode
)

from data_synth import make_scaling_demo

from plotting import (
    plot_attention_heatmap,
    plot_attention_pipeline,
    plot_entropy_vs_dk,
    plot_mask,
    plot_qkv_attention,
    plot_qkv_weights
)

FIG_DIR = Path(__file__).resolve().parent / "fig"



def build_qkv_weights() -> tuple[
    str,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    '''
    step 1: show where Q, K, V actually come from 
    weight matrices W_Q, W_K, W_V projecting a one-hot input X
    '''

    # get 1 sequence (25 aa)
    sequence = load_sequences(max_len=25, n=1, seed=7)[0]

    # convert each aa as one-hot, shape (1, 25, 20)
    tokens, _ = one_hot_encode([sequence])

    # remove the outer []
    X = tokens[0]  # (25, 20)

    d_model = X.shape[-1]  # 20
    d_k = 16  # dimensions of weight's feature

    W_Q, W_K, W_V = init_qkv_weights(d_model, d_k, seed=7)

    plot_qkv_weights(
        X.numpy(), 
        W_Q.numpy(), 
        W_K.numpy(), 
        W_V.numpy(), 
        sequence, 
        FIG_DIR / "qkv_weights.png"
    )

    return sequence, X, W_Q, W_K, W_V



def scaled_attention_demo(
    sequence: str,
    X: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
) -> None:
    '''
    step 2: manually calculate scaled self-attention
    '''

    Q = X @ W_Q  # (25, 16)
    K = X @ W_K  # (25, 16)
    V = X @ W_V  # (25, 16)

    output, attn_weights = scaled_dot_product_attention(Q, K, V)

    L, d_k = Q.shape
    assert Q.shape == K.shape == V.shape == (L, d_k)
    assert output.shape == (L, d_k)
    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(
        row_sums, 
        torch.ones_like(row_sums), 
        atol=1e-5
    )

    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    plot_qkv_attention(
        Q.numpy(),
        K.numpy(),
        V.numpy(),
        scores.numpy(),
        attn_weights.numpy(),
        output.numpy(),
        sequence,
        FIG_DIR / "qkv_attention.png",
    )



def check_against_pytorch() -> None:
    '''
    step 3: compare masked self-attention computed from scratch vs
    pytorch's built-in function
    '''

    torch.manual_seed(1)
    B = 3
    L = 10
    d_k = 16
    d_v = 12
    Q = torch.randn(B, L, d_k)
    K = torch.randn(B, L, d_k)
    V = torch.randn(B, L, d_v)

    # boolean matrix to prevent each position from attending to future positions
    mask = causal_mask(L)

    attn_scratch, _ = scaled_dot_product_attention(
        Q,
        K,
        V,
        mask=mask,
    )

    attn_torch = F.scaled_dot_product_attention(
        Q,
        K,
        V,
        attn_mask=mask,
    )

    max_diff = (attn_scratch - attn_torch).abs().max().item()

    assert torch.allclose(
        attn_scratch,
        attn_torch,
        atol=1e-4
    ), max_diff



def scaling_demo() -> None:
    '''
    step 4: show why dividing attention scores by sqrt(d_k) prevents
    softmax saturation, by comparing scaled vs unscaled self-attention
    '''
    d_k_values = [4, 8, 16, 32, 64, 128, 256]
    demo = make_scaling_demo(d_k_values, seq_len=20, seed=4)

    scaled_entropy, unscaled_entropy = [], []
    for d_k in d_k_values:
        Q, K = demo[d_k]
        V = torch.eye(Q.shape[0])  # arbitrary fixed V -- only attn_weights matter here

        # scaled attn
        _, scaled_attn = scaled_dot_product_attention(Q, K, V)

        # unscaled attn
        unscaled_scores = Q @ K.transpose(-2, -1)  # no 1/sqrt(d_k)
        unscaled_attn = torch.softmax(unscaled_scores, dim=-1)

        # attn entropy - how evenly attn weights are distributed across positions
            # high entropy - attn is evenly distributed
            # low entropy - attn is concentrated on one position
        scaled_entropy.append(attention_entropy(scaled_attn).mean().item())
        unscaled_entropy.append(attention_entropy(unscaled_attn).mean().item())

    gap_small_dk = scaled_entropy[0] - unscaled_entropy[0]
    gap_large_dk = scaled_entropy[-1] - unscaled_entropy[-1]
    assert gap_large_dk > gap_small_dk, "the scaled/unscaled entropy gap should widen as d_k grows"

    plot_entropy_vs_dk(
        d_k_values,
        scaled_entropy,
        unscaled_entropy,
        FIG_DIR / "scaling_entropy.png"
    )



def causal_mask_demo() -> np.ndarray:
    '''
    step 5: show how a causal mask prevents a position from attending to future positions
    '''

    # pick first seq
    sequence = load_sequences(max_len=25, n=1, seed=3)[0]

    tokens, _ = one_hot_encode([sequence])  # (1, L, 20)
    L = tokens.shape[1]

    W_Q, W_K, W_V = init_qkv_weights(d_model=20, d_k=16, seed=3)
    Q, K, V = tokens[0] @ W_Q, tokens[0] @ W_K, tokens[0] @ W_V

    mask = causal_mask(L)

    _, attn_weights = scaled_dot_product_attention(
        Q,
        K,
        V,
        mask=mask,
    )

    # verify masking. identify positions above the diagonal
    upper_triangle = torch.triu(
        torch.ones(L, L, dtype=torch.bool),
        diagonal=1
    )
    assert torch.all(attn_weights[upper_triangle] == 0.0)

    plot_mask(
        attn_weights.numpy(),
        'causal mask',
        FIG_DIR / 'causal_mask.png',
    )



def show_sequence_lengths() -> list[str]:
    '''
    step 6a: show varying seq lens for 6 proteins
    (before padding forces them into a single tensor shape)
    '''
    sequences = load_variable_length_sequences(
        max_len=25,
        min_len=10,
        n=6,
        seed=2
    )
    lengths = [len(seq) for seq in sequences]
    print(f"[sequence lengths] {len(sequences)} real sequences, lengths = {lengths}")
    return sequences



def padding_mask_demo(sequences: list[str]) -> tuple[
    np.ndarray,
    int,
]:
    '''
    step 6b: show how attention ignores padded positions once sequences
    of different lengths are packed into one padded batch
    '''
    tokens, lengths = one_hot_encode(sequences)
    L_max = tokens.shape[1]  # len of longest seq

    print(f"[padding] padded every sequence to length {L_max}; tokens shape = {tuple(tokens.shape)}")

    mask = padding_mask(lengths, L_max)  # (B, 1, L_max)

    _, attn_weights = scaled_dot_product_attention(
        tokens,
        tokens,
        tokens,
        mask=mask  # use mask to prevent attn from being assigned to padded key positions
    )

    for i, length in enumerate(lengths.tolist()):
        weight_on_padding = attn_weights[i, :, length:].sum().item()
        assert weight_on_padding < 1e-5, f"sequence {i}: {weight_on_padding:.2e} weight leaked onto padding"

    # plot the most-padded sequence (shortest real length) -- sequence 0
    # can happen to be exactly L_max long (no padding at all), which would
    # show nothing
    idx = int(lengths.argmin())
    length = int(lengths[idx])
    print(f"[padding] plotting sequence {idx} (length {length}/{L_max} -- most padded in this batch)")

    plot_mask(
        attn_weights[idx].numpy(),
        f"padded mask (sequence {idx}, length {length}/{L_max})",
        FIG_DIR / 'padded_mask.png'
    )

    return attn_weights[idx].numpy(), length


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    # 1. where do Q, K, V come from
    sequence, X, W_Q, W_K, W_V = build_qkv_weights()

    # 2. scaled self-attention mechanics
    scaled_attention_demo(sequence, X, W_Q, W_K, W_V)

    # 3. from-scratch vs pytorch's built-in
    check_against_pytorch()

    # 4. scaled vs unscaled self-attention
    scaling_demo()

    # 5. demonstrate masking
    causal_mask_demo()

    # 6. padding: lengths first, then its effect on attention
    sequences = show_sequence_lengths()
    padding_mask_demo(sequences)



if __name__ == "__main__":
    main()
