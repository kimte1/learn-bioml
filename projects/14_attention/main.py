"""
Scaled dot-product attention from scratch.

Forward-pass only, no prediction
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attention import (
    attention_entropy, 
    causal_mask, 
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
    plot_entropy_vs_dk, 
    plot_mask_comparison
)

FIG_DIR = Path(__file__).resolve().parent / "fig"



def check_shapes_and_softmax() -> None:
    '''
    verify that matmul bw Q, K, and V produce expected dimensions
    and softmax over attn weights = 1
    '''
    torch.manual_seed(0)
    B = 4 # batch_size
    L_q = 6 # query positions
    L_k = 7 # key positions
    d_k = 8 # query/key features
    d_v = 5 # value features

    # make random inputs 
    Q = torch.randn(B, L_q, d_k) 
    K = torch.randn(B, L_k, d_k)
    V = torch.randn(B, L_k, d_v)

    # compute attention
    output, attn_weights = scaled_dot_product_attention(Q, K, V)

    # check whether output dimensions matches Q, K, V's dimensions
    # , output.shape is the actual shape in the error message, if aseert failes
    assert output.shape == (B, L_q, d_v), output.shape
    assert attn_weights.shape == (B, L_q, L_k), attn_weights.shape

    # check that sum of attn weights == 1 
    row_sums = attn_weights.sum(dim=-1)

    assert torch.allclose(
        row_sums, 
        torch.ones_like(row_sums), 
        atol=1e-5
    )



def check_against_pytorch() -> None:
    ''' 
    compare masked self-attention computed from scratch vs pytorch's built-in function
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



def same_residue_probe() -> None:
    '''
    show that attention can identify matching residues w/o model training
    '''

    sequences = load_sequences(max_len=25, n=20, seed=0)

    # pick 1 seq with len 25
    sequence = next(seq for seq in sequences if len(seq) == 25)

    # one-hot encode with shape (batch, max_len, 20 pos)
    tokens, _ = one_hot_encode([sequence]) 
    
    # Define Q, K, V 
    # Q = K = qk. Matching residues have large dot products
    # V = tokens[0]. Original one-hot encoding
    scale = 8.0
    qk = tokens[0] * scale  # (25, 20)
    _, attn_weights = scaled_dot_product_attention(qk, qk, tokens[0])

    # boolean matrix identifying pairs of identical residues
    same_residue = torch.tensor([[a == b for b in sequence] for a in sequence])  # (25, 25) bool

    weight_on_same = attn_weights[same_residue].sum().item()
    weight_on_diff = attn_weights[~same_residue].sum().item()
    assert weight_on_diff < 1e-3, f"expected ~0 weight on different residues, got {weight_on_diff:.2e}"
    
    plot_attention_heatmap(
        attn_weights.numpy(),
        FIG_DIR / "same_residue_probe.png",
        title="Attention concentrates on matching amino acids\n(real sequence, hand-built Q=K)",
        query_labels=list(sequence),
        key_labels=list(sequence),
    )



def padding_mask_demo() -> tuple[np.ndarray, int]:
    '''
    show how attention ignores padded positions in seqs of diff lengths

    return attn_weights and seq length of the first protein
    '''

    sequences = load_variable_length_sequences(
        max_len=25, 
        min_len=10, 
        n=6, 
        seed=2
    )
    tokens, lengths = one_hot_encode(sequences)  # (B, L_max, 20), (B,)
    L_max = tokens.shape[1] # len of longest seq
    
    mask = padding_mask(lengths, L_max)  # (B, 1, L_max)

    _, attn_weights = scaled_dot_product_attention(
        tokens, 
        tokens, 
        tokens, 
        mask=mask # use mask to prevent attn from being assigned to padded key positions
    )

    for i, length in enumerate(lengths.tolist()):
        weight_on_padding = attn_weights[i, :, length:].sum().item()
        assert weight_on_padding < 1e-5, f"sequence {i}: {weight_on_padding:.2e} weight leaked onto padding"

    # keep first protein's attn_weights, and length
    return attn_weights[0].numpy(), int(lengths[0])



def causal_mask_demo() -> np.ndarray:
    '''
    verify that positions can't attend to future positions
    '''
    torch.manual_seed(3)

    # pick first seq
    sequence = load_sequences(max_len=25, n=1, seed=3)[0]

    tokens, _ = one_hot_encode([sequence])  # (1, L, 20)
    L = tokens.shape[1]

    # create random Q, K, V
    Wq, Wk, Wv = (torch.randn(20, 16) for _ in range(3))
    Q, K, V = tokens[0] @ Wq, tokens[0] @ Wk, tokens[0] @ Wv

    mask = causal_mask(L)

    _, attn_weights = scaled_dot_product_attention(
        Q, 
        K, 
        V, 
        mask=mask
    )

    # verify masking. identify positions above the diagonal
    upper_triangle = torch.triu(
        torch.ones(L, L, dtype=torch.bool), 
        diagonal=1
    )
    print(upper_triangle)
    assert torch.all(attn_weights[upper_triangle] == 0.0)
    
    return attn_weights.numpy()



def scaling_demo() -> None:
    '''
    show why dividing attentions cores by sqrt(d_k) prevents softmax saturation
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


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    check_shapes_and_softmax()
    check_against_pytorch()
    same_residue_probe()

    padding_attn, _ = padding_mask_demo()
    causal_attn = causal_mask_demo()
    plot_mask_comparison(
        causal_attn, 
        padding_attn, 
        FIG_DIR / "mask_comparison.png"
    )
    scaling_demo()




if __name__ == "__main__":
    main()
