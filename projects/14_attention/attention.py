"""
Scaled dot-product attention
"""

from __future__ import annotations

import math

import torch


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:

    d_k = Q.shape[-1]

    # (-2, -1) means swap the last and second-to-last dimensions
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    output = attn_weights @ V  

    return output, attn_weights



def causal_mask(seq_len: int) -> torch.Tensor:
    """
    boolean mask, True on/below the diagonal 
    """
    return torch.tril(
        torch.ones(
            seq_len, 
            seq_len, 
            dtype=torch.bool
        ),
    )



def padding_mask(lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    """`lengths`: (batch,) true (non-padded) length of each sequence.

    Returns a (batch, 1, seq_len) boolean mask, True for real key
    positions -- the singleton middle dim broadcasts over every query
    position, so every query in a sequence is blocked from attending to
    that sequence's padded key positions.
    """
    positions = torch.arange(seq_len).unsqueeze(0)  # (1, seq_len)
    mask = positions < lengths.unsqueeze(1)  # (batch, seq_len)
    return mask.unsqueeze(1)  # (batch, 1, seq_len)



def init_qkv_weights(
    d_model: int,
    d_k: int,
    d_v: int | None = None,
    seed: int = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    
    d_v = d_k if d_v is None else d_v
    generator = torch.Generator().manual_seed(seed)
    W_Q = torch.randn(d_model, d_k, generator=generator)
    W_K = torch.randn(d_model, d_k, generator=generator)
    W_V = torch.randn(d_model, d_v, generator=generator)

    return W_Q, W_K, W_V


def attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy (nats) of each row of `attn_weights`. Since softmax
    output is already a valid probability distribution over key positions,
    this needs no extra normalization. Low entropy means attention has
    concentrated (peaked) on a few keys; high entropy means it's close to
    uniform over all of them.
    """
    eps = 1e-12
    return -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)
