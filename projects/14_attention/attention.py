"""
Scaled dot-product attention (Vaswani et al., 2017)
"""

from __future__ import annotations

import math

import torch


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    d_k = Q.shape[-1]
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
