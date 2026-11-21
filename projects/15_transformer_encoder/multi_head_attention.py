"""
Multi-head self-attention (Vaswani et al., 2017, sec 3.2.2): run h parallel
scaled-dot-product attentions on learned linear projections of Q/K/V, each
in a smaller subspace (d_k = d_model / h), then concatenate the h outputs
and project back to d_model.

Uses `torch.nn.functional.scaled_dot_product_attention` for the actual
attention computation, rather than the from-scratch version in
`14_attention/attention.py`. That function doesn't return the attention
weights -- it may dispatch to a fused/flash-attention kernel that never
materializes the full (L, L) matrix (see Week 22, FlashAttention) -- so the
weights used for visualization/inspection below are recomputed separately
with the same formula; they play no part in computing `output`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divide evenly into num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        '''
        splits input tensor x into multiple heads
            - initial x.shape (B, L, d_model)
            - after .view(): (B, L, num_heads, d_k)
            - after .transpose(): (B, num_heads, L, d_k)
        '''
        B, L, _ = x.shape
        return x.view(
            B, 
            L, 
            self.num_heads, 
            self.d_k
        ).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, num_heads, L, d_k) -> (B, L, d_model)
        B, H, L, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * d_k)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, L, d_model). 

        Returns (output, attn_weights)
        - output: (B, L, d_model)
        - attn_weights: (B, num_heads, L, L)

        kept per-head (not averaged) so each head can be inspected
        separately.
        """
        Q = self._split_heads(self.W_q(x))
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        if mask is not None and mask.dim() == 3:
            # (B, 1, L) from `padding_mask` -> (B, 1, 1, L), broadcasts
            # over heads and query positions. A 2D `causal_mask` (L, L)
            # already broadcasts fine against (B, num_heads, L, L).
            mask = mask.unsqueeze(1)

        context = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)

        # recomputed only for inspection/visualization -- not part of the
        # output's compute graph
        with torch.no_grad():
            scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            attn_weights = torch.softmax(scores, dim=-1)

        output = self.W_o(self._merge_heads(context))
        return output, attn_weights
