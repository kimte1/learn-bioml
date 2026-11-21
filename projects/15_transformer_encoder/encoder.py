"""
Minimal Transformer encoder (Vaswani et al., 2017, sec 3.1): a stack of
identical layers, each with two sublayers -- multi-head self-attention and
a position-wise feed-forward network -- each wrapped in a residual
connection followed by LayerNorm: `x = LayerNorm(x + Sublayer(x))`
(post-norm, as in the original paper). The residual path is what lets the
stack go deep: even if a sublayer's output were exactly zero, information
from `x` still passes straight through.
"""

from __future__ import annotations

import torch
from torch import nn

from feed_forward import PositionwiseFeedForward
from multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int
    ):
        ''' defines a single encoder layer '''
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns `(x_after_ffn, attn_weights, x_after_attn)` -- both
        post-sublayer representations are returned (not just the final
        one) so a caller can inspect what each sublayer did on its own.
        """
        attn_out, attn_weights = self.self_attn(x, mask=mask)
        x_after_attn = self.norm1(x + attn_out)

        ff_out = self.feed_forward(x_after_attn)
        x_after_ffn = self.norm2(x_after_attn + ff_out)

        return x_after_ffn, attn_weights, x_after_attn



class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        num_layers: int, 
        d_model: int, 
        num_heads: int, 
        d_ff: int
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[dict]]:
        """Returns `(output, stages)` where `stages` has one dict per
        layer: `{"attn_weights", "after_attn", "after_ffn"}`, so every
        sublayer's output along the stack can be inspected, not just the
        final one.
        """
        stages: list[dict] = []
        for layer in self.layers:
            x, attn_weights, x_after_attn = layer(x, mask=mask)
            stages.append({
                "attn_weights": attn_weights, 
                "after_attn": x_after_attn, 
                "after_ffn": x,
            })
        return x, stages
