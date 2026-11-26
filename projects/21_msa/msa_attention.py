"""
MSA Transformer's two attention operations (Rao et al. 2021) at toy
scale, run once with random weights -- no training, just the shapes and
the operation itself, over a slice of the real alignment loaded by
`data_load_prep.py`.

Row attention here is TIED: unlike ordinary self-attention, which would
give each sequence its own independent (num_cols, num_cols) attention
map, tied row attention averages the attention logits across every
sequence in the MSA first, then applies ONE shared map to every row --
the specific mechanism that lets the model use the whole MSA's
statistics rather than just one sequence's own context. (Built from the
same scaled-dot-product idea as `14_attention/attention.py`, just with
that extra averaging step.)

Column attention is the complementary operation: ordinary self-attention
across the sequence axis, run independently for each alignment column --
structurally the piece that can pick up coevolution between positions.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class TiedRowSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int = 2):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (num_seqs, num_cols, d_model).

        Returns (output, tied_weights). tied_weights: (nhead, num_cols,
        num_cols) -- one map per head, shared across every sequence.
        """
        num_seqs, num_cols, d_model = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(num_seqs, num_cols, self.nhead, self.d_head).permute(2, 0, 1, 3)
            # (nhead, num_seqs, num_cols, d_head)

        Q = split_heads(self.q_proj(x))
        K = split_heads(self.k_proj(x))
        V = split_heads(self.v_proj(x))

        # per-sequence logits, then averaged across sequences -- this
        # averaging is the "tying"
        logits = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)  # (nhead, num_seqs, num_cols, num_cols)
        tied_logits = logits.mean(dim=1)  # (nhead, num_cols, num_cols)
        tied_weights = torch.softmax(tied_logits, dim=-1)

        context = tied_weights.unsqueeze(1) @ V  # (nhead, num_seqs, num_cols, d_head)
        context = context.permute(1, 2, 0, 3).reshape(num_seqs, num_cols, d_model)
        return self.out_proj(context), tied_weights


class ColumnSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int = 2):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (num_seqs, num_cols, d_model).

        Returns (output, weights). weights: (num_cols, nhead, num_seqs,
        num_seqs) -- NOT tied, every column gets its own attention map
        over the sequence axis.
        """
        num_seqs, num_cols, d_model = x.shape
        x_t = x.transpose(0, 1)  # (num_cols, num_seqs, d_model) -- columns become the "batch"

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(num_cols, num_seqs, self.nhead, self.d_head).permute(0, 2, 1, 3)
            # (num_cols, nhead, num_seqs, d_head)

        Q = split_heads(self.q_proj(x_t))
        K = split_heads(self.k_proj(x_t))
        V = split_heads(self.v_proj(x_t))

        logits = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)  # (num_cols, nhead, num_seqs, num_seqs)
        weights = torch.softmax(logits, dim=-1)
        context = weights @ V  # (num_cols, nhead, num_seqs, d_head)

        context = context.permute(2, 0, 1, 3).reshape(num_seqs, num_cols, d_model)
        return self.out_proj(context), weights


class MSARowColumnBlock(nn.Module):
    """One MSA Transformer-style block: tied row attention, then column
    attention, each with a residual connection. No feed-forward/layer
    norm -- this is a shape/operation demo, not a model meant to be
    trained (see curriculum: "data analysis and a small attention
    operation, not training").
    """

    def __init__(self, d_model: int, nhead: int = 2):
        super().__init__()
        self.row_attn = TiedRowSelfAttention(d_model, nhead)
        self.col_attn = ColumnSelfAttention(d_model, nhead)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_out, row_weights = self.row_attn(x)
        x = x + row_out
        col_out, col_weights = self.col_attn(x)
        x = x + col_out
        return x, row_weights, col_weights
