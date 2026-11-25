"""
A small BERT-style protein Transformer encoder, built from PyTorch's own
`nn.TransformerEncoder` -- same reasoning as
`16_transformer_train_encoder/model.py`: this week's named key skills
are about the pretraining objective and pipeline, not re-deriving the
attention/encoder architecture already built from scratch in
`14_attention`/`15_transformer_encoder`.

The one real architectural difference from Week 16's classifier: this
model predicts something at EVERY position (the original, possibly-
hidden residue there), not one label for the whole sequence -- so there
is no pooling step. The output head runs on the full (B, L, d_model)
tensor, giving (B, L, vocab_size) logits.
"""

from __future__ import annotations

import torch
from torch import nn

from positional_encoding import sinusoidal_positional_encoding


class BertStyleEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_idx,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        # the masked language model output head: one vocabulary-sized prediction PER POSITION
        self.mlm_head = nn.Linear(
            d_model,
            vocab_size,
        )

    def forward(
        self,
        ids: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """ids: (B, L) long, possibly containing <mask>/randomly-replaced
        tokens from `masking.mask_batch`. key_padding_mask: (B, L) bool,
        True = pad.

        Returns logits: (B, L, vocab_size) -- a full vocabulary
        distribution at every position, most of which get ignored by the
        loss (see `masking.py`'s IGNORE_INDEX), except the ~15% the
        training loop actually corrupted and wants predicted back.
        """
        x = self.embedding(ids) * (self.d_model**0.5)  # (B, L, d_model)

        seq_len = ids.shape[1]
        x = x + sinusoidal_positional_encoding(seq_len, self.d_model).to(x.device)

        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, L, d_model)

        return self.mlm_head(encoded)  # (B, L, vocab_size)
