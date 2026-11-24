"""
A Transformer encoder classifier built from PyTorch's own built-in
modules (`nn.TransformerEncoder`) rather than the from-scratch versions in
`14_attention`/`15_transformer_encoder` -- this week is about training
mechanics (padding, masking, batching, gradient accumulation, memory/
length scaling), not re-deriving the architecture.
"""

from __future__ import annotations

import torch
from torch import nn

from positional_encoding import sinusoidal_positional_encoding


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
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
        # enable_nested_tensor converts padded tensor into a nested tensor 
        # and a batch only keeps non-padded seqs for computation, to speed up processing
        # MPS doesn't do this. So set to False
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers, 
            enable_nested_tensor=False 
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, ids: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """ids: (B, L) long. key_padding_mask: (B, L) bool, True = pad.
        Returns logits: (B, num_classes).
        """
        L = ids.shape[1]
        x = self.embedding(ids) * (self.d_model**0.5)  # (B, L, d_model) -- sec 3.4 scaling
        x = x + sinusoidal_positional_encoding(L, self.d_model).to(x.device)

        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, L, d_model)

        # masked mean-pool over real (non-padded) positions only
        real = (~key_padding_mask).unsqueeze(-1).float()  # (B, L, 1)
        pooled = (encoded * real).sum(dim=1) / real.sum(dim=1).clamp(min=1.0)  # (B, d_model)

        return self.classifier(pooled)
