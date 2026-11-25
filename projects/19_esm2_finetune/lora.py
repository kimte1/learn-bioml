"""
LoRA (Hu et al., 2022, "LoRA: Low-Rank Adaptation of Large Language
Models") implemented from scratch -- not via HuggingFace's `peft`
library, since this is a named key skill to *learn*, not just use.

# KEY SKILL: parameter-efficient fine-tuning
The idea: a pretrained weight matrix W (here, one of ESM-2's attention
projections, 320x320 = 102,400 numbers) is frozen -- never updated. Next
to it, we add two much smaller trainable matrices A (r x 320) and B (320
x r), where r ("rank") is a small number like 4 or 8. The adapted layer
computes `W @ x + (B @ A) @ x` instead of just `W @ x`. Since `B @ A` is
the product of a (320 x r) and an (r x 320) matrix, it can only ever
represent rank-r corrections to W -- far less expressive than retraining
all of W, but it needs only `2 * r * 320` trainable numbers instead of
`320 * 320`. At r=4, that's 2,560 trainable numbers standing in for
102,400 -- about 2.5%.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wraps an existing, already-trained `nn.Linear` layer. The original
    layer's weight and bias are frozen (`requires_grad = False`) --
    # KEY SKILL: layer freezing -- so gradients never update the
    pretrained weights at all; only `lora_A` and `lora_B` are trainable.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
    ):
        super().__init__()
        self.base_layer = base_layer

        # KEY SKILL: layer freezing -- the original pretrained projection
        # never gets a gradient update under any LoRA strategy
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # A is initialized with small random values (same scale PyTorch
        # uses for a fresh Linear layer's weights) so the adapter starts
        # by contributing something nonzero but small and learnable; B is
        # initialized to all zeros so that, at the very start of training,
        # `B @ A == 0` and the adapted layer computes EXACTLY the same
        # output as the original frozen layer -- fine-tuning starts from
        # "no change at all" and gradually learns a correction, rather
        # than starting from a random perturbation of a perfectly good
        # pretrained layer.
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features) / math.sqrt(in_features),
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank),
        )

        # `alpha / rank` rescales the adapter's contribution so that
        # changing `rank` alone (to make the adapter more or less
        # expressive) doesn't also silently change how large its
        # numerical contribution to the output is -- this is the
        # normalization Hu et al. (2022) use, not an arbitrary choice.
        self.scaling = alpha / rank

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        base_output = self.base_layer(
            x,
        )
        lora_correction = (
            x @ self.lora_A.T @ self.lora_B.T
        ) * self.scaling
        return base_output + lora_correction


def apply_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
) -> int:
    """Walks every `EsmSelfAttention` module in `model` and replaces its
    `query` and `value` projections with `LoRALinear`-wrapped versions, in
    place. `key` is deliberately left untouched: Hu et al. (2022) found
    adapting query and value alone captures most of LoRA's benefit, and
    adapting fewer matrices means fewer trainable parameters -- the whole
    point of this method.

    Returns the number of attention layers modified (6, for this
    6-layer checkpoint) as a sanity-checkable count.
    """
    # KEY SKILL: layer freezing -- freeze the ENTIRE backbone first. Only
    # wrapping query/value in LoRALinear (next) is not enough on its own:
    # every other parameter in the model (key projections, the
    # feed-forward layers, embeddings, LayerNorms) defaults to
    # `requires_grad = True` the moment the checkpoint is loaded, and
    # would otherwise stay fully trainable -- silently turning "LoRA" into
    # "full fine-tuning plus a couple of extra matrices." Confirmed this
    # was happening (83.66% of parameters trainable instead of ~2%) before
    # adding this loop.
    for parameter in model.parameters():
        parameter.requires_grad = False

    n_layers_modified = 0
    for module in model.modules():
        if type(module).__name__ == "EsmSelfAttention":
            module.query = LoRALinear(
                module.query,
                rank,
                alpha,
            )
            module.value = LoRALinear(
                module.value,
                rank,
                alpha,
            )
            n_layers_modified += 1

    return n_layers_modified
