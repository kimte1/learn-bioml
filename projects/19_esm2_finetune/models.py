"""
Four ways to adapt the same pretrained ESM-2 backbone to one task --
predicting a mutant protein's fitness score -- that differ only in which
parameters are allowed to change during training.

# KEY SKILL: pretrained checkpoints -- every strategy below starts from
the exact same `biohub/esm2_t6_8M_UR50D` weights used in `18_esm2`, loaded
fresh each time. None of them ever train a protein language model from
scratch; the whole point of this week is to reuse what ESM-2 already
learned from masked-language-modeling over huge sequence databases, and
only adapt it.

# KEY SKILL: transfer learning -- "fine-tuning" in general means exactly
this: take a model already trained on a large, generic objective, and
continue training it (fully or partially) on a smaller, task-specific
labeled dataset, instead of training a new model from random
initialization for that task. The 4 functions below are 4 different
*amounts* of adaptation, from "barely adapt it at all" (linear probe) to
"adapt everything" (full fine-tune).
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from lora import apply_lora

CHECKPOINT = "biohub/esm2_t6_8M_UR50D"
HIDDEN_SIZE = 320


class FitnessRegressor(nn.Module):
    """
    Wraps an ESM-2 backbone plus a regression head: 
        backbone -> masked mean-pool over real residues -> head -> one scalar (`DMS_score`).

    The pooling step is the same idea as `18_esm2/embeddings.py`'s
    `sequence_embedding`: average the per-residue contextual vectors,
    excluding `<cls>` (index 0) and `<eos>` (the last index), to get one
    fixed-size vector per sequence regardless of how it was tokenized.
    Every sequence in this dataset happens to tokenize to the same length
    (286 residues + 2 special tokens, always), so there's no padding to
    additionally exclude here, unlike a dataset with variable-length
    sequences.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = output.last_hidden_state  # (B, L+2, 320)

        real_residues = hidden_states[:, 1:-1, :]  # drop <cls> and <eos>
        pooled = real_residues.mean(dim=1)  # (B, 320)

        return self.head(pooled).squeeze(-1)


def _count_trainable_parameters(
    model: nn.Module,
) -> int:
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )


def build_linear_probe() -> tuple[FitnessRegressor, int]:
    """
    Strategy 1: the cheapest possible adaptation. 
    The entire backbone is frozen, sinalge `nn.Linear(320, 1)` 
    reads the (fixed) pooled embedding directly. 

    This is "linear probing": a standard way to ask "how good is this
    pretrained representation on its own, with no fine-tuning of the
    representation itself?"
    """
    backbone = AutoModel.from_pretrained(CHECKPOINT)

    for parameter in backbone.parameters():
        parameter.requires_grad = False

    head = nn.Linear(HIDDEN_SIZE, 1)

    model = FitnessRegressor(
        backbone,
        head,
    )
    return model, _count_trainable_parameters(model)


def build_frozen_backbone() -> tuple[FitnessRegressor, int]:
    """Strategy 2: same frozen backbone as the linear probe, but a head
    with more capacity (one hidden layer instead of none). This isolates
    "how much does head capacity alone matter" from "how much does
    touching the backbone matter" -- strategies 1 and 2 never touch the
    backbone at all; strategies 3 and 4 do.
    """
    backbone = AutoModel.from_pretrained(
        CHECKPOINT,
    )
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    head = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

    model = FitnessRegressor(
        backbone,
        head,
    )
    return model, _count_trainable_parameters(model)


def build_lora(
    rank: int = 4,
    alpha: float = 8.0,
) -> tuple[FitnessRegressor, int]:
    """Strategy 3: # KEY SKILL: parameter-efficient fine-tuning. The
    backbone IS adapted this time -- unlike strategies 1 and 2 -- but only
    through the tiny LoRA matrices `apply_lora` installs (see `lora.py`
    for the full explanation); everything else in the backbone stays
    frozen. Same head architecture as `build_frozen_backbone`, so any
    performance difference between the two comes from the backbone
    adaptation, not a bigger head.
    """
    backbone = AutoModel.from_pretrained(
        CHECKPOINT,
    )
    apply_lora(
        backbone,
        rank,
        alpha,
    )

    head = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

    model = FitnessRegressor(
        backbone,
        head,
    )
    return model, _count_trainable_parameters(model)


def build_full_finetune() -> tuple[FitnessRegressor, int]:
    """Strategy 4: the most expensive adaptation -- nothing is frozen.
    Every one of the backbone's ~7.5M parameters, plus the head, can move
    during training. This is the strategy curriculum.md's Week 19 Hardware
    note calls out as needing an HPC GPU "for reasonable memory/speed" --
    but that note is calibrated for ESM-2's larger checkpoints (650M+
    parameters); at this 8M-parameter size, full fine-tuning is small
    enough to run on MPS too.
    """
    backbone = AutoModel.from_pretrained(
        CHECKPOINT,
    )
    # explicitly leaving every parameter's requires_grad at its loaded
    # default (True) -- nothing frozen, the opposite of every other
    # strategy above

    head = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

    model = FitnessRegressor(
        backbone,
        head,
    )
    return model, _count_trainable_parameters(model)
