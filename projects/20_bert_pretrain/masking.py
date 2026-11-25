"""
masked-language-modeling as self-supervised: 
    - the input and label sequence are the same, but the former has some positions hidden in the input. 
    - the labels are just "what was actually here before we hid it," which
is why this can scale to any amount of unlabeled sequence data.
"""

from __future__ import annotations

import torch

from data_load_prep import AMINO_ACIDS
from data_load_prep import MASK_IDX

IGNORE_INDEX = -100  # nn.CrossEntropyLoss's default ignore_index


def mask_batch(
    ids: torch.Tensor,
    key_padding_mask: torch.Tensor,
    mask_prob: float = 0.15,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ids: (B, L) amino-acid/pad indices (no masking applied
        yet). 
    key_padding_mask: (B, L) bool, True = padding.

    Returns:
        corrupted_ids: (B, L) long, same shape as `ids`, with ~15% of
            the *real* (non-pad) positions corrupted.
        labels: (B, L) long, holding the ORIGINAL token id at every
            corrupted position and IGNORE_INDEX (-100) everywhere else.
            Passing IGNORE_INDEX to `nn.CrossEntropyLoss` makes it skip
            those positions entirely when computing loss/gradients --
            this is what keeps the training signal restricted to exactly
            the positions we hid, and nowhere else.

    Of the positions selected for corruption (not of ALL positions),
    BERT's recipe splits them:
        80% -> replaced with <mask>
        10% -> replaced with a random (possibly wrong) amino acid
        10% -> left unchanged, but STILL scored in the loss

    Why not just always use <mask>? Because at fine-tuning/inference
    time there IS no <mask> token in real input -- if the model only
    ever learned to produce real representations at <mask> positions, it
    would have no idea what to do with an ordinary, unmasked residue.
    The 10% "random" and 10% "unchanged" cases force the model to build
    a genuinely contextual representation at EVERY position, since any
    position might secretly be corrupted (or might just look unchanged
    while actually being graded) -- it can't just special-case the
    <mask> token's shape.
    """
    real_positions = ~key_padding_mask  # (B, L), True = not padding

    # KEY SKILL: masking strategies -- decide which real positions get
    # corrupted at all
    select_roll = torch.rand(
        ids.shape,
        generator=generator,
    )
    selected = (select_roll < mask_prob) & real_positions

    labels = torch.full_like(
        ids,
        IGNORE_INDEX,
    )
    labels[selected] = ids[selected]

    # KEY SKILL: masking strategies -- the 80/10/10 split among selected
    # positions, so the model can't just learn "ignore anything shaped
    # like <mask>"
    split_roll = torch.rand(
        ids.shape,
        generator=generator,
    )
    to_mask_token = selected & (split_roll < 0.8)
    to_random_token = selected & (split_roll >= 0.8) & (split_roll < 0.9)
    # the remaining 10% (split_roll >= 0.9) stays unchanged in the input,
    # but `labels` still scores it above -- no further action needed here

    corrupted_ids = ids.clone()
    corrupted_ids[to_mask_token] = MASK_IDX

    random_aa_ids = torch.randint(
        0,
        len(AMINO_ACIDS),
        ids.shape,
        generator=generator,
    )
    corrupted_ids[to_random_token] = random_aa_ids[to_random_token]

    return corrupted_ids, labels
