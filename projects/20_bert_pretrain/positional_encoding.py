from __future__ import annotations

import math
import torch


def sinusoidal_positional_encoding(
    seq_len: int,
    d_model: int
) -> torch.Tensor:

    pe = torch.zeros(seq_len, d_model)

    for pos in range(seq_len):
        for d in range(d_model):
            i = d // 2 # determine if i is even or odd

            angle = pos / (10000 ** (2 * i / d_model))

            if d % 2 == 0:
                pe[pos, d] = math.sin(angle)
            else:
                pe[pos, d] = math.cos(angle)

    return pe
