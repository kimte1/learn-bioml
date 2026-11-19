"""
create a synthetic random Gaussian Q, K to demonstrate why attention is scaled by 1/sqrt(d_k)
"""

from __future__ import annotations

import torch


def make_scaling_demo(
    d_k_values: list[int],
    seq_len: int = 20,
    seed: int = 0,
) -> dict[
    int,
    tuple[torch.Tensor, torch.Tensor],
]:

    generator = torch.Generator().manual_seed(seed)
    result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    for d_k in d_k_values:
        Q = torch.randn(seq_len, d_k, generator=generator)
        K = torch.randn(seq_len, d_k, generator=generator)
        result[d_k] = (Q, K)
    return result
