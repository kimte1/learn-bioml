"""
Two representative neural-network operations, timed end-to-end rather
than as raw matmuls: a small MLP with forward AND backward (training
runs backward every step, and autograd's bookkeeping is itself part of
why GPUs matter -- not just the matmuls) and self-attention, the O(L^2)
operation this curriculum's own models (`14_attention` onward) are
built from.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from device_utils import median_time

BATCH_SIZES = [32, 128, 512, 2048, 8192]
SEQ_LENGTHS = [32, 64, 128, 256, 512, 1024]
D_MODEL = 256
NHEAD = 8


def benchmark_mlp(
    device: torch.device, 
    batch_size: int
) -> float:
    """Forward + backward through Linear(D_MODEL, 4*D_MODEL) -> ReLU ->
    Linear(4*D_MODEL, D_MODEL) -- the same widen-then-narrow shape as a
    Transformer feed-forward sublayer.
    """
    model = nn.Sequential(
        nn.Linear(D_MODEL, D_MODEL * 4),
        nn.ReLU(),
        nn.Linear(D_MODEL * 4, D_MODEL),
    ).to(device)
    x = torch.randn(batch_size, D_MODEL, device=device)

    def step() -> None:
        model.zero_grad()
        model(x).sum().backward()

    return median_time(step, device)


def benchmark_attention(
    device: torch.device, 
    seq_len: int, 
    batch_size: int = 32
) -> float:
    """Scaled dot-product self-attention, forward only. Shapes are
    (batch, nhead, seq_len, d_head) -- `F.scaled_dot_product_attention`
    is used directly
    """
    d_head = D_MODEL // NHEAD
    q = torch.randn(batch_size, NHEAD, seq_len, d_head, device=device)
    k = torch.randn(batch_size, NHEAD, seq_len, d_head, device=device)
    v = torch.randn(batch_size, NHEAD, seq_len, d_head, device=device)

    return median_time(
        lambda: F.scaled_dot_product_attention(q, k, v), 
        device
    )


def run_mlp_benchmark(devices: list[torch.device]) -> dict[torch.device, dict[str, list[float]]]:
    results = {}
    for device in devices:
        times = [benchmark_mlp(device, b) for b in BATCH_SIZES]
        results[device] = {"batch_sizes": BATCH_SIZES, "times": times}
    return results


def run_attention_benchmark(devices: list[torch.device]) -> dict[torch.device, dict[str, list[float]]]:
    results = {}
    for device in devices:
        times = [benchmark_attention(device, seq_len) for seq_len in SEQ_LENGTHS]
        results[device] = {"seq_lengths": SEQ_LENGTHS, "times": times}
    return results
