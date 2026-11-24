"""
Sequence-length x batch-size sweep: how does one forward+backward pass's
wall-clock time and MPS memory scale with L and B? Dummy, full-length
(unpadded) random-token batches are used here on purpose -- this sweep is
about compute/memory scaling with tensor *shape*, not about the DeepLoc
task's accuracy, and dense (unpadded) sequences are attention's worst case
for a given L.
"""

from __future__ import annotations

import time

import torch
from data_load_prep import VOCAB_SIZE
from torch import nn


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def profile_shape(
    model: nn.Module, 
    device: torch.device, 
    seq_len: int, 
    batch_size: int
) -> tuple[float, int]:
    """One forward+backward pass at this (seq_len, batch_size), full dense
    sequences, no padding. Returns (seconds, activation_bytes).

    `torch.mps` has no CUDA-style peak-memory watermark (no
    `max_memory_allocated`/`reset_peak_memory_stats`), only a snapshot via
    `current_allocated_memory()`. 

    Measuring right after the forward pass to get the dominant memory cost 
    and a reasonable stand-in for "peak" given MPS's more limited introspection than CUDA.
    """
    model.to(device)
    model.train()

    ids = torch.randint(
        0, 
        VOCAB_SIZE - 1, 
        (batch_size, seq_len), 
        device=device
    )  # never samples PAD_IDX
    key_padding_mask = torch.zeros(
        batch_size, 
        seq_len, 
        dtype=torch.bool, 
        device=device
    )
    labels = torch.randint(
        0, 
        model.classifier.out_features, 
        (batch_size,), 
        device=device,
    )
    loss_fn = nn.CrossEntropyLoss()

    if device.type == "mps":
        torch.mps.empty_cache()
        start_mem = torch.mps.current_allocated_memory()

    _sync(device)
    start = time.perf_counter()

    logits = model(ids, key_padding_mask)

    _sync(device)
    if device.type == "mps":
        activation_bytes = torch.mps.current_allocated_memory() - start_mem
    else:
        activation_bytes = 0

    loss = loss_fn(logits, labels)
    loss.backward()

    _sync(device)
    elapsed = time.perf_counter() - start

    model.zero_grad(set_to_none=True)
    return elapsed, activation_bytes


def sweep(
    model: nn.Module,
    device: torch.device,
    seq_lens: list[int],
    batch_sizes: list[int],
    warmup: bool = True,
    repeats: int = 5,
) -> list[dict]:
    """Runs `profile_shape` over every (L, B) combination, `repeats` times
    each, and keeps the median time (single-trial timings are noisy at
    small shapes, dominated by dispatch/scheduling jitter rather than
    actual compute -- the median is far more stable than one sample).
    `bytes` is deterministic given the shape, so any one repeat's value
    is used as-is. Returns a list of dicts with keys
    seq_len/batch_size/seconds/bytes.
    """
    results = []
    if warmup:
        profile_shape(model, device, seq_lens[0], batch_sizes[0])  # first-call overhead (kernel compilation etc.) shouldn't pollute the real measurements

    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            trials = [profile_shape(model, device, seq_len, batch_size) for _ in range(repeats)]
            seconds = sorted(t[0] for t in trials)[len(trials) // 2]  # median
            peak_bytes = trials[0][1]
            results.append({
                    "seq_len": seq_len, 
                    "batch_size": batch_size, 
                    "seconds": seconds, 
                    "bytes": peak_bytes
            })
            print(f"L={seq_len:4d} B={batch_size:3d}: {seconds * 1000:7.2f} ms, {peak_bytes / 1e6:7.2f} MB")

    return results
