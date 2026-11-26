"""
Host (CPU RAM) -> device (GPU VRAM) transfer cost -- the thing that
makes "just move everything to the GPU" not free. `main.py` plots this
directly against matmul compute time at the same size, to see where
transfer stops being the bottleneck.
"""

from __future__ import annotations

import torch

from device_utils import median_time

TRANSFER_SIZES = [64, 256, 1024, 2048, 4096, 8192]


def benchmark_host_to_device(
    device: torch.device, 
    size: int
) -> float:
    """Median seconds to move one (size, size) tensor from CPU to `device`."""
    cpu_tensor = torch.randn(size, size)
    return median_time(lambda: cpu_tensor.to(device), device)


def run_transfer_benchmark(devices: list[torch.device]) -> dict[torch.device, dict[str, list[float]]]:
    """
    Only meaningful for actual GPU devices -- CPU is the host, so
    there's no transfer for it to make.
    """
    results = {}
    for device in devices:
        if device.type == "cpu":
            continue
        times = [
            benchmark_host_to_device(device, size) 
            for size in TRANSFER_SIZES
        ]
        results[device] = {"sizes": TRANSFER_SIZES, "times": times}
    return results
