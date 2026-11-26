"""
Raw matrix-multiplication throughput across square sizes and devices
"""

from __future__ import annotations

import torch

from device_utils import median_time

MATRIX_SIZES = [64, 128, 256, 512, 1024, 2048, 4096]


def benchmark_matmul(
    device: torch.device, 
    size: int
) -> float:
    """Median seconds for one (size, size) @ (size, size) matmul."""
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    return median_time(lambda: a @ b, device)


def matmul_gflops(size: int, seconds: float) -> float:
    """An (n,n)@(n,n) matmul produces n^2 output entries, each an
    n-length dot product -- n multiplies + n adds. 2*n^3 FLOPs total.
    """
    return 2 * size**3 / seconds / 1e9


def run_matmul_benchmark(devices: list[torch.device]) -> dict[torch.device, dict[str, list[float]]]:
    results = {}
    for device in devices:
        times = [benchmark_matmul(device, size) for size in MATRIX_SIZES]
        gflops = [matmul_gflops(size, t) for size, t in zip(MATRIX_SIZES, times)]
        results[device] = {"sizes": MATRIX_SIZES, "times": times, "gflops": gflops}
    return results
