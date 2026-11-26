"""
A GPU is thousands of weak, simple cores (vs CPU's handful)
The advantage of using a GPU occurs when there's enough independent
work to spread across all those cores, and when the cost of moving data
from host RAM to device VRAM is small relative to the compute it buys.

THE SPECIFIC TASK HERE: 
- benchmark matrix multiplication, a small MLP, and self-attention across a range of size
- measure host <-> device transfer cost
"""

from __future__ import annotations

from pathlib import Path

from device_utils import available_devices
from matmul_bench import run_matmul_benchmark
from nn_ops_bench import run_attention_benchmark
from nn_ops_bench import run_mlp_benchmark
from plotting import plot_matmul_benchmark
from plotting import plot_nn_ops_benchmark
from plotting import plot_speedup
from plotting import plot_transfer_vs_compute
from transfer_bench import run_transfer_benchmark

FIG_DIR = Path(__file__).resolve().parent / "fig"


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    devices = available_devices()
    print(f"Benchmarking on: {[str(d) for d in devices]}")

    print("Matrix multiplication...")
    matmul_results = run_matmul_benchmark(devices)
    plot_matmul_benchmark(
        matmul_results, 
        FIG_DIR / "matmul_benchmark.png"
    )
    plot_speedup(
        matmul_results, 
        FIG_DIR / "matmul_speedup.png"
    )

    print("MLP forward + backward...")
    mlp_results = run_mlp_benchmark(devices)

    print("Self-attention forward...")
    attn_results = run_attention_benchmark(devices)
    plot_nn_ops_benchmark(
        mlp_results, 
        attn_results, 
        FIG_DIR / "nn_ops_benchmark.png"
    )

    print("Host <-> device transfer...")
    gpu_devices = [d for d in devices if d.type != "cpu"]
    transfer_results = run_transfer_benchmark(gpu_devices)
    plot_transfer_vs_compute(
        transfer_results, 
        matmul_results, 
        FIG_DIR / "transfer_vs_compute.png"
    )



if __name__ == "__main__":
    main()
