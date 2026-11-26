"""
Visualizations for the CPU/MPS benchmark suite. Log-log axes throughout
-- these benchmarks span orders of magnitude in both size and time, and
a linear axis would crush the small-size end into invisibility.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

DEVICE_COLORS = {"cpu": "tab:orange", "mps": "tab:blue", "cuda": "tab:green"}


def _color(device: torch.device) -> str:
    return DEVICE_COLORS.get(device.type, "tab:gray")


def plot_matmul_benchmark(results: dict[torch.device, dict[str, list[float]]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for device, data in results.items():
        axes[0].plot(data["sizes"], data["times"], marker="o", label=str(device), color=_color(device))
        axes[1].plot(data["sizes"], data["gflops"], marker="o", label=str(device), color=_color(device))

    for ax, ylabel, title in [
        (axes[0], "Time (s, median of 10)", "Matmul wall-clock time"),
        (axes[1], "Throughput (GFLOPS)", "Matmul throughput"),
    ]:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Matrix size (n, for an n x n @ n x n matmul)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Matrix multiplication: CPU vs. GPU")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_speedup(results: dict[torch.device, dict[str, list[float]]], path: Path) -> None:
    """Speedup relative to CPU, matmul benchmark -- the actual answer to
    "when does a GPU help": below the break-even line, kernel-launch and
    dispatch overhead dominate and the GPU is actually slower; above it,
    there's enough independent work to make the GPU's extra parallelism
    pay off.
    """
    cpu_times = results[torch.device("cpu")]["times"]
    sizes = results[torch.device("cpu")]["sizes"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(1.0, color="gray", linestyle="--", label="break-even with CPU")
    for device, data in results.items():
        if device.type == "cpu":
            continue
        speedup = [cpu_t / gpu_t for cpu_t, gpu_t in zip(cpu_times, data["times"])]
        ax.plot(sizes, speedup, marker="o", label=f"{device} vs. cpu", color=_color(device))

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size (n, for an n x n @ n x n matmul)")
    ax.set_ylabel("Speedup over CPU (CPU time / device time)")
    ax.set_title("GPU only wins once there's enough work to parallelize")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_nn_ops_benchmark(
    mlp_results: dict[torch.device, dict[str, list[float]]],
    attn_results: dict[torch.device, dict[str, list[float]]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for device, data in mlp_results.items():
        axes[0].plot(data["batch_sizes"], data["times"], marker="o", label=str(device), color=_color(device))
    axes[0].set_xlabel("Batch size")
    axes[0].set_ylabel("Time (s, median of 10)")
    axes[0].set_title("MLP forward + backward")

    for device, data in attn_results.items():
        axes[1].plot(data["seq_lengths"], data["times"], marker="o", label=str(device), color=_color(device))
    axes[1].set_xlabel("Sequence length")
    axes[1].set_ylabel("Time (s, median of 10)")
    axes[1].set_title("Self-attention forward (O(L^2))")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend()

    fig.suptitle("Representative neural-network operations: CPU vs. GPU")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_transfer_vs_compute(
    transfer_results: dict[torch.device, dict[str, list[float]]],
    matmul_results: dict[torch.device, dict[str, list[float]]],
    path: Path,
) -> None:
    """Host->device transfer time and on-device matmul compute time, at
    the same sizes, on the same (seconds) axis -- directly comparable,
    unlike time-vs-GFLOPS above. Where the transfer line sits above the
    compute line, moving the data to the GPU costs more than the GPU
    saves you on that one matmul.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for device, data in transfer_results.items():
        ax.plot(
            data["sizes"],
            data["times"],
            marker="o",
            linestyle="--",
            label=f"{device}: host -> device transfer",
            color=_color(device),
        )
        compute = matmul_results[device]
        ax.plot(
            compute["sizes"],
            compute["times"],
            marker="s",
            label=f"{device}: matmul compute",
            color=_color(device),
            alpha=0.6,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size (n, for an n x n tensor)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Host<->device transfer vs. on-device compute")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)
