from __future__ import annotations

import time
from typing import Callable

import torch


def available_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def synchronize(device: torch.device) -> None:
    """GPU work is dispatched ASYNCHRONOUSLY -- the CPU hands off a
    kernel and moves on without waiting for it to finish. Without this,
    a wall-clock timer around a GPU op would measure "how long it took
    to launch the kernel", not "how long the kernel took to actually
    run". CPU ops have no such gap, so there's nothing to synchronize.
    """
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def median_time(
    fn: Callable[[], None],
    device: torch.device,
    warmup_iters: int = 3,
    timed_iters: int = 10,
) -> float:
    """
    Median wall-clock seconds for one call to `fn`. 

    Warmup iterations absorb one-time costs (CUDA/MPS context setup, kernel compilation) that have nothing to do with the operation's steady-state speed.

    Median (not mean) of several timed repeats is robust to the
    occasional slow iteration from OS scheduling noise.
    """
    for _ in range(warmup_iters):
        fn()
    synchronize(device)

    times = []
    for _ in range(timed_iters):
        start = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - start)

    return sorted(times)[len(times) // 2]
