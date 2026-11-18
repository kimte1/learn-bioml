"""Plot best-val-loss-so-far across a sweep's trials, ordered by start
time -- shows whether the search has plateaued (more trials aren't
finding better configs) or is still improving. A diagnostic for deciding
how many `wandb agent --count N` trials are "enough," not part of the
training pipeline itself.

Usage:
    python plot_sweep_progress.py tekim/learn-bioml/<sweep_id>
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python plot_sweep_progress.py <entity>/<project>/<sweep_id>")
    sweep_path = sys.argv[1]

    api = wandb.Api()
    sweep = api.sweep(sweep_path)
    runs = sorted(sweep.runs, key=lambda r: r.created_at)

    # best_val_loss is the per-trial summary field main.py sets explicitly
    # (the genuine best val loss that run saw), not the sweep's
    # optimization target (val_loss), which is noisier (last-epoch value,
    # chosen instead so Hyperband has a per-step signal to prune on --
    # see the comment in sweep.yaml).
    trial_losses = [r.summary.get("best_val_loss") for r in runs]
    trial_losses = [v for v in trial_losses if v is not None]  # drop crashed/incomplete trials

    if not trial_losses:
        raise SystemExit(f"No completed trials with a best_val_loss found in sweep {sweep_path}")

    running_best: list[float] = []
    current_best = float("inf")
    for v in trial_losses:
        current_best = min(current_best, v)
        running_best.append(current_best)

    trial_numbers = range(1, len(trial_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.scatter(trial_numbers, trial_losses, alpha=0.5, label="this trial's best_val_loss")
    plt.plot(trial_numbers, running_best, color="firebrick", label="best so far")
    plt.yscale("log")
    plt.xlabel("Trial number")
    plt.ylabel("Validation MSE (log scale)")
    plt.legend()
    plt.title(f"Sweep progress: {sweep_path}")
    plt.tight_layout()

    out_path = Path(__file__).resolve().parent / "sweep_progress.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path} ({len(trial_losses)} trials, best so far = {running_best[-1]:.4f})")


if __name__ == "__main__":
    main()
