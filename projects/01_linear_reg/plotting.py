
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

DIVERGED_CAP = 1.0e4



def plot_loss(
    loss_history: list[float],
    path: Path,
) -> None:

    plt.figure(figsize=(6, 4))
    plt.plot(loss_history)

    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (standardized x)")
    plt.title("Batch gradient descent convergence")

    # plt.yscale("log")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()



def plot_fit(
    x: np.ndarray,
    y: np.ndarray,
    gd_intercept: float,
    gd_slope: float,
    path: Path,
) -> None:

    plt.figure(figsize=(6, 5))

    plt.scatter(
        x,
        y,
        s=12,
        alpha=0.5,
        label=f"Proteins (n={x.size})")

    # predicted fit
    x_line = np.linspace(x.min(), x.max(), 100)
    plt.plot(
        x_line,
        gd_intercept + gd_slope * x_line,
        color="crimson",
        label="Gradient descent fit",
    )

    plt.xlabel("Molecular Weight")
    plt.ylabel("measured log solubility (mol/L")

    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()



def plot_fit_evolution(
    x: np.ndarray,
    y: np.ndarray,
    snapshots: list[tuple[int, float, float]],
    x_mean: float,
    x_scale: float,
    path: Path,
) -> None:
    """Scatter plot with the best-fit line overlaid at several epochs, colored by epoch."""
    plt.figure(figsize=(6, 5))

    # raw data
    plt.scatter(
        x,
        y,
        s=12,
        alpha=0.4,
        color="gray",
        label=f"Proteins (n={x.size})",
        zorder=1
    )

    x_line = np.linspace(x.min(), x.max(), 100)
    epochs = [epoch for epoch, _, _ in snapshots]
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(epochs), vmax=max(epochs))

    for epoch, intercept_std, slope_std in snapshots:
        # convert standardized-x weights back to raw-x scale, as in main()
        slope = slope_std / x_scale
        intercept = intercept_std - slope_std * x_mean / x_scale
        plt.plot(
            x_line,
            intercept + slope * x_line,
            color=cmap(norm(epoch)),
            zorder=2
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=plt.gca(), label="Epoch")
    plt.xlabel("log2 LFQ intensity, NoDOX replicate 1 (uninduced control)")
    plt.ylabel("log2 LFQ intensity, RNF4-WT replicate 1 (induced)")
    plt.title("Best-fit line vs. training epoch")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()




def plot_lr_sweep(
    learning_rates: np.ndarray,
    final_losses: np.ndarray,
    path: Path) -> None:

    # diverged = final_losses >= DIVERGED_CAP

    plt.figure(figsize=(6.5, 5))
    plt.plot(
        learning_rates,
        final_losses,
        "o-",
        color="steelblue",
        markersize=4,
    )
    # if diverged.any():
    #     plt.scatter(
    #         learning_rates[diverged],
    #         final_losses[diverged],
    #         color="crimson",
    #         marker="x",
    #         s=50,
    #         label="Diverged (capped for display)",
    #     )
    # plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("MSE loss at final epoch")
    plt.title("Effect of learning rate on final training loss")
    # plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
