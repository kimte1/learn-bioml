
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import seaborn as sns


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

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()



def plot_weight_evolution(
    snapshots: list[tuple[int, float, float]],
    path: Path,
) -> None:

    plt.figure(figsize=(6, 4))

    for epoch, intercept, slope in snapshots:
        sns.scatterplot(
            x=[epoch], 
            y=[slope],
            color="blue",
            alpha=0.5)
        sns.scatterplot(
            x=[epoch],
            y=[intercept],
            color="crimson",
            alpha=0.5)

    plt.xlabel("Epoch")
    plt.ylabel("Weight or Slope value")
    plt.legend(["slope", "intercept"])
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




def plot_standardization(
    x_raw: np.ndarray,
    x_std: np.ndarray,
    feature_name: str,
    path: Path,
) -> None:
    """Side-by-side histograms of a feature before and after z-score standardization.

    standardize() only re-centers (subtracts the mean) and re-scales (divides
    by the std) the data -- it does not change the shape of the distribution.
    This plot makes that visible: same histogram shape in both panels, just
    shifted so the mean lands on 0 and the +/-1 std lines land on +/-1.
    """
    mean_raw, std_raw = x_raw.mean(), x_raw.std()
    mean_z, std_z = x_std.mean(), x_std.std()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(x_raw, bins=30, color="steelblue", edgecolor="white")
    axes[0].axvline(mean_raw, color="crimson", linewidth=1.5, label=f"mean={mean_raw:.2f}")
    axes[0].axvline(mean_raw - std_raw, color="crimson", linestyle="--", linewidth=1, label=f"mean ± 1 std (std={std_raw:.2f})")
    axes[0].axvline(mean_raw + std_raw, color="crimson", linestyle="--", linewidth=1)
    axes[0].set_title(f"Raw {feature_name}")
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)

    axes[1].hist(x_std, bins=30, color="steelblue", edgecolor="white")
    axes[1].axvline(mean_z, color="crimson", linewidth=1.5, label=f"mean={mean_z:.2f}")
    axes[1].axvline(mean_z - std_z, color="crimson", linestyle="--", linewidth=1, label=f"mean ± 1 std (std={std_z:.2f})")
    axes[1].axvline(mean_z + std_z, color="crimson", linestyle="--", linewidth=1)
    axes[1].set_title(f"Standardized {feature_name}")
    axes[1].set_xlabel(f"({feature_name} - mean) / std")
    axes[1].legend(fontsize=8)

    fig.suptitle("What standardize() does: re-center to mean 0, re-scale to std 1 (same shape)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
