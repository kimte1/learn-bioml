"""Visualization functions for the bias-variance tradeoff demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_underfit_vs_overfit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_true: np.ndarray,
    fits: dict[int, np.ndarray],
    path: Path,
) -> None:
    """One training sample, fit with a few different polynomial degrees.

    This is the plot that makes "bias" and "variance" concrete as shapes:
    a low-degree polynomial is too rigid to bend into the true curve (it
    underfits -- high bias), a very high-degree polynomial has so much
    freedom it bends to chase individual noisy points instead of the
    underlying trend (it overfits -- high variance), and something in
    between tracks the true curve without chasing the noise.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(x_train, y_train, color="black", s=20, zorder=5, label="Training data (noisy)")
    plt.plot(x_test, y_true, color="black", linestyle="--", linewidth=2, label="True function (unknown in practice)")

    labels = {
        min(fits): f"degree {min(fits)} (underfit: high bias)",
        max(fits): f"degree {max(fits)} (overfit: high variance)",
    }
    colors = plt.get_cmap("tab10")
    for i, (degree, y_pred) in enumerate(sorted(fits.items())):
        label = labels.get(degree, f"degree {degree} (good fit)")
        plt.plot(x_test, y_pred, color=colors(i), linewidth=2, label=label)

    plt.ylim(-2.0, 2.0)  # clip the y-axis: high-degree fits can swing wildly near the edges
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Same training data, different model complexity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_many_fits(
    x_test: np.ndarray,
    y_true: np.ndarray,
    low_degree_fits: np.ndarray,
    high_degree_fits: np.ndarray,
    low_degree: int,
    high_degree: int,
    path: Path,
) -> None:
    """Refit on many independent noisy training samples, overlay every fit.

    This is "variance" made visible directly: for the low-degree model,
    every fit lands in roughly the same (wrong) place, because the model
    is too simple to react much to which particular sample it saw -- low
    variance, but consistently biased. For the high-degree model, the
    fits fan out all over the place, because the model has enough
    flexibility to chase each sample's specific noise -- high variance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, fits, degree, title in (
        (axes[0], low_degree_fits, low_degree, f"Degree {low_degree}: consistent, but consistently wrong"),
        (axes[1], high_degree_fits, high_degree, f"Degree {high_degree}: wildly different fit per sample"),
    ):
        for y_pred in fits:
            ax.plot(x_test, y_pred, color="steelblue", alpha=0.25, linewidth=1)
        ax.plot(x_test, y_true, color="black", linestyle="--", linewidth=2, label="True function")
        ax.set_xlabel("x")
        ax.set_title(title)
        ax.legend()

    axes[0].set_ylabel("y")
    axes[0].set_ylim(-2.0, 2.0)
    fig.suptitle(f"Variance = how much the fit changes across {len(low_degree_fits)} different training samples")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bias_variance_curve(
    degrees: np.ndarray,
    bias_sq: np.ndarray,
    variance: np.ndarray,
    noise_floor: float,
    total_error: np.ndarray,
    path: Path,
) -> None:
    """The textbook bias-variance tradeoff curve: error vs. model complexity.

    Bias^2 falls as the model is allowed to be more flexible (it can
    finally bend to match the true curve). Variance rises as the model
    gets more flexible (it has more freedom to chase noise). Total
    expected error is their sum plus the irreducible noise floor -- it's
    U-shaped, with the minimum marking the best bias-variance tradeoff,
    not the most "accurate-looking" model on the training set.
    """
    best_idx = int(np.argmin(total_error))

    plt.figure(figsize=(8, 6))
    plt.plot(degrees, bias_sq, marker="o", color="steelblue", label="Bias$^2$")
    plt.plot(degrees, variance, marker="o", color="crimson", label="Variance")
    plt.axhline(noise_floor, color="gray", linestyle=":", label="Irreducible noise")
    plt.plot(degrees, total_error, marker="o", color="black", linewidth=2, label="Total expected error")
    plt.axvline(
        degrees[best_idx],
        color="darkgreen",
        linestyle="--",
        linewidth=1,
        label=f"Best degree = {degrees[best_idx]}",
    )

    # Log scale: as degree approaches the training-set size, the polynomial's
    # design matrix becomes ill-conditioned (the same p-approaches-n
    # instability as 02_multivar_lin_reg's limitation 2) and variance
    # explodes by orders of magnitude. A linear y-axis would squash the
    # interesting low-degree U-shape down to a flat line at that scale.
    plt.yscale("log")
    plt.xlabel("Polynomial degree (model complexity)")
    plt.ylabel("Error (log scale)")
    plt.title("Bias-variance tradeoff vs. model complexity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
