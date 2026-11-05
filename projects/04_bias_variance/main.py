"""
Bias-variance tradeoff, demonstrated with polynomial regression on synthetic sinusoidal data:
   -  y = sin(1.5*pi*x) + noise

Here, model complexity (the degree of a fitted polynomial) plays the same role alpha (reg strength) played in lasso/ridge:
     - changing alpha will change how bias and variance respond, in opposite directions.


  - Bias^2    = (truth(x) - E[prediction(x)])^2
                how far off the model's AVERAGE prediction is from the truth
                -- error from the model being too simple/rigid to represent
                the real relationship, no matter how much data it sees.
  - Variance  = E[(prediction(x) - E[prediction(x)])^2]
                how much the prediction at x swings around depending on
                which particular noisy training set was used to fit it --
                error from the model being too sensitive to sampling noise.
  
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from plotting import (
    plot_underfit_vs_overfit, 
    plot_many_fits,
    plot_bias_variance_curve
)


FIG_DIR = Path(__file__).resolve().parent / "fig"


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of (x, y) points in each simulated training set",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.3,
        help="Standard deviation of the irreducible noise added to the true function",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=300,
        help="Number of independent training sets drawn per polynomial degree, "
        "when estimating bias^2 and variance",
    )
    return parser.parse_args()



def true_function(x: np.ndarray) -> np.ndarray:
    return np.sin(1.5 * np.pi * x)



def sample_data(
    n_samples: int,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make a noisy training set 
    
    y = true_function(x) + irreducible noise. 
    """

    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=n_samples)
    y = true_function(x) + rng.normal(0, noise_std, size=n_samples)

    return x, y



def polyfit(
    x: np.ndarray, 
    y: np.ndarray, 
    degree: int
) -> np.ndarray:
    """
    Fit y = w0 + w1*x + w2*x^2 + ... + w_degree * x^degree by closed-form least squares on a Vandermonde design matrix. 
    """
    X = np.vander(
        x, 
        N=degree + 1, 
        increasing=True # columns increase from left to right
    )
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return w



def poly_predict(
    w: np.ndarray, 
    x: np.ndarray
) -> np.ndarray:

    X = np.vander(
        x, 
        N=len(w), 
        increasing=True
    )
    return X @ w




def bias_variance_per_model_complexity(
    degree: int,
    n_repeats: int,
    n_samples: int,
    noise_std: float,
    x_test: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    """
    Estimate Bias^2 and Variance for one polynomial degree, by simulating
    "many different training sets" directly
    """
    rng = np.random.default_rng(seed)
    predictions = np.zeros((n_repeats, len(x_test)))

    for i in range(n_repeats):
        x_train, y_train = sample_data(
            n_samples, 
            noise_std, 
            seed=rng.integers(0, 1_000_000)
        )
        w = polyfit(x_train, y_train, degree)
        predictions[i] = poly_predict(w, x_test)

    avg_pred = predictions.mean(axis=0)
    bias_sq = np.mean((avg_pred - true_function(x_test)) ** 2)
    variance = np.mean(predictions.var(axis=0))

    return bias_sq, variance
#

def main():
    
    args = build_parser()
    FIG_DIR.mkdir(exist_ok=True)

    x_test = np.linspace(0, 1, 200)
    y_true = true_function(x_test)

    x_train, y_train = sample_data(
        args.n_samples, 
        args.noise_std, 
        seed=0
    )
    
    # Plot 1: true function and several functions of various degrees
    # to show how different degrees relate to the true function
    # degree 1 is underfit (high bias, underfit)
    # higher degrees will bend so much it tries to fit noisy points (high variance, overfit)
    
    degrees_to_show = [1, 2, 5, 10, 25]

    fits = {}
    for d in degrees_to_show:
        w = polyfit(x_train, y_train, d)
        y = poly_predict(w, x_test)
        fits[d] = y

    plot_underfit_vs_overfit(
        x_train,
        y_train,
        x_test,
        y_true,
        fits,
        FIG_DIR / "01_underfit_vs_overfit.png",
    )

    # Plot 2: Show variance 
    # Refit the SAME degree on different train sets

    low_degree = 1
    high_degree = 25

    low_d_fits = []
    for i in range(30):
        x_train, y_train = sample_data(
            args.n_samples, 
            args.noise_std, 
            seed=i,
        )
        w = polyfit(x_train, y_train, low_degree)
        y = poly_predict(w, x_test)
        low_d_fits.append(y)
    
    high_d_fits = []
    for i in range(30):
        x_train, y_train = sample_data(
            args.n_samples, 
            args.noise_std, 
            seed=i,
        )
        w = polyfit(x_train, y_train, high_degree)
        y = poly_predict(w, x_test)
        high_d_fits.append(y)


    plot_many_fits(
        x_test,
        y_true,
        low_d_fits,
        high_d_fits,
        low_degree,
        high_degree,
        FIG_DIR / "02_many_fits_per_degree.png",
    )

    # Plot 3: train multiple models (different degrees)
    # plot Error vs model's degree (hue is bias or variance )
    degrees = np.arange(1, 25 + 1)
    bias_sq = np.zeros(len(degrees))
    variance = np.zeros(len(degrees))

    for i, degree in enumerate(degrees):
        bias_sq[i], variance[i] = bias_variance_per_model_complexity(
            degree,
            args.n_repeats,
            args.n_samples,
            args.noise_std,
            x_test,
            seed=degree,
        )

    noise_floor = args.noise_std**2 # square stdev to get variance
    total_error = bias_sq + variance + noise_floor
    best_degree = degrees[np.argmin(total_error)]

    plot_bias_variance_curve(
        degrees,
        bias_sq,
        variance,
        noise_floor,
        total_error,
        FIG_DIR / "03_bias_variance_vs_complexity.png",
    )



if __name__ == "__main__":
    main()
