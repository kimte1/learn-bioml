"""
Ridge and Lasso regression from scratch, cross-checked against scikit-learn.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge

from plotting import (
    plot_coef_comparison,
    plot_coefficient_stability,
    plot_cv_curve,
    plot_loss_history,
    plot_regularization_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
FIG_DIR = Path(__file__).resolve().parent / "fig"

DATA_FILE = DATA_DIR / "delaney-esol.csv"
FEATURE_COLS = [
    "Minimum Degree",
    "Molecular Weight",
    "Number of H-Bond Donors",
    "Number of Rings",
    "Number of Rotatable Bonds",
    "Polar Surface Area",
]
TARGET_COL = "measured log solubility in mols per litre"



def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ridge-alpha", 
        type=float,
        default=10.0,
        help="Ridge alpha for the headline coefficient comparison"
    )
    parser.add_argument(
        "--lasso-alpha",
        type=float,
        default=0.1,
        help="Lasso alpha for the headline coefficient comparison"
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation"
    )
    parser.add_argument(
        "--n-alphas",
        type=int,
        default=30,
        help="Number of alpha values to sweep for the regularization path / CV curve"
    )
    return parser.parse_args()



def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found")
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=float)
    return X, y, FEATURE_COLS



def standardize_matrix(X: np.ndarray) -> tuple:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std, mean, std



def soft_threshold(rho: float, alpha: float) -> float:
    """
    Proximal operator for the L1 penalty: 
        - shrinks rho toward zero by alpha,
        - clipping to exactly zero if |rho| <= alpha.
    """
    if rho > alpha:
        return rho - alpha
    if rho < -alpha:
        return rho + alpha
    return 0.0



def lasso(
    X: np.ndarray, 
    y: np.ndarray, 
    alpha: float, 
    epochs: int = 1000, 
    tol: float = 1e-7
) -> tuple[float, np.ndarray, list[float]]:
    """
    minimize (1 / (2n)) * ||y - (wx + b||^2 + alpha * ||w||_1
    """
    n, n_features = X.shape
    x_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - x_mean
    yc = y - y_mean

    w = np.zeros(n_features) # init w
    col_sq = (Xc**2).sum(axis=0) / n  # normalization factor for each coordinate update
    loss_history: list[float] = []

    for _ in range(epochs):
        w_prev = w.copy()
        for j in range(n_features):
            residual = yc - Xc @ w + Xc[:, j] * w[j]  # residual excluding feature j's contribution
            rho_j = (Xc[:, j] @ residual) / n
            w[j] = soft_threshold(rho_j, alpha) / col_sq[j] if col_sq[j] > 0 else 0.0

        # loss 
        obj = (1.0 / (2 * n)) * np.sum((yc - Xc @ w) ** 2) + alpha * np.sum(np.abs(w))
        loss_history.append(float(obj))
        if np.max(np.abs(w - w_prev)) < tol:
            break

    b = y_mean - x_mean @ w
    return b, w, loss_history



def ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    lr: float = 0.1,
    epochs: int = 1000,
    tol: float = 1e-7,
) -> tuple[float, np.ndarray, list[float]]:
    """
    minimize w and b: ||y - (wX + b)||^2 + alpha * ||w||^2
    """
    n, n_features = X.shape
    x_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - x_mean
    yc = y - y_mean

    w = np.zeros(n_features)
    loss_history: list[float] = []

    for _ in range(epochs):
        w_prev = w.copy()

        error = Xc @ w - yc
        grad_w = (Xc.T @ error + alpha * w) / n # the 1/n is a scaling factor for stabilization
        w -= lr * grad_w

        error = Xc @ w - yc  # recomputed post-update, so the logged loss matches the new w
        obj = np.sum(error**2) + alpha * np.sum(w**2)
        loss_history.append(float(obj))

        if np.max(np.abs(w - w_prev)) < tol:
            break

    b = y_mean - x_mean @ w
    return b, w, loss_history



def ols_closed_form(X: np.ndarray, y: np.ndarray, alpha: float = 0.0) -> tuple[float, np.ndarray]:
    """
    Exact, unregularized OLS solution via SVD-based least squares.

    Used only as the alpha=0 baseline for bootstrap_coefficient_stability.
    ridge(alpha=0.0) would instead give gradient descent stopped after a
    fixed epoch budget, which acts as an implicit regularizer and badly
    understates real OLS's instability under multicollinearity -- GD
    converges very slowly along a near-duplicate feature pair's shared
    direction (see weight_split.png in 02_multivar_lin_reg), so 1000
    epochs lands far short of the true, highly unstable closed-form
    solution. `alpha` is accepted and ignored, only so this matches the
    (X, y, alpha) signature bootstrap_coefficient_stability expects.
    """
    x_mean = X.mean(axis=0)
    y_mean = y.mean()
    w, *_ = np.linalg.lstsq(X - x_mean, y - y_mean, rcond=None)
    b = y_mean - x_mean @ w
    return b, w



def add_near_duplicate_feature(
    X: np.ndarray,
    feature_names: list[str],
    source_col: str,
    noise_frac: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """
    Append a near-duplicate of `source_col` (its values plus a little noise).

    The real feature set's multicollinearity is too mild to visibly
    destabilize OLS, so this engineers a clearly redundant column (e.g.
    Molecular Weight plus 2% noise, r~0.9998 with the original) to make
    the coefficient-instability effect in bootstrap_coefficient_stability
    show up clearly.
    """
    rng = np.random.default_rng(seed)
    j = feature_names.index(source_col)
    duplicate = X[:, j] + rng.normal(0, noise_frac * X[:, j].std(), size=X.shape[0])
    X_aug = np.column_stack([X, duplicate])
    names_aug = feature_names + [f"{source_col} (near-duplicate)"]
    return X_aug, names_aug



def bootstrap_coefficient_stability(
    X: np.ndarray, 
    y: np.ndarray, 
    fit_fn, 
    alpha: float, 
    n_bootstraps: int = 200, 
    seed: int = 0,
) -> np.ndarray:
    """
    - `n_bootstraps` resamples (with replacement) 
    - use lasso or ridge to fit the bootstrapped data
    - return: the resulting weights, with shape (n_bootstraps, n_features).
    """
    rng = np.random.default_rng(seed)
    n, n_features = X.shape

    coefs = np.zeros((n_bootstraps, n_features))

    for i in range(n_bootstraps):
        boot_idx = rng.integers(0, n, size=n)
        X_boot, y_boot = X[boot_idx], y[boot_idx]
        X_boot_std, _, _ = standardize_matrix(X_boot)
        result = fit_fn(X_boot_std, y_boot, alpha)
        coefs[i] = result[1]
    return coefs



def make_folds(
    n: int, 
    k: int, 
    seed: int = 0
) -> list[np.ndarray]:

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    return np.array_split(indices, k)



def cross_validate(
    X: np.ndarray, 
    y: np.ndarray, 
    alphas: np.ndarray, 
    fit_fn, 
    k: int = 5, 
    seed: int = 0,
) -> np.ndarray:
    """
    Mean k-fold validation MSE for each alpha
    """
    n = X.shape[0]
    folds = make_folds(n, k, seed)
    mean_mse = np.zeros(len(alphas))

    for a_idx, alpha in enumerate(alphas):
        fold_mses = []
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            X_train_std, mean, std = standardize_matrix(X_train)
            X_val_std = (X_val - mean) / std

            result = fit_fn(X_train_std, y_train, alpha)
            b, w = result[0], result[1]

            y_pred = b + X_val_std @ w
            fold_mses.append(np.mean((y_val - y_pred) ** 2))

        mean_mse[a_idx] = np.mean(fold_mses)

    return mean_mse

    
    
def main() -> None:
    args = build_parser()
    FIG_DIR.mkdir(exist_ok=True)

    X, y, feature_names = load_data()
    X_std, x_mean, x_scale = standardize_matrix(X)

    # lasso regression at a particular alpha
    lasso_b, lasso_w, lasso_loss_history = lasso(X_std, y, args.lasso_alpha)
    sk_lasso = Lasso(
        alpha=args.lasso_alpha, 
        fit_intercept=True, 
        max_iter=10000
    ).fit(X_std, y)
    
    plot_coef_comparison(
        feature_names, 
        lasso_w, 
        sk_lasso.coef_, 
        "Lasso", 
        FIG_DIR / "lasso_coef_comparison.png"
    )

    plot_loss_history(lasso_loss_history, FIG_DIR / "lasso_loss_history.png")
    
    # ridge 
    ridge_b, ridge_w, ridge_loss_history = ridge(X_std, y, args.ridge_alpha)
    sk_ridge = Ridge(
        alpha=args.ridge_alpha, 
        fit_intercept=True
    ).fit(X_std, y)

    plot_coef_comparison(
        feature_names,
        ridge_w,
        sk_ridge.coef_,
        "Ridge",
        FIG_DIR / "ridge_coef_comparison.png"
    )

    plot_loss_history(ridge_loss_history, FIG_DIR / "ridge_loss_history.png")


    # test different alphas
    # the range of values differs bc Ridge 
    alphas_lasso = np.logspace(-3, 1, args.n_alphas)
    alphas_ridge = np.logspace(-2, 4, args.n_alphas)
    
    coefs_lasso = np.array([lasso(X_std, y, a)[1] for a in alphas_lasso])
    coefs_ridge = np.array([ridge(X_std, y, a)[1] for a in alphas_ridge])

    plot_regularization_path(
        alphas_ridge, 
        coefs_ridge, 
        alphas_lasso, 
        coefs_lasso, 
        feature_names, 
        FIG_DIR / "regularization_path.png"
    )

    # find the optimal alpha 
    cv_mse_lasso = cross_validate(X, y, alphas_lasso, lasso, k=args.k_folds)
    cv_mse_ridge = cross_validate(X, y, alphas_ridge, ridge, k=args.k_folds)
    
    best_alpha_ridge = alphas_ridge[np.argmin(cv_mse_ridge)]
    best_alpha_lasso = alphas_lasso[np.argmin(cv_mse_lasso)]

    plot_cv_curve(
        alphas_ridge, cv_mse_ridge, best_alpha_ridge,
        alphas_lasso, cv_mse_lasso, best_alpha_lasso,
        FIG_DIR / "cv_curve.png",
    )


    # how lasso and ridge addresses regular linear regression  limitations
    X_dup, feature_names_dup = add_near_duplicate_feature(
        X,
        feature_names,
        "Molecular Weight",
        noise_frac=0.02,
    )

    n_bootstraps = 200
    ols_coefs = bootstrap_coefficient_stability(
        X_dup,
        y,
        ols_closed_form,
        alpha=0.0,
        n_bootstraps=n_bootstraps
)
    ridge_coefs = bootstrap_coefficient_stability(
        X_dup,
        y,
        ridge,
        alpha=best_alpha_ridge,
        n_bootstraps=n_bootstraps
    )
    lasso_coefs = bootstrap_coefficient_stability(
        X_dup, 
        y, 
        lasso, 
        alpha=best_alpha_lasso, 
        n_bootstraps=n_bootstraps
    )

    plot_coefficient_stability(
        feature_names_dup, 
        ols_coefs, 
        ridge_coefs, 
        lasso_coefs, 
        best_alpha_ridge, 
        best_alpha_lasso,
        FIG_DIR / "coefficient_stability.png",
    )



if __name__ == "__main__":
    main()
