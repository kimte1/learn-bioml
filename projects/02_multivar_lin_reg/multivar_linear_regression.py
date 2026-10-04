"""
Multivariable linear regression using 2 variables

Also demonstrates why linear regression breaks down if features are highly correlated (foreshadowing Ridge/Lasso, which are implemented in 03_regularized_reg).

Dataset: Delaney (2004) ESOL aqueous solubility dataset (data/delaney-esol.csv).

Target  : measured log solubility in mols per litre
Features: Molecular Weight, Number of Rings
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from plotting import (
    plot_fit_3d,
    plot_loss,
    plot_predicted_vs_actual,
    plot_feat_corr,
    plot_weight_trajectories,
    plot_weight_split,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
FIG_DIR = Path(__file__).resolve().parent / "fig"

DATA_FILE = DATA_DIR / "delaney-esol.csv"
FEATURE_COLS = ["Molecular Weight", "Number of Rings"]
TARGET_COL = "measured log solubility in mols per litre"



def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.1, 
        help="Gradient descent learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of gradient descent epochs"
    )
    parser.add_argument(
        "--limitation-epochs",
        type=int,
        default=3000,
        help="Number of gradient descent epochs for the near-duplicate-feature demo "
        "(needs to be much larger: the near-duplicate pair converges far slower than the rest)",
    )
    return parser.parse_args()



def load_data() -> tuple[np.ndarray, np.ndarray]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found")
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=float)
    return X, y



def standardize_matrix(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std, mean, std



def multivar_lin_reg(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    n_epochs: int,
) -> tuple[float, np.ndarray, list[float], np.ndarray]:
    """
    Fit y = w@X + b by minimizing MSE with full-batch gradient descent.
    """
    n, n_features = X.shape
    b = 0.0
    w = np.zeros(n_features)
    loss_history: list[float] = []
    weight_history = [w.copy()]

    for _ in range(n_epochs):
        y_pred = b + X @ w
        error = y_pred - y

        loss_history.append(float(np.mean(error**2)))

        grad_b = (2.0 / n) * np.sum(error)
        grad_w = (2.0 / n) * (X.T @ error)
        b -= lr * grad_b
        w -= lr * grad_w

        weight_history.append(w.copy())

    return b, w, loss_history, np.array(weight_history)



def unstandardize_weights(
    b_std: float, 
    w_std: np.ndarray,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Convert standardized (intercept, weights) back to raw-feature units.
    """
    w_raw = w_std / x_scale
    b_raw = b_std - np.sum(w_std * x_mean / x_scale)
    return b_raw, w_raw



def add_near_duplicate_feature(
    X: np.ndarray, 
    feature_names: list[str], 
    source_col: str, 
    noise_frac: float = 0.02, 
    seed: int = 0
) -> tuple[np.ndarray, list[str]]:
    """
    Create a new feature column that is a near-duplicate of `source_col` (its values plus a little noise).
    """
    rng = np.random.default_rng(seed)
    j = feature_names.index(source_col)
    duplicate = X[:, j] + rng.normal(0, noise_frac * X[:, j].std(), size=X.shape[0])
    X_aug = np.column_stack([X, duplicate])
    names_aug = feature_names + [f"{source_col} (near-duplicate)"]
    return X_aug, names_aug



def main() -> None:
    args = build_parser()
    FIG_DIR.mkdir(exist_ok=True)

    X, y = load_data()
    X_std, x_mean, x_scale = standardize_matrix(X)

   # 1, initial multivar lin reg
    b_gd, w_gd, loss_history, _ = multivar_lin_reg(
        X_std,
        y,
        args.lr,
        args.epochs
    )

    plot_loss(
        loss_history,
        FIG_DIR / "loss_vs_epoch.png",
    )

    b_raw, w_raw = unstandardize_weights(
        b_gd,
        w_gd,
        x_mean,
        x_scale,
    )

    plot_fit_3d(
        X[:, 0],
        X[:, 1],
        y,
        b_raw,
        w_raw,
        FEATURE_COLS,
        TARGET_COL,
        FIG_DIR / "fit_plane_3d.png",
    )

    y_pred = X_std @ w_gd + b_gd

    plot_predicted_vs_actual(
        y,
        y_pred,
        FIG_DIR / "predicted_vs_actual.png",
    )

    # what happens when a feature is near-duplicate of another?
    X_dup, feature_names_dup = add_near_duplicate_feature(
        X,
        FEATURE_COLS,
        "Molecular Weight",
        noise_frac=0.2
    )

    X_dup_std, _, _ = standardize_matrix(X_dup)

    plot_feat_corr(
        X_dup,
        feature_names_dup,
        FIG_DIR / "near_duplicate_corr.png",
    )

    b_dup, w_dup, loss_history_dup, weight_history_dup = multivar_lin_reg(
        X_dup_std,
        y,
        args.lr,
        args.limitation_epochs,
    )

    predictions_dup = X_dup_std @ w_dup + b_dup

    plot_predicted_vs_actual(
        predictions_dup,
        y,
        FIG_DIR / "predicted_vs_actual_dup.png",
    )

    plot_weight_trajectories(
        weight_history_dup,
        feature_names_dup,
        FIG_DIR / "weight_trajectories.png",
    )
    plot_weight_split(
        weight_history_dup,
        idx_a=0,
        idx_b=2,
        name_a=feature_names_dup[0],
        name_b=feature_names_dup[2],
        path=FIG_DIR / "weight_split.png",
    )



if __name__ == "__main__":
    main()
