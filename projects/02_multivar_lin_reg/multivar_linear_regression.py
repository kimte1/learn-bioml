"""
Multivariable linear regression using 2 variables (for visualization)

Demonstrates 4 reasons gradient descent breaks down as the feature set grows
  1. Multicollinearity -> undetermined weight split
  2. More features than samples (e.g. p > n) -> X^T X singular, no unique solution
  3. Overfitting / high variance -- no complexity penalty -> train error down, test error up
  4. No feature selection -> irrelevant/noise features still get nonzero weight

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
    plot_null_space_solutions,
    plot_overfitting_curve,
    plot_noise_feature_weights,
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
        default=1000,
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

    # X_aug = [
    #     molecular weight,
    #     num_rings,
    #     molecular_weight_dup,
    # ]
    return X_aug, names_aug



def add_random_noise_features(
    X: np.ndarray,
    feature_names: list[str],
    n_noise: int,
    seed: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """
    Add `n_noise` columns of pure random Gaussian noise -- unrelated to
    y or to any real feature -- to X.

    These columns carry no real signal. 
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(X.shape[0], n_noise))
    X_aug = np.column_stack([X, noise])
    names_aug = feature_names + [f"noise_{i}" for i in range(n_noise)]
    return X_aug, names_aug



def ols_closed_form(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Exact OLS solution via SVD-based least squares (np.linalg.lstsq).

    Used instead of multivar_lin_reg's gradient descent for the demos
    below: as p grows, gradient descent needs far more epochs to fully
    converge, which would conflate "hasn't converged yet" with "this is
    what OLS's overfitting / no-selection limitations actually look like."
    lstsq also still returns an answer (the minimum-norm one) even when
    X^T X is singular, which is itself the point of limitation 2.
    """
    x_mean = X.mean(axis=0)
    y_mean = y.mean()
    w, *_ = np.linalg.lstsq(X - x_mean, y - y_mean, rcond=None)
    b = y_mean - x_mean @ w
    return b, w



def demo_p_greater_than_n(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_samples: int,
    n_noise: int,
    seed: int = 0,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Subsample down to just `n_samples` rows and pad with `n_noise` random columns so there are more features (p) than data points (n)
    """
    rng = np.random.default_rng(seed)

    # pick n_samples random rows from X
    idx = rng.choice(
        X.shape[0], 
        size=n_samples, 
        replace=False
    )
    # take n_samples rows from X and add n_noise random columns of noise
    X_small, names_small = add_random_noise_features(
        X[idx],
        feature_names,
        n_noise,
        seed=seed,
    )
    # get the corresponding y values
    y_small = y[idx]
    # standardize the features
    X_small_std, _, _ = standardize_matrix(X_small)

    # get n samples and p features
    n, p = X_small_std.shape
    
    # compute a "rank" (how many independent columns)
    rank = np.linalg.matrix_rank(X_small_std)

    # solve for one set of weights that fit the data
    # here lstsq returns the smallest magnitude solution
    w1, *_ = np.linalg.lstsq(
        X_small_std,
        y_small,
        rcond=None,
    )

    # use SVD to get Vt, which contains the directions in feature space
    _, _, Vt = np.linalg.svd(
        X_small_std,
        full_matrices=True,
    )

    # grab one specific direction from Vt (e.g., the first null-space direction)
    v = Vt[n]  
    # rescale to w1's size
    v = v / np.linalg.norm(v) * np.linalg.norm(w1)  # 
    # build second weight vector by shifting w1 along that zero-effect direction
    w2 = w1 + v

    pred1 = X_small_std @ w1
    pred2 = X_small_std @ w2

    return names_small, w1, w2



def demo_overfitting_vs_p(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    max_noise: int,
    step: int,
    train_frac: float = 0.7,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Limitation 3: overfitting / high variance as useless features pile up.

    OLS minimizes training error with no penalty on coefficient size or
    model complexity, so adding more features -- even pure noise -- can
    only ever help it fit the training set better. Track train vs. held-out
    MSE as random noise columns are added: train MSE keeps dropping while
    test MSE eventually turns back up once the model starts fitting
    training-set noise instead of the real relationship.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    noise_counts = np.arange(0, max_noise + 1, step)
    train_mse = np.zeros(len(noise_counts))
    test_mse = np.zeros(len(noise_counts))

    for i, n_noise in enumerate(noise_counts):
        X_aug, _ = add_random_noise_features(X, feature_names, int(n_noise), seed=seed)
        X_train, X_test = X_aug[train_idx], X_aug[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize using train-set statistics only, applied to both splits --
        # fitting on all the data (including test) would leak information.
        X_train_std, mean, scale = standardize_matrix(X_train)
        X_test_std = (X_test - mean) / scale

        b, w = ols_closed_form(X_train_std, y_train)

        train_pred = b + X_train_std @ w
        test_pred = b + X_test_std @ w
        train_mse[i] = np.mean((y_train - train_pred) ** 2)
        test_mse[i] = np.mean((y_test - test_pred) ** 2)

    print(f"\nLimitation 3 (overfitting): train MSE {train_mse[0]:.4f} -> {train_mse[-1]:.4f} "
          f"as {max_noise} noise features are added, test MSE {test_mse[0]:.4f} -> {test_mse[-1]:.4f}")

    return noise_counts, train_mse, test_mse, n_train



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

    # Limitation 1: multicollinearity.
    # duplicate 'molecular weight' feature (but with noise)
    X_dup, feature_names_dup = add_near_duplicate_feature(
        X,
        FEATURE_COLS,
        "Molecular Weight",
        noise_frac=0.2
    )

    # show noise in 'molecular weight' dup feat
    plot_feat_corr(
        X_dup,
        feature_names_dup,
        FIG_DIR / "near_duplicate_corr.png",
    )
    
    X_dup_std, _, _ = standardize_matrix(X_dup)

    
    b_dup, w_dup, loss_history_dup, weight_history_dup = multivar_lin_reg(
        X_dup_std,
        y,
        args.lr,
        args.limitation_epochs,
    )

    plot_loss(
        loss_history_dup,
        FIG_DIR / "loss_history_dup.png",
    )

    y_pred_dup = X_dup_std @ w_dup + b_dup
    plot_predicted_vs_actual(
        y,
        y_pred_dup,
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

    # Limitation 2: (features > n) -- p > n.
    
    # Shrink down to a handful of samples and pad with random features until there are more features (p) than data points (n). 
    
    names_small, w1, w2 = demo_p_greater_than_n(
        X,
        y,
        FEATURE_COLS,
        n_samples=10,
        n_noise=20,
    )
    plot_null_space_solutions(
        names_small,
        w1,
        w2,
        FIG_DIR / "p_greater_than_n_solutions.png",
    )

 
    # Limitation 3: overfitting / high variance.
    # as more features are added, model keeps fitting the training set better,
    
    noise_counts, train_mse, test_mse, n_train = demo_overfitting_vs_p(
        X,
        y,
        FEATURE_COLS,
        max_noise=150,
        step=5,
    )
    
    plot_overfitting_curve(
        noise_counts,
        train_mse,
        test_mse,
        n_train,
        FIG_DIR / "overfitting_vs_num_features.png",
    )


    # Limitation 4: no built-in feature selection.
    # Fit a model with a batch of pure-noise features mixed in and look at the resulting weights directly: 
    # OLS has no mechanism to recognize a feature is irrelevant and zero it out, 
    # so even noise columns get a weight, just from chance correlation with y
    
    X_noisy, feature_names_noisy = add_random_noise_features(
        X,
        FEATURE_COLS,
        n_noise=20,
        seed=0,
    )
    X_noisy_std, _, _ = standardize_matrix(X_noisy)
    _, w_noisy = ols_closed_form(X_noisy_std, y)
    
    plot_noise_feature_weights(
        feature_names_noisy,
        w_noisy,
        len(FEATURE_COLS),
        FIG_DIR / "noise_feature_weights.png",
    )



if __name__ == "__main__":
    main()
