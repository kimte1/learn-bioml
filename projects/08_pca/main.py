"""
PCA from scratch, cross-checked against scikit-learn, on the DeepLoc
subcellular localization dataset (Almagro Armenteros et al., 2017,
Bioinformatics).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

from data_load_prep import make_dataset
from pca import PCA
from plotting import plot_pca_scatter, plot_scree

FIG_DIR = Path(__file__).resolve().parent / "fig"


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=4000,
        help="Class-stratified subsample size",
    )
    return parser.parse_args()


def main() -> None:
    args = build_parser()
    FIG_DIR.mkdir(exist_ok=True)

    rng_seed = 0
    pca_components = 10  # how many components to compress to

    X, y, feature_names, ids = make_dataset(
        n_samples=args.n_samples,
        seed=rng_seed,
    )
    print(f"Loaded {len(y)} proteins, {len(feature_names)} features, {len(set(y))} localization classes")

    # standardize feature data
    # bc e.g. MW is measured in 100s but AA composition is 0-1
    # so scale it
    X_scaled = StandardScaler().fit_transform(X)

    ### PCA from scratch, cross-checked against scikit-learn
    pca = PCA(n_components=pca_components).fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    sklearn_pca = SklearnPCA(
        n_components=pca_components,
        random_state=rng_seed,
    ).fit(X_scaled)

    ratio_diff = np.abs(pca.explained_variance_ratio_ - sklearn_pca.explained_variance_ratio_).max()

    print(f"PCA cross-check. How far off PCA (from scratch) is from scikit-learn: {ratio_diff:.2e}")

    plot_scree(
        pca.explained_variance_ratio_full_,
        FIG_DIR / "01_scree.png",
    )
    plot_pca_scatter(
        X_pca[:, :2],
        y,
        pca.explained_variance_ratio_,
        FIG_DIR / "02_pca_scatter.png",
    )


if __name__ == "__main__":
    main()
