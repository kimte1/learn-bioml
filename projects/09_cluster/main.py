"""
K-means clustering from scratch, cross-checked against scikit-learn, on
the DeepLoc subcellular localization dataset (Almagro Armenteros et al.,
2017, Bioinformatics).

Pipeline: sequence-derived physicochemical/compositional features ->
standardize -> k-means directly on the standardized features (clustering,
run unsupervised, never shown the true label) -> compare clusters
against the true localization labels. PCA on this same data lives in
08_pca.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from data_load_prep import make_dataset
from kmeans import KMeans
from plotting import plot_cluster_vs_label, plot_kmeans_convergence, plot_kmeans_elbow

FIG_DIR = Path(__file__).resolve().parent / "fig"


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=4000, help="Class-stratified subsample size")
    parser.add_argument("--n-clusters", type=int, default=10, help="k for the main k-means run (10 = number of true classes)")
    return parser.parse_args()


def main() -> None:
    args = build_parser()
    FIG_DIR.mkdir(exist_ok=True)

    rng_seed = 0

    X, y, feature_names, ids = make_dataset(
        n_samples=args.n_samples,
        seed=rng_seed,
    )
    print(f"Loaded {len(y)} proteins, {len(feature_names)} features, {len(set(y))} localization classes")

    # standardize feature data
    # bc e.g. MW is measured in 100s but AA composition is 0-1
    # so scale it
    X_scaled = StandardScaler().fit_transform(X)

    ### K-means elbow sweep: inertia vs. k, directly on the standardized
    ### features (no PCA step -- clustering happens in the full feature
    ### space, not a compressed one)
    k_values = np.arange(2, 13)
    inertias = np.zeros(len(k_values))

    for i, k in enumerate(k_values):
        km = KMeans(
            n_clusters=int(k),
            n_init=3,
            max_iter=100,
            random_state=rng_seed,
        ).fit(X_scaled)
        inertias[i] = km.inertia_

    plot_kmeans_elbow(
        k_values,
        inertias,
        args.n_clusters,
        FIG_DIR / "01_kmeans_elbow.png",
    )

    ### Final k-means run, cross-checked against scikit-learn
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=rng_seed).fit(X_scaled)
    plot_kmeans_convergence(kmeans.inertia_history_, FIG_DIR / "02_kmeans_convergence.png")

    sklearn_kmeans = SklearnKMeans(n_clusters=args.n_clusters, n_init=10, random_state=rng_seed).fit(X_scaled)
    print(f"K-means cross-check: scratch inertia = {kmeans.inertia_:.1f}, scikit-learn inertia = {sklearn_kmeans.inertia_:.1f}")

    ### Unsupervised clusters vs. true localization labels -- k-means never
    ### sees y, so any structure here was discovered from features alone
    ari = adjusted_rand_score(y, kmeans.labels_)
    nmi = normalized_mutual_info_score(y, kmeans.labels_)
    print(f"Cluster vs. true label agreement: adjusted Rand index = {ari:.3f}, normalized mutual info = {nmi:.3f}")
    plot_cluster_vs_label(y, kmeans.labels_, FIG_DIR / "03_cluster_vs_label.png")


if __name__ == "__main__":
    main()
