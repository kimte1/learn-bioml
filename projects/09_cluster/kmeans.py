"""
K-means clustering from scratch (Lloyd, 1957/1982, "Least Squares
Quantization in PCM"; also MacQueen, 1967), with k-means++ initialization
(Arthur & Vassilvitskii, 2007).

K-means partitions points into k clusters by alternating two steps
(Lloyd's algorithm) until the assignments stop changing:
  1. Assign: give each point to its nearest centroid.
  2. Update: move each centroid to the mean of the points now assigned
     to it.
Each step can only decrease (or hold constant) the inertia -- the total
squared distance from every point to its own centroid -- so the loop is
guaranteed to converge, just not necessarily to the global optimum.
That non-guarantee is exactly why initialization matters and why we
re-run from several random starts (n_init) and keep the lowest-inertia
result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """(n_samples, n_centers) matrix of squared Euclidean distances."""
    diff = X[:, None, :] - centers[None, :, :]
    return np.sum(diff**2, axis=2)


def _kmeans_plusplus_init(X: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++: pick the first center uniformly at random, then each
    subsequent center with probability proportional to its squared
    distance from the nearest center already chosen. This spreads the
    initial centers out across the data instead of letting them land by
    chance in the same region, which is what random initialization can
    do and which slow, poor convergence tends to follow from.
    """
    n_samples = X.shape[0]
    centers = np.empty((n_clusters, X.shape[1]))
    first = rng.integers(n_samples)
    centers[0] = X[first]

    closest_sq_dist = np.sum((X - centers[0]) ** 2, axis=1)
    for i in range(1, n_clusters):
        probs = closest_sq_dist / closest_sq_dist.sum()
        next_idx = rng.choice(n_samples, p=probs)
        centers[i] = X[next_idx]
        new_sq_dist = np.sum((X - centers[i]) ** 2, axis=1)
        closest_sq_dist = np.minimum(closest_sq_dist, new_sq_dist)

    return centers


@dataclass
class KMeans:
    n_clusters: int
    max_iter: int = 300
    n_init: int = 10
    tol: float = 1e-4
    random_state: int | None = None

    inertia_history_: list[float] = field(default_factory=list, init=False)

    def _single_run(self, X: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
        centers = _kmeans_plusplus_init(X, self.n_clusters, rng)
        history = []
        labels = np.zeros(X.shape[0], dtype=int)

        for _ in range(self.max_iter):
            # Assign step: nearest centroid per point.
            dists = _squared_distances(X, centers)
            labels = np.argmin(dists, axis=1)
            inertia = dists[np.arange(len(X)), labels].sum()
            history.append(float(inertia))

            # Update step: each centroid becomes the mean of its members.
            # An empty cluster keeps its previous center rather than
            # becoming NaN.
            new_centers = centers.copy()
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members) > 0:
                    new_centers[k] = members.mean(axis=0)

            shift = np.sqrt(np.sum((new_centers - centers) ** 2, axis=1)).max()
            centers = new_centers
            if shift < self.tol:
                break

        final_dists = _squared_distances(X, centers)
        labels = np.argmin(final_dists, axis=1)
        inertia = float(final_dists[np.arange(len(X)), labels].sum())
        return labels, centers, inertia, history

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        # Lloyd's algorithm only finds a local optimum of inertia, so we
        # restart from several k-means++ initializations and keep the
        # run with the lowest final inertia.
        best_inertia = np.inf
        for _ in range(self.n_init):
            labels, centers, inertia, history = self._single_run(X, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                self.labels_ = labels
                self.cluster_centers_ = centers
                self.inertia_ = inertia
                self.inertia_history_ = history

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = _squared_distances(np.asarray(X, dtype=np.float64), self.cluster_centers_)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
