"""
Principal Component Analysis from scratch (Pearson, 1901, "On Lines and
Planes of Closest Fit to Systems of Points in Space").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCA:
    n_components: int

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=np.float64)

        # Center the data: PCA's "variance" is only meaningful around the
        # mean, and an off-center cloud would bias the covariance matrix
        # toward the direction of the mean itself rather than the spread.
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Covariance matrix: cov[i, j] is how much features i and j vary
        # together. Its eigenvectors are exactly the uncorrelated
        # ("orthogonal") directions of variance that PCA looks for.
        n_samples = X_centered.shape[0]
        cov = (X_centered.T @ X_centered) / (n_samples - 1)

        # Covariance matrices are symmetric, so eigh (symmetric-matrix
        # eigendecomposition) is both faster and numerically more stable
        # than a general eigensolver. eigh returns eigenvalues ascending.
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self.components_ = eigenvectors[:, : self.n_components].T  # (n_components, n_features)
        self.explained_variance_ = eigenvalues[: self.n_components]
        self.explained_variance_ratio_ = eigenvalues[: self.n_components] / eigenvalues.sum()
        self.explained_variance_ratio_full_ = eigenvalues / eigenvalues.sum()
        return self


    def transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = np.asarray(X, dtype=np.float64) - self.mean_
        return X_centered @ self.components_.T


    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
