"""
Pairwise column covariation via mutual information (MI), with the
Average Product Correction (Dunn et al. 2008) that removes the generic
"both columns are just globally variable" bias so what's left is closer
to real coevolution.
"""

from __future__ import annotations

import numpy as np

from data_load_prep import NUM_STATES


def pairwise_mutual_information(encoded: np.ndarray) -> np.ndarray:
    """encoded: (num_sequences, num_cols) int in [0, NUM_STATES).

    Returns an (num_cols, num_cols) matrix of MI in bits. One-hot encodes
    every column, then gets ALL pairwise joint frequencies at once via a
    single big matmul (X.T @ X) rather than looping column pairs in
    Python -- the only way this is fast enough at ~250 columns.
    """
    num_seqs, num_cols = encoded.shape
    onehot = np.eye(NUM_STATES, dtype=np.float32)[encoded]  # (N, L, 21)
    marginals = onehot.mean(axis=0)  # (L, 21) -- p_i(a)

    flat = onehot.reshape(num_seqs, num_cols * NUM_STATES)
    joint_counts = flat.T @ flat  # (L*21, L*21)
    joint = (joint_counts / num_seqs).reshape(num_cols, NUM_STATES, num_cols, NUM_STATES)
    joint = joint.transpose(0, 2, 1, 3)  # joint[i, j, a, b] = P(col_i=a, col_j=b)

    mi = np.zeros((num_cols, num_cols))
    for i in range(num_cols):
        joint_i = joint[i]  # (L, 21, 21)
        outer_i = marginals[i][None, :, None] * marginals[:, None, :]  # (L, 21, 21)
        mask = joint_i > 1e-12
        ratio = np.ones_like(joint_i)
        ratio[mask] = joint_i[mask] / outer_i[mask]
        terms = np.zeros_like(joint_i)
        terms[mask] = joint_i[mask] * np.log2(ratio[mask])
        mi[i] = terms.sum(axis=(1, 2))

    return mi


def apc_correction(mi: np.ndarray) -> np.ndarray:
    """Average Product Correction: subtracts off each pair's expected MI
    under the assumption it's driven only by how "chatty" (high average
    MI with everything else) columns i and j are individually.
    """
    mi = mi.copy()
    np.fill_diagonal(mi, 0.0)
    num_cols = mi.shape[0]

    row_mean = mi.sum(axis=1) / (num_cols - 1)
    overall_mean = mi.sum() / (num_cols * (num_cols - 1))

    corrected = mi - np.outer(row_mean, row_mean) / overall_mean
    np.fill_diagonal(corrected, 0.0)
    return corrected
