"""
Column-wise conservation: how much the amino-acid identity varies across
the family at each aligned position. Low entropy = conserved.
"""

from __future__ import annotations

import numpy as np

from data_load_prep import GAP_IDX


def column_entropy(encoded: np.ndarray) -> np.ndarray:
    """Shannon entropy (bits) of the amino-acid distribution at each
    column, computed only over the non-gap residues -- gap frequency
    itself is reported separately by `gap_fraction` (coverage), not
    folded into "how conserved is the residue identity".
    """
    num_cols = encoded.shape[1]
    entropy = np.full(num_cols, np.nan)
    for col in range(num_cols):
        residues = encoded[:, col]
        residues = residues[residues != GAP_IDX]
        if len(residues) == 0:
            continue
        counts = np.bincount(residues, minlength=GAP_IDX)
        probs = counts[counts > 0] / counts.sum()
        entropy[col] = -(probs * np.log2(probs)).sum()
    return entropy


def gap_fraction(encoded: np.ndarray) -> np.ndarray:
    return (encoded == GAP_IDX).mean(axis=0)
