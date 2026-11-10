"""
Synthetic protein-solubility dataset generation.
"""

from __future__ import annotations

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def sample_sequences(
    n_samples: int,
    min_length: int = 50,
    max_length: int = 500,
    seed: int | None = None,
) -> list[str]:

    rng = np.random.default_rng(seed)

    # randomly generate intergers, with range min_length-max_length
    lengths = rng.integers(
        min_length, 
        max_length + 1, 
        size=n_samples
    )
    
    letters = np.array(list(AMINO_ACIDS))

    sequences = []
    for length in lengths:
        residues = rng.choice(letters, size=length)
        seq = ''.join(residues)
        sequences.append(seq)

    return sequences



def compute_features(sequences: list[str]) -> tuple[np.ndarray, list[str]]:
    feature_names = [
        "length",
        "molecular_weight",
        "gravy", # grand average of hydropathy
        "aromaticity",
        "instability_index",
        "isoelectric_point",
        "charge_at_ph7",
        "helix_fraction",
        "turn_fraction",
        "sheet_fraction",
    ] + [f"frac_{aa}" for aa in AMINO_ACIDS]

    rows = []
    for seq in sequences:
        pa = ProteinAnalysis(seq)
        
        helix, turn, sheet = pa.secondary_structure_fraction()
        comp = pa.amino_acids_percent

        rows.append(
            [
                len(seq),
                pa.molecular_weight(),
                pa.gravy(),
                pa.aromaticity(),
                pa.instability_index(),
                pa.isoelectric_point(),
                pa.charge_at_pH(7.0),
                helix,
                turn,
                sheet,
            ]
            + [comp[aa] for aa in AMINO_ACIDS]
        )

    return np.array(rows), feature_names



def make_labels(
    X: np.ndarray,
    feature_names: list[str],
    noise_std: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Make synthetic solubility labels (1 = soluble, 0 = not soluble) using a logistic function of a few biologically-plausible signal features: 
        - lower hydrophobicity (gravy)
        - lower aromaticity
        - lower instability
        - less extreme charge
        - shorter length 
        
    """
    rng = np.random.default_rng(seed)
    idx = {name: i for i, name in enumerate(feature_names)}

    def z(col: str) -> np.ndarray:
        values = X[:, idx[col]]
        return (values - values.mean()) / values.std()

    latent_score = (
        -1.5 * z("gravy")
        - 1.0 * z("aromaticity")
        - 0.5 * z("instability_index")
        - 0.5 * np.abs(z("charge_at_ph7"))
        - 0.3 * z("length")
    )
    noisy_score = latent_score + rng.normal(0, noise_std, size=len(latent_score))
    p_soluble = 1 / (1 + np.exp(-noisy_score))
    return (rng.uniform(size=len(p_soluble)) < p_soluble).astype(int)



def make_protein_dataset(
    n_samples: int,
    seed: int,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate a synthetic protein-solubility classification dataset.

    Returns 
        - X (values of features
        - y (labels; 1 = soluble; 0 = insoluble)
        - feature_names (list of features) 
    """
    sequences = sample_sequences(n_samples, seed=seed)
    X, feature_names = compute_features(sequences)
    y = make_labels(
        X, 
        feature_names, 
        noise_std=noise_std, 
        seed=seed
    )
    return X, y, feature_names
