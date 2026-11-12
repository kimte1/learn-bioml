"""
Load the Meltome Atlas (Jarzab et al., 2020, Nature Methods) thermal
proteome stability dataset -- already joined to UniProt sequences by
fetch_uniprot_sequences.py into data/meltome_with_sequences.csv.zip -- and
turn it into (X, y) for gradient boosting: one row per gene, physicochemical
features computed from its sequence as X, its mean quantile-normalized
melting point across the cell lines/replicates it was measured in as y.

Purely local/offline: run fetch_uniprot_sequences.py once beforehand to
build the CSV, and main.py can be re-run as many times as you like from
there with no UniProt access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MELTOME_WITH_SEQUENCES_CSV = REPO_ROOT / "data" / "meltome_with_sequences.csv.zip"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
CHARGED = "DEKR"


def load_gene_table(csv_path: Path = MELTOME_WITH_SEQUENCES_CSV) -> pd.DataFrame:
    """One row per (gene, sequence): mean quantile-normalized melting
    point, averaged across every cell line and replicate it was measured
    in. Quantile normalization is what the Meltome Atlas uses to make Tm
    comparable across cell lines that were run as separate experiments.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python fetch_uniprot_sequences.py` once "
            "to build it from data/meltome.csv.zip + UniProt sequences."
        )
    df = pd.read_csv(csv_path, usecols=["gene_name", "sequence", "quan_norm_meltPoint"])
    df = df.dropna(subset=["quan_norm_meltPoint"])
    return df.groupby(["gene_name", "sequence"], as_index=False)["quan_norm_meltPoint"].mean()


def compute_features(sequences: list[str]) -> tuple[np.ndarray, list[str]]:
    feature_names = [
        "length",
        "molecular_weight",
        "gravy",
        "aromaticity",
        "instability_index",
        "isoelectric_point",
        "charge_at_ph7",
        "helix_fraction",
        "turn_fraction",
        "sheet_fraction",
        "charged_fraction",
    ] + [f"frac_{aa}" for aa in AMINO_ACIDS]

    rows = []
    for seq in sequences:
        pa = ProteinAnalysis(seq)
        helix, turn, sheet = pa.secondary_structure_fraction()
        comp = pa.amino_acids_percent
        charged_fraction = sum(comp[aa] for aa in CHARGED)

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
                charged_fraction,
            ]
            + [comp[aa] for aa in AMINO_ACIDS]
        )

    return np.array(rows), feature_names


def make_protein_tm_dataset(
    csv_path: Path = MELTOME_WITH_SEQUENCES_CSV,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Returns (X, y, feature_names, gene_names), one row per gene."""
    gene_table = load_gene_table(csv_path)
    X, feature_names = compute_features(gene_table["sequence"].tolist())
    y = gene_table["quan_norm_meltPoint"].to_numpy()
    genes = gene_table["gene_name"].tolist()
    return X, y, feature_names, genes
