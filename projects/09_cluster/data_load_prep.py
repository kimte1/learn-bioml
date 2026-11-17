"""
Load the DeepLoc dataset and turn it into (X, y) for unsupervised
learning. 

One row:
   - protein
   - physicochemical/compositional features (computed from seq)
   - subcellular localization label 
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEEPLOC_FASTA = REPO_ROOT / "data" / "deeploc_data.fasta"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
CHARGED = "DEKR"

# parse DeepLoc FASTA header
_HEADER_RE = re.compile(r"^(\S+)\s+([^\s]+?)-[MSU](?:\s+test)?$")


def parse_fasta(path: Path = DEEPLOC_FASTA) -> tuple[list[str], list[str], list[str]]:
    
    ids: list[str] = []
    labels: list[str] = []
    sequences: list[str] = []

    header: str | None = None
    seq_chunks: list[str] = []

    def flush() -> None:
        if header is None:
            return
        match = _HEADER_RE.match(header)
        if match is None or "-" in match.group(2):
            return
        sequence = "".join(seq_chunks)
        if set(sequence) - set(AMINO_ACIDS):
            return
        ids.append(match.group(1))
        labels.append(match.group(2))
        sequences.append(sequence)

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
    flush()

    return ids, labels, sequences


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


def make_dataset(
    path: Path = DEEPLOC_FASTA,
    n_samples: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    
    ids, labels, sequences = parse_fasta(path)
    X, feature_names = compute_features(sequences)
    y = np.array(labels)
    ids = np.array(ids)

    if n_samples is not None and n_samples < len(y):
        rng = np.random.default_rng(seed)
        keep = []
        for label in np.unique(y):
            class_idx = np.where(y == label)[0]
            n_keep = max(1, round(n_samples * len(class_idx) / len(y)))
            keep.append(rng.choice(class_idx, size=min(n_keep, len(class_idx)), replace=False))
        keep_idx = np.concatenate(keep)
        rng.shuffle(keep_idx)
        X, y, ids = X[keep_idx], y[keep_idx], ids[keep_idx]

    return X, y, feature_names, list(ids)
