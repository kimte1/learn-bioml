from __future__ import annotations

from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEEPLOC_FASTA = REPO_ROOT / "data" / "deeploc_data.fasta"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # the 20 standard amino acids
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
VOCAB_SIZE = len(AMINO_ACIDS)


def _parse_fasta(path: Path) -> list[str]:
    sequences: list[str] = []
    seq_chars: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_chars:
                    sequences.append("".join(seq_chars))
                seq_chars = []
            else:
                seq_chars.append(line)
    if seq_chars:
        sequences.append("".join(seq_chars))
    return sequences


def load_sequence(
    path: Path = DEEPLOC_FASTA, 
    length: int = 12, 
    seed: int = 0
) -> str:
    """
    One protein sequence truncated to exactly `length` residues.    
    """
    sequences = _parse_fasta(path)
    candidates = [seq[:length] for seq in sequences if len(seq) >= length]
    candidates = [seq for seq in candidates if all(c in AA_TO_INDEX for c in seq)]

    generator = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(candidates), (1,), generator=generator).item()
    return candidates[idx]


def tokenize(sequence: str) -> torch.Tensor:
    """(1, L) long tensor of amino-acid indices -- batch size 1."""
    return torch.tensor(
        [[AA_TO_INDEX[aa] for aa in sequence]], 
        dtype=torch.long
    )
