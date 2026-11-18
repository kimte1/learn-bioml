"""Load real protein sequences from DeepLoc for the attention demos in
`main.py`. No label is used -- this is Week 10, forward-pass mechanics
only -- so only the raw sequences and their lengths matter here.

REUSES: `data/deeploc_data.fasta`, already in the repo (see
`data/README.md`) and otherwise unused until Week 13 (protein language
models). It supplies real amino-acid content and real variable lengths for
hand-constructed attention probes -- it is not a prediction task.
"""

from __future__ import annotations

from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEEPLOC_FASTA = REPO_ROOT / "data" / "deeploc_data.fasta"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


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



def load_sequences(
    path: Path = DEEPLOC_FASTA,
    max_len: int = 25,
    n: int = 8,
    seed: int = 0,
) -> list[str]:
    """
    Return `n` protein sequences truncated to at most `max_len`
    residues
    """
    all_sequences = _parse_fasta(path)
    candidates = [seq[:max_len] for seq in all_sequences]

    # skip seqs with non-standard aa 
    candidates = [seq for seq in candidates if all(c in AA_TO_INDEX for c in seq)]

    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(candidates), generator=generator)[:n]
    return [candidates[i] for i in idx.tolist()]



def load_variable_length_sequences(
    path: Path = DEEPLOC_FASTA,
    max_len: int = 25,
    min_len: int = 10,
    n: int = 6,
    seed: int = 0,
) -> list[str]:
    
    base = load_sequences(path, max_len=max_len, n=n, seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    lengths = torch.randint(
        min_len, 
        max_len + 1, 
        (len(base),), 
        generator=generator
    ).tolist()

    return [seq[:length] for seq, length in zip(base, lengths)]



def one_hot_encode(sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad `sequences` to their batch max length and one-hot encode over
    the 20 standard amino acids.

    Returns:
        tokens: (batch, max_len, 20) float one-hot tensor, zero-padded.
        lengths: (batch,) long tensor of each sequence's true length.
    """
    lengths = torch.tensor(
        [len(seq) for seq in sequences], 
        dtype=torch.long
    )
    max_len = int(lengths.max())

    tokens = torch.zeros(len(sequences), max_len, len(AMINO_ACIDS))
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            tokens[i, j, AA_TO_INDEX[aa]] = 1.0

    return tokens, lengths
