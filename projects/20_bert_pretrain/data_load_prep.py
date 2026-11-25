"""
Loads a toy-scale, UniRef-subset stand-in for masked-language-model
(MLM) pretraining: a filtered, randomly sampled slice of real human
SwissProt sequences.

REUSES: `data/uniprot_human_sprot.fasta`, same source as
`07_boosting`/`18_esm2`. Unlike every earlier project, this week's task
needs no labels at all -- self-supervised pretraining manufactures its
own supervisory signal out of the raw sequences themselves (see
`masking.py`), which is exactly why it can use orders of magnitude more
data than any labeled task.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UNIPROT_FASTA = REPO_ROOT / "data" / "uniprot_human_sprot.fasta"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # the 20 standard amino acids
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
PAD_IDX = len(AMINO_ACIDS)  # 20 -- reserved index for padding
MASK_IDX = len(AMINO_ACIDS) + 1  # 21 -- reserved index for the <mask> token
VOCAB_SIZE = len(AMINO_ACIDS) + 2  # 22 -- 20 real residues + <pad> + <mask>


def _parse_fasta(
    path: Path,
) -> list[str]:
    """Plain header/sequence accumulation, no labels -- simpler than
    `16_transformer_train_encoder/data_load_prep.py`'s `_parse_fasta_with_labels`
    since there's nothing to extract from the header here.
    """
    sequences: list[str] = []
    seq_chars: list[str] = []

    def flush() -> None:
        if seq_chars:
            sequences.append("".join(seq_chars))

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                seq_chars.clear()
            else:
                seq_chars.append(line)
    flush()

    return sequences


@dataclass
class PretrainDataset:
    train_sequences: list[str]
    val_sequences: list[str]


def load_sequences(
    path: Path = UNIPROT_FASTA,
    max_len: int = 200,
    min_len: int = 20,
    n_sequences: int = 4000,
    val_frac: float = 0.1,
    seed: int = 0,
) -> PretrainDataset:
    """
    Filters to sequences made entirely of the 20 standard residues
    within `[min_len, max_len]` (dropping anything containing a
    non-standard residue like X/B/Z/U, same filtering idea as
    `14_attention`/`16_transformer_train_encoder`), randomly samples
    `n_sequences` of them, and splits off `val_frac` for validation.

    No stratification here, unlike every earlier `train_test_split` in
    this curriculum -- there's no label to stratify by. The only thing
    that matters for a self-supervised objective is that train and val
    are disjoint, real sequences.
    """
    all_sequences = _parse_fasta(
        path,
    )

    filtered = [
        seq
        for seq in all_sequences
        if min_len <= len(seq) <= max_len and all(c in AA_TO_INDEX for c in seq)
    ]

    print(
        f"{len(filtered)}/{len(all_sequences)} sequences pass the "
        f"length [{min_len}, {max_len}] + standard-residue filter",
    )

    rng = random.Random(
        seed,
    )
    sampled = rng.sample(
        filtered,
        min(n_sequences, len(filtered)),
    )
    rng.shuffle(
        sampled,
    )

    n_val = int(len(sampled) * val_frac)
    val_sequences = sampled[:n_val]
    train_sequences = sampled[n_val:]

    return PretrainDataset(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
    )
