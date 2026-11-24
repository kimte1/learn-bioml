"""
Load real, labeled protein sequences from DeepLoc for subcellular-
localization classification -- unlike `14_attention`/`15_transformer_encoder`,
which only used the sequences, this week needs the labels too.

REUSES: `data/deeploc_data.fasta`, same source as 08_pca/09_cluster/
14_attention/15_transformer_encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEEPLOC_FASTA = REPO_ROOT / "data" / "deeploc_data.fasta"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # the 20 standard amino acids
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
PAD_IDX = len(AMINO_ACIDS)  # 20 -- one reserved index for padding
VOCAB_SIZE = len(AMINO_ACIDS) + 1  # 21


def _parse_fasta_with_labels(path: Path, max_len: int) -> tuple[list[str], list[str]]:
    """Each header looks like `>ACCESSION LABEL-FLAG` or
    `>ACCESSION LABEL-FLAG test` (FLAG is M/S/U, and "test" marks
    DeepLoc's own train/test split, which we don't use -- see
    `load_dataset`). Returns (sequences, labels), each truncated/filtered
    to be cleanly usable: sequences longer than `max_len` are truncated,
    and sequences containing non-standard residues (X, B, Z, U) within
    that truncated window are dropped, matching 14_attention's approach.
    """
    sequences: list[str] = []
    labels: list[str] = []

    header: str | None = None
    seq_chars: list[str] = []

    def flush() -> None:
        if header is None:
            return
        label = header.split()[1].rsplit("-", 1)[0]  # drop the trailing -M/-S/-U flag
        if label == "Cytoplasm-Nucleus":  # the one dual-localization category -- keep this a clean single-label problem
            return
        seq = "".join(seq_chars)[:max_len]
        if not all(c in AA_TO_INDEX for c in seq):
            return
        sequences.append(seq)
        labels.append(label)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line
                seq_chars = []
            else:
                seq_chars.append(line)
    flush()

    return sequences, labels


@dataclass
class Dataset:
    train_sequences: list[str]
    train_labels: torch.Tensor
    val_sequences: list[str]
    val_labels: torch.Tensor
    test_sequences: list[str]
    test_labels: torch.Tensor
    label_encoder: LabelEncoder


def load_dataset(
    path: Path = DEEPLOC_FASTA,
    max_len: int = 512,
    seed: int = 0,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Dataset:
    sequences, labels = _parse_fasta_with_labels(path, max_len)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    # first split off the test set, then split the remainder into
    # train/val -- same two-step approach as 13_mlp_wb/data_load_prep.py,
    # stratified throughout since DeepLoc's classes are very imbalanced
    # (4043 Nucleus vs. 154 Peroxisome)
    test_frac = 1.0 - train_frac - val_frac
    train_val_seqs, test_seqs, train_val_y, test_y = train_test_split(
        sequences, y, test_size=test_frac, random_state=seed, stratify=y
    )
    train_seqs, val_seqs, train_y, val_y = train_test_split(
        train_val_seqs,
        train_val_y,
        test_size=val_frac / (train_frac + val_frac),
        random_state=seed,
        stratify=train_val_y,
    )

    return Dataset(
        train_sequences=train_seqs,
        train_labels=torch.tensor(train_y, dtype=torch.long),
        val_sequences=val_seqs,
        val_labels=torch.tensor(val_y, dtype=torch.long),
        test_sequences=test_seqs,
        test_labels=torch.tensor(test_y, dtype=torch.long),
        label_encoder=encoder,
    )


def tokenize_batch(sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad `sequences` to their batch max length.

    Returns:
        ids: (batch, max_len) long tensor, amino-acid indices, PAD_IDX
            where padded.
        key_padding_mask: (batch, max_len) bool tensor, **True means
            "this position is padding, ignore it"** -- the convention
            `nn.TransformerEncoderLayer`'s `src_key_padding_mask` argument
            expects. This is the opposite of `14_attention`/
            `15_transformer_encoder`'s `padding_mask()`, which used True
            for "real token, attend to it" (the convention
            `F.scaled_dot_product_attention`'s `attn_mask` expects).
            Deliberately not reusing that helper here.
    """
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    max_len = int(lengths.max())

    ids = torch.full((len(sequences), max_len), PAD_IDX, dtype=torch.long)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            ids[i, j] = AA_TO_INDEX[aa]

    positions = torch.arange(max_len).unsqueeze(0)  # (1, max_len)
    key_padding_mask = positions >= lengths.unsqueeze(1)  # (batch, max_len), True = pad

    return ids, key_padding_mask
