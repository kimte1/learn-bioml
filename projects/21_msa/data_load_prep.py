"""
Obtains the MSA that TEM-1 beta-lactamase (`BLAT_ECOLX`, the same protein
as `19_esm2_finetune`'s DMS assay) belongs to -- Pfam family PF13354
("Beta-lactamase2"), 21,357 member sequences -- and turns it into a
fixed-width, 21-state (20 amino acids + gap) integer matrix ready for
conservation/covariation statistics.
"""

from __future__ import annotations

import gzip
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
ALIGNMENT_CACHE = DATA_DIR / "pfam_PF13354_beta_lactamase2_full.sto.gz"
ALIGNMENT_URL = "https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam/PF13354/?annotation=alignment:full"
ASSAY_CSV = DATA_DIR / "proteingym_blat_ecolx_jacquier_2013.csv"

# this row IS TEM-1 -- UniProt P62593, residues 28-286 (the catalytic
# domain, after the signal peptide), same numbering ProteinGym's mutation
# codes below use
TEM1_ROW_ID = "BLAT_ECOLX/28-286"
TEM1_FIRST_POSITION = 28

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # same ordering as 20_bert_pretrain
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
GAP_IDX = len(AMINO_ACIDS)  # 20 -- gap AND any non-canonical letter (X/B/Z/...)
NUM_STATES = len(AMINO_ACIDS) + 1  # 21

MUTANT_PATTERN = re.compile(r"^[A-Z](\d+)[A-Z]$")


@dataclass
class MSA:
    ids: list[str]
    encoded: np.ndarray  # (num_sequences, num_columns) int8 in [0, NUM_STATES)
    tem1_index: int
    column_to_uniprot_position: dict[int, int]  # only columns where TEM-1 has a real residue


def _download_alignment(path: Path = ALIGNMENT_CACHE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(ALIGNMENT_URL, path)


def _read_stockholm(path: Path) -> tuple[list[str], np.ndarray]:
    """
    Stockholm format is the MSA file format 
    """
    chunks: dict[str, list[str]] = {}
    order: list[str] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#") or line.startswith("//") or not line.strip():
                continue
            seq_id, chunk = line.split()
            if seq_id not in chunks:
                order.append(seq_id)
                chunks[seq_id] = []
            chunks[seq_id].append(chunk)

    sequences = ["".join(chunks[seq_id]) for seq_id in order]
    length = len(sequences[0])
    raw = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8)
    return order, raw.reshape(len(order), length)


def _match_columns(raw: np.ndarray) -> np.ndarray:
    """A column is a Pfam match-state column iff no sequence has a
    lowercase (insert-state) character there.
    """
    is_lower = (raw >= ord("a")) & (raw <= ord("z"))
    return ~is_lower.any(axis=0)


def _encode(raw: np.ndarray) -> np.ndarray:
    """Raw ASCII codes -> integers in [0, NUM_STATES): canonical amino
    acids keep their `AMINO_ACIDS` index, everything else (gap '-'/'.'
    and non-canonical letters) becomes GAP_IDX.
    """
    encoded = np.full(raw.shape, GAP_IDX, dtype=np.int8)
    for aa, idx in AA_TO_INDEX.items():
        encoded[raw == ord(aa)] = idx
    return encoded


def load_msa(
    max_gap_frac: float = 0.5,
    max_sequences: int = 2000,
    seed: int = 0,
) -> MSA:
    """Downloads (or reuses the cached copy of) the PF13354 alignment,
    keeps only match-state columns, drops columns that are mostly gap
    across the whole family, then subsamples sequences down to a
    laptop-friendly count -- always keeping the TEM-1 row itself.
    """
    if not ALIGNMENT_CACHE.exists():
        _download_alignment()

    ids, raw = _read_stockholm(ALIGNMENT_CACHE)
    raw = raw[:, _match_columns(raw)]
    tem1_idx = ids.index(TEM1_ROW_ID)

    # TEM-1's own residue numbering must walk every match column (even
    # ones we're about to drop for being too gappy family-wide), so this
    # has to happen before the gap-fraction column filter below
    full_col_to_position: dict[int, int] = {}
    position = TEM1_FIRST_POSITION
    for col, char in enumerate(raw[tem1_idx]):
        if char not in (ord("-"), ord(".")):
            full_col_to_position[col] = position
            position += 1

    is_gap = (raw == ord("-")) | (raw == ord("."))
    keep_cols = np.where(is_gap.mean(axis=0) <= max_gap_frac)[0]
    raw = raw[:, keep_cols]
    old_to_new = {old: new for new, old in enumerate(keep_cols)}
    column_to_uniprot_position = {
        old_to_new[old]: pos for old, pos in full_col_to_position.items() if old in old_to_new
    }

    rng = np.random.default_rng(seed)
    n = raw.shape[0]
    if n > max_sequences:
        other = [i for i in range(n) if i != tem1_idx]
        sampled = rng.choice(other, size=max_sequences - 1, replace=False)
        keep_rows = np.sort(np.append(sampled, tem1_idx))
        raw = raw[keep_rows]
        ids = [ids[i] for i in keep_rows]
        tem1_idx = int(np.nonzero(keep_rows == tem1_idx)[0][0])

    return MSA(
        ids=ids,
        encoded=_encode(raw),
        tem1_index=tem1_idx,
        column_to_uniprot_position=column_to_uniprot_position,
    )


def load_dms_scores_by_position(path: Path = ASSAY_CSV) -> dict[int, list[float]]:
    """ProteinGym mutation codes (e.g. `H24Y`) are 1-indexed positions
    into TEM-1's own sequence -- the same numbering `column_to_uniprot_position`
    uses -- so these keys line up directly with the MSA's position numbers.
    """
    df = pd.read_csv(path)
    scores_by_position: dict[int, list[float]] = {}
    for mutant_code, score in zip(df["mutant"], df["DMS_score"]):
        position = int(MUTANT_PATTERN.match(mutant_code).group(1))
        scores_by_position.setdefault(position, []).append(score)
    return scores_by_position
