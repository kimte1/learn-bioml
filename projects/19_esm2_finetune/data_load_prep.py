"""
Loads the BLAT_ECOLX_Jacquier_2013 ProteinGym assay -- 989 single
point-mutant variants of TEM-1 beta-lactamas, each with a measured `DMS_score`
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSAY_CSV = REPO_ROOT / "data" / "proteingym_blat_ecolx_jacquier_2013.csv"

# the wild-type (un-mutated) TEM-1 beta-lactamase sequence, pulled directly
# from ProteinGym's own reference file for this assay (DMS_id
# BLAT_ECOLX_Jacquier_2013) -- this is the sequence every mutant below is
# a one-residue edit away from
WILD_TYPE_SEQUENCE = (
    "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAG"
    "QEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAI"
    "PNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIV"
    "VIYTTGSQATMDERNRQIAEIGASLIKHW"
)

MUTANT_PATTERN = re.compile(
    r"^([A-Z])(\d+)([A-Z])$",
)


def _apply_mutation(
    mutant_code: str,
) -> str:
    """
    Turns a ProteinGym mutation code like `H24Y` into a full mutant
    sequence
    """
    match = MUTANT_PATTERN.match(
        mutant_code,
    )
    wild_type_residue, position_str, mutant_residue = match.groups()
    position = int(position_str) - 1  # ProteinGym positions are 1-indexed

    actual_residue = WILD_TYPE_SEQUENCE[position]
    assert actual_residue == wild_type_residue, (
        f"{mutant_code}: wild type has '{actual_residue}' at position {position + 1}, "
        f"not '{wild_type_residue}' as the mutation code claims"
    )

    return (
        WILD_TYPE_SEQUENCE[:position]
        + mutant_residue
        + WILD_TYPE_SEQUENCE[position + 1 :]
    )


@dataclass
class FitnessDataset:
    train_sequences: list[str]
    train_scores: list[float]
    val_sequences: list[str]
    val_scores: list[float]
    test_sequences: list[str]
    test_scores: list[float]


def load_fitness_dataset(
    path: Path = ASSAY_CSV,
    seed: int = 0,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> FitnessDataset:
    """
    Loads the assay CSV, reconstructs every mutant's full sequence, and
    splits into train/val/test.

    """
    df = pd.read_csv(
        path,
    )
    df["sequence"] = df["mutant"].apply(
        _apply_mutation,
    )

    test_frac = 1.0 - train_frac - val_frac
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_frac,
        random_state=seed,
        stratify=df["DMS_score_bin"],
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_frac / (train_frac + val_frac),
        random_state=seed,
        stratify=train_val_df["DMS_score_bin"],
    )

    return FitnessDataset(
        train_sequences=train_df["sequence"].tolist(),
        train_scores=train_df["DMS_score"].tolist(),
        val_sequences=val_df["sequence"].tolist(),
        val_scores=val_df["DMS_score"].tolist(),
        test_sequences=test_df["sequence"].tolist(),
        test_scores=test_df["DMS_score"].tolist(),
    )
