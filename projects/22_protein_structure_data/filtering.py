"""
filters by resolution, length, and completeness, same as
AlphaFold/OpenFold's own training-set curation. 
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from data_load_prep import PDB_DIR
from data_load_prep import Structure
from structural_features import find_missing_residues


@dataclass
class FilterCriteria:
    max_resolution: float = 1.8  # angstrom -- lower is better/sharper
    min_length: int = 50  # resolved residues
    max_missing_fraction: float = 0.05  # fraction of the [min,max] residue-number range with no resolved coordinates


def get_resolution(
    pdb_id: str,
) -> float:
    """
    Reads the X-ray resolution straight out of the already-cached
    `.cif` file's own metadata (`_refine.ls_d_res_high`) -- verified
    during planning to match RCSB's reported values exactly for all 4
    structures used here.
    """
    path = PDB_DIR / f"{pdb_id.lower()}.cif"
    mmcif_dict = MMCIF2Dict(
        path,
    )
    return float(mmcif_dict["_refine.ls_d_res_high"][0])


def evaluate_filters(
    structure: Structure,
    resolution: float,
    criteria: FilterCriteria,
) -> dict[str, bool]:
    """
    # KEY SKILL: filtering -- every criterion evaluated and returned
    independently, so an excluded structure's report says exactly WHY
    (one failed criterion, several, or none) instead of a single
    opaque True/False.
    """
    missing = find_missing_residues(structure.residue_numbers)
    full_range = max(structure.residue_numbers) - min(structure.residue_numbers) + 1
    missing_fraction = len(missing) / full_range

    return {
        "resolution": resolution <= criteria.max_resolution,
        "length": len(structure.residue_numbers) >= criteria.min_length,
        "completeness": missing_fraction <= criteria.max_missing_fraction,
    }


def apply_filters(
    structures: list[Structure],
    criteria: FilterCriteria,
) -> tuple[list[Structure], dict[str, dict[str, bool]]]:
    report: dict[str, dict[str, bool]] = {}
    kept: list[Structure] = []

    for structure in structures:
        resolution = get_resolution(structure.pdb_id)
        per_criterion = evaluate_filters(structure, resolution, criteria)
        report[structure.pdb_id] = per_criterion

        if all(per_criterion.values()):
            kept.append(structure)

    return kept, report
