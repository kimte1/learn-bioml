"""
Turns a parsed `Structure` (from `data_load_prep.py`) into the numeric
representations an ML pipeline consumes: which residues are
missing, how far apart every pair of residues is, which pairs are close
enough to call "in contact", and which residues sit near a bound ligand.
"""

from __future__ import annotations

import numpy as np

from data_load_prep import Structure


def find_missing_residues(
    residue_numbers: list[int],
) -> list[int]:
    """
    # KEY SKILL: missing coordinates -- a structure's residue numbers are
    not guaranteed to be consecutive: disordered loops with no resolved
    electron density simply have no entry at all, and classical
    numbering schemes (like beta-lactamase's Ambler numbering) can
    reserve numbers on purpose. Either way, this has to be found by
    checking the FULL expected range, not by counting entries.
    """
    full_range = set(
        range(min(residue_numbers), max(residue_numbers) + 1),
    )
    return sorted(full_range - set(residue_numbers))


def compute_distance_map(
    ca_coordinates: np.ndarray,
) -> np.ndarray:
    """
    # KEY SKILL: structural representations -- the (L, L) pairwise
    Euclidean distance between every pair of alpha-carbons, vectorized
    via broadcasting 
    """
    diff = ca_coordinates[:, None, :] - ca_coordinates[None, :, :]  # (L, L, 3)
    return np.sqrt(
        (diff**2).sum(axis=-1),
    )


def compute_contact_map(
    distance_map: np.ndarray,
    threshold: float = 5.0,
) -> np.ndarray:
    """
    A pair of residues is "in contact" if their alpha-carbons are within
    `threshold` angstroms
    """
    return distance_map < threshold


def ligand_contact_residues(
    structure: Structure,
    threshold: float = 5.0,
) -> list[int]:
    """
    # KEY SKILL: chain/interface annotations -- which residues (by PDB
    number) have an alpha-carbon within `threshold` angstroms of ANY atom
    in a bound heteroatom group (e.g. 1BTL's sulfate ion). This is a
    coarse, CA-only proxy for the rigorous all-atom ligand-contact
    definition real structural biology uses, and it's a LIGAND contact,
    not a protein-protein interface -- none of this project's 4
    structures is a multi-chain complex, so there is no second protein
    chain to define an interface against.
    """
    if not structure.hetero_atoms:
        return []

    contact_residues: set[int] = set()
    for _resname, hetero_coords in structure.hetero_atoms:
        # (L, 1, 3) - (1, n_hetero_atoms, 3) -> (L, n_hetero_atoms, 3)
        diff = structure.ca_coordinates[:, None, :] - hetero_coords[None, :, :]
        distances = np.sqrt(
            (diff**2).sum(axis=-1),
        )
        close_residues = np.where(
            (distances < threshold).any(axis=1),
        )[0]
        for index in close_residues:
            contact_residues.add(
                structure.residue_numbers[index],
            )

    return sorted(contact_residues)
