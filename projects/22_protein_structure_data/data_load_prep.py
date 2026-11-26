"""
Downloads and parses 4 single-chain PDB structures into
numeric arrays for downstream structural
representation: 
- distance maps, contact maps, batches


3 of the 4 are small, completely clean structures (no missing residues,
no ligands) used mainly to give "batching variable-size structures"
something real to batch. 

1BTL, is TEM-1 beta-lactamase has missing residues, a bound ligand, and its own numbering scheme doesn't match either of those two earlier projects' numbering. 
"""

from __future__ import annotations

import urllib.request
import warnings
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PDB_DIR = REPO_ROOT / "data" / "pdb"
STRUCTURE_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


STRUCTURES = {
    "1BTL": "TEM-1 beta-lactamase",
    "1CRN": "Crambin (Crambe hispanica)",
    "1UBQ": "Ubiquitin (human)",
    "1LYZ": "Hen egg-white lysozyme",
}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# the same 286-residue TEM-1 precursor sequence as
# `19_esm2_finetune/data_load_prep.py`'s WILD_TYPE_SEQUENCE and
# `21_msa`'s Pfam-alignment TEM-1 row -- duplicated here rather than
# imported, since no project in this repo imports from another project's
# directory (each one is meant to run standalone from its own folder)
TEM1_WILD_TYPE_SEQUENCE = (
    "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAG"
    "QEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAI"
    "PNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIV"
    "VIYTTGSQATMDERNRQIAEIGASLIKHW"
)


@dataclass
class Structure:
    pdb_id: str
    chain_id: str
    residue_numbers: list[int]  # the REAL PDB numbers -- not necessarily 1..L, may skip integers
    sequence: str  # one-letter, same length/order as residue_numbers
    ca_coordinates: np.ndarray  # (L, 3) float, one alpha-carbon per residue
    hetero_atoms: list[tuple[str, np.ndarray]] = field(default_factory=list)  # (resname, (n_atoms, 3)), water excluded


@dataclass
class ResidueAlignment:
    position_map: dict[int, int]  # PDB residue number -> reference 1-indexed position
    mismatches: list[tuple[int, int, str, str]]  # (pdb_resnum, ref_position, pdb_aa, ref_aa)


def _download_structure(
    pdb_id: str,
) -> Path:
    PDB_DIR.mkdir(parents=True, exist_ok=True)

    path = PDB_DIR / f"{pdb_id.lower()}.cif"

    if not path.exists():
        urllib.request.urlretrieve(
            STRUCTURE_URL.format(pdb_id=pdb_id),
            path,
        )
    return path


def parse_structure(
    pdb_id: str,
) -> Structure:
    """
    # KEY SKILL: PDB/mmCIF parsing -- uses Biopython's MMCIFParser rather
    than hand-rolling a parser for mmCIF's own fixed-field text format
    """
    path = _download_structure(pdb_id)

    with warnings.catch_warnings():
        # Biopython warns about minor mmCIF spec deviations in older
        # depositions (1BTL is from 1995) that don't affect anything we
        # read here (coordinates, residue identity/numbering)
        warnings.simplefilter("ignore")
        parser = MMCIFParser(QUIET=True)
        model = parser.get_structure(pdb_id, path)[0]

    chain = next(iter(model))  # all 4 structures used here are single-chain (verified against RCSB)
    chain_id = chain.id

    residue_numbers: list[int] = []
    sequence_chars: list[str] = []
    ca_coords: list[np.ndarray] = []
    hetero_atoms: list[tuple[str, np.ndarray]] = []

    for residue in chain:
        is_standard = residue.id[0] == " "
        is_water = residue.id[0] == "W"

        if is_standard:
            # KEY SKILL: missing coordinates -- a residue can be present
            # (it has a number, it has SOME atoms) but still be missing
            # the specific atom we need; checking explicitly here, rather
            # than assuming every residue has a CA, is what keeps a
            # KeyError from surfacing deep inside a distance-map
            # computation instead of here, where it's obvious what went
            # wrong and for which residue
            if "CA" not in residue:
                continue
            residue_numbers.append(residue.id[1])
            sequence_chars.append(
                THREE_TO_ONE.get(residue.resname, "X"),
            )
            ca_coords.append(
                residue["CA"].coord,
            )
        elif not is_water:
            # a real heteroatom group (ligand, ion, cofactor) -- e.g.
            # 1BTL's bound sulfate (SO4) from the crystallization buffer
            atom_coords = np.array(
                [atom.coord for atom in residue],
            )
            hetero_atoms.append(
                (residue.resname, atom_coords),
            )

    return Structure(
        pdb_id=pdb_id,
        chain_id=chain_id,
        residue_numbers=residue_numbers,
        sequence="".join(sequence_chars),
        ca_coordinates=np.array(ca_coords),
        hetero_atoms=hetero_atoms,
    )


def load_all_structures() -> dict[str, Structure]:
    return {
        pdb_id: parse_structure(pdb_id) for pdb_id in STRUCTURES
    }


def align_to_reference(
    structure: Structure,
    reference_sequence: str,
) -> ResidueAlignment:
    """
    # KEY SKILL: residue indexing -- a structure's own residue numbers
    (here, 1BTL's Ambler-style numbering, starting at 26 with 2 internal
    gaps) can NOT be assumed to line up with another source's numbering
    for the "same" protein by simple arithmetic (e.g. "just subtract 2").
    The only correct way to cross-reference them is to align the actual
    SEQUENCES and read the correspondence off the alignment -- which is
    also what naturally surfaces real mismatches (genuine sequence
    differences between this exact crystallized construct and the
    reference) instead of silently papering over them.
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.match_score = 2
    aligner.mismatch_score = -1

    alignment = aligner.align(
        reference_sequence,
        structure.sequence,
    )[0]
    ref_coords, structure_coords = alignment.aligned

    position_map: dict[int, int] = {}
    mismatches: list[tuple[int, int, str, str]] = []

    for (ref_start, ref_end), (struct_start, struct_end) in zip(ref_coords, structure_coords):
        # each aligned BLOCK is gap-free on both sides, so corresponding
        # offsets within a block walk both sequences in lockstep
        for offset in range(int(ref_end) - int(ref_start)):
            ref_index = int(ref_start) + offset  # 0-indexed into reference_sequence
            struct_index = int(struct_start) + offset  # 0-indexed into structure.sequence

            ref_position = ref_index + 1  # 1-indexed, matches ProteinGym's own convention
            pdb_resnum = structure.residue_numbers[struct_index]
            position_map[pdb_resnum] = ref_position

            ref_aa = reference_sequence[ref_index]
            pdb_aa = structure.sequence[struct_index]
            if ref_aa != pdb_aa:
                mismatches.append(
                    (pdb_resnum, ref_position, pdb_aa, ref_aa),
                )

    return ResidueAlignment(
        position_map=position_map,
        mismatches=mismatches,
    )
