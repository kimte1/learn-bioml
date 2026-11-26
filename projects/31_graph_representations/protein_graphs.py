"""
Builds TWO granularities of protein graph from the same parsed
structure: a RESIDUE graph (one node per residue, CA-level -- matching
`22_protein_structure_data`'s view) and an ATOMIC graph (one node per
heavy atom). Unlike SMILES, a bare mmCIF file has no declared bonds at
all -- `molecular_graphs.py`'s bonds came straight from SMILES' own
grammar, but mmCIF just gives positions. So every edge here, at BOTH
granularities, is DISTANCE-BASED: inferred from geometry, not read off
a chemistry record -- except the residue graph's "sequence" edge
scheme, which uses real covalent (peptide-bond) adjacency instead, kept
deliberately for contrast.
"""

from __future__ import annotations

import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser

from graph_types import Graph

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PDB_DIR = REPO_ROOT / "data" / "pdb"
STRUCTURE_URL = "https://files.rcsb.org/download/{pdb_id}.cif"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
UNKNOWN_AA_INDEX = len(AMINO_ACIDS)
NUM_AA_SLOTS = len(AMINO_ACIDS) + 1

PROTEIN_ELEMENTS = ("C", "N", "O", "S")
PROTEIN_ELEMENT_TO_INDEX = {element: index for index, element in enumerate(PROTEIN_ELEMENTS)}
OTHER_PROTEIN_ELEMENT_INDEX = len(PROTEIN_ELEMENTS)
NUM_PROTEIN_ELEMENT_SLOTS = len(PROTEIN_ELEMENTS) + 1

# KEY SKILL: distance-based edges -- CA-CA contact threshold, same
# convention `22_protein_structure_data` used for its contact map, now
# used directly as a graph's actual edge-construction rule
RESIDUE_CONTACT_THRESHOLD = 8.0

# KEY SKILL: distance-based edges -- verified directly against 1CRN's
# real geometry: every true covalent bond's nearest-neighbor distance is
# <= 1.82A (a C-S bond), and the structure's 3 disulfide bridges (S-S)
# sit at 2.00-2.05A; the very next-closest non-bonded pairs (atoms two
# bonds apart, not directly bonded) don't start appearing until >2.1A.
# 2.1A is therefore a threshold that captures every real bond INCLUDING
# the disulfides, without yet picking up any non-bonded pair.
ATOM_BOND_THRESHOLD = 2.1


@dataclass
class ParsedResidue:
    residue_number: int
    amino_acid: str  # one-letter
    ca_coordinate: np.ndarray  # (3,)
    atoms: list[tuple[str, np.ndarray]]  # EVERY atom in this residue (backbone + sidechain): (element, xyz)


@dataclass
class ParsedStructure:
    pdb_id: str
    residues: list[ParsedResidue]


def _download_structure(
    pdb_id: str,
) -> Path:
    PDB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    path = PDB_DIR / f"{pdb_id.lower()}.cif"
    if not path.exists():
        urllib.request.urlretrieve(
            STRUCTURE_URL.format(pdb_id=pdb_id),
            path,
        )
    return path


def parse_structure(
    pdb_id: str,
) -> ParsedStructure:
    path = _download_structure(pdb_id)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parser = MMCIFParser(QUIET=True)
        model = parser.get_structure(pdb_id, path)[0]

    chain = next(iter(model))

    residues: list[ParsedResidue] = []
    for residue in chain:
        if residue.id[0] != " " or "CA" not in residue:
            continue  # skip hetero groups/water and any residue missing a resolved CA

        atoms = [(atom.element, atom.coord) for atom in residue]
        residues.append(
            ParsedResidue(
                residue_number=residue.id[1],
                amino_acid=THREE_TO_ONE.get(residue.resname, "X"),
                ca_coordinate=residue["CA"].coord,
                atoms=atoms,
            ),
        )

    return ParsedStructure(
        pdb_id=pdb_id,
        residues=residues,
    )


def _residue_node_features(
    structure: ParsedStructure,
) -> np.ndarray:
    indices = [
        AA_TO_INDEX.get(residue.amino_acid, UNKNOWN_AA_INDEX)
        for residue in structure.residues
    ]
    features = np.zeros(
        (len(structure.residues), NUM_AA_SLOTS),
        dtype=np.float32,
    )
    features[np.arange(len(indices)), indices] = 1.0
    return features


def _distance_edges(
    coordinates: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """KEY SKILL: geometric features -- builds BOTH the edge list and a
    (distance, unit-direction-vector) feature for every kept edge, in
    one vectorized pass. Only possible because real 3D coordinates
    exist; `molecular_graphs.py`'s SMILES-only bonds have nothing
    equivalent to compute.
    """
    diff = coordinates[:, None, :] - coordinates[None, :, :]  # (N, N, 3)
    distance = np.sqrt((diff**2).sum(axis=-1))  # (N, N)

    n = coordinates.shape[0]
    within_threshold = (distance < threshold) & ~np.eye(n, dtype=bool)
    sources, destinations = np.nonzero(within_threshold)

    edge_distance = distance[sources, destinations]
    unit_direction = diff[sources, destinations] / edge_distance[:, None]
    edge_features = np.concatenate(
        [edge_distance[:, None], unit_direction],
        axis=1,
    ).astype(np.float32)

    edge_index = np.array(
        [sources, destinations],
        dtype=np.int64,
    )
    return edge_index, edge_features


def build_residue_graph(
    structure: ParsedStructure,
    edge_scheme: str,
) -> Graph:
    """edge_scheme: "sequence" (real peptide-bond adjacency -- a
    TOPOLOGICAL edge, correctly skipped across a numbering gap, no
    geometric features attached, same spirit as a molecular bond) or
    "distance" (CA-CA contact within `RESIDUE_CONTACT_THRESHOLD`, a
    purely GEOMETRIC criterion that also carries distance + direction as
    edge features).
    """
    coordinates = np.stack([residue.ca_coordinate for residue in structure.residues])
    node_labels = [
        f"{residue.amino_acid}{residue.residue_number}" for residue in structure.residues
    ]

    if edge_scheme == "sequence":
        residue_numbers = [residue.residue_number for residue in structure.residues]
        sources, destinations = [], []
        for i in range(len(residue_numbers) - 1):
            if residue_numbers[i + 1] == residue_numbers[i] + 1:  # skip across numbering gaps
                sources += [i, i + 1]
                destinations += [i + 1, i]
        edge_index = np.array(
            [sources, destinations],
            dtype=np.int64,
        )
        edge_features = None
    elif edge_scheme == "distance":
        edge_index, edge_features = _distance_edges(
            coordinates,
            RESIDUE_CONTACT_THRESHOLD,
        )
    else:
        raise ValueError(f"unknown edge_scheme: {edge_scheme!r}")

    return Graph(
        name=f"{structure.pdb_id} residue graph ({edge_scheme} edges)",
        node_labels=node_labels,
        node_features=_residue_node_features(structure),
        edge_index=edge_index,
        edge_features=edge_features,
        node_coordinates=coordinates,
    )


def build_atomic_graph(
    structure: ParsedStructure,
) -> Graph:
    elements: list[str] = []
    coordinate_rows: list[np.ndarray] = []
    for residue in structure.residues:
        for element, coordinate in residue.atoms:
            elements.append(element)
            coordinate_rows.append(coordinate)
    coordinates = np.stack(coordinate_rows)

    element_indices = [
        PROTEIN_ELEMENT_TO_INDEX.get(element, OTHER_PROTEIN_ELEMENT_INDEX)
        for element in elements
    ]
    node_features = np.zeros(
        (len(elements), NUM_PROTEIN_ELEMENT_SLOTS),
        dtype=np.float32,
    )
    node_features[np.arange(len(elements)), element_indices] = 1.0

    edge_index, edge_features = _distance_edges(
        coordinates,
        ATOM_BOND_THRESHOLD,
    )

    return Graph(
        name=f"{structure.pdb_id} atomic graph (distance-based edges)",
        node_labels=elements,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        node_coordinates=coordinates,
    )
