"""
KEY SKILL: molecular graphs -- a small molecule's graph is naturally
ATOMIC (every atom is its own node; there's no coarser grouping the way
a protein has residues), and its edges come directly from SMILES' own
declared bonds, a purely TOPOLOGICAL graph. A bare SMILES string carries
NO 3D information at all -- that's the main contrast this week draws
against `protein_graphs.py`'s graphs, which have real coordinates and
can carry genuine geometric edge features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Atom
from rdkit.Chem.rdchem import Bond

from graph_types import Graph

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ESOL_CSV = REPO_ROOT / "data" / "delaney-esol.csv"

ELEMENTS = (
    "C",
    "O",
    "N",
    "Cl",
    "S",
    "F",
    "Br",
    "P",
    "I",
)
ELEMENT_TO_INDEX = {element: index for index, element in enumerate(ELEMENTS)}
OTHER_ELEMENT_INDEX = len(ELEMENTS)
NUM_ELEMENT_SLOTS = len(ELEMENTS) + 1

BOND_TYPES = (
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "AROMATIC",
)
BOND_TYPE_TO_INDEX = {bond_type: index for index, bond_type in enumerate(BOND_TYPES)}

MAX_DEGREE_BUCKET = 4
NUM_DEGREE_SLOTS = MAX_DEGREE_BUCKET + 1


def _one_hot(
    index: int,
    num_slots: int,
) -> np.ndarray:
    vector = np.zeros(num_slots, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _atom_features(
    atom: Atom,
    feature_scheme: str,
) -> np.ndarray:
    element_index = ELEMENT_TO_INDEX.get(
        atom.GetSymbol(),
        OTHER_ELEMENT_INDEX,
    )
    element_onehot = _one_hot(
        element_index,
        NUM_ELEMENT_SLOTS,
    )

    if feature_scheme == "minimal":
        return element_onehot

    degree_onehot = _one_hot(
        min(atom.GetDegree(), MAX_DEGREE_BUCKET),
        NUM_DEGREE_SLOTS,
    )
    num_h_onehot = _one_hot(
        min(atom.GetTotalNumHs(), MAX_DEGREE_BUCKET),
        NUM_DEGREE_SLOTS,
    )
    extra = np.array(
        [
            float(atom.GetFormalCharge()),
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [element_onehot, degree_onehot, extra, num_h_onehot],
    )


def _bond_features(
    bond: Bond,
) -> np.ndarray:
    bond_type_index = BOND_TYPE_TO_INDEX[str(bond.GetBondType())]
    return _one_hot(
        bond_type_index,
        len(BOND_TYPES),
    )


def build_molecular_graph(
    name: str,
    smiles: str,
    feature_scheme: str = "rich",
) -> Graph:

    mol = Chem.MolFromSmiles(smiles)

    node_features = np.stack(
        [_atom_features(atom, feature_scheme) for atom in mol.GetAtoms()],
    )
    node_labels = [atom.GetSymbol() for atom in mol.GetAtoms()]

    sources: list[int] = []
    destinations: list[int] = []
    bond_feature_rows: list[np.ndarray] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = _bond_features(bond)

        sources += [i, j]
        destinations += [j, i]
        bond_feature_rows += [features, features]

    return Graph(
        name=name,
        node_labels=node_labels,
        node_features=node_features,
        edge_index=np.array(
            [sources, destinations],
            dtype=np.int64,
        ),
        edge_features=np.stack(bond_feature_rows),
        node_coordinates=None,  # KEY POINT: SMILES carries no 3D geometry at all
    )
