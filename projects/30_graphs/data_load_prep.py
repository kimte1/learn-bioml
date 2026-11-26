"""
Loads Delaney (2004) ESOL dataset (1128 molecules, SMILES + measured aqueous
solubility) but this time the only input is the raw SMILES string.

Build molecular graph (atoms as nodes, bonds as edges) from SMILES
The model learns its own representation via message passing rather than relying on descriptors someone else already computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ESOL_CSV = REPO_ROOT / "data" / "delaney-esol.csv"

# every element actually present in this dataset (verified directly --
# 1128/1128 SMILES parse cleanly with RDKit, no unknown elements), plus
# one "OTHER" bucket so an unexpected element elsewhere would degrade
# gracefully instead of crashing
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

# every RDKit bond type actually present in this dataset
BOND_TYPES = (
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "AROMATIC",
)
BOND_TYPE_TO_INDEX = {bond_type: index for index, bond_type in enumerate(BOND_TYPES)}

MAX_DEGREE_BUCKET = 4  # degrees/H-counts above this are clamped into one "4+" bucket
NUM_DEGREE_SLOTS = MAX_DEGREE_BUCKET + 1

# KEY SKILL: nodes -- element identity, local connectivity (degree),
# charge, aromaticity/ring membership, and attached-hydrogen count, all
# read directly off the RDKit atom object
NODE_FEATURE_DIM = NUM_ELEMENT_SLOTS + NUM_DEGREE_SLOTS + 1 + 1 + 1 + NUM_DEGREE_SLOTS

# KEY SKILL: edges -- bond order plus two structural flags
EDGE_FEATURE_DIM = len(BOND_TYPES) + 1 + 1


@dataclass
class MoleculeGraph:
    compound_id: str
    smiles: str
    atom_features: np.ndarray  # (num_atoms, NODE_FEATURE_DIM)
    bond_index: np.ndarray  # (2, num_directed_edges) -- each real bond stored in BOTH directions
    bond_features: np.ndarray  # (num_directed_edges, EDGE_FEATURE_DIM)
    target: float  # measured log solubility (mol/L)


def _one_hot(
    index: int,
    num_slots: int,
) -> list[float]:
    vector = [0.0] * num_slots
    vector[index] = 1.0
    return vector


def _atom_features(
    atom: Chem.Atom,
) -> list[float]:
    element_index = ELEMENT_TO_INDEX.get(
        atom.GetSymbol(),
        OTHER_ELEMENT_INDEX,
    )
    degree_index = min(
        atom.GetDegree(),
        MAX_DEGREE_BUCKET,
    )
    num_h_index = min(
        atom.GetTotalNumHs(),
        MAX_DEGREE_BUCKET,
    )

    return (
        _one_hot(element_index, NUM_ELEMENT_SLOTS)
        + _one_hot(degree_index, NUM_DEGREE_SLOTS)
        + [float(atom.GetFormalCharge())]
        + [float(atom.GetIsAromatic())]
        + [float(atom.IsInRing())]
        + _one_hot(num_h_index, NUM_DEGREE_SLOTS)
    )


def _bond_features(
    bond: Chem.Bond,
) -> list[float]:
    bond_type_index = BOND_TYPE_TO_INDEX[str(bond.GetBondType())]
    return (
        _one_hot(bond_type_index, len(BOND_TYPES))
        + [float(bond.GetIsConjugated())]
        + [float(bond.IsInRing())]
    )


def _mol_to_graph(
    compound_id: str,
    smiles: str,
    target: float,
) -> MoleculeGraph:

    mol = Chem.MolFromSmiles(smiles)

    atom_features = np.array(
        [_atom_features(atom) for atom in mol.GetAtoms()],
        dtype=np.float32,
    )

    # KEY SKILL: adjacency -- an undirected bond becomes TWO directed
    # edges (i->j and j->i, same bond features both ways), since message
    # passing needs every node to receive messages from every neighbor
    sources: list[int] = []
    destinations: list[int] = []
    bond_feature_rows: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = _bond_features(bond)

        sources += [i, j]
        destinations += [j, i]
        bond_feature_rows += [features, features]

    bond_index = np.array(
        [sources, destinations],
        dtype=np.int64,
    )
    bond_features = np.array(
        bond_feature_rows,
        dtype=np.float32,
    ).reshape(-1, EDGE_FEATURE_DIM)

    return MoleculeGraph(
        compound_id=compound_id,
        smiles=smiles,
        atom_features=atom_features,
        bond_index=bond_index,
        bond_features=bond_features,
        target=target,
    )


def load_molecule_graphs(
    path: Path = ESOL_CSV,
) -> list[MoleculeGraph]:
    df = pd.read_csv(path)

    graphs = []
    for _, row in df.iterrows():
        graph = _mol_to_graph(
            compound_id=row["Compound ID"],
            smiles=row["smiles"].strip(),
            target=float(row["measured log solubility in mols per litre"]),
        )
        graphs.append(graph)

    return graphs


def split_molecule_graphs(
    graphs: list[MoleculeGraph],
    seed: int = 0,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[
    list[MoleculeGraph],
    list[MoleculeGraph],
    list[MoleculeGraph],
]:
    test_frac = 1.0 - train_frac - val_frac
    train_graphs, remaining = train_test_split(
        graphs,
        test_size=val_frac + test_frac,
        random_state=seed,
    )
    val_graphs, test_graphs = train_test_split(
        remaining,
        test_size=test_frac / (val_frac + test_frac),
        random_state=seed,
    )
    return train_graphs, val_graphs, test_graphs
