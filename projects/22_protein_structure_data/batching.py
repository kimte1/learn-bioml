"""
Pads a list of different-length structures into one fixed-size batch
tensor plus a mask -- the same core idea as `tokenize_batch`'s
`key_padding_mask` in `14_attention`-`20_bert_pretrain`, applied to
continuous 3D coordinates and 2D distance maps instead of discrete token
ids.

MASK CONVENTION: here, `True` means "this is a real residue" --
the OPPOSITE of `16_transformer_train_encoder`'s `key_padding_mask`
(`True` = padding). That earlier convention existed specifically to match
`nn.TransformerEncoderLayer`'s `src_key_padding_mask` argument; there's
no such API to match here, so this project uses the more directly
readable polarity instead. Worth remembering if these masks are ever
passed into that kind of layer later -- they would need inverting first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_load_prep import Structure


@dataclass
class BatchedStructures:
    pdb_ids: list[str]
    coordinates: np.ndarray  # (B, L_max, 3), zero-padded
    mask: np.ndarray  # (B, L_max) bool, True = real residue
    residue_numbers: list[list[int]]  # length B, each entry length L_i (not padded)


@dataclass
class BatchedDistanceMaps:
    pdb_ids: list[str]
    distance_maps: np.ndarray  # (B, L_max, L_max), zero-padded
    mask: np.ndarray  # (B, L_max) bool, True = real residue -- same mask broadcasts over both axes


def collate_structures(
    structures: list[Structure],
) -> BatchedStructures:
    """
    # KEY SKILL: batching variable-size structures -- real structures
    almost never share a length (here: 46, 76, 129, 263 residues), so a
    single `(B, L, 3)` tensor can only exist once every structure is
    padded out to the batch's longest one, with a mask recording which
    positions are real so padded "residues" at the origin never get
    mistaken for real geometry downstream.
    """
    lengths = [
        len(structure.residue_numbers) for structure in structures
    ]
    max_len = max(lengths)
    batch_size = len(structures)

    coordinates = np.zeros(
        (batch_size, max_len, 3),
    )
    mask = np.zeros(
        (batch_size, max_len),
        dtype=bool,
    )

    for i, structure in enumerate(structures):
        length = lengths[i]
        coordinates[i, :length] = structure.ca_coordinates
        mask[i, :length] = True

    return BatchedStructures(
        pdb_ids=[structure.pdb_id for structure in structures],
        coordinates=coordinates,
        mask=mask,
        residue_numbers=[structure.residue_numbers for structure in structures],
    )


def collate_distance_maps(
    structures: list[Structure],
    distance_maps: list[np.ndarray],
) -> BatchedDistanceMaps:
    """Same padding idea as `collate_structures`, one rank higher: each
    structure's own `(L_i, L_i)` distance map gets placed in the
    top-left corner of a shared `(L_max, L_max)` block, so the same 1D
    mask used for coordinates also correctly marks real vs. padded ROWS
    AND COLUMNS here (a padded position is invalid on either axis).
    """
    lengths = [
        len(structure.residue_numbers) for structure in structures
    ]
    max_len = max(lengths)
    batch_size = len(structures)

    padded = np.zeros(
        (batch_size, max_len, max_len),
    )
    mask = np.zeros(
        (batch_size, max_len),
        dtype=bool,
    )

    for i, (structure, distance_map) in enumerate(zip(structures, distance_maps)):
        length = lengths[i]
        padded[i, :length, :length] = distance_map
        mask[i, :length] = True

    return BatchedDistanceMaps(
        pdb_ids=[structure.pdb_id for structure in structures],
        distance_maps=padded,
        mask=mask,
    )
