"""
Cropping is picking a fixed-size subset of residues to actually use for later processing.

The reason for cropping is that if a sequence is very long compared to other sequences in a particular batch, then padding may not be memory efficient. 

So cropping makes training really long sequence possible

The 2 crop strategies here is what AlphaFold does
- contiguous crop (get a random contiguous of window of residues to crop)
- spatial crop (pick a random residue, then take its crop_size nearest neighbors in 3d space (CA-CA distance)) and keep those
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_load_prep import Structure


@dataclass
class Crop:
    pdb_id: str
    kept_indices: np.ndarray  # indices into the ORIGINAL structure's arrays, ascending order
    residue_numbers: list[int]
    sequence: str
    ca_coordinates: np.ndarray  # (crop_size, 3)


def _build_crop(
    structure: Structure,
    kept_indices: np.ndarray,
) -> Crop:
    kept_indices = np.sort(kept_indices)
    return Crop(
        pdb_id=structure.pdb_id,
        kept_indices=kept_indices,
        residue_numbers=[structure.residue_numbers[i] for i in kept_indices],
        sequence="".join(structure.sequence[i] for i in kept_indices),
        ca_coordinates=structure.ca_coordinates[kept_indices],
    )


def contiguous_crop(
    structure: Structure,
    crop_size: int,
    seed: int = 0,
) -> Crop:
    """
    # KEY SKILL: cropping
    """
    length = len(structure.residue_numbers)
    if crop_size >= length:
        return _build_crop(
            structure, 
            np.arange(length),
        )

    rng = np.random.default_rng(seed)
    start = rng.integers(0, length - crop_size + 1)
    return _build_crop(
        structure,
        np.arange(start, start + crop_size),
    )


def spatial_crop(
    structure: Structure,
    crop_size: int,
    seed: int = 0,
) -> Crop:
    """
    # KEY SKILL: cropping -- pick one random residue, then take its
    `crop_size - 1` nearest neighbors by CA-CA distance
    """
    length = len(structure.residue_numbers)
    if crop_size >= length:
        return _build_crop(structure, np.arange(length))

    rng = np.random.default_rng(seed)
    seed_index = rng.integers(0, length)

    distances = np.linalg.norm(
        structure.ca_coordinates - structure.ca_coordinates[seed_index],
        axis=1,
    )
    nearest_indices = np.argsort(distances)[:crop_size]
    return _build_crop(structure, nearest_indices)
