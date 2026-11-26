"""
Take 4 single-chain CIF files and converts to representations that an ML model can use:
    - per-residue coordinates (CA-only)
    - mask for what's missing
    - pairwise distance/contact maps
    - clusters (structures by sequence identity, residues by spatial proximity)
    - crops (contiguous and spatial, AlphaFold-style)
    - a filtered, curated subset by resolution/length/completeness
    - padded batch across structures of different lengths.

Some of this is the type of preprocessing done in AF2 and AF3
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from batching import collate_distance_maps
from batching import collate_structures
from clustering import cluster_by_identity
from clustering import pairwise_sequence_identity
from clustering import spatial_kmeans_clusters
from cropping import contiguous_crop
from cropping import spatial_crop
from data_load_prep import STRUCTURES
from data_load_prep import TEM1_WILD_TYPE_SEQUENCE
from data_load_prep import align_to_reference
from data_load_prep import load_all_structures
from filtering import FilterCriteria
from filtering import apply_filters
from plotting import plot_batch_padding
from plotting import plot_crop_selection
from plotting import plot_distance_and_contact_map
from plotting import plot_filtering_summary
from plotting import plot_ligand_contacts
from plotting import plot_residue_indexing
from plotting import plot_sequence_identity_clusters
from plotting import plot_spatial_clusters
from plotting import plot_structure_size_comparison
from structural_features import compute_contact_map
from structural_features import compute_distance_map
from structural_features import find_missing_residues
from structural_features import ligand_contact_residues

FIG_DIR = Path(__file__).resolve().parent / "fig"


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    structures_by_id = load_all_structures()

    structures = list(structures_by_id.values())

    print("=== structures loaded ===")
    for pdb_id, structure in structures_by_id.items():
        missing = find_missing_residues(structure.residue_numbers)
        print(
            f"  {pdb_id}: {len(structure.residue_numbers)} residues "
            f"({structure.residue_numbers[0]}-{structure.residue_numbers[-1]}), "
            f"{len(missing)} missing {missing}, "
            f"{len(structure.hetero_atoms)} ligand groups",
        )

    plot_structure_size_comparison(
        structures,
        FIG_DIR / "structure_size_comparison.png",
    )

    # per-structure distance/contact maps (KEY SKILL: structural
    # representations) -- computed and plotted for every structure, since
    # the idea is the same regardless of size
    distance_maps = []
    print("\n=== distance / contact maps ===")
    for structure in structures:
        distance_map = compute_distance_map(structure.ca_coordinates)
        contact_map = compute_contact_map(distance_map)
        distance_maps.append(distance_map)

        n_contacts = contact_map.sum()
        print(f"  {structure.pdb_id}: {n_contacts} contact pairs (< 8A, including self-pairs)")

        plot_distance_and_contact_map(
            distance_map,
            contact_map,
            f"{structure.pdb_id} ({STRUCTURES[structure.pdb_id]})",
            FIG_DIR / f"distance_contact_{structure.pdb_id.lower()}.png",
        )

    # TEM-1-specific: residue indexing + ligand contacts (KEY SKILL:
    # residue indexing, chain/interface annotations)
    tem1 = structures_by_id["1BTL"]
    alignment = align_to_reference(
        tem1,
        TEM1_WILD_TYPE_SEQUENCE,
    )
    missing = find_missing_residues(tem1.residue_numbers)

    print("\n=== 1BTL / TEM-1 residue indexing ===")
    print(f"  {len(alignment.position_map)} residues aligned to the 286-residue reference")
    print(f"  {len(alignment.mismatches)} real sequence mismatches: {alignment.mismatches}")
    print(f"  missing PDB residues (no resolved coordinates): {missing}")

    plot_residue_indexing(
        tem1,
        alignment,
        missing,
        FIG_DIR / "residue_indexing.png",
    )

    contact_residues = ligand_contact_residues(tem1)
    print(f"\n=== 1BTL ligand contacts === \n  {len(contact_residues)} residues within 5A of bound SO4: {contact_residues}")

    plot_ligand_contacts(
        tem1,
        contact_residues,
        FIG_DIR / "ligand_contacts.png",
    )

    # batching variable-size structures (KEY SKILL)
    batch = collate_structures(structures)
    batched_distance_maps = collate_distance_maps(structures, distance_maps)
    print(
        f"\n=== batched {len(structures)} structures ===\n"
        f"  coordinates: {batch.coordinates.shape}, mask: {batch.mask.shape}, "
        f"distance maps: {batched_distance_maps.distance_maps.shape}",
    )

    plot_batch_padding(
        batch.mask,
        batch.pdb_ids,
        FIG_DIR / "batch_padding.png",
    )

    # cropping (KEY SKILL): only meaningful on 1BTL, the one structure
    # longer than a reasonable crop size
    contiguous = contiguous_crop(tem1, crop_size=100)
    spatial = spatial_crop(tem1, crop_size=100)

    print(
        f"\n=== 1BTL cropping (crop_size=100 of {len(tem1.residue_numbers)}) ===\n"
        f"  contiguous: residues {contiguous.residue_numbers[0]}-{contiguous.residue_numbers[-1]}\n"
        f"  spatial: residues {spatial.residue_numbers[:5]}...{spatial.residue_numbers[-5:]} "
        f"(scattered in sequence, compact in 3D)",
    )
    plot_crop_selection(
        tem1,
        contiguous,
        spatial,
        FIG_DIR / "crop_selection.png",
    )

    # clustering (KEY SKILL), both senses
    identity_matrix = pairwise_sequence_identity(structures)
    clusters = cluster_by_identity(structures)
    print(
        f"\n=== dataset clustering (sequence identity) ===\n"
        f"  pairwise identity range: {identity_matrix[identity_matrix < 1].min():.1%}"
        f"-{identity_matrix[identity_matrix < 1].max():.1%}\n"
        f"  clusters: {clusters} (4 singletons -- correct for 4 unrelated real proteins)",
    )
    plot_sequence_identity_clusters(
        structures,
        identity_matrix,
        clusters,
        FIG_DIR / "sequence_identity_clusters.png",
    )

    spatial_labels = spatial_kmeans_clusters(tem1, n_clusters=4)
    print(
        f"\n=== 1BTL spatial clustering (k-means, k=4) ===\n"
        f"  cluster sizes: {np.bincount(spatial_labels).tolist()}",
    )
    plot_spatial_clusters(
        tem1,
        spatial_labels,
        FIG_DIR / "spatial_clusters.png",
    )

    # filtering (KEY SKILL): a real curation pass over the dataset
    kept, filter_report = apply_filters(
        structures,
        FilterCriteria(),
    )
    print(
        f"\n=== filtering (max_resolution=1.8, min_length=50, max_missing_fraction=0.05) ===\n"
        f"  kept: {[s.pdb_id for s in kept]}\n"
        f"  report: {filter_report}",
    )
    plot_filtering_summary(
        filter_report,
        FIG_DIR / "filtering_summary.png",
    )
    


if __name__ == "__main__":
    main()
