"""
Visualizations for the structure-dataset pipeline. Same conventions as
earlier projects: `Blues` for non-negative magnitudes (distances), a
binary colormap for boolean contact/mask images, `tab10` for the small
set of named structures, independently-scaled panels whenever two
plotted quantities live on very different numeric ranges.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from cropping import Crop
from data_load_prep import ResidueAlignment
from data_load_prep import Structure

HEATMAP_CMAP = "Blues"
CATEGORICAL_CMAP = "tab10"


def plot_structure_size_comparison(
    structures: list[Structure],
    path: Path,
) -> None:
    pdb_ids = [structure.pdb_id for structure in structures]
    lengths = [len(structure.residue_numbers) for structure in structures]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(
        pdb_ids,
        lengths,
        color="tab:blue",
    )
    for bar, length in zip(bars, lengths):
        ax.annotate(
            str(length),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Residues resolved in the crystal structure")
    ax.set_title("Why batching needs padding: 4 real, different lengths")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distance_and_contact_map(
    distance_map: np.ndarray,
    contact_map: np.ndarray,
    title: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        constrained_layout=True,
    )

    im0 = axes[0].imshow(
        distance_map,
        cmap=HEATMAP_CMAP,
    )
    axes[0].set_title("CA-CA distance (angstrom)")
    axes[0].set_xlabel("Residue index")
    axes[0].set_ylabel("Residue index")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    axes[1].imshow(
        contact_map,
        cmap="Greys",
    )
    axes[1].set_title("Contact map (< 5A)")
    axes[1].set_xlabel("Residue index")
    axes[1].set_ylabel("Residue index")

    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residue_indexing(
    structure: Structure,
    alignment: ResidueAlignment,
    missing_residues: list[int],
    path: Path,
) -> None:
    pdb_resnums = sorted(alignment.position_map.keys())
    ref_positions = [alignment.position_map[r] for r in pdb_resnums]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        pdb_resnums,
        ref_positions,
        color="tab:blue",
        marker=".",
        linestyle="none",
        label="aligned residues",
    )

    mismatch_pdb = [m[0] for m in alignment.mismatches]
    mismatch_ref = [m[1] for m in alignment.mismatches]
    ax.scatter(
        mismatch_pdb,
        mismatch_ref,
        color="tab:red",
        s=80,
        zorder=5,
        label=f"real sequence mismatches (n={len(alignment.mismatches)})",
    )
    for pdb_resnum, ref_position, pdb_aa, ref_aa in alignment.mismatches:
        ax.annotate(
            f"  PDB {pdb_aa}{pdb_resnum} vs. ref {ref_aa}{ref_position}",
            (pdb_resnum, ref_position),
            color="tab:red",
            fontsize=8,
        )

    for missing in missing_residues:
        ax.axvline(
            missing,
            color="gray",
            linestyle="--",
            alpha=0.6,
        )
    if missing_residues:
        ax.axvline(
            missing_residues[0],
            color="gray",
            linestyle="--",
            alpha=0.6,
            label=f"missing PDB residues ({missing_residues})",
        )

    ax.set_xlabel(f"{structure.pdb_id} PDB residue number")
    ax.set_ylabel("Reference (ProteinGym/21_msa) position, 1-286")
    ax.set_title("Residue indexing: PDB numbering vs. reference numbering")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ligand_contacts(
    structure: Structure,
    contact_residues: list[int],
    path: Path,
) -> None:
    is_contact = np.array(
        [resnum in contact_residues for resnum in structure.residue_numbers],
    )

    fig, ax = plt.subplots(figsize=(10, 1.8))
    ax.imshow(
        is_contact[None, :],
        cmap="Reds",
        aspect="auto",
        extent=[structure.residue_numbers[0], structure.residue_numbers[-1], 0, 1],
    )
    ax.set_yticks([])
    ax.set_xlabel(f"{structure.pdb_id} PDB residue number")
    ligand_names = ", ".join(sorted({name for name, _ in structure.hetero_atoms}))
    ax.set_title(
        f"Residues within 5A of bound ligand ({ligand_names}) -- n={len(contact_residues)}",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_batch_padding(
    mask: np.ndarray,
    pdb_ids: list[str],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(
        mask,
        cmap="Greys",
        aspect="auto",
    )
    ax.set_yticks(
        range(len(pdb_ids)),
        pdb_ids,
    )
    ax.set_xlabel("Padded residue index (L_max)")
    # Greys colormap: higher value (mask=True, real residue) renders dark
    ax.set_title("Batch padding mask: dark = real residue, white = padding")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_crop_selection(
    structure: Structure,
    contiguous: Crop,
    spatial: Crop,
    path: Path,
) -> None:
    """A 1D strip per crop type (same idea as `plot_ligand_contacts`),
    making the key visual difference obvious: the contiguous crop is one
    solid block, the spatial crop is scattered along the sequence axis
    (since it's selected by 3D proximity, not sequence position) even
    though it's just as compact in actual 3D space.
    """
    all_resnums = structure.residue_numbers
    extent = [all_resnums[0], all_resnums[-1], 0, 1]
    cmap = plt.get_cmap("Reds")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 3.2),
        constrained_layout=True,
    )

    for ax, crop, label in [
        (axes[0], contiguous, "contiguous crop"),
        (axes[1], spatial, "spatial crop"),
    ]:
        is_kept = np.array([resnum in set(crop.residue_numbers) for resnum in all_resnums])
        ax.imshow(
            is_kept[None, :],
            cmap=cmap,
            aspect="auto",
            extent=extent,
        )
        ax.set_yticks([])
        ax.set_title(f"{label} (n={len(crop.residue_numbers)})")

    legend_handles = [
        Patch(facecolor=cmap(1.0), edgecolor="black", label="kept in crop"),
        Patch(facecolor=cmap(0.0), edgecolor="black", label="not kept"),
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )

    axes[-1].set_xlabel(f"{structure.pdb_id} PDB residue number")

    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sequence_identity_clusters(
    structures: list[Structure],
    identity_matrix: np.ndarray,
    clusters: dict[str, int],
    path: Path,
) -> None:
    pdb_ids = [structure.pdb_id for structure in structures]
    labels = [f"{pdb_id}\n(cluster {clusters[pdb_id]})" for pdb_id in pdb_ids]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(
        identity_matrix,
        cmap=HEATMAP_CMAP,
        vmin=0,
        vmax=1,
    )
    ax.set_xticks(range(len(pdb_ids)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(pdb_ids)), labels)

    for i in range(len(pdb_ids)):
        for j in range(len(pdb_ids)):
            ax.text(
                j,
                i,
                f"{identity_matrix[i, j]:.0%}",
                ha="center",
                va="center",
                color="white" if identity_matrix[i, j] > 0.5 else "black",
            )

    fig.colorbar(im, ax=ax, label="Sequence identity", fraction=0.046)
    ax.set_title("Pairwise sequence identity -- every structure is its own cluster")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spatial_clusters(
    structure: Structure,
    labels: np.ndarray,
    path: Path,
) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")

    cmap = plt.colormaps[CATEGORICAL_CMAP]
    coords = structure.ca_coordinates
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=[cmap(label) for label in labels],
        s=20,
    )
    ax.set_xlabel("x (angstrom)")
    ax.set_ylabel("y (angstrom)")
    ax.set_zlabel("z (angstrom)")
    ax.set_title(f"{structure.pdb_id}: spatial k-means clusters (n={labels.max() + 1})")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_filtering_summary(
    report: dict[str, dict[str, bool]],
    path: Path,
) -> None:
    pdb_ids = list(report.keys())
    criteria = list(next(iter(report.values())).keys())

    grid = np.array(
        [[report[pdb_id][criterion] for criterion in criteria] for pdb_id in pdb_ids],
    )

    # pass/fail is a 2-value STATUS, not a continuous/diverging quantity,
    # so a plain 2-color categorical map is used rather than a
    # diverging colormap (which would misleadingly imply a continuous
    # scale between the two states)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(
        grid,
        cmap=ListedColormap(["lightcoral", "lightgreen"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set_xticks(range(len(criteria)), criteria)
    ax.set_yticks(range(len(pdb_ids)), pdb_ids)

    for i in range(len(pdb_ids)):
        for j in range(len(criteria)):
            ax.text(
                j,
                i,
                "pass" if grid[i, j] else "FAIL",
                ha="center",
                va="center",
            )

    ax.set_title("Filtering report -- per-criterion pass/fail")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
