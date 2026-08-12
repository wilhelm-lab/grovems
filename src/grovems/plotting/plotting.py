from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib_venn import venn2

from ..postprocess import postprocess

logger = logging.getLogger(__name__)

_TARGET_COLOR, _DECOY_COLOR = "#2a78d6", "#c44e52"
_SIDE_CMAP = {"database": "Blues", "denovo": "Oranges"}
_PERCOLATOR_THRESHOLD = 0.0


def _plot_shared_venn(shared_scan_count: int, shared_psm_count: int, out_path: Path) -> None:
    """Venn: shared scans (both engines hit this scan) vs shared PSM (they agree).

    shared PSM is a strict subset of shared scans (agreement requires co-occurrence),
    so the "only in shared PSM" region is always 0.
    """
    n_conflict = shared_scan_count - shared_psm_count
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    venn2(
        subsets=(n_conflict, 0, shared_psm_count),
        set_labels=("shared scans", "shared PSM"),
        set_colors=(postprocess.DETECTION_LEVEL_COLORS["shared"], postprocess.DETECTION_LEVEL_COLORS["database"]),
        ax=ax,
    )
    frac = shared_psm_count / shared_scan_count if shared_scan_count else float("nan")
    ax.set_title(
        f"Shared scans vs shared PSM\nshared scans={shared_scan_count:,}, "
        f"shared PSM={shared_psm_count:,} ({frac:.1%} agree)"
    )
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _plot_levenshtein_distribution(distances: np.ndarray, out_path: Path) -> None:
    """Histogram of PSA_LEVENSHTEIN (database vs de novo sequence, shared PSMs).

    Log-scale y-axis: distance-0 (agreement) dominates the count and would otherwise
    flatten the rest of the histogram.
    """
    bins = np.arange(0, int(distances.max()) + 2) - 0.5
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(distances, bins=bins, color=postprocess.DETECTION_LEVEL_COLORS["shared"])
    ax.set_yscale("log")
    ax.set_xlabel("Levenshtein distance (database vs de novo sequence)")
    ax.set_ylabel("count (log scale)")
    ax.set_title(f"Levenshtein distance, shared PSMs (n={len(distances):,})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _plot_peptide_length_distribution(database_lengths: np.ndarray, denovo_lengths: np.ndarray, out_path: Path) -> None:
    """Peptide length distribution, database vs de novo, all PSMs (no filtering)."""
    lo = int(min(database_lengths.min(), denovo_lengths.min()))
    hi = int(max(database_lengths.max(), denovo_lengths.max()))
    bins = np.arange(lo, hi + 2) - 0.5
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, values in (("database", database_lengths), ("denovo", denovo_lengths)):
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            lw=1.8,
            color=postprocess.DETECTION_LEVEL_COLORS[name],
            label=f"{name} (n={len(values):,}, median {np.median(values):.0f})",
        )
    ax.set_xlabel("peptide length")
    ax.set_ylabel("density")
    ax.set_title("Peptide length distribution -- database vs de novo, all PSMs")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _plot_peptide_length_distribution_denovo(denovo_lengths: np.ndarray, out_path: Path) -> None:
    """Peptide length distribution, de novo only (denovo_only mode -- no database side)."""
    lo, hi = int(denovo_lengths.min()), int(denovo_lengths.max())
    bins = np.arange(lo, hi + 2) - 0.5
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        denovo_lengths,
        bins=bins,
        density=True,
        histtype="step",
        lw=1.8,
        color=postprocess.DETECTION_LEVEL_COLORS["denovo"],
        label=f"denovo (n={len(denovo_lengths):,}, median {np.median(denovo_lengths):.0f})",
    )
    ax.set_xlabel("peptide length")
    ax.set_ylabel("density")
    ax.set_title("Peptide length distribution -- de novo")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _plot_perc_vs_iso(
    percolator: np.ndarray,
    iso: np.ndarray,
    cutoff: float,
    side: str,
    out_path: Path,
    target: Optional[np.ndarray] = None,
) -> None:
    """Joint density: ISO_scores_<side> vs percolator_score_<side>, with marginal histograms.

    ``target`` (bool array) splits the hexbin into target/decoy layers on a shared color
    scale, same as the database side's Percolator target-decoy competition; ``None``
    (de novo has no decoys) draws a single layer.
    """
    groups = (
        [("target", target, _TARGET_COLOR, _SIDE_CMAP[side]), ("decoy", ~target, _DECOY_COLOR, "Reds")]
        if target is not None
        else [("all", np.ones(len(iso), dtype=bool), postprocess.DETECTION_LEVEL_COLORS[side], _SIDE_CMAP[side])]
    )

    extent = (0.0, 1.0, float(percolator.min()), float(percolator.max()))
    fig = plt.figure(figsize=(7.5, 8.5))
    gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
    ax_joint = fig.add_subplot(gs[1:, :3])
    ax_marg_x = fig.add_subplot(gs[0, :3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:, 3], sharey=ax_joint)

    layers = []
    for _, mask, _, cmap in groups:
        n = int(mask.sum())
        weights = np.full(n, 1.0 / n) if n else np.array([])
        hb = ax_joint.hexbin(
            iso[mask], percolator[mask], C=weights, reduce_C_function=np.sum, gridsize=90, extent=extent,
            cmap=cmap, mincnt=1e-9, linewidths=0, alpha=0.55 if cmap == "Reds" else 1.0,
        )
        layers.append(hb)
    if len(layers) > 1:
        arrays = [hb.get_array() for hb in layers if (hb.get_array() > 0).any()]
        if arrays:
            norm = LogNorm(vmin=min(a[a > 0].min() for a in arrays), vmax=max(a.max() for a in arrays))
            for hb in layers:
                hb.set_norm(norm)

    ax_joint.axvline(cutoff, color="red", lw=1.3)
    ax_joint.axhline(_PERCOLATOR_THRESHOLD, color="red", lw=1.3)

    # quadrant counts, split by target/decoy where that split applies
    quads = {
        "top-right": (iso >= cutoff) & (percolator >= _PERCOLATOR_THRESHOLD),
        "top-left": (iso < cutoff) & (percolator >= _PERCOLATOR_THRESHOLD),
        "bottom-left": (iso < cutoff) & (percolator < _PERCOLATOR_THRESHOLD),
        "bottom-right": (iso >= cutoff) & (percolator < _PERCOLATOR_THRESHOLD),
    }
    for name, quad_mask in quads.items():
        x = 0.97 if "right" in name else 0.03
        y = 0.97 if "top" in name else 0.03
        ha = "right" if "right" in name else "left"
        va = "top" if "top" in name else "bottom"
        if target is not None:
            text = f"target: {int((quad_mask & target).sum()):,}\ndecoy: {int((quad_mask & ~target).sum()):,}"
        else:
            text = f"{int(quad_mask.sum()):,}"
        ax_joint.text(x, y, text, transform=ax_joint.transAxes, ha=ha, va=va, fontsize=8.5,
                      bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    ax_joint.set_xlabel(f"isolation score, ISO_scores_{side} (0 = bad, 1 = good)")
    ax_joint.set_ylabel(f"percolator_score_{side}")

    iso_bins = np.linspace(0.0, 1.0, 101)
    perc_bins = np.linspace(percolator.min(), percolator.max(), 121)
    for name, mask, color, _ in groups:
        ax_marg_x.hist(iso[mask], bins=iso_bins, color=color, histtype="step", lw=1.6,
                        label=f"{name} (n={int(mask.sum()):,})")
        ax_marg_y.hist(percolator[mask], bins=perc_bins, orientation="horizontal", color=color, histtype="step", lw=1.6)
    ax_marg_x.axvline(cutoff, color="red", lw=1.3)
    ax_marg_y.axhline(_PERCOLATOR_THRESHOLD, color="red", lw=1.3)
    ax_marg_x.tick_params(labelbottom=False)
    ax_marg_y.tick_params(labelleft=False)
    ax_marg_x.set_ylabel("count")
    ax_marg_y.set_xlabel("count")
    if target is not None:
        ax_marg_x.legend(fontsize=7, loc="upper left")
    ax_marg_x.set_title(
        f"Isolation score vs percolator score -- {side}, {len(iso):,} PSMs, cutoff in red\n"
        f"(centre: hexbin density; margins: raw count)",
        fontsize=10,
    )
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def run(grove_forest_dir: Path, qc_dir: Optional[Path] = None) -> Path:
    """Write extra QC plots to ``grove_forest_dir/qc``: shared-scan-vs-PSM Venn,
    Levenshtein distance, peptide length, and percolator-vs-isolation-score joint
    density (database and de novo each on their own terms).

    Reads straight from ``grove_forest_dir/results/*.parquet`` -- independent of whether
    ``grovems.postprocess`` already ran in this invocation; cutoffs come from
    :func:`grovems.postprocess.postprocess.compute_cutoffs`, computed read-only here if
    needed.
    """
    results_dir = grove_forest_dir / "results"
    files = sorted(results_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No grove_forest parquet files found in {results_dir}")

    qc_dir = qc_dir or grove_forest_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "_merge",
        "sequence_match",
        "PSA_LEVENSHTEIN",
        "Label_database",
        "SEQUENCE_database",
        "SEQUENCE_denovo",
        "percolator_score_database",
        "percolator_score_denovo",
        "ISO_scores_database",
        "ISO_scores_denovo",
    ]
    logger.info("Reading %d grove_forest file(s) for plotting", len(files))
    combined = pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)

    shared_scan = combined["_merge"] == "shared"
    shared_psm = shared_scan & combined["sequence_match"]
    logger.info("Shared scans: %d, shared PSM: %d", int(shared_scan.sum()), int(shared_psm.sum()))
    _plot_shared_venn(int(shared_scan.sum()), int(shared_psm.sum()), qc_dir / "shared_venn.svg")

    lev = combined["PSA_LEVENSHTEIN"].dropna().to_numpy()
    if len(lev):
        _plot_levenshtein_distribution(lev, qc_dir / "levenshtein_distribution.svg")
    else:
        logger.warning("No PSA_LEVENSHTEIN values found; skipping levenshtein_distribution.svg")

    db_len = combined["SEQUENCE_database"].dropna().str.len().to_numpy()
    dn_len = combined["SEQUENCE_denovo"].dropna().str.len().to_numpy()
    _plot_peptide_length_distribution(db_len, dn_len, qc_dir / "peptide_length_distribution.svg")

    logger.info("Computing per-side GOOD/BAD cutoffs")
    cutoff = postprocess.compute_cutoffs(files)

    has_db = combined["percolator_score_database"].notna() & combined["ISO_scores_database"].notna()
    db = combined.loc[has_db]
    _plot_perc_vs_iso(
        db["percolator_score_database"].to_numpy(),
        db["ISO_scores_database"].to_numpy(),
        cutoff["database"],
        "database",
        qc_dir / "perc_vs_iso_database.svg",
        target=db["Label_database"].eq(1).to_numpy(),
    )

    has_dn = combined["percolator_score_denovo"].notna() & combined["ISO_scores_denovo"].notna()
    dn = combined.loc[has_dn]
    _plot_perc_vs_iso(
        dn["percolator_score_denovo"].to_numpy(),
        dn["ISO_scores_denovo"].to_numpy(),
        cutoff["denovo"],
        "denovo",
        qc_dir / "perc_vs_iso_denovo.svg",
    )

    logger.info("Plotting QC written to %s", qc_dir)
    return qc_dir


def run_denovo_only(grove_forest_dir: Path, qc_dir: Optional[Path] = None) -> Path:
    """De novo-only variant of :func:`run` -- no database side exists (see ``grovems.runner.run``).

    Every other plot :func:`run` produces is inherently two-engine: the shared-scan Venn
    and Levenshtein distance both need a database sequence to compare de novo against,
    and both percolator-vs-iso plots need a ``percolator_score`` -- Percolator itself
    doesn't run in denovo_only mode (see ``grovems.rescoring.rescoring.run``), on either
    side. Only the peptide length distribution survives, de novo alone.
    """
    results_dir = grove_forest_dir / "results"
    files = sorted(results_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No grove_forest parquet files found in {results_dir}")

    qc_dir = qc_dir or grove_forest_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat([pd.read_parquet(path, columns=["SEQUENCE_denovo"]) for path in files], ignore_index=True)
    dn_len = combined["SEQUENCE_denovo"].dropna().str.len().to_numpy()
    if len(dn_len):
        _plot_peptide_length_distribution_denovo(dn_len, qc_dir / "peptide_length_distribution.svg")
    else:
        logger.warning("No SEQUENCE_denovo values found; skipping peptide_length_distribution.svg")

    logger.info("Plotting QC written to %s", qc_dir)
    return qc_dir
