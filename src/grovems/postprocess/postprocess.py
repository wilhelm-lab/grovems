from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import ks_2samp

from ..iforest import trusted_denovo_mask, trusted_shared_mask

logger = logging.getLogger(__name__)

ID_COLUMNS = ["SpecId", "RAW_FILE", "SCAN_NUMBER"]
SIDES = ("database", "denovo")
# Each side's own uncorroborated-hit population, compared against that side's own
# trusted-shared reference -- keeps the two KS tests (and the two cutoffs they produce)
# from being blended across sides that were scored on different feature vectors.
SIDE_ONLY_MERGE = {"database": "database_only", "denovo": "denovo_only"}
DETECTION_LEVELS = {"shared": "shared", "database_only": "database", "denovo_only": "denovo"}
DETECTION_LEVEL_ORDER = ("shared", "database", "denovo")
DETECTION_LEVEL_LABELS = {"shared": "Shared PSM", "database": "Database only", "denovo": "De novo only"}
DETECTION_LEVEL_COLORS = {"shared": "#2a78d6", "database": "#eb6834", "denovo": "#1baf7a"}
SCORE_COLUMNS = {
    "SCORE_denovo": "CASANOVO_SCORE",
    "SCORE_database": "DATABASE_SCORE",
    "percolator_score_database": "PERCOLATOR_SCORE_DATABASE",
}
AA_SCORE_LOWERCASE_THRESHOLD = 0.6


def _lowercase_low_confidence_aa(sequence: str, aa_scores: str, threshold: float = AA_SCORE_LOWERCASE_THRESHOLD) -> str:
    """Lowercase each residue in ``sequence`` whose aligned Casanovo AA score is below ``threshold``."""
    scores = aa_scores.split(",")
    if len(scores) != len(sequence):
        logger.warning(
            "SEQUENCE/AA_SCORE length mismatch (%d residues vs %d scores) for %r; leaving case as-is",
            len(sequence),
            len(scores),
            sequence,
        )
        return sequence
    return "".join(aa.lower() if float(score) < threshold else aa for aa, score in zip(sequence, scores))


def _add_casanovo_sequence_columns(combined: pd.DataFrame) -> None:
    """Add CASANOVO_AA_SCORE and lowercase low-confidence residues in combined['SEQUENCE'], in place."""
    combined["CASANOVO_AA_SCORE"] = combined.pop("AA_SCORE_denovo").str.replace("|", ",", regex=False)
    has_aa_score = combined["CASANOVO_AA_SCORE"].notna()
    combined.loc[has_aa_score, "SEQUENCE"] = [
        _lowercase_low_confidence_aa(sequence, aa_scores)
        for sequence, aa_scores in zip(
            combined.loc[has_aa_score, "SEQUENCE"], combined.loc[has_aa_score, "CASANOVO_AA_SCORE"]
        )
    ]


def _gather_reference_scores(files: list[Path]) -> dict[str, np.ndarray]:
    """Pass 1: sorted per-side ISO_scores of the trusted-shared PSMs across every file."""
    reference: dict[str, list[pd.Series]] = {side: [] for side in SIDES}
    for path in files:
        columns = ["_merge", "Label_database", "percolator_score_database", *(f"ISO_scores_{side}" for side in SIDES)]
        df = pd.read_parquet(path, columns=columns)
        trusted = df.loc[trusted_shared_mask(df)]
        for side in SIDES:
            reference[side].append(trusted[f"ISO_scores_{side}"].dropna())
    return {side: np.sort(pd.concat(parts, ignore_index=True).to_numpy()) for side, parts in reference.items()}


def _goodness(reference_sorted: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Fraction of the reference at least as anomalous (<=) as each of ``scores``; NaN scores stay NaN."""
    n = len(reference_sorted)
    filled = np.nan_to_num(scores, nan=-np.inf)
    result = np.searchsorted(reference_sorted, filled, side="right") / n
    return np.where(np.isnan(scores), np.nan, result)


def _add_goodness_column(files: list[Path], reference_sorted: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Pass 2: add per-side TP_GOODNESS_<side> to every file; return each *_only class's ISO_scores for the KS test."""
    other_scores: dict[str, list[np.ndarray]] = {only_name: [] for only_name in SIDE_ONLY_MERGE.values()}
    for path in files:
        df = pd.read_parquet(path)
        for side in SIDES:
            df[f"TP_GOODNESS_{side}"] = _goodness(reference_sorted[side], df[f"ISO_scores_{side}"].to_numpy())
        df.to_parquet(path, index=False, engine="pyarrow")
        for side, only_name in SIDE_ONLY_MERGE.items():
            other_scores[only_name].append(df.loc[df["_merge"] == only_name, f"ISO_scores_{side}"].to_numpy())
    return {name: np.concatenate(parts) for name, parts in other_scores.items()}


def compute_cutoffs(files: list[Path]) -> dict[str, float]:
    """Per-side GOOD/BAD cutoff, read-only -- lets grovems.plotting reuse run()'s cutoffs without rerunning it."""
    reference_sorted = _gather_reference_scores(files)

    other_scores: dict[str, list[np.ndarray]] = {only_name: [] for only_name in SIDE_ONLY_MERGE.values()}
    for path in files:
        columns = ["_merge", *(f"ISO_scores_{side}" for side in SIDES)]
        df = pd.read_parquet(path, columns=columns)
        for side, only_name in SIDE_ONLY_MERGE.items():
            other_scores[only_name].append(df.loc[df["_merge"] == only_name, f"ISO_scores_{side}"].to_numpy())
    other_scores = {name: np.concatenate(parts) for name, parts in other_scores.items()}

    divergence = {
        only_name: _ks_divergence(reference_sorted[side], other_scores[only_name])
        for side, only_name in SIDE_ONLY_MERGE.items()
    }
    return {side: float(divergence[only_name]["statistic_location"]) for side, only_name in SIDE_ONLY_MERGE.items()}


def _ks_divergence(reference_scores: np.ndarray, other_scores: np.ndarray) -> dict:
    result = ks_2samp(reference_scores, other_scores, alternative="two-sided")
    return {
        "n_reference": len(reference_scores),
        "n_other": len(other_scores),
        "statistic_D": float(result.statistic),
        "pvalue": float(result.pvalue),
        "statistic_location": float(result.statistic_location),
    }


def _ecdf_at(sample_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.searchsorted(sample_sorted, x, side="right") / len(sample_sorted)


def _plot_ks_vs_tp(
    reference_scores: dict[str, np.ndarray], other: dict[str, np.ndarray], cutoff: dict[str, float], out_path: Path
) -> None:
    """One ECDF panel per side, each against its own side-matched reference and cutoff."""
    pairs = list(SIDE_ONLY_MERGE.items())
    all_values = [reference_scores[side] for side, _ in pairs] + [other[name] for _, name in pairs]
    grid = np.linspace(min(v.min() for v in all_values), max(v.max() for v in all_values), 501)

    fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (side, name) in zip(axes, pairs):
        reference_sorted = reference_scores[side]
        scores = other[name]
        ax.plot(
            grid,
            _ecdf_at(reference_sorted, grid),
            lw=1.8,
            label=f"trusted shared, {side} (n={len(reference_sorted):,})",
        )
        ax.plot(grid, _ecdf_at(np.sort(scores), grid), lw=1.8, label=f"{name} (n={len(scores):,})")
        ax.axvline(cutoff[side], color="0.4", ls="--", lw=0.9, label=f"cutoff: ISO_scores_{side}={cutoff[side]:.4f}")
        ax.set_xlabel(f"ISO_scores_{side} (0-1 scale; 1 = good, 0 = bad)")
        ax.set_title(f"ECDF: trusted shared vs {name}")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("cumulative fraction (ECDF)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _psm_detection_level_counts(combined: pd.DataFrame) -> pd.Series:
    """Count actual shared PSMs, not shared scans -- a shared scan where the two engines called
    different peptides is split into the database/denovo buckets via sequence_match instead.
    """
    shared_scan = combined["DETECTION_LEVEL"] == "shared"
    agree = shared_scan & combined["sequence_match"]
    conflicting = shared_scan & ~combined["sequence_match"]
    return pd.Series(
        {
            "shared": int(agree.sum()),
            "database": int(((combined["DETECTION_LEVEL"] == "database") | conflicting).sum()),
            "denovo": int(((combined["DETECTION_LEVEL"] == "denovo") | conflicting).sum()),
        }
    )


def _plot_detection_level_counts(counts: pd.Series, out_path: Path) -> None:
    """Bar chart of PSM counts per detection level (shared / database-only / denovo-only)."""
    values = [int(counts.get(level, 0)) for level in DETECTION_LEVEL_ORDER]
    labels = [DETECTION_LEVEL_LABELS[level] for level in DETECTION_LEVEL_ORDER]
    colors = [DETECTION_LEVEL_COLORS[level] for level in DETECTION_LEVEL_ORDER]

    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=3)
    ax.set_ylabel("PSM count")
    ax.set_title("PSMs by detection level")
    ax.margins(y=0.12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _write_good_bad_lists(files: list[Path], cutoff: dict[str, float], grove_forest_dir: Path, qc_dir: Path) -> None:
    """Pass 3: classify every PSM per side, plot the detection-level split, write good.csv/bad.csv."""
    columns = [
        *ID_COLUMNS,
        "_merge",
        "ISO_scores_side",
        *(f"ISO_scores_{side}" for side in SIDES),
        *(f"TP_GOODNESS_{side}" for side in SIDES),
        "AA_SCORE_denovo",
        "SEQUENCE_database",
        "SEQUENCE_denovo",
        "sequence_match",
        *SCORE_COLUMNS,
    ]
    parts = [pd.read_parquet(path, columns=columns) for path in files]

    combined = pd.concat(parts, ignore_index=True)
    combined["DETECTION_LEVEL"] = combined.pop("_merge").map(DETECTION_LEVELS)
    psm_counts = _psm_detection_level_counts(combined)
    combined = combined.drop(columns=["sequence_match"])
    combined["SEQUENCE"] = combined.pop("SEQUENCE_denovo").combine_first(combined.pop("SEQUENCE_database"))
    _add_casanovo_sequence_columns(combined)
    combined = combined.rename(columns=SCORE_COLUMNS)

    # Each side is called GOOD/BAD against its own side-matched KS cutoff, not a blended average.
    for side in SIDES:
        scores = combined[f"ISO_scores_{side}"]
        combined[f"CALL_{side}"] = np.where(scores.notna(), np.where(scores >= cutoff[side], "GOOD", "BAD"), None)

    # Legacy single-value view: whichever side is this row's primary identification (ISO_scores_side).
    is_denovo_side = combined["ISO_scores_side"].eq("denovo")
    combined["ISO_scores"] = np.where(is_denovo_side, combined["ISO_scores_denovo"], combined["ISO_scores_database"])
    combined["TP_GOODNESS"] = np.where(is_denovo_side, combined["TP_GOODNESS_denovo"], combined["TP_GOODNESS_database"])
    combined["CALL"] = np.where(is_denovo_side, combined["CALL_denovo"], combined["CALL_database"])
    combined = combined.sort_values("TP_GOODNESS", ascending=False).reset_index(drop=True)

    _plot_detection_level_counts(psm_counts, qc_dir / "psm_overlap.svg")

    for call, name in (("GOOD", "good"), ("BAD", "bad")):
        subset = combined.loc[combined["CALL"] == call].drop(columns=["CALL"])
        out_path = grove_forest_dir / f"{name}.csv"
        subset.to_csv(out_path, index=False)
        logger.info(
            "%s.csv: %d rows (cutoffs: database>=%.4f, denovo>=%.4f for GOOD) -> %s",
            name,
            len(subset),
            cutoff["database"],
            cutoff["denovo"],
            out_path,
        )


def run(grove_forest_dir: Path) -> Path:
    """Add per-side TP_GOODNESS to every grove_forest PSM and write KS-based good/bad calls.

    Requires ISO_scores_database/ISO_scores_denovo (from the IForest stage) already
    present in grove_forest_dir/results/*.parquet.
    """
    results_dir = grove_forest_dir / "results"
    files = sorted(results_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No grove_forest parquet files found in {results_dir}")
    first_file_columns = pq.ParquetFile(files[0]).schema_arrow.names
    missing = [f"ISO_scores_{side}" for side in SIDES if f"ISO_scores_{side}" not in first_file_columns]
    if missing:
        raise ValueError(f"{files[0]} is missing {missing} -- run the IForest stage first")

    qc_dir = grove_forest_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Gathering trusted-shared reference scores from %d file(s)", len(files))
    reference_sorted = _gather_reference_scores(files)
    logger.info("Trusted-shared reference: %s", {side: len(scores) for side, scores in reference_sorted.items()})
    empty_sides = [side for side, scores in reference_sorted.items() if len(scores) == 0]
    if empty_sides:
        raise ValueError(f"No trusted-shared reference PSMs found for side(s) {empty_sides} in {results_dir}")

    logger.info("Scoring per-side TP_GOODNESS and updating %d grove_forest file(s)", len(files))
    other_scores = _add_goodness_column(files, reference_sorted)
    empty_others = [name for name, scores in other_scores.items() if len(scores) == 0]
    if empty_others:
        raise ValueError(
            f"No uncorroborated PSMs found for {empty_others} in {results_dir} -- cannot compute a KS cutoff"
        )

    divergence = {
        only_name: _ks_divergence(reference_sorted[side], other_scores[only_name])
        for side, only_name in SIDE_ONLY_MERGE.items()
    }
    cutoff = {side: float(divergence[only_name]["statistic_location"]) for side, only_name in SIDE_ONLY_MERGE.items()}
    logger.info("KS divergence: %s", divergence)
    logger.info("Good/bad cutoffs (per side): %s", cutoff)

    summary_rows = [{"comparison": f"trusted_shared vs {name}", **stats} for name, stats in divergence.items()]
    summary_rows.extend({"comparison": f"cutoff_{side}", "cutoff_iso_scores": value} for side, value in cutoff.items())
    pd.DataFrame(summary_rows).to_csv(qc_dir / "ks_vs_tp_summary.csv", index=False)

    _plot_ks_vs_tp(reference_sorted, other_scores, cutoff, qc_dir / "ks_vs_tp.svg")
    _write_good_bad_lists(files, cutoff, grove_forest_dir, qc_dir)

    logger.info("Postprocess QC written to %s", qc_dir)
    return qc_dir


# ---- denovo_only mode: no database side exists (see grovems.runner.run). Reference =
# SCORE_denovo >= threshold (the population IForest trained on), other = the rest of
# the de novo PSMs -- no database column is read or written below. ----


def _add_denovo_goodness_column(files: list[Path], reference_sorted: np.ndarray, score_threshold: float) -> np.ndarray:
    """Write TP_GOODNESS_denovo to every file; return ISO_scores_denovo of the untrusted rest, for the KS test."""
    other_scores = []
    for path in files:
        df = pd.read_parquet(path)
        df["TP_GOODNESS_denovo"] = _goodness(reference_sorted, df["ISO_scores_denovo"].to_numpy())
        df.to_parquet(path, index=False, engine="pyarrow")
        has_call = df["_merge"].isin(("denovo_only", "shared"))
        other_mask = has_call & ~trusted_denovo_mask(df, score_threshold)
        other_scores.append(df.loc[other_mask, "ISO_scores_denovo"].to_numpy())
    return np.concatenate(other_scores)


def _plot_denovo_ks_vs_tp(reference_sorted: np.ndarray, other: np.ndarray, cutoff: float, out_path: Path) -> None:
    grid = np.linspace(min(reference_sorted.min(), other.min()), max(reference_sorted.max(), other.max()), 501)

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    ax.plot(grid, _ecdf_at(reference_sorted, grid), lw=1.8, label=f"trusted de novo (n={len(reference_sorted):,})")
    ax.plot(grid, _ecdf_at(np.sort(other), grid), lw=1.8, label=f"rest of de novo (n={len(other):,})")
    ax.axvline(cutoff, color="0.4", ls="--", lw=0.9, label=f"cutoff: ISO_scores_denovo={cutoff:.4f}")
    ax.set_xlabel("ISO_scores_denovo (0-1 scale; 1 = good, 0 = bad)")
    ax.set_ylabel("cumulative fraction (ECDF)")
    ax.set_title("ECDF: trusted de novo vs rest of de novo")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _write_denovo_good_bad_lists(files: list[Path], cutoff: float, grove_forest_dir: Path, qc_dir: Path) -> None:
    columns = [
        *ID_COLUMNS,
        "ISO_scores_denovo",
        "TP_GOODNESS_denovo",
        "AA_SCORE_denovo",
        "SEQUENCE_denovo",
        "SCORE_denovo",
    ]
    combined = pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)

    combined["SEQUENCE"] = combined.pop("SEQUENCE_denovo")
    _add_casanovo_sequence_columns(combined)
    combined = combined.rename(
        columns={
            "SCORE_denovo": SCORE_COLUMNS["SCORE_denovo"],
            "ISO_scores_denovo": "ISO_scores",
            "TP_GOODNESS_denovo": "TP_GOODNESS",
        }
    )
    combined["CALL"] = np.where(
        combined["ISO_scores"].notna(), np.where(combined["ISO_scores"] >= cutoff, "GOOD", "BAD"), None
    )
    combined = combined.sort_values("TP_GOODNESS", ascending=False).reset_index(drop=True)

    for call, name in (("GOOD", "good"), ("BAD", "bad")):
        subset = combined.loc[combined["CALL"] == call].drop(columns=["CALL"])
        out_path = grove_forest_dir / f"{name}.csv"
        subset.to_csv(out_path, index=False)
        logger.info("%s.csv: %d rows (cutoff: denovo>=%.4f for GOOD) -> %s", name, len(subset), cutoff, out_path)


def run_denovo_only(grove_forest_dir: Path, score_threshold: float) -> Path:
    """De novo-only variant of run() -- score_threshold is normally config.iforest_denovo_score_threshold."""
    results_dir = grove_forest_dir / "results"
    files = sorted(results_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No grove_forest parquet files found in {results_dir}")
    first_file_columns = pq.ParquetFile(files[0]).schema_arrow.names
    if "ISO_scores_denovo" not in first_file_columns:
        raise ValueError(f"{files[0]} is missing ISO_scores_denovo -- run the IForest stage first")

    qc_dir = grove_forest_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    reference_parts = []
    for path in files:
        df = pd.read_parquet(path, columns=["_merge", "SCORE_denovo", "ISO_scores_denovo"])
        trusted = df.loc[trusted_denovo_mask(df, score_threshold)]
        reference_parts.append(trusted["ISO_scores_denovo"].dropna())
    reference_sorted = np.sort(pd.concat(reference_parts, ignore_index=True).to_numpy())
    logger.info("Trusted de novo (SCORE_denovo >= %.4f) reference: %d rows", score_threshold, len(reference_sorted))
    if len(reference_sorted) == 0:
        raise ValueError(
            f"No PSMs with SCORE_denovo >= {score_threshold} found in {results_dir} -- lower "
            "iforest_denovo_score_threshold or check the SCORE_denovo distribution in this dataset."
        )

    other_scores = _add_denovo_goodness_column(files, reference_sorted, score_threshold)
    if len(other_scores) == 0:
        raise ValueError(f"No uncorroborated de novo PSMs found in {results_dir} -- cannot compute a KS cutoff")

    divergence = _ks_divergence(reference_sorted, other_scores)
    cutoff = float(divergence["statistic_location"])
    logger.info("KS divergence: %s", divergence)
    logger.info("Good/bad cutoff (denovo): %s", cutoff)

    summary_rows = [
        {"comparison": "trusted_denovo vs rest", **divergence},
        {"comparison": "cutoff_denovo", "cutoff_iso_scores": cutoff},
    ]
    pd.DataFrame(summary_rows).to_csv(qc_dir / "ks_vs_tp_summary.csv", index=False)

    _plot_denovo_ks_vs_tp(reference_sorted, other_scores, cutoff, qc_dir / "ks_vs_tp.svg")
    _write_denovo_good_bad_lists(files, cutoff, grove_forest_dir, qc_dir)

    logger.info("Postprocess QC written to %s", qc_dir)
    return qc_dir
