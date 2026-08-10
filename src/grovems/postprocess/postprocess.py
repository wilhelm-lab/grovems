from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ..iforest import trusted_shared_mask

logger = logging.getLogger(__name__)

ID_COLUMNS = ["SpecId", "RAW_FILE", "SCAN_NUMBER"]
OTHER_CLASSES = ("database_only", "denovo_only")
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


def _gather_reference_scores(files: list[Path]) -> np.ndarray:
    """Pass 1: sorted ISO_scores of the trusted-shared PSMs across every file (small)."""
    reference = []
    for path in files:
        df = pd.read_parquet(path, columns=["_merge", "Label_database", "percolator_score_database", "ISO_scores"])
        reference.append(df.loc[trusted_shared_mask(df), "ISO_scores"])
    return np.sort(pd.concat(reference, ignore_index=True).to_numpy())


def _goodness(reference_sorted: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Fraction of the reference at least as anomalous (<=) as each of ``scores``.

    ``ISO_scores`` is on the 0-1 scale produced by ``iforest._rescale_iso_scores``
    (1=good/least anomalous, 0=bad/most anomalous), so "at least as anomalous as x"
    means a reference score <= x.
    """
    n = len(reference_sorted)
    return np.searchsorted(reference_sorted, scores, side="right") / n


def _add_goodness_column(files: list[Path], reference_sorted: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Pass 2: add TP_GOODNESS to every file in place; collect ISO_scores for the KS test."""
    other_scores: dict[str, list[np.ndarray]] = {name: [] for name in OTHER_CLASSES}
    for path in files:
        df = pd.read_parquet(path)
        df["TP_GOODNESS"] = _goodness(reference_sorted, df["ISO_scores"].to_numpy())
        df.to_parquet(path, index=False, engine="pyarrow")
        for name in OTHER_CLASSES:
            other_scores[name].append(df.loc[df["_merge"] == name, "ISO_scores"].to_numpy())
    return other_scores


def _ks_divergence(reference_scores: np.ndarray, other_scores: np.ndarray) -> dict:
    result = ks_2samp(reference_scores, other_scores, alternative="two-sided")
    return {
        "n_reference": len(reference_scores),
        "n_other": len(other_scores),
        "statistic_D": float(result.statistic),
        "pvalue": float(result.pvalue),
        "statistic_location": float(result.statistic_location),
    }


def _plot_ks_vs_tp(reference_scores: np.ndarray, other: dict[str, np.ndarray], cutoff: float, out_path: Path) -> None:
    grid = np.linspace(
        min(reference_scores.min(), *(scores.min() for scores in other.values())),
        max(reference_scores.max(), *(scores.max() for scores in other.values())),
        501,
    )
    reference_sorted = np.sort(reference_scores)

    def ecdf_at(sample_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.searchsorted(sample_sorted, x, side="right") / len(sample_sorted)

    fig, axes = plt.subplots(1, len(other), figsize=(5.5 * len(other), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, scores) in zip(axes, other.items()):
        ax.plot(grid, ecdf_at(reference_sorted, grid), lw=1.8, label=f"trusted shared (n={len(reference_scores):,})")
        ax.plot(grid, ecdf_at(np.sort(scores), grid), lw=1.8, label=f"{name} (n={len(scores):,})")
        ax.axvline(cutoff, color="0.4", ls="--", lw=0.9, label=f"cutoff: ISO_scores={cutoff:.4f}")
        ax.set_xlabel("ISO_scores (0-1 scale; 1 = good, 0 = bad)")
        ax.set_title(f"ECDF: trusted shared vs {name}")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("cumulative fraction (ECDF)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _psm_detection_level_counts(combined: pd.DataFrame) -> pd.Series:
    """Count actual shared PSMs, not shared scans.

    ``DETECTION_LEVEL`` only reflects whether a *scan* produced a hit from both search
    engines. A shared scan where the two engines called different peptides is not a
    shared PSM -- it's one database PSM and one de novo PSM that happen to share a
    spectrum, uncorroborated by the other engine. ``sequence_match`` (identical modified
    sequence + charge, from :func:`grovems.psa.psa_merge.add_sequence_match_columns`)
    is what actually decides agreement, so split shared-but-conflicting scans into the
    database/denovo buckets instead of counting them as shared.
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


def _write_good_bad_lists(files: list[Path], cutoff: float, grove_forest_dir: Path, qc_dir: Path) -> None:
    """Pass 3: classify every PSM, plot the detection-level split, write good.csv/bad.csv."""
    columns = [
        *ID_COLUMNS,
        "_merge",
        "ISO_scores",
        "TP_GOODNESS",
        "AA_SCORE",
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
    combined["CASANOVO_AA_SCORE"] = combined.pop("AA_SCORE").str.replace("|", ",", regex=False)
    combined["SEQUENCE"] = combined.pop("SEQUENCE_denovo").combine_first(combined.pop("SEQUENCE_database"))
    has_aa_score = combined["CASANOVO_AA_SCORE"].notna()
    combined.loc[has_aa_score, "SEQUENCE"] = [
        _lowercase_low_confidence_aa(sequence, aa_scores)
        for sequence, aa_scores in zip(
            combined.loc[has_aa_score, "SEQUENCE"], combined.loc[has_aa_score, "CASANOVO_AA_SCORE"]
        )
    ]
    combined = combined.rename(columns=SCORE_COLUMNS)
    # ISO_scores is 0-1 (1=good, 0=bad); GOOD is now the >= side of the cutoff.
    combined["CALL"] = np.where(combined["ISO_scores"] >= cutoff, "GOOD", "BAD")
    combined = combined.sort_values("TP_GOODNESS", ascending=False).reset_index(drop=True)

    _plot_detection_level_counts(psm_counts, qc_dir / "psm_overlap.svg")

    for call, name in (("GOOD", "good"), ("BAD", "bad")):
        subset = combined.loc[combined["CALL"] == call].drop(columns=["CALL"])
        out_path = grove_forest_dir / f"{name}.csv"
        subset.to_csv(out_path, index=False)
        logger.info(
            "%s.csv: %d rows (cutoff ISO_scores>=%.4f for GOOD) -> %s",
            name,
            len(subset),
            cutoff,
            out_path,
        )


def run(grove_forest_dir: Path) -> Path:
    """Add TP_GOODNESS to every grove_forest PSM and write KS-based good/bad calls.

    Args:
        grove_forest_dir: Directory of IForest-scored ``grove_forest/results/*.parquet``
            files (must already have ``ISO_scores``/``ISO_labels`` columns from the
            IForest stage).

    Returns:
        The ``grove_forest_dir/qc`` directory the outputs were written to.
    """
    results_dir = grove_forest_dir / "results"
    files = sorted(results_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No grove_forest parquet files found in {results_dir}")
    if "ISO_scores" not in pd.read_parquet(files[0], columns=None).columns:
        raise ValueError(f"{files[0]} has no ISO_scores column -- run the IForest stage first")

    qc_dir = grove_forest_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Gathering trusted-shared reference scores from %d file(s)", len(files))
    reference_sorted = _gather_reference_scores(files)
    logger.info("Trusted-shared reference: %d PSMs", len(reference_sorted))

    logger.info("Scoring TP_GOODNESS and updating %d grove_forest file(s)", len(files))
    other_scores_parts = _add_goodness_column(files, reference_sorted)
    other_scores = {name: np.concatenate(parts) for name, parts in other_scores_parts.items()}

    divergence = {name: _ks_divergence(reference_sorted, scores) for name, scores in other_scores.items()}
    cutoff = float(np.mean([d["statistic_location"] for d in divergence.values()]))
    logger.info("KS divergence: %s", divergence)
    logger.info("Good/bad cutoff (average of statistic_location): ISO_scores>=%.4f is GOOD", cutoff)

    summary_rows = [{"comparison": f"trusted_shared vs {name}", **stats} for name, stats in divergence.items()]
    summary_rows.append({"comparison": "average", "cutoff_iso_scores": cutoff})
    pd.DataFrame(summary_rows).to_csv(qc_dir / "ks_vs_tp_summary.csv", index=False)

    _plot_ks_vs_tp(reference_sorted, other_scores, cutoff, qc_dir / "ks_vs_tp.svg")
    _write_good_bad_lists(files, cutoff, grove_forest_dir, qc_dir)

    logger.info("Postprocess QC written to %s", qc_dir)
    return qc_dir
