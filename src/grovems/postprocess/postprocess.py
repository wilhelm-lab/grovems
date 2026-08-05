"""Post-IForest QC: empirical-CDF goodness + KS-divergence good/bad calls.

Builds an empirical CDF of ``ISO_scores`` from the trusted shared PSMs (the same
population IForest itself trains on -- see :func:`grovems.iforest.trusted_shared_mask`)
and uses it two ways:

1. **Goodness** -- every PSM in every ``grove_forest/*.parquet`` file gets a
   ``TP_GOODNESS`` column: the fraction of the trusted-shared reference that is at
   least as anomalous (``ISO_scores >=``) as this PSM. 1.0 means better than every
   reference PSM, 0.0 means worse than all of them.
2. **Divergence / good-bad calls** -- a two-sample Kolmogorov-Smirnov test compares
   the reference distribution against ``database_only`` and against ``denovo_only``
   ``ISO_scores``. Each comparison's point of maximum divergence
   (``ks_2samp``'s ``statistic_location``) is that comparison's natural accept/reject
   boundary; averaging the two locations gives one cutoff applied to both classes: a
   PSM is ``GOOD`` if ``ISO_scores <= cutoff``, ``BAD`` otherwise.

Reads/updates ``grove_forest_dir/*.parquet`` in place (adds ``TP_GOODNESS``) and
writes ``grove_forest_dir/qc/ks_vs_tp.svg``, ``qc/ks_vs_tp_summary.csv``, and
``qc/good_bad_database_only.csv`` / ``qc/good_bad_denovo_only.csv``.
"""
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


def _gather_reference_scores(files: list[Path]) -> np.ndarray:
    """Pass 1: sorted ISO_scores of the trusted-shared PSMs across every file (small)."""
    reference = []
    for path in files:
        df = pd.read_parquet(path, columns=["_merge", "Label_database", "percolator_score_database", "ISO_scores"])
        reference.append(df.loc[trusted_shared_mask(df), "ISO_scores"])
    return np.sort(pd.concat(reference, ignore_index=True).to_numpy())


def _goodness(reference_sorted: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Fraction of the reference at least as anomalous (>=) as each of ``scores``."""
    n = len(reference_sorted)
    return 1.0 - np.searchsorted(reference_sorted, scores, side="left") / n


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
        ax.set_xlabel("ISO_scores (higher = more anomalous)")
        ax.set_title(f"ECDF: trusted shared vs {name}")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("cumulative fraction (ECDF)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _write_good_bad_lists(files: list[Path], cutoff: float, out_dir: Path) -> None:
    """Pass 3: narrow columns only, classify database_only/denovo_only PSMs, write two CSVs."""
    columns = [*ID_COLUMNS, "_merge", "ISO_scores", "TP_GOODNESS"]
    rows: dict[str, list[pd.DataFrame]] = {name: [] for name in OTHER_CLASSES}
    for path in files:
        df = pd.read_parquet(path, columns=columns)
        for name in OTHER_CLASSES:
            rows[name].append(df.loc[df["_merge"] == name])

    for name in OTHER_CLASSES:
        combined = pd.concat(rows[name], ignore_index=True).drop(columns=["_merge"])
        combined["CALL"] = np.where(combined["ISO_scores"] <= cutoff, "GOOD", "BAD")
        combined = combined.sort_values("TP_GOODNESS", ascending=False).reset_index(drop=True)
        combined.to_csv(out_dir / f"good_bad_{name}.csv", index=False)
        logger.info(
            "%s: %d GOOD, %d BAD (cutoff ISO_scores<=%.4f) -> %s",
            name,
            (combined["CALL"] == "GOOD").sum(),
            (combined["CALL"] == "BAD").sum(),
            cutoff,
            out_dir / f"good_bad_{name}.csv",
        )


def run(grove_forest_dir: Path) -> Path:
    """Add TP_GOODNESS to every grove_forest PSM and write KS-based good/bad calls.

    Args:
        grove_forest_dir: Directory of IForest-scored ``grove_forest/*.parquet`` files
            (must already have ``ISO_scores``/``ISO_labels`` columns from the IForest
            stage).

    Returns:
        The ``grove_forest_dir/qc`` directory the outputs were written to.
    """
    files = sorted(grove_forest_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No grove_forest parquet files found in {grove_forest_dir}")
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
    logger.info("Good/bad cutoff (average of statistic_location): ISO_scores<=%.4f", cutoff)

    summary_rows = [{"comparison": f"trusted_shared vs {name}", **stats} for name, stats in divergence.items()]
    summary_rows.append({"comparison": "average", "cutoff_iso_scores": cutoff})
    pd.DataFrame(summary_rows).to_csv(qc_dir / "ks_vs_tp_summary.csv", index=False)

    _plot_ks_vs_tp(reference_sorted, other_scores, cutoff, qc_dir / "ks_vs_tp.svg")
    _write_good_bad_lists(files, cutoff, qc_dir)

    logger.info("Postprocess QC written to %s", qc_dir)
    return qc_dir
