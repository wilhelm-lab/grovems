from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    """Add CASANOVO_AA_SCORE and lowercase low-confidence residues in combined['SEQUENCE'], in place.

    Shared by the two-sided and denovo_only good/bad writers -- both first populate a
    'SEQUENCE' column (two-sided: de novo's sequence with database's as fallback;
    denovo_only: just de novo's) before calling this.
    """
    combined["CASANOVO_AA_SCORE"] = combined.pop("AA_SCORE_denovo").str.replace("|", ",", regex=False)
    has_aa_score = combined["CASANOVO_AA_SCORE"].notna()
    combined.loc[has_aa_score, "SEQUENCE"] = [
        _lowercase_low_confidence_aa(sequence, aa_scores)
        for sequence, aa_scores in zip(
            combined.loc[has_aa_score, "SEQUENCE"], combined.loc[has_aa_score, "CASANOVO_AA_SCORE"]
        )
    ]


def _gather_reference_scores(
    files: list[Path],
    sides: tuple[str, ...],
    trusted_mask: Callable[[pd.DataFrame], pd.Series],
    extra_columns: list[str],
) -> dict[str, np.ndarray]:
    """Pass 1: sorted per-side ISO_scores of ``trusted_mask``'s trusted population across every file (small).

    Shared by both QC modes: two-sided :func:`run` calls this with ``sides=SIDES``,
    ``trusted_mask=trusted_shared_mask`` (shared, agreeing PSMs, needing
    Label_database/percolator_score_database); denovo_only's :func:`run_denovo_only`
    calls it with ``sides=("denovo",)`` and a ``SCORE_denovo >= threshold`` mask (the
    same population IForest trained on) -- ``shared`` rows get scored on both sides
    independently (see ``iforest.score_grove_forest_file``), so with two sides this
    yields a separate reference distribution per side: comparing a side's scores
    against a reference computed on the *other* side's features would mix two
    different scorings.
    """
    reference: dict[str, list[pd.Series]] = {side: [] for side in sides}
    for path in files:
        columns = ["_merge", *extra_columns, *(f"ISO_scores_{side}" for side in sides)]
        df = pd.read_parquet(path, columns=columns)
        trusted = df.loc[trusted_mask(df)]
        for side in sides:
            reference[side].append(trusted[f"ISO_scores_{side}"].dropna())
    return {side: np.sort(pd.concat(parts, ignore_index=True).to_numpy()) for side, parts in reference.items()}


def _goodness(reference_sorted: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Fraction of the reference at least as anomalous (<=) as each of ``scores``; NaN scores stay NaN.

    ``ISO_scores*`` is on the 0-1 scale produced by ``iforest._rescale_iso_scores``
    (1=good/least anomalous, 0=bad/most anomalous), so "at least as anomalous as x"
    means a reference score <= x.
    """
    n = len(reference_sorted)
    filled = np.nan_to_num(scores, nan=-np.inf)
    result = np.searchsorted(reference_sorted, filled, side="right") / n
    return np.where(np.isnan(scores), np.nan, result)


def _add_goodness_column(files: list[Path], reference_sorted: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Pass 2: add per-side TP_GOODNESS_<side> to every file in place.

    Also collects each ``*_only`` class's own-side ISO_scores for the KS test (e.g.
    ``denovo_only`` rows contribute their ``ISO_scores_denovo``, not a database-side
    value they don't have).
    """
    other_scores: dict[str, list[np.ndarray]] = {only_name: [] for only_name in SIDE_ONLY_MERGE.values()}
    for path in files:
        df = pd.read_parquet(path)
        for side in SIDES:
            df[f"TP_GOODNESS_{side}"] = _goodness(reference_sorted[side], df[f"ISO_scores_{side}"].to_numpy())
        df.to_parquet(path, index=False, engine="pyarrow")
        for side, only_name in SIDE_ONLY_MERGE.items():
            other_scores[only_name].append(df.loc[df["_merge"] == only_name, f"ISO_scores_{side}"].to_numpy())
    return {name: np.concatenate(parts) for name, parts in other_scores.items()}


def _gather_only_scores(files: list[Path]) -> dict[str, np.ndarray]:
    """Each ``*_only`` class's own-side ISO_scores, read-only (no TP_GOODNESS side effect).

    Same population :func:`_add_goodness_column` collects for the KS test, but without
    that function's write pass -- for callers (e.g. ``grovems.plotting``) that just want
    the cutoffs, not to re-run postprocess's TP_GOODNESS write.
    """
    other_scores: dict[str, list[np.ndarray]] = {only_name: [] for only_name in SIDE_ONLY_MERGE.values()}
    for path in files:
        columns = ["_merge", *(f"ISO_scores_{side}" for side in SIDES)]
        df = pd.read_parquet(path, columns=columns)
        for side, only_name in SIDE_ONLY_MERGE.items():
            other_scores[only_name].append(df.loc[df["_merge"] == only_name, f"ISO_scores_{side}"].to_numpy())
    return {name: np.concatenate(parts) for name, parts in other_scores.items()}


def compute_cutoffs(files: list[Path]) -> dict[str, float]:
    """Per-side GOOD/BAD cutoff, read-only: each side's own KS-divergence point (trusted-
    shared reference vs that side's uncorroborated population), never averaged across
    sides. Lets ``grovems.plotting`` reuse the same cutoffs ``run()`` computes without
    re-running the full postprocess stage (and its TP_GOODNESS write).
    """
    reference_sorted = _gather_reference_scores(
        files, SIDES, trusted_shared_mask, ["Label_database", "percolator_score_database"]
    )
    other_scores = _gather_only_scores(files)
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


class _EcdfPanel(NamedTuple):
    reference_sorted: np.ndarray
    reference_label: str
    other: np.ndarray
    other_label: str
    cutoff: float
    xlabel: str
    title: str


def _plot_ks_vs_tp(panels: list[_EcdfPanel], out_path: Path) -> None:
    """One ECDF panel per entry in ``panels``, each against its own reference and cutoff.

    Used for both the two-sided (one panel per side) and denovo_only (single panel) QC
    plots -- the maths and layout are identical either way, just a different panel count.
    """
    all_values = [p.reference_sorted for p in panels] + [p.other for p in panels]
    grid = np.linspace(min(v.min() for v in all_values), max(v.max() for v in all_values), 501)

    def ecdf_at(sample_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.searchsorted(sample_sorted, x, side="right") / len(sample_sorted)

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, panel in zip(axes, panels):
        ax.plot(
            grid,
            ecdf_at(panel.reference_sorted, grid),
            lw=1.8,
            label=f"{panel.reference_label} (n={len(panel.reference_sorted):,})",
        )
        ax.plot(grid, ecdf_at(np.sort(panel.other), grid), lw=1.8, label=f"{panel.other_label} (n={len(panel.other):,})")
        ax.axvline(panel.cutoff, color="0.4", ls="--", lw=0.9, label=f"cutoff: {panel.xlabel}={panel.cutoff:.4f}")
        ax.set_xlabel(panel.xlabel)
        ax.set_title(panel.title)
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

    # Each side is called GOOD/BAD against its own side-matched KS cutoff -- averaging
    # the two cutoffs into one blended number (the old behavior) would compare a side's
    # scores against a boundary calibrated partly from the other side's distribution.
    for side in SIDES:
        scores = combined[f"ISO_scores_{side}"]
        combined[f"CALL_{side}"] = np.where(scores.notna(), np.where(scores >= cutoff[side], "GOOD", "BAD"), None)

    # Legacy single-value view: whichever side is this row's primary identification
    # (ISO_scores_side, from the IForest stage), now compared against that side's own
    # cutoff instead of the old blended average.
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

    Args:
        grove_forest_dir: Directory of IForest-scored ``grove_forest/results/*.parquet``
            files (must already have ``ISO_scores_database``/``ISO_scores_denovo``
            columns from the IForest stage).

    Returns:
        The ``grove_forest_dir/qc`` directory the outputs were written to.
    """
    results_dir = grove_forest_dir / "results"
    files = sorted(results_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No grove_forest parquet files found in {results_dir}")
    first_file_columns = pd.read_parquet(files[0], columns=None).columns
    missing = [f"ISO_scores_{side}" for side in SIDES if f"ISO_scores_{side}" not in first_file_columns]
    if missing:
        raise ValueError(f"{files[0]} is missing {missing} -- run the IForest stage first")

    qc_dir = grove_forest_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Gathering trusted-shared reference scores from %d file(s)", len(files))
    reference_sorted = _gather_reference_scores(
        files, SIDES, trusted_shared_mask, ["Label_database", "percolator_score_database"]
    )
    logger.info("Trusted-shared reference: %s", {side: len(scores) for side, scores in reference_sorted.items()})

    logger.info("Scoring per-side TP_GOODNESS and updating %d grove_forest file(s)", len(files))
    other_scores = _add_goodness_column(files, reference_sorted)

    # Each side's cutoff comes only from that side's own KS test -- no averaging across
    # sides, since ISO_scores_database and ISO_scores_denovo are independent scorings.
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

    panels = [
        _EcdfPanel(
            reference_sorted=reference_sorted[side],
            reference_label=f"trusted shared, {side}",
            other=other_scores[name],
            other_label=name,
            cutoff=cutoff[side],
            xlabel=f"ISO_scores_{side} (0-1 scale; 1 = good, 0 = bad)",
            title=f"ECDF: trusted shared vs {name}",
        )
        for side, name in SIDE_ONLY_MERGE.items()
    ]
    _plot_ks_vs_tp(panels, qc_dir / "ks_vs_tp.svg")
    _write_good_bad_lists(files, cutoff, grove_forest_dir, qc_dir)

    logger.info("Postprocess QC written to %s", qc_dir)
    return qc_dir


# ---- denovo_only mode: no database side exists at all (see grovems.runner.run) ----
# Mirrors the two-sided methodology above (a "trusted" reference population KS-tested
# against an "uncorroborated" population, to find a GOOD/BAD cutoff), but entirely within
# de novo: trusted_denovo_mask's SCORE_denovo-threshold population (the same one IForest
# trained on) stands in for trusted_shared_mask's shared-and-agreeing population, and the
# rest of the de novo PSMs stand in for that side's *_only population. No database
# column (Label_database, percolator_score_database, SEQUENCE_database, ISO_scores_database,
# ...) is read or written anywhere in this section.


def _has_denovo_call_mask(df: pd.DataFrame) -> pd.Series:
    """Rows with any de novo identification at all (denovo_only or shared)."""
    return df["_merge"].isin(("denovo_only", "shared"))


def _gather_denovo_other_scores(files: list[Path], score_threshold: float) -> np.ndarray:
    """ISO_scores_denovo of every de novo PSM that isn't in the trusted reference."""
    parts = []
    for path in files:
        df = pd.read_parquet(path, columns=["_merge", "SCORE_denovo", "ISO_scores_denovo"])
        other_mask = _has_denovo_call_mask(df) & ~trusted_denovo_mask(df, score_threshold)
        parts.append(df.loc[other_mask, "ISO_scores_denovo"].to_numpy())
    return np.concatenate(parts)


def _add_denovo_goodness_column(files: list[Path], reference_sorted: np.ndarray, score_threshold: float) -> np.ndarray:
    """Pass 2: add TP_GOODNESS_denovo to every file in place; return the "other" population for the KS test."""
    other_scores = []
    for path in files:
        df = pd.read_parquet(path)
        df["TP_GOODNESS_denovo"] = _goodness(reference_sorted, df["ISO_scores_denovo"].to_numpy())
        df.to_parquet(path, index=False, engine="pyarrow")
        other_mask = _has_denovo_call_mask(df) & ~trusted_denovo_mask(df, score_threshold)
        other_scores.append(df.loc[other_mask, "ISO_scores_denovo"].to_numpy())
    return np.concatenate(other_scores)


def compute_denovo_cutoff(files: list[Path], score_threshold: float) -> float:
    """Read-only GOOD/BAD cutoff for denovo_only mode, for ``grovems.plotting`` to reuse."""
    reference_sorted = _gather_reference_scores(
        files, ("denovo",), lambda df: trusted_denovo_mask(df, score_threshold), ["SCORE_denovo"]
    )["denovo"]
    other_scores = _gather_denovo_other_scores(files, score_threshold)
    divergence = _ks_divergence(reference_sorted, other_scores)
    return float(divergence["statistic_location"])


def _write_denovo_good_bad_lists(files: list[Path], cutoff: float, grove_forest_dir: Path, qc_dir: Path) -> None:
    """Classify every de novo PSM, write good.csv/bad.csv."""
    columns = [*ID_COLUMNS, "ISO_scores_denovo", "TP_GOODNESS_denovo", "AA_SCORE_denovo", "SEQUENCE_denovo", "SCORE_denovo"]
    combined = pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)

    combined["SEQUENCE"] = combined.pop("SEQUENCE_denovo")
    _add_casanovo_sequence_columns(combined)
    combined = combined.rename(
        columns={"SCORE_denovo": "CASANOVO_SCORE", "ISO_scores_denovo": "ISO_scores", "TP_GOODNESS_denovo": "TP_GOODNESS"}
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
    """De novo-only variant of :func:`run` -- no database side exists (see ``grovems.runner.run``).

    Args:
        grove_forest_dir: Directory of IForest-scored ``grove_forest/results/*.parquet``
            files (must already have an ``ISO_scores_denovo`` column from the IForest stage).
        score_threshold: Minimum ``SCORE_denovo`` for the trusted reference population
            (see :func:`grovems.iforest.trusted_denovo_mask`) -- normally
            ``config.iforest_denovo_score_threshold``, the same value IForest trained on.

    Returns:
        The ``grove_forest_dir/qc`` directory the outputs were written to.
    """
    results_dir = grove_forest_dir / "results"
    files = sorted(results_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No grove_forest parquet files found in {results_dir}")
    first_file_columns = pd.read_parquet(files[0], columns=None).columns
    if "ISO_scores_denovo" not in first_file_columns:
        raise ValueError(f"{files[0]} is missing ISO_scores_denovo -- run the IForest stage first")

    qc_dir = grove_forest_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Gathering trusted de novo (SCORE_denovo >= %.4f) reference scores from %d file(s)", score_threshold, len(files)
    )
    reference_sorted = _gather_reference_scores(
        files, ("denovo",), lambda df: trusted_denovo_mask(df, score_threshold), ["SCORE_denovo"]
    )["denovo"]
    logger.info("Trusted de novo reference: %d rows", len(reference_sorted))

    logger.info("Scoring TP_GOODNESS_denovo and updating %d grove_forest file(s)", len(files))
    other_scores = _add_denovo_goodness_column(files, reference_sorted, score_threshold)

    divergence = _ks_divergence(reference_sorted, other_scores)
    cutoff = float(divergence["statistic_location"])
    logger.info("KS divergence: %s", divergence)
    logger.info("Good/bad cutoff (denovo): %s", cutoff)

    summary_rows = [
        {"comparison": "trusted_denovo vs rest", **divergence},
        {"comparison": "cutoff_denovo", "cutoff_iso_scores": cutoff},
    ]
    pd.DataFrame(summary_rows).to_csv(qc_dir / "ks_vs_tp_summary.csv", index=False)

    panel = _EcdfPanel(
        reference_sorted=reference_sorted,
        reference_label="trusted de novo",
        other=other_scores,
        other_label="rest of de novo",
        cutoff=cutoff,
        xlabel="ISO_scores_denovo (0-1 scale; 1 = good, 0 = bad)",
        title="ECDF: trusted de novo vs rest of de novo",
    )
    _plot_ks_vs_tp([panel], qc_dir / "ks_vs_tp.svg")
    _write_denovo_good_bad_lists(files, cutoff, grove_forest_dir, qc_dir)

    logger.info("Postprocess QC written to %s", qc_dir)
    return qc_dir
