from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

ID_COLUMNS = ["SpecId", "RAW_FILE", "SCAN_NUMBER"]
SIDES = ("database", "denovo")
SIDE_ONLY_MERGE = {"database": "database_only", "denovo": "denovo_only"}
SCORE_COLUMNS = {
    "SCORE_denovo": "CASANOVO_SCORE",
    "SCORE_database": "DATABASE_SCORE",
    "percolator_score_database": "PERCOLATOR_SCORE_DATABASE",
}


def _gather_reference_scores(files: list[Path]) -> dict[str, np.ndarray]:
    """Pass 1: sorted per-side ISO_scores of the trusted-shared PSMs across every file."""
    reference: dict[str, list[pd.Series]] = {side: [] for side in SIDES}
    for path in files:
        columns = ["_merge", "Label_database", "percolator_score_database", *(f"ISO_scores_{side}" for side in SIDES)]
        df = pd.read_parquet(path, columns=columns)
        trusted = df.loc[
            (df["_merge"] == "shared") & (df["Label_database"] == 1) & (df["percolator_score_database"] > 0)
        ]
        for side in SIDES:
            reference[side].append(trusted[f"ISO_scores_{side}"].dropna())
    return {side: np.sort(pd.concat(parts, ignore_index=True).to_numpy()) for side, parts in reference.items()}


def _goodness(reference_sorted: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Fraction of the reference at least as anomalous (<=) as each of ``scores``; NaN scores stay NaN."""
    n = len(reference_sorted)
    filled = np.nan_to_num(scores, nan=-np.inf)
    result = np.searchsorted(reference_sorted, filled, side="right") / n
    return np.where(np.isnan(scores), np.nan, result)


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


def _write_good_bad_lists(files: list[Path], cutoff: dict[str, float], grove_forest_dir: Path, qc_dir: Path) -> None:
    """Pass 3: classify every PSM per side, plot the detection-level split, write good.csv/bad.csv."""
    from ..plotting.plotting import plot_detection_level_counts  # local: plotting.py imports this module at load time

    columns = [
        *ID_COLUMNS,
        "_merge",
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
    combined["DETECTION_LEVEL"] = combined.pop("_merge").map(
        {"shared": "shared", "database_only": "database", "denovo_only": "denovo"}
    )

    shared_scan = combined["DETECTION_LEVEL"] == "shared"
    conflicting = shared_scan & ~combined["sequence_match"]
    psm_counts = pd.Series(
        {
            "shared": int((shared_scan & combined["sequence_match"]).sum()),
            "database": int(((combined["DETECTION_LEVEL"] == "database") | conflicting).sum()),
            "denovo": int(((combined["DETECTION_LEVEL"] == "denovo") | conflicting).sum()),
        }
    )
    combined = combined.drop(columns=["sequence_match"])
    combined["SEQUENCE"] = combined.pop("SEQUENCE_denovo").combine_first(combined.pop("SEQUENCE_database"))
    combined["CASANOVO_AA_SCORE"] = combined.pop("AA_SCORE_denovo").str.replace("|", ",", regex=False)
    combined = combined.rename(columns=SCORE_COLUMNS)

    # Each side is called GOOD/BAD against its own side-matched KS cutoff, not a blended average.
    for side in SIDES:
        scores = combined[f"ISO_scores_{side}"]
        combined[f"CALL_{side}"] = np.where(scores.notna(), np.where(scores >= cutoff[side], "GOOD", "BAD"), None)

    is_denovo_side = combined["DETECTION_LEVEL"].eq("denovo")
    combined["TP_GOODNESS"] = np.where(is_denovo_side, combined["TP_GOODNESS_denovo"], combined["TP_GOODNESS_database"])
    combined["CALL"] = np.where(is_denovo_side, combined["CALL_denovo"], combined["CALL_database"])
    combined = combined.sort_values("TP_GOODNESS", ascending=False).reset_index(drop=True)

    plot_detection_level_counts(psm_counts, qc_dir / "psm_overlap.svg")

    for call, name in (("GOOD", "good"), ("BAD", "bad")):
        subset = combined.loc[combined["CALL"] == call].drop(columns=["CALL", "TP_GOODNESS"])
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
    from ..plotting.plotting import plot_ks_vs_tp

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
    other_scores: dict[str, list[np.ndarray]] = {only_name: [] for only_name in SIDE_ONLY_MERGE.values()}
    for path in files:
        df = pd.read_parquet(path)
        for side in SIDES:
            df[f"TP_GOODNESS_{side}"] = _goodness(reference_sorted[side], df[f"ISO_scores_{side}"].to_numpy())
        df.to_parquet(path, index=False, engine="pyarrow")
        for side, only_name in SIDE_ONLY_MERGE.items():
            other_scores[only_name].append(df.loc[df["_merge"] == only_name, f"ISO_scores_{side}"].to_numpy())
    other_scores = {name: np.concatenate(parts) for name, parts in other_scores.items()}
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

    plot_ks_vs_tp(reference_sorted, other_scores, cutoff, qc_dir / "ks_vs_tp.svg")
    _write_good_bad_lists(files, cutoff, grove_forest_dir, qc_dir)

    logger.info("Postprocess QC written to %s", qc_dir)
    return qc_dir


def _add_denovo_goodness_column(files: list[Path], reference_sorted: np.ndarray, score_threshold: float) -> np.ndarray:
    """Write TP_GOODNESS_denovo to every file; return ISO_scores_denovo of the untrusted rest, for the KS test."""
    other_scores = []
    for path in files:
        df = pd.read_parquet(path)
        df["TP_GOODNESS_denovo"] = _goodness(reference_sorted, df["ISO_scores_denovo"].to_numpy())
        df.to_parquet(path, index=False, engine="pyarrow")
        has_call = df["_merge"].isin(("denovo_only", "shared"))
        other_mask = has_call & ~pd.to_numeric(df["SCORE_denovo"], errors="coerce").ge(score_threshold)
        other_scores.append(df.loc[other_mask, "ISO_scores_denovo"].to_numpy())
    return np.concatenate(other_scores)


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
    combined["CASANOVO_AA_SCORE"] = combined.pop("AA_SCORE_denovo").str.replace("|", ",", regex=False)
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
    from ..plotting.plotting import plot_denovo_ks_vs_tp  # local: plotting.py imports this module at load time

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
        trusted = df.loc[
            df["_merge"].isin(("denovo_only", "shared"))
            & pd.to_numeric(df["SCORE_denovo"], errors="coerce").ge(score_threshold)
        ]
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

    plot_denovo_ks_vs_tp(reference_sorted, other_scores, cutoff, qc_dir / "ks_vs_tp.svg")
    _write_denovo_good_bad_lists(files, cutoff, grove_forest_dir, qc_dir)

    logger.info("Postprocess QC written to %s", qc_dir)
    return qc_dir
