from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import dump
from pyod.models.iforest import IForest
from pyod.models.suod import SUOD

logger = logging.getLogger(__name__)

_SIDE_MERGE_VALUES = {
    "database": ("database_only", "shared"),
    "denovo": ("denovo_only", "shared"),
}


def select_training_candidates(
    merged_df: pd.DataFrame, training_source: str = "percolator_percentile", denovo_score_threshold: float = 0.9
) -> pd.DataFrame:
    """Pick one file's high-confidence PSM rows, before the global percentile cutoff (percolator_percentile only)."""
    base_cols = ["SpecId", "_merge"]
    if training_source == "denovo_score":
        suffix = "_denovo"
        mask = merged_df["_merge"].isin(_SIDE_MERGE_VALUES["denovo"]) & pd.to_numeric(
            merged_df["SCORE_denovo"], errors="coerce"
        ).ge(denovo_score_threshold)
    else:
        suffix = "_database"
        mask = (
            (merged_df["_merge"] == "shared")
            & (merged_df["Label_database"] == 1)
            & (merged_df["percolator_score_database"] > 0)
        )
    cols = base_cols + [
        col for col in merged_df.columns if col.endswith(suffix) and col.removesuffix(suffix) not in base_cols
    ]
    return merged_df.loc[mask, cols].copy()


def build_training_set(candidates: pd.DataFrame, training_source: str = "percolator_percentile") -> pd.DataFrame:
    """Apply the Percolator-percentile cutoff (percolator_percentile only) and unsuffix feature columns."""
    if training_source == "denovo_score":
        train_set = candidates
        suffix = "_denovo"
    else:
        cutoff = np.percentile(candidates.percolator_score_database.to_numpy(), 70)
        train_set = candidates.query(f"percolator_score_database > {cutoff}").copy()
        suffix = "_database"
    train_set.columns = train_set.columns.str.replace(suffix, "")
    duplicated = train_set.columns[train_set.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"training set has duplicate columns after suffix normalization: {duplicated}")
    logger.info("Training set: %d rows, %d columns", train_set.shape[0], train_set.shape[1])
    return train_set


def build_suod_model(train_row_count: int) -> SUOD:
    """Build an unfit SUOD ensemble of up to 8 isolation forests, max_samples as powers of 2 up to train_row_count."""
    max_samples: list[int] = []
    power = 8
    while (value := 2**power) < train_row_count:
        max_samples.append(value)
        power += 1

    detector_list = [
        IForest(n_estimators=100, max_samples=sample, contamination=1e-9, n_jobs=30) for sample in max_samples
    ]

    model = SUOD(base_estimators=detector_list[:8], n_jobs=1, combination="average", verbose=True)
    logger.info("Model summary:\n%s", model)
    return model


def score_grove_forest_file(merged_df: pd.DataFrame, model: SUOD, feature_cols: list[str]) -> None:
    """Add per-side ISO_labels_<side>/ISO_scores_<side> columns to ``merged_df`` in place.

    Each side (database, denovo) is scored independently on its own suffixed feature
    columns, wherever that side has an identification at all: database_only/shared for
    the database side, denovo_only/shared for the denovo side. ``shared`` rows -- a scan
    with a hit from both engines, not necessarily the same peptide -- therefore get both
    an ``ISO_scores_database`` and an ``ISO_scores_denovo``, each scored from that row's
    own feature vector for that side.

    ``ISO_scores*`` here are still the raw (unbounded) SUOD ``decision_function``
    output, higher = more anomalous; :func:`run` rescales every score
    column to the final 0-1 (1=good, 0=bad) scale once every file has been scored.
    """
    for side in ("database", "denovo"):
        merged_df[f"ISO_labels_{side}"] = np.nan
        merged_df[f"ISO_scores_{side}"] = np.nan
        mask = merged_df["_merge"].isin(_SIDE_MERGE_VALUES[side])
        if not mask.any():
            continue
        suffixed_cols = [f"{col}_{side}" for col in feature_cols]
        missing = [col for col in suffixed_cols if col not in merged_df.columns]
        if missing:
            raise KeyError(f"Missing feature columns for {side}: {missing}")
        subset = merged_df.loc[mask, suffixed_cols]
        subset.columns = feature_cols
        merged_df.loc[mask, f"ISO_labels_{side}"] = model.predict(subset)
        merged_df.loc[mask, f"ISO_scores_{side}"] = model.decision_function(subset)


def run(
    grove_forest_dir: Path,
    feature_cols: list[str],
    model_dir: Path,
    *,
    training_source: str = "percolator_percentile",
    denovo_score_threshold: float = 0.9,
    results_output_dir: Optional[Path] = None,
) -> None:
    """Fit a SUOD model on grove_forest/results/*.parquet, then score those files.

    Written back in place unless ``results_output_dir`` is given, in which case scored
    copies are written there instead, leaving ``grove_forest_dir`` untouched.
    """
    results_dir = grove_forest_dir / "results"
    files = sorted(results_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No grove_forest parquet files found in {results_dir}")

    logger.info("Gathering SUOD training candidates from %d file(s) (source=%s)", len(files), training_source)
    candidates = [
        select_training_candidates(pd.read_parquet(path), training_source, denovo_score_threshold) for path in files
    ]
    train_set = build_training_set(pd.concat(candidates, ignore_index=True), training_source)
    del candidates
    if train_set.empty:
        raise ValueError(
            f"No training candidates found (training_source={training_source!r}, "
            f"denovo_score_threshold={denovo_score_threshold}) -- check the threshold against the "
            "actual SCORE_denovo/percolator_score_database distribution in this dataset."
        )

    logger.info("Training SUOD model on %d rows", len(train_set))
    model = build_suod_model(train_set.shape[0])
    model.fit(train_set[feature_cols])
    del train_set

    model_dir.mkdir(parents=True, exist_ok=True)
    dump(model, model_dir / "SUOD_model.pkl")

    output_dir = results_output_dir or results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Scoring %d grove_forest file(s) into %s", len(files), output_dir)
    score_min, score_max = np.inf, -np.inf
    output_files = []
    for path in files:
        merged_df = pd.read_parquet(path)
        score_grove_forest_file(merged_df, model, feature_cols)
        combined_scores = pd.concat([merged_df["ISO_scores_database"], merged_df["ISO_scores_denovo"]]).dropna()
        score_min = min(score_min, combined_scores.min())
        score_max = max(score_max, combined_scores.max())
        out_path = output_dir / path.name
        merged_df.to_parquet(out_path, index=False, engine="pyarrow")
        output_files.append(out_path)

    spread = score_max - score_min
    for out_path in output_files:
        df = pd.read_parquet(out_path)
        for column in ("ISO_scores_database", "ISO_scores_denovo"):
            series = df[column]
            df[column] = series.where(series.isna(), 1.0) if spread <= 0 else 1.0 - (series - score_min) / spread
        df.to_parquet(out_path, index=False, engine="pyarrow")

    logger.info(
        "Rescaling ISO_scores to [0, 1] (1=good, 0=bad) across %d file(s); raw range [%.6f, %.6f]",
        len(output_files),
        score_min,
        score_max,
    )
