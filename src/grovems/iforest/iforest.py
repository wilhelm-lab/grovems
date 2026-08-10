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

_MERGE_CATEGORIES = (("database_only", "database"), ("denovo_only", "denovo"), ("shared", "database"))


def select_suffixed_search_columns(
    merged_df: pd.DataFrame,
    search: str,
    existing_columns: Optional[set[str]] = None,
) -> list[str]:
    """List ``merged_df`` columns ending in ``_{search}`` whose unsuffixed name is new.

    Args:
        merged_df: The merged (``_database``/``_denovo``-suffixed) dataframe.
        search: Suffix to match, e.g. ``"database"`` or ``"denovo"``.
        existing_columns: Unsuffixed names to exclude even if a ``_{search}`` column exists.

    Returns:
        Matching (still-suffixed) column names.
    """
    suffix = f"_{search}"
    existing_columns = existing_columns or set()
    return [
        col for col in merged_df.columns if col.endswith(suffix) and col.removesuffix(suffix) not in existing_columns
    ]


def require_unique_columns(df: pd.DataFrame, frame_name: str) -> None:
    """Raise if ``df`` has duplicate column names (typically after suffix stripping).

    Args:
        df: DataFrame to check.
        frame_name: Name used in the error message, for context.

    Raises:
        ValueError: If any column name is duplicated.
    """
    duplicated = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"{frame_name} has duplicate columns after suffix normalization: {duplicated}")


def trusted_shared_mask(merged_df: pd.DataFrame) -> pd.Series:
    """Rows counted as high-confidence shared PSMs: shared, target, positive database score.

    The single definition of "trustworthy shared PSM" used both to pick IForest's
    training candidates (:func:`select_training_candidates`) and, downstream, as the
    ECDF reference population in ``grovems.postprocess`` -- kept in one place so the
    two stages can't silently drift apart.
    """
    return (
        (merged_df["_merge"] == "shared")
        & (merged_df["Label_database"] == 1)
        & (merged_df["percolator_score_database"] > 0)
    )


def select_training_candidates(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Pick one file's high-confidence shared-PSM rows, before the global percentile cutoff.

    The cheap per-file filter for pass 1 -- only :func:`trusted_shared_mask` rows are
    candidates at all; :func:`build_training_set` applies the actual cutoff once
    candidates from every file have been combined.
    """
    base_cols = ["SpecId", "_merge"]
    cols = base_cols + select_suffixed_search_columns(merged_df, "database", set(base_cols))
    return merged_df.loc[trusted_shared_mask(merged_df), cols].copy()


def build_training_set(candidates: pd.DataFrame) -> pd.DataFrame:
    """Apply the 70th-percentile Percolator-score cutoff and unsuffix feature columns.

    Args:
        candidates: Training candidates from every raw file, concatenated (see
            :func:`select_training_candidates`) -- a stricter subset than what gets
            scored later (every shared PSM, in :func:`score_grove_forest_file`).

    Returns:
        Unsuffixed feature matrix (plus ``SpecId``/``_merge``) for training.
    """
    cutoff = np.percentile(candidates.percolator_score_database.to_numpy(), 70)
    train_set = candidates.query(f"percolator_score_database > {cutoff}").copy()
    train_set.columns = train_set.columns.str.replace("_database", "")
    require_unique_columns(train_set, "training set")
    logger.info("Training set: %d rows, %d columns", train_set.shape[0], train_set.shape[1])
    return train_set


def build_suod_model(train_row_count: int) -> SUOD:
    """Build a SUOD ensemble of isolation forests, sized to the training set.

    Args:
        train_row_count: Number of rows in the training set; used to pick a spread of
            ``max_samples`` values (powers of 2 up to the training set size).

    Returns:
        An unfit :class:`pyod.models.suod.SUOD` model with up to 8 IForest base estimators.
    """
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
    """Add ISO_labels/ISO_scores columns to ``merged_df`` in place, for every row.

    Each ``_merge`` category (database_only/denovo_only/shared) is scored on its own
    suffixed feature columns -- shared rows use the database-side ones, the same
    convention used elsewhere for shared PSMs. ``ISO_scores`` here is still the raw
    (unbounded) SUOD ``decision_function`` output, higher = more anomalous;
    :func:`_rescale_iso_scores` rescales it to the final 0-1 (1=good, 0=bad) scale once
    every file has been scored.
    """
    merged_df["ISO_labels"] = np.nan
    merged_df["ISO_scores"] = np.nan
    for merge_value, suffix_source in _MERGE_CATEGORIES:
        mask = merged_df["_merge"] == merge_value
        if not mask.any():
            continue
        suffixed_cols = [f"{col}_{suffix_source}" for col in feature_cols]
        missing = [col for col in suffixed_cols if col not in merged_df.columns]
        if missing:
            raise KeyError(f"Missing feature columns for {merge_value}: {missing}")
        subset = merged_df.loc[mask, suffixed_cols]
        subset.columns = feature_cols
        merged_df.loc[mask, "ISO_labels"] = model.predict(subset)
        merged_df.loc[mask, "ISO_scores"] = model.decision_function(subset)


def _rescale_iso_scores(files: list[Path], score_min: float, score_max: float) -> None:
    """Min-max normalize ISO_scores to [0, 1] across every file, in place.

    Inverted relative to the raw SUOD output so the final scale reads naturally:
    ``1.0`` = least anomalous (good), ``0.0`` = most anomalous (bad).
    """
    spread = score_max - score_min
    for path in files:
        df = pd.read_parquet(path)
        df["ISO_scores"] = 1.0 if spread <= 0 else 1.0 - (df["ISO_scores"] - score_min) / spread
        df.to_parquet(path, index=False, engine="pyarrow")


def run(grove_forest_dir: Path, feature_cols: list[str], model_dir: Path) -> None:
    """Fit a SUOD model on grove_forest/results/*.parquet, then score and update those files in place."""
    results_dir = grove_forest_dir / "results"
    files = sorted(results_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No grove_forest parquet files found in {results_dir}")

    logger.info("Gathering SUOD training candidates from %d file(s)", len(files))
    candidates = [select_training_candidates(pd.read_parquet(path)) for path in files]
    train_set = build_training_set(pd.concat(candidates, ignore_index=True))
    del candidates

    logger.info("Training SUOD model on %d rows", len(train_set))
    model = build_suod_model(train_set.shape[0])
    model.fit(train_set[feature_cols])
    del train_set

    model_dir.mkdir(parents=True, exist_ok=True)
    dump(model, model_dir / "SUOD_model.pkl")

    logger.info("Scoring and updating %d grove_forest file(s)", len(files))
    score_min, score_max = np.inf, -np.inf
    for path in files:
        merged_df = pd.read_parquet(path)
        score_grove_forest_file(merged_df, model, feature_cols)
        score_min = min(score_min, merged_df["ISO_scores"].min())
        score_max = max(score_max, merged_df["ISO_scores"].max())
        merged_df.to_parquet(path, index=False, engine="pyarrow")

    logger.info(
        "Rescaling ISO_scores to [0, 1] (1=good, 0=bad) across %d file(s); raw range [%.6f, %.6f]",
        len(files),
        score_min,
        score_max,
    )
    _rescale_iso_scores(files, score_min, score_max)
