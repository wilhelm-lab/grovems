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

# Merge categories that carry each side's own identification -- "shared" rows have
# both (a scan with a hit from both engines, not necessarily the same peptide), so
# they get scored independently on each side (see score_grove_forest_file).
_SIDE_MERGE_VALUES = {
    "database": ("database_only", "shared"),
    "denovo": ("denovo_only", "shared"),
}

# Which independent per-side score backs the legacy single-value ISO_scores/ISO_labels
# columns, kept for existing consumers (grovems.postprocess): database-side for
# database_only/shared, denovo-side for denovo_only -- the same selection the old
# single-score _MERGE_CATEGORIES made before both sides were scored independently.
_LEGACY_SCORE_SIDE = {"database_only": "database", "denovo_only": "denovo", "shared": "database"}


def select_suffixed_search_columns(
    merged_df: pd.DataFrame,
    search: str,
    existing_columns: Optional[set[str]] = None,
) -> list[str]:
    """List merged_df columns ending in _{search} whose unsuffixed name isn't already in existing_columns."""
    suffix = f"_{search}"
    existing_columns = existing_columns or set()
    return [
        col for col in merged_df.columns if col.endswith(suffix) and col.removesuffix(suffix) not in existing_columns
    ]


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


def trusted_denovo_mask(merged_df: pd.DataFrame, threshold: float) -> pd.Series:
    """Rows counted as high-confidence de novo PSMs: has a de novo call, SCORE_denovo >= threshold."""
    return merged_df["_merge"].isin(_SIDE_MERGE_VALUES["denovo"]) & pd.to_numeric(
        merged_df["SCORE_denovo"], errors="coerce"
    ).ge(threshold)


def select_training_candidates(
    merged_df: pd.DataFrame, training_source: str = "percolator_percentile", denovo_score_threshold: float = 0.9
) -> pd.DataFrame:
    """Pick one file's high-confidence PSM rows, before the global percentile cutoff (percolator_percentile only)."""
    base_cols = ["SpecId", "_merge"]
    if training_source == "denovo_score":
        mask = trusted_denovo_mask(merged_df, denovo_score_threshold)
        cols = base_cols + select_suffixed_search_columns(merged_df, "denovo", set(base_cols))
    else:
        mask = trusted_shared_mask(merged_df)
        cols = base_cols + select_suffixed_search_columns(merged_df, "database", set(base_cols))
    return merged_df.loc[mask, cols].copy()


def build_training_set(candidates: pd.DataFrame, training_source: str = "percolator_percentile") -> pd.DataFrame:
    """Apply the Percolator-percentile cutoff (percolator_percentile only) and unsuffix feature columns."""
    if training_source == "denovo_score":
        train_set = candidates  # uniquely owned by this call (caller passes a fresh pd.concat(...) result)
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
    """Add per-side and legacy ISO_labels/ISO_scores columns to ``merged_df`` in place.

    Each side (database, denovo) is scored independently on its own suffixed feature
    columns, wherever that side has an identification at all: database_only/shared for
    the database side, denovo_only/shared for the denovo side. ``shared`` rows -- a scan
    with a hit from both engines, not necessarily the same peptide -- therefore get both
    an ``ISO_scores_database`` and an ``ISO_scores_denovo``, each scored from that row's
    own feature vector for that side.

    ``ISO_scores``/``ISO_labels`` remain the legacy single-value view existing
    consumers (``grovems.postprocess``) read: for every row, the same side the old
    single-score ``_MERGE_CATEGORIES`` picked (database-side for database_only/shared,
    denovo-side for denovo_only). ``ISO_scores_side`` records which of the two
    independent columns that legacy value came from.

    ``ISO_scores*`` here are still the raw (unbounded) SUOD ``decision_function``
    output, higher = more anomalous; :func:`_rescale_iso_scores` rescales every score
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

    merged_df["ISO_scores_side"] = merged_df["_merge"].map(_LEGACY_SCORE_SIDE)
    is_denovo_side = merged_df["ISO_scores_side"].eq("denovo")
    merged_df["ISO_scores"] = np.where(is_denovo_side, merged_df["ISO_scores_denovo"], merged_df["ISO_scores_database"])
    merged_df["ISO_labels"] = np.where(is_denovo_side, merged_df["ISO_labels_denovo"], merged_df["ISO_labels_database"])


def _rescale_series(series: pd.Series, score_min: float, score_max: float) -> pd.Series:
    spread = score_max - score_min
    if spread <= 0:
        return series.where(series.isna(), 1.0)
    return 1.0 - (series - score_min) / spread


def _rescale_iso_scores(files: list[Path], score_min: float, score_max: float) -> None:
    """Min-max normalize every ISO_scores* column to [0, 1] across every file, in place.

    Inverted relative to the raw SUOD output so the final scale reads naturally:
    ``1.0`` = least anomalous (good), ``0.0`` = most anomalous (bad).
    ``ISO_scores_database``/``ISO_scores_denovo`` share the legacy ``ISO_scores``
    column's ``score_min``/``score_max`` so all three stay on one comparable scale;
    rows missing a side (e.g. ``ISO_scores_denovo`` on a database_only row) stay null.
    """
    for path in files:
        df = pd.read_parquet(path)
        for column in ("ISO_scores", "ISO_scores_database", "ISO_scores_denovo"):
            df[column] = _rescale_series(df[column], score_min, score_max)
        df.to_parquet(path, index=False, engine="pyarrow")


def run(
    grove_forest_dir: Path,
    feature_cols: list[str],
    model_dir: Path,
    *,
    training_source: str = "percolator_percentile",
    denovo_score_threshold: float = 0.9,
) -> None:
    """Fit a SUOD model on grove_forest/results/*.parquet, then score and update those files in place."""
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

    logger.info("Scoring and updating %d grove_forest file(s)", len(files))
    score_min, score_max = np.inf, -np.inf
    for path in files:
        merged_df = pd.read_parquet(path)
        score_grove_forest_file(merged_df, model, feature_cols)
        combined_scores = pd.concat([merged_df["ISO_scores_database"], merged_df["ISO_scores_denovo"]]).dropna()
        score_min = min(score_min, combined_scores.min())
        score_max = max(score_max, combined_scores.max())
        merged_df.to_parquet(path, index=False, engine="pyarrow")

    logger.info(
        "Rescaling ISO_scores to [0, 1] (1=good, 0=bad) across %d file(s); raw range [%.6f, %.6f]",
        len(files),
        score_min,
        score_max,
    )
    _rescale_iso_scores(files, score_min, score_max)
