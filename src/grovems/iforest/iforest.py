from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from joblib import dump
from pyod.models.iforest import IForest
from pyod.models.suod import SUOD

from ..utils import configure_logging

logger = logging.getLogger(__name__)


def load_yaml_config(path: str) -> dict:
    """Read a YAML config file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def select_feature_columns(config: dict, feature_list: Optional[str]) -> list[str]:
    """Pick the SUOD feature columns: from ``--feature_list`` if given, else the config.

    Args:
        config: Parsed ``config_train.yaml`` (needs ``features.selected`` if
            ``feature_list`` is not given).
        feature_list: Comma-separated feature names, or ``None`` to use the config.

    Returns:
        List of feature column names.
    """
    if feature_list:
        return [feature.strip() for feature in feature_list.split(",") if feature.strip()]
    return config["features"]["selected"]


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


def build_test_set(
    merged_df: pd.DataFrame,
    merge_value: str,
    suffix_source: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Build the feature matrix (plus ``SpecId``/``_merge``) for one ``_merge`` category.

    Args:
        merged_df: The merged (``_database``/``_denovo``-suffixed) dataframe.
        merge_value: ``_merge`` value to select, e.g. ``"database_only"`` or ``"shared"``.
        suffix_source: Which suffix's feature columns to pull (``"database"`` or ``"denovo"``).
        feature_cols: Feature column names (unsuffixed) to keep.

    Returns:
        DataFrame with ``SpecId``, ``_merge``, and the requested (now unsuffixed) feature columns.
    """
    base_cols = ["SpecId", "_merge"]
    cols = base_cols + select_suffixed_search_columns(merged_df, suffix_source, set(base_cols))
    x_test = merged_df.query(f"_merge == '{merge_value}'")[cols].copy()
    logger.info("Prepared %s test section with shape %s", merge_value, x_test.shape)
    x_test.columns = x_test.columns.str.replace(f"_{suffix_source}", "", regex=False)
    require_unique_columns(x_test, f"{merge_value} test set")
    selected_feature_cols = ["SpecId", "_merge"] + feature_cols
    return x_test[selected_feature_cols]


def build_search_output(merged_df: pd.DataFrame, merge_value: str, suffix_source: str) -> pd.DataFrame:
    """Build the full (non-feature-filtered) output rows for one ``_merge`` category.

    Args:
        merged_df: The merged (``_database``/``_denovo``-suffixed) dataframe.
        merge_value: ``_merge`` value to select, e.g. ``"database_only"`` or ``"shared"``.
        suffix_source: Which suffix's columns to pull for the metadata/output fields.

    Returns:
        DataFrame with the common (unsuffixed) columns plus the ``suffix_source``-suffixed
        ones, all unsuffixed.
    """
    common_columns = [col for col in merged_df.columns if not col.endswith("_database") and not col.endswith("_denovo")]
    search_columns = select_suffixed_search_columns(merged_df, suffix_source, set(common_columns))

    search_df = merged_df.loc[merged_df["_merge"] == merge_value, common_columns + search_columns].copy()
    search_df.rename(columns=lambda col: col.removesuffix(f"_{suffix_source}"), inplace=True)
    require_unique_columns(search_df, f"{merge_value} output")
    return search_df


def predict_search_set(
    model: SUOD,
    merged_df: pd.DataFrame,
    merge_value: str,
    suffix_source: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Score every PSM in one ``_merge`` category with a fitted SUOD model.

    Args:
        model: A fitted :class:`pyod.models.suod.SUOD` model.
        merged_df: The merged (``_database``/``_denovo``-suffixed) dataframe.
        merge_value: ``_merge`` value to select, e.g. ``"database_only"`` or ``"shared"``.
        suffix_source: Which suffix's feature/output columns to use.
        feature_cols: Feature column names the model expects.

    Returns:
        The category's output rows (see :func:`build_search_output`) with ``ISO_labels``/
        ``ISO_scores`` columns added.
    """
    x_test = build_test_set(merged_df, merge_value, suffix_source, feature_cols)
    logger.info("Predicting %s test set with %d rows", merge_value, len(x_test))
    x_test["ISO_labels"] = model.predict(x_test[feature_cols])
    x_test["ISO_scores"] = model.decision_function(x_test[feature_cols])

    search_output = build_search_output(merged_df, merge_value, suffix_source)
    predictions = x_test.loc[:, ["SpecId", "ISO_labels", "ISO_scores"]]
    search_output.drop_duplicates(subset=["SpecId"], inplace=True)
    predictions.drop_duplicates(subset=["SpecId"], inplace=True)
    return search_output.merge(predictions, how="left", on="SpecId")


def build_training_set(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Select the high-confidence shared-PSM training set for the SUOD model.

    Only ``_merge == "shared"`` target (``Label_database == 1``) PSMs above the 70th
    percentile of database Percolator score are used for training -- a stricter subset
    than what gets scored later (every shared PSM, via :func:`predict_search_set`).

    Args:
        merged_df: The merged (``_database``/``_denovo``-suffixed) dataframe.

    Returns:
        Unsuffixed feature matrix (plus ``SpecId``/``_merge``) for training.
    """
    base_cols = ["SpecId", "_merge"]
    cols = base_cols + select_suffixed_search_columns(merged_df, "database", set(base_cols))

    train_initial = merged_df.query("_merge == 'shared' and Label_database == 1 and percolator_score_database > 0")[
        cols
    ].copy()
    cutoff = np.percentile(
        train_initial.query("percolator_score_database > 0").percolator_score_database.to_numpy(), 70
    )
    train_set = train_initial.query(f"percolator_score_database > {cutoff}").copy()
    train_set.columns = train_set.columns.str.replace("_database", "")
    require_unique_columns(train_set, "training set")
    logger.info("Loaded training set with %d rows and %d columns", train_set.shape[0], train_set.shape[1])
    logger.info("Training columns: %s", list(train_set.columns))
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


def write_partitioned(df: pd.DataFrame, output_path: Path) -> None:
    """Write ``df`` as parquet, partitioned by ``RAW_FILE``.

    Args:
        df: DataFrame to write; must have a ``RAW_FILE`` column.
        output_path: Output directory.
    """
    logger.info("Writing %d rows to %s", len(df), output_path)
    df.to_parquet(output_path, index=False, engine="pyarrow", partition_cols=["RAW_FILE"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--njobs", type=int, default=None)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--allinone", type=str, required=True, help="merged_SCAN parquet dataset to train/score on")
    parser.add_argument("--config", type=str, default=None, help="config_train.yaml (features.selected)")
    parser.add_argument("--feature_list", type=str, default=None, help="Comma-separated list of features")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    config = load_yaml_config(args.config) if args.config else {}
    logger.info("Final configuration loaded:\n%s", yaml.dump(config, sort_keys=False))

    workdir = Path(args.workdir)
    merged_df = pd.read_parquet(args.allinone)
    feature_cols = select_feature_columns(config, args.feature_list)

    train_set = build_training_set(merged_df)

    logger.info("Creating model: SUOD")
    model = build_suod_model(train_set.shape[0])

    logger.info("Training model...")
    model.fit(train_set[feature_cols])

    logger.info("Saving model")
    dump(model, workdir / "SUOD_model.pkl")

    logger.info("Predicting on database-only, denovo-only, and shared PSMs")
    database_results = predict_search_set(model, merged_df, "database_only", "database", feature_cols)
    denovo_results = predict_search_set(model, merged_df, "denovo_only", "denovo", feature_cols)
    # Shared PSMs are scored on their database-side features -- the same convention used
    # elsewhere for shared PSMs (e.g. analysis/scripts/load_iforest.py). They are not
    # dropped: every _merge == "shared" row gets a prediction here, not just the
    # high-confidence subset used above to train the model.
    shared_results = predict_search_set(model, merged_df, "shared", "database", feature_cols)

    write_partitioned(database_results, workdir / "iforest_database")
    write_partitioned(denovo_results, workdir / "iforest_denovo")
    write_partitioned(shared_results, workdir / "iforest_shared")

    logger.info("Writing combined iforest_all (database-only + denovo-only + shared)")
    common_cols = [
        col for col in database_results.columns if col in denovo_results.columns and col in shared_results.columns
    ]
    iforest_all = pd.concat(
        [database_results[common_cols], denovo_results[common_cols], shared_results[common_cols]],
        ignore_index=True,
    )
    write_partitioned(iforest_all, workdir / "iforest_all")

    logger.info("Done.")


if __name__ == "__main__":
    main()
