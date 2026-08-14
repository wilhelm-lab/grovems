from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .psa_classifier import PSA

logger = logging.getLogger(__name__)

PSA_MAX_WORKERS = 8
_WORKER_PSA = None


def add_sequence_match_columns(df: pd.DataFrame) -> None:
    """Add unmodified/modified sequence and precursor-charge match columns, in place."""
    sequence_columns = ["SEQUENCE_database", "SEQUENCE_denovo"]
    if not set(sequence_columns).issubset(df.columns):
        df["unmodified_sequence_match"] = False
        df["modified_sequence_match"] = False
        df["precursor_charge_match"] = False
        df["sequence_match"] = False
        return

    df["unmodified_sequence_match"] = (
        df["SEQUENCE_database"].notna() & df["SEQUENCE_denovo"].notna() & df["SEQUENCE_database"].eq(df["SEQUENCE_denovo"])
    )

    modified_columns = ["MODIFIED_SEQUENCE_database", "MODIFIED_SEQUENCE_denovo"]
    if set(modified_columns).issubset(df.columns):
        df["modified_sequence_match"] = (
            df["MODIFIED_SEQUENCE_database"].notna()
            & df["MODIFIED_SEQUENCE_denovo"].notna()
            & df["MODIFIED_SEQUENCE_database"].eq(df["MODIFIED_SEQUENCE_denovo"])
        )
    else:
        df["modified_sequence_match"] = False

    charge_columns = ["PRECURSOR_CHARGE_database", "PRECURSOR_CHARGE_denovo"]
    if set(charge_columns).issubset(df.columns):
        database_charge = pd.to_numeric(df["PRECURSOR_CHARGE_database"], errors="coerce")
        denovo_charge = pd.to_numeric(df["PRECURSOR_CHARGE_denovo"], errors="coerce")
        df["precursor_charge_match"] = database_charge.notna() & denovo_charge.notna() & database_charge.eq(denovo_charge)
    else:
        df["precursor_charge_match"] = False

    df["sequence_match"] = df["modified_sequence_match"] & df["precursor_charge_match"]


_MERGE_KEY_COLUMNS = ("RAW_FILE", "SCAN_NUMBER")


def _merge_search_results(merged_database: pd.DataFrame, merged_denovo: pd.DataFrame) -> pd.DataFrame:
    def suffix_non_key_columns(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        return df.rename(columns={c: f"{c}{suffix}" for c in df.columns if c not in _MERGE_KEY_COLUMNS})

    # Suffix columns ourselves rather than pandas' merge(suffixes=...), which only
    # suffixes columns that collide by name -- if a raw file has no data at all on one
    # side, that side has nothing to collide with, so its columns would come through
    # unsuffixed instead of e.g. SEQUENCE_denovo.
    merged = suffix_non_key_columns(merged_database, "_database").merge(
        suffix_non_key_columns(merged_denovo, "_denovo"),
        how="outer",
        on=list(_MERGE_KEY_COLUMNS),
        indicator=True,
    )

    # Coalesce SpecId (present, suffixed, on whichever side(s) had it).
    left_specid = merged["SpecId_database"] if "SpecId_database" in merged.columns else None
    right_specid = merged["SpecId_denovo"] if "SpecId_denovo" in merged.columns else None
    if left_specid is not None or right_specid is not None:
        empty = pd.Series(index=merged.index, dtype="object")
        merged["SpecId"] = (left_specid if left_specid is not None else empty).combine_first(
            right_specid if right_specid is not None else empty
        )

    # A side with no data at all for this file never had a SEQUENCE column to suffix
    # above -- add it as all-null so required-column checks below don't see this as
    # a missing-data bug.
    for column in ("SEQUENCE_database", "SEQUENCE_denovo"):
        if column not in merged.columns:
            merged[column] = None

    # Flag scans where the database side called more than one distinct peptide (chimeric spectra).
    chimeric_required = ["RAW_FILE", "SCAN_NUMBER", "SEQUENCE"]
    chimeric_scan_keys: set[tuple[str, int]] = set()
    if not merged_database.empty and set(chimeric_required).issubset(merged_database.columns):
        database = merged_database.dropna(subset=chimeric_required).copy()
        database["SCAN_NUMBER"] = pd.to_numeric(database["SCAN_NUMBER"], errors="coerce")
        database = database.dropna(subset=["SCAN_NUMBER"])
        if not database.empty:
            counts = database.groupby(["RAW_FILE", "SCAN_NUMBER"], observed=True)["SEQUENCE"].nunique()
            chimeric_scan_keys = {(str(raw), int(scan)) for raw, scan in counts[counts > 1].index}

    if {"RAW_FILE", "SCAN_NUMBER"}.issubset(merged.columns) and chimeric_scan_keys:
        scan_number = pd.to_numeric(merged["SCAN_NUMBER"], errors="coerce")
        keys = pd.Series(zip(merged["RAW_FILE"].astype(str), scan_number), index=merged.index, dtype="object")
        merged["chimeric"] = keys.map(lambda key: pd.notna(key[1]) and (key[0], int(key[1])) in chimeric_scan_keys)
    else:
        merged["chimeric"] = False

    add_sequence_match_columns(merged)

    merged["_merge"] = merged["_merge"].map({"right_only": "denovo_only", "left_only": "database_only", "both": "shared"})
    return merged


def _init_psa_worker() -> None:
    global _WORKER_PSA
    _WORKER_PSA = PSA()


def _run_psa_pair(sequence1, sequence2):
    global _WORKER_PSA
    if _WORKER_PSA is None:
        _WORKER_PSA = PSA()
    try:
        _WORKER_PSA.set_sequences(str(sequence1), str(sequence2))
        _WORKER_PSA.classify()
    except Exception as exc:
        return ("PSA_ERROR", np.nan, f"{type(exc).__name__}: {exc}")
    return (_WORKER_PSA.result.label, _WORKER_PSA.result.levenshtein_distance, None)


def _run_psa_for_pairs(sequences_database: pd.Series, sequences_denovo: pd.Series) -> pd.DataFrame:
    n_workers = PSA_MAX_WORKERS
    total_items = len(sequences_database)
    chunksize = 1 if total_items <= 0 else max(1, total_items // max(1, n_workers * 8))
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_psa_worker) as executor:
        rows = list(
            tqdm(
                executor.map(_run_psa_pair, sequences_database.tolist(), sequences_denovo.tolist(), chunksize=chunksize),
                total=len(sequences_database),
                desc="Running PSA",
            )
        )
    return pd.DataFrame(rows, columns=["PSA", "PSA_LEVENSHTEIN", "PSA_ERROR"], index=sequences_database.index)


def _add_psa_columns(merged_scan: pd.DataFrame) -> None:
    """Classify PSA on shared, sequence-differing scans; add PSA columns to every row.

    Rows outside "shared" (database_only/denovo_only), or shared rows missing a
    sequence on either side, get null PSA columns -- PSA only applies where both a
    database and a de novo call exist for the same scan.
    """
    required = ["_merge", "SEQUENCE_database", "SEQUENCE_denovo"]
    missing = [column for column in required if column not in merged_scan.columns]
    if missing:
        raise KeyError(f"Missing columns needed for PSA: {missing}")

    merged_scan["PSA"] = None
    merged_scan["PSA_LEVENSHTEIN"] = np.nan
    merged_scan["PSA_ERROR"] = None

    shared_mask = (
        merged_scan["_merge"].eq("shared") & merged_scan["SEQUENCE_database"].notna() & merged_scan["SEQUENCE_denovo"].notna()
    )
    same_sequence_mask = shared_mask & merged_scan["sequence_match"]
    different_sequence_mask = shared_mask & ~merged_scan["sequence_match"]

    merged_scan.loc[same_sequence_mask, "PSA"] = "PSA - Tier 0 - IDENTICAL"
    merged_scan.loc[same_sequence_mask, "PSA_LEVENSHTEIN"] = 0

    if different_sequence_mask.any():
        scored = _run_psa_for_pairs(
            merged_scan.loc[different_sequence_mask, "SEQUENCE_database"],
            merged_scan.loc[different_sequence_mask, "SEQUENCE_denovo"],
        )
        merged_scan.loc[different_sequence_mask, "PSA"] = scored["PSA"]
        merged_scan.loc[different_sequence_mask, "PSA_LEVENSHTEIN"] = scored["PSA_LEVENSHTEIN"]
        merged_scan.loc[different_sequence_mask, "PSA_ERROR"] = scored["PSA_ERROR"]


def _branch_schema_columns(directory: Optional[Path]) -> list[str]:
    """Column names of any one file in ``directory`` -- every file in a branch shares the same
    schema (same keep_columns/pin/percolator columns throughout), so one file's schema stands
    in as the template for backfilling a raw file that's entirely missing from this branch.
    """
    if directory is not None:
        first = next(directory.glob("*.parquet"), None)
        if first is not None:
            return pq.ParquetFile(first).schema_arrow.names
    return ["SpecId", "RAW_FILE", "SCAN_NUMBER"]


def _read_parquet_or_empty(path: Optional[Path], template_columns: list[str]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=template_columns)
    return pd.read_parquet(path)


def run(
    database_merged_dir: Optional[Path],
    denovo_merged_dir: Path,
    grove_forest_dir: Path,
    *,
    max_raw_files: Optional[int] = None,
    max_workers: Optional[int] = None,
    overwrite_outputs: bool = False,
) -> Path:
    """Merge every raw file's database+de novo data, run PSA, write grove_forest/results/<raw>.parquet.

    ``database_merged_dir=None`` (denovo_only mode) makes every row come out
    ``_merge="denovo_only"``. Deletes both merged dirs once done -- their data now
    lives in ``grove_forest_dir/results``.
    """
    global PSA_MAX_WORKERS
    PSA_MAX_WORKERS = max_workers or os.cpu_count() or 1
    results_dir = grove_forest_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    database_raw_files = {path.stem for path in database_merged_dir.glob("*.parquet")} if database_merged_dir else set()
    raw_files = sorted(database_raw_files | {path.stem for path in denovo_merged_dir.glob("*.parquet")})
    if max_raw_files is not None:
        raw_files = raw_files[:max_raw_files]

    # A raw file missing entirely from one branch (e.g. database found zero PSMs for it, or
    # denovo_only mode where the whole branch is absent) still needs that side's full column
    # set so every _database/_denovo-suffixed column downstream stages expect actually exists.
    database_template = _branch_schema_columns(database_merged_dir)
    denovo_template = _branch_schema_columns(denovo_merged_dir)

    for raw_file in tqdm(raw_files, desc="Merging + PSA"):
        out_path = results_dir / f"{raw_file}.parquet"
        if out_path.exists() and not overwrite_outputs:
            continue
        try:
            database_path = database_merged_dir / f"{raw_file}.parquet" if database_merged_dir else None
            merged_database = _read_parquet_or_empty(database_path, database_template)
            merged_denovo = _read_parquet_or_empty(denovo_merged_dir / f"{raw_file}.parquet", denovo_template)
            merged_scan = _merge_search_results(merged_database, merged_denovo)
            _add_psa_columns(merged_scan)
            merged_scan.to_parquet(out_path, index=False, engine="pyarrow")
        except Exception:
            logger.exception("%s: merge/PSA failed; skipping", raw_file)
            continue

    if database_merged_dir is not None:
        shutil.rmtree(database_merged_dir, ignore_errors=True)
    shutil.rmtree(denovo_merged_dir, ignore_errors=True)
    return grove_forest_dir
