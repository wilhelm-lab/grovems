from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Must be set before importing anything that might touch numba/matplotlib at import
# time (below), hence these two imports are not at the very top of the file.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.cwd() / ".numba-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mpl"))

from .eval import DATABASE_COLUMNS, DENOVO_COLUMNS, PERCOLATOR_COLUMNS, ResultLoader  # noqa: E402
from .psa_classifier import PSA  # noqa: E402
from ..utils import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Set from CLI args in main() (or by runner.run() before calling run_pipeline()) --
# referenced as module globals by the functions below (mirrors the original script's
# structure).
OUTPUT_DIR: Path
CACHE_DIR: Path
PSA_MAX_WORKERS = 8

PIN_CHUNKSIZE = 250_000
PERCOLATOR_CHUNKSIZE = 500_000
SPECID_RAW_PATTERN = r"^(?P<RAW_FILE>.+?)-\d+-.+-\d+$"

MERGED_DATASET_NAMES = ("merged_database", "merged_denovo", "merged_PSM", "merged_SCAN")
RAW_ERROR_COLUMNS = ["RAW_FILE", "error_type", "error_message", "traceback", "failed_at_utc"]
_WORKER_PSA = None


def safe_raw_file(raw_file: str) -> str:
    return quote(str(raw_file), safe="")


def raw_partition_dir(root: Path, raw_file: str) -> Path:
    return root / safe_raw_file(raw_file)


def success_marker(path: Path) -> Path:
    return path / "_SUCCESS"


def reset_dir(path: Path, rebuild: bool) -> None:
    if rebuild and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_raw_file(spec_ids: pd.Series) -> pd.Series:
    return spec_ids.astype("string").str.extract(SPECID_RAW_PATTERN, expand=False)


def write_part(df: pd.DataFrame, root: Path, raw_file: str, filename: str) -> None:
    if df.empty:
        return
    out_dir = raw_partition_dir(root, raw_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / filename, index=False, engine="pyarrow")


def read_partition(root: Path, raw_file: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = raw_partition_dir(root, raw_file)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path, columns=columns)


def materialize_table_by_raw(
    input_path: Path,
    output_root: Path,
    *,
    id_column: str,
    sep: str,
    chunksize: int,
    rebuild: bool,
    usecols: list[str] | None = None,
    source_label: str = "part",
) -> None:
    if success_marker(output_root).exists() and not rebuild:
        logger.info("Using cached partitions in %s", output_root)
        return

    reset_dir(output_root, rebuild=True)
    reader = pd.read_csv(input_path, sep=sep, chunksize=chunksize, usecols=usecols)

    for chunk_idx, chunk in enumerate(tqdm(reader, desc=f"Partitioning {input_path.name}")):
        if chunk.empty:
            continue
        chunk["RAW_FILE"] = extract_raw_file(chunk[id_column])
        chunk = chunk.loc[chunk["RAW_FILE"].notna()].copy()
        for raw_file, part in chunk.groupby("RAW_FILE", sort=False):
            write_part(part, output_root, raw_file, f"{source_label}-{chunk_idx:06d}.parquet")
        del chunk
        gc.collect()

    success_marker(output_root).touch()


def materialize_pin_and_percolator_caches(loader: ResultLoader, *, rebuild: bool = False) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for prefix in ("database", "denovo"):
        pin_path = loader.path_mapper[f"{prefix}_pin"]
        materialize_table_by_raw(
            pin_path,
            CACHE_DIR / f"{prefix}_pin_by_raw",
            id_column="SpecId",
            sep="\t",
            chunksize=PIN_CHUNKSIZE,
            rebuild=rebuild,
            source_label="pin",
        )

        percolator_root = CACHE_DIR / f"{prefix}_percolator_by_raw"
        if success_marker(percolator_root).exists() and not rebuild:
            logger.info("Using cached partitions in %s", percolator_root)
            continue

        reset_dir(percolator_root, rebuild=True)
        for label, key in (("target", f"{prefix}_percolator_target"), ("decoy", f"{prefix}_percolator_decoy")):
            percolator_path = loader.path_mapper[key]
            if percolator_path is None or not percolator_path.exists():
                continue
            reader = pd.read_csv(
                percolator_path,
                sep="\t",
                chunksize=PERCOLATOR_CHUNKSIZE,
                usecols=PERCOLATOR_COLUMNS,
            )
            for chunk_idx, chunk in enumerate(tqdm(reader, desc=f"Partitioning {percolator_path.name}")):
                if chunk.empty:
                    continue
                chunk["RAW_FILE"] = extract_raw_file(chunk["PSMId"])
                chunk = chunk.loc[chunk["RAW_FILE"].notna()].copy()
                for raw_file, part in chunk.groupby("RAW_FILE", sort=False):
                    write_part(part, percolator_root, raw_file, f"{label}-{chunk_idx:06d}.parquet")
                del chunk
                gc.collect()

        success_marker(percolator_root).touch()


def load_percolator_scores(prefix: str, raw_file: str) -> pd.Series:
    scores = read_partition(
        CACHE_DIR / f"{prefix}_percolator_by_raw",
        raw_file,
        columns=["PSMId", "score"],
    )
    if scores.empty:
        return pd.Series(dtype="float64")
    scores = scores.drop_duplicates("PSMId", keep="last")
    return scores.set_index("PSMId")["score"]


def load_pin_with_scores(prefix: str, raw_file: str) -> pd.DataFrame:
    pin = read_partition(CACHE_DIR / f"{prefix}_pin_by_raw", raw_file)
    if pin.empty:
        return pin.drop(columns=["RAW_FILE"], errors="ignore")
    pin = pin.drop(columns=["RAW_FILE"], errors="ignore")
    pin["percolator_score"] = pin["SpecId"].map(load_percolator_scores(prefix, raw_file))
    return pin


def read_search_rescore(search_dir: Path, raw_file: str, keep_columns: list[str]) -> pd.DataFrame:
    path = search_dir / f"{raw_file}.rescore"
    columns = [column for column in keep_columns if column in keep_columns]
    if not path.exists():
        return pd.DataFrame(columns=columns + ["SpecId"])

    df = pd.read_csv(path, sep=",", usecols=keep_columns)
    if df.empty:
        df["SpecId"] = pd.Series(dtype="object")
        return df

    specid = df["RAW_FILE"].astype(str)
    for column in ("SCAN_NUMBER", "MODIFIED_SEQUENCE", "PRECURSOR_CHARGE"):
        specid = specid.str.cat(df[column].astype(str), sep="-")
    df["SpecId"] = specid
    return df


def merge_search_and_pin(search: pd.DataFrame, pin: pd.DataFrame) -> pd.DataFrame:
    if search.empty and pin.empty:
        return pd.DataFrame(columns=["SpecId", "RAW_FILE", "SCAN_NUMBER"])
    return search.merge(pin, how="outer", on="SpecId", indicator=False)


def init_psa_worker() -> None:
    global _WORKER_PSA
    _WORKER_PSA = PSA()


def run_psa_pair(sequence1, sequence2):
    global _WORKER_PSA
    if _WORKER_PSA is None:
        _WORKER_PSA = PSA()
    try:
        _WORKER_PSA.set_sequences(str(sequence1), str(sequence2))
        _WORKER_PSA.classify()
    except Exception as exc:
        return ("PSA_ERROR", np.nan, f"{type(exc).__name__}: {exc}")
    return (_WORKER_PSA.result.label, _WORKER_PSA.result.similarity, None)


def choose_chunksize(total_items: int, n_workers: int) -> int:
    if total_items <= 0:
        return 1
    return max(1, total_items // max(1, n_workers * 8))


def run_psa_for_pairs(df: pd.DataFrame, max_workers: int | None = None) -> pd.DataFrame:
    result = df.copy()
    if result.empty:
        result["PSA"] = pd.Series(dtype="object")
        result["PSA_SIMILARITY"] = pd.Series(dtype="float64")
        result["PSA_ERROR"] = pd.Series(dtype="object")
        return result

    n_workers = max_workers or PSA_MAX_WORKERS
    chunksize = choose_chunksize(len(result), n_workers)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=init_psa_worker) as executor:
        psa_rows = list(
            tqdm(
                executor.map(
                    run_psa_pair,
                    result["SEQUENCE_database"].tolist(),
                    result["SEQUENCE_denovo"].tolist(),
                    chunksize=chunksize,
                ),
                total=len(result),
                desc="Running PSA",
            )
        )

    result[["PSA", "PSA_SIMILARITY", "PSA_ERROR"]] = pd.DataFrame(
        psa_rows,
        columns=["PSA", "PSA_SIMILARITY", "PSA_ERROR"],
        index=result.index,
    )
    return result


def add_sequence_match_columns(df: pd.DataFrame) -> None:
    if {"SEQUENCE_database", "SEQUENCE_denovo"}.issubset(df.columns):
        df["unmodified_sequence_match"] = (
            df["SEQUENCE_database"].notna()
            & df["SEQUENCE_denovo"].notna()
            & df["SEQUENCE_database"].eq(df["SEQUENCE_denovo"])
        )
    else:
        df["unmodified_sequence_match"] = False

    if {"MODIFIED_SEQUENCE_database", "MODIFIED_SEQUENCE_denovo"}.issubset(df.columns):
        df["modified_sequence_match"] = (
            df["MODIFIED_SEQUENCE_database"].notna()
            & df["MODIFIED_SEQUENCE_denovo"].notna()
            & df["MODIFIED_SEQUENCE_database"].eq(df["MODIFIED_SEQUENCE_denovo"])
        )
    else:
        df["modified_sequence_match"] = False

    if {"PRECURSOR_CHARGE_database", "PRECURSOR_CHARGE_denovo"}.issubset(df.columns):
        database_charge = pd.to_numeric(df["PRECURSOR_CHARGE_database"], errors="coerce")
        denovo_charge = pd.to_numeric(df["PRECURSOR_CHARGE_denovo"], errors="coerce")
        df["precursor_charge_match"] = (
            database_charge.notna() & denovo_charge.notna() & database_charge.eq(denovo_charge)
        )
    else:
        df["precursor_charge_match"] = False

    df["sequence_match"] = df["modified_sequence_match"] & df["precursor_charge_match"]


def add_shared_scan_psa(merged_scan: pd.DataFrame) -> pd.DataFrame:
    required = ["_merge", "SEQUENCE_database", "SEQUENCE_denovo"]
    missing = [column for column in required if column not in merged_scan.columns]
    if missing:
        raise KeyError(f"Missing columns in merged_SCAN needed for PSA: {missing}")

    shared = merged_scan.loc[
        merged_scan["_merge"].eq("shared")
        & merged_scan["SEQUENCE_database"].notna()
        & merged_scan["SEQUENCE_denovo"].notna()
    ].copy()
    add_sequence_match_columns(shared)

    same_sequence = shared.loc[shared["sequence_match"]].copy()
    same_sequence["PSA"] = "PSA - Tier 0 - IDENTICAL"
    same_sequence["PSA_SIMILARITY"] = 1.0
    same_sequence["PSA_ERROR"] = None

    different_sequence = shared.loc[~shared["sequence_match"]].copy()
    different_sequence = run_psa_for_pairs(different_sequence)
    return pd.concat([same_sequence, different_sequence], ignore_index=True, copy=False)


def list_raw_files(loader: ResultLoader) -> list[str]:
    raw_files = set()
    for key in ("database_search", "denovo_search"):
        search_dir = loader.path_mapper[key]
        if search_dir is not None and search_dir.exists():
            raw_files.update(path.stem for path in search_dir.glob("*.rescore"))
    return sorted(raw_files)


def write_dataset_part(dataset_name: str, raw_file: str, df: pd.DataFrame) -> Path:
    root = OUTPUT_DIR / dataset_name
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / f"{safe_raw_file(raw_file)}.parquet"
    temp_path = root / f".{out_path.name}.{os.getpid()}.tmp"
    try:
        df.to_parquet(temp_path, index=False, engine="pyarrow")
        temp_path.replace(out_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return out_path


def merge_one_raw(loader: ResultLoader, raw_file: str) -> dict[str, int]:
    database_search = read_search_rescore(loader.path_mapper["database_search"], raw_file, DATABASE_COLUMNS)
    denovo_search = read_search_rescore(loader.path_mapper["denovo_search"], raw_file, DENOVO_COLUMNS)
    database_pin = load_pin_with_scores("database", raw_file)
    denovo_pin = load_pin_with_scores("denovo", raw_file)

    merged_database = merge_search_and_pin(database_search, database_pin)
    merged_denovo = merge_search_and_pin(denovo_search, denovo_pin)

    chimeric_scan_keys = loader._chimeric_scan_keys(merged_database)
    merged_psm = loader._merge_search_results(
        merged_database,
        merged_denovo,
        left_on=["SpecId"],
        right_on=["SpecId"],
        chimeric_scan_keys=chimeric_scan_keys,
    )
    merged_scan = loader._merge_search_results(
        merged_database,
        merged_denovo,
        left_on=["RAW_FILE", "SCAN_NUMBER"],
        right_on=["RAW_FILE", "SCAN_NUMBER"],
        chimeric_scan_keys=chimeric_scan_keys,
        add_sequence_match=True,
    )
    shared_scan_psa = add_shared_scan_psa(merged_scan)

    write_dataset_part("merged_database", raw_file, merged_database)
    write_dataset_part("merged_denovo", raw_file, merged_denovo)
    write_dataset_part("merged_PSM", raw_file, merged_psm)
    write_dataset_part("merged_SCAN", raw_file, merged_scan)
    write_dataset_part("shared_scan_psa", raw_file, shared_scan_psa)

    counts = {
        "RAW_FILE": raw_file,
        "merged_database_rows": len(merged_database),
        "merged_denovo_rows": len(merged_denovo),
        "merged_PSM_rows": len(merged_psm),
        "merged_SCAN_rows": len(merged_scan),
        "shared_scan_psa_rows": len(shared_scan_psa),
    }

    del (
        database_search,
        denovo_search,
        database_pin,
        denovo_pin,
        merged_database,
        merged_denovo,
        merged_psm,
        merged_scan,
        shared_scan_psa,
    )
    gc.collect()
    return counts


def format_raw_error(raw_file: str, exc: Exception) -> dict[str, str]:
    return {
        "RAW_FILE": raw_file,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_raw_errors(new_errors: list[dict[str, str]], successful_raw_files: set[str]) -> pd.DataFrame:
    errors_path = OUTPUT_DIR / "raw_errors.csv"
    if errors_path.exists():
        errors_df = pd.read_csv(errors_path)
    else:
        errors_df = pd.DataFrame(columns=RAW_ERROR_COLUMNS)

    if successful_raw_files and not errors_df.empty:
        errors_df = errors_df.loc[~errors_df["RAW_FILE"].isin(successful_raw_files)].copy()

    if new_errors:
        new_errors_df = pd.DataFrame(new_errors, columns=RAW_ERROR_COLUMNS)
        if not errors_df.empty:
            errors_df = errors_df.loc[~errors_df["RAW_FILE"].isin(new_errors_df["RAW_FILE"])].copy()
        errors_df = pd.concat([errors_df, new_errors_df], ignore_index=True)

    errors_df = errors_df.reindex(columns=RAW_ERROR_COLUMNS)
    errors_df.to_csv(errors_path, index=False)
    return errors_df


def write_manifest(row_counts: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.DataFrame(
        [
            {"name": name, "path": str(OUTPUT_DIR / name), "kind": "parquet_parts_by_raw_file"}
            for name in (*MERGED_DATASET_NAMES, "shared_scan_psa")
        ]
        + [
            {"name": "row_counts", "path": str(OUTPUT_DIR / "row_counts.csv"), "kind": "csv"},
            {"name": "raw_errors", "path": str(OUTPUT_DIR / "raw_errors.csv"), "kind": "csv"},
            {"name": "psa_summary", "path": str(OUTPUT_DIR / "shared_scan_psa_summary.csv"), "kind": "csv"},
        ]
    )
    row_counts.to_csv(OUTPUT_DIR / "row_counts.csv", index=False)
    manifest.to_csv(OUTPUT_DIR / "manifest.csv", index=False)
    return manifest


def summarize_psa() -> pd.DataFrame:
    psa_dir = OUTPUT_DIR / "shared_scan_psa"
    frames = []
    for path in sorted(psa_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(path, columns=["sequence_match", "PSA"])
        except Exception as exc:
            logger.warning("Skipping unreadable PSA summary input %s: %s: %s", path, type(exc).__name__, exc)
            continue
        if df.empty:
            continue
        frames.append(df.groupby(["sequence_match", "PSA"], dropna=False).size().rename("count").reset_index())

    if not frames:
        summary = pd.DataFrame(columns=["sequence_match", "PSA", "count"])
    else:
        summary = (
            pd.concat(frames, ignore_index=True)
            .groupby(["sequence_match", "PSA"], dropna=False)["count"]
            .sum()
            .reset_index()
            .sort_values(["sequence_match", "count"], ascending=[False, False])
        )
    summary.to_csv(OUTPUT_DIR / "shared_scan_psa_summary.csv", index=False)
    return summary


def run_pipeline(
    loader: ResultLoader,
    *,
    raw_files: Iterable[str] | None = None,
    max_raw_files: int | None = None,
    rebuild_cache: bool = False,
    overwrite_outputs: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    materialize_pin_and_percolator_caches(loader, rebuild=rebuild_cache)
    selected_raw_files = list(raw_files) if raw_files is not None else list_raw_files(loader)
    if max_raw_files is not None:
        selected_raw_files = selected_raw_files[:max_raw_files]

    row_counts = []
    raw_errors = []
    successful_raw_files = set()
    for raw_file in tqdm(selected_raw_files, desc="Merging RAW files"):
        expected_output = OUTPUT_DIR / "merged_SCAN" / f"{safe_raw_file(raw_file)}.parquet"
        expected_psa = OUTPUT_DIR / "shared_scan_psa" / f"{safe_raw_file(raw_file)}.parquet"
        if expected_output.exists() and expected_psa.exists() and not overwrite_outputs:
            successful_raw_files.add(raw_file)
            continue
        try:
            row_counts.append(merge_one_raw(loader, raw_file))
            successful_raw_files.add(raw_file)
        except Exception as exc:
            raw_errors.append(format_raw_error(raw_file, exc))
            logger.error(
                "%s: failed with %s: %s; continuing with next RAW.",
                raw_file,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            gc.collect()
            continue

    if row_counts:
        new_counts = pd.DataFrame(row_counts)
        existing_counts_path = OUTPUT_DIR / "row_counts.csv"
        if existing_counts_path.exists() and not overwrite_outputs:
            existing_counts = pd.read_csv(existing_counts_path)
            row_counts_df = pd.concat([existing_counts, new_counts], ignore_index=True)
            row_counts_df = row_counts_df.drop_duplicates("RAW_FILE", keep="last")
        else:
            row_counts_df = new_counts
    elif (OUTPUT_DIR / "row_counts.csv").exists():
        row_counts_df = pd.read_csv(OUTPUT_DIR / "row_counts.csv")
    else:
        row_counts_df = pd.DataFrame(columns=["RAW_FILE"])

    failed_raw_files = {error["RAW_FILE"] for error in raw_errors}
    if failed_raw_files and not row_counts_df.empty:
        row_counts_df = row_counts_df.loc[~row_counts_df["RAW_FILE"].isin(failed_raw_files)].copy()

    raw_errors_df = write_raw_errors(raw_errors, successful_raw_files)
    if not raw_errors_df.empty:
        logger.warning("Recorded %d RAW-level error(s) in %s", len(raw_errors_df), OUTPUT_DIR / "raw_errors.csv")

    psa_summary = summarize_psa()
    manifest = write_manifest(row_counts_df)
    return row_counts_df, psa_summary, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine ProteomeTools search and rescoring outputs into partitioned PSA dataframes."
    )
    parser.add_argument("--database-config", required=True, type=Path, help="database rescoring_config_*.json")
    parser.add_argument("--denovo-config", required=True, type=Path, help="de novo rescoring_config_*.json")
    parser.add_argument("--database-pin", required=True, type=Path, help="database rescore.filtered.tab")
    parser.add_argument("--denovo-pin", required=True, type=Path, help="de novo rescore.filtered.tab")
    parser.add_argument("--database-percolator-target", required=True, type=Path)
    parser.add_argument("--database-percolator-decoy", type=Path, default=None)
    parser.add_argument("--denovo-percolator-target", required=True, type=Path)
    parser.add_argument(
        "--denovo-percolator-decoy",
        type=Path,
        default=None,
        help="optional; the de novo Percolator run is typically static/no decoy output",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-workers", type=int, default=8, help="workers for caching and PSA alignment")
    parser.add_argument(
        "--raw-file",
        dest="raw_files",
        action="append",
        help="Process one RAW file stem. May be supplied multiple times. Defaults to all available RAW files.",
    )
    parser.add_argument(
        "--max-raw-files",
        type=int,
        default=None,
        help="Process only the first N RAW files. Useful for testing the large ProteomeTools outputs.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild cached PIN/percolator partitions under the PSA output directory.",
    )
    parser.add_argument(
        "--overwrite-outputs",
        action="store_true",
        help="Overwrite existing per-RAW output partitions.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    global OUTPUT_DIR, CACHE_DIR, PSA_MAX_WORKERS
    OUTPUT_DIR = args.output_dir
    CACHE_DIR = OUTPUT_DIR / "_cache"
    PSA_MAX_WORKERS = args.max_workers

    logger.info("Writing partitioned outputs to: %s", OUTPUT_DIR)
    logger.info(
        "raw_files=%s max_raw_files=%s rebuild_cache=%s overwrite_outputs=%s",
        args.raw_files,
        args.max_raw_files,
        args.rebuild_cache,
        args.overwrite_outputs,
    )

    loader = ResultLoader(
        database_oktoberfest_config=args.database_config,
        denovo_oktoberfest_config=args.denovo_config,
        database_pin_name="rescore.filtered.tab",
        denovo_pin_name="rescore.filtered.tab",
        max_workers=args.max_workers,
    )

    loader.set_path("database_pin", args.database_pin)
    loader.set_path("denovo_pin", args.denovo_pin)
    loader.set_path("database_percolator_target", args.database_percolator_target)
    loader.set_path("database_percolator_decoy", args.database_percolator_decoy)
    loader.set_path("denovo_percolator_target", args.denovo_percolator_target)
    loader.set_path("denovo_percolator_decoy", args.denovo_percolator_decoy)

    row_counts, psa_summary, manifest = run_pipeline(
        loader,
        raw_files=args.raw_files,
        max_raw_files=args.max_raw_files,
        rebuild_cache=args.rebuild_cache,
        overwrite_outputs=args.overwrite_outputs,
    )

    logger.info("Wrote %d row-count records", len(row_counts))
    logger.info("Wrote %d PSA summary rows", len(psa_summary))
    logger.info("Manifest: %s", OUTPUT_DIR / "manifest.csv")
    logger.info("Manifest contents:\n%s", manifest.to_string(index=False))


if __name__ == "__main__":
    main()
