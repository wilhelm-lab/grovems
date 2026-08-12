from __future__ import annotations

import dataclasses
import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from oktoberfest import runner as oktoberfest_runner

from ..config import GrovemsConfig

logger = logging.getLogger(__name__)

PIN_CHUNKSIZE = 250_000
PERCOLATOR_CHUNKSIZE = 500_000
SPECID_RAW_PATTERN = r"^(?P<RAW_FILE>.+?)-\d+-.+-\d+$"

DENOVO_COLUMNS = [
    "RAW_FILE",
    "SCAN_NUMBER",
    "MODIFIED_SEQUENCE",
    "PRECURSOR_CHARGE",
    "MASS",
    "SCORE",
    "REVERSE",
    "SEQUENCE",
    "AA_SCORE",
]

DATABASE_COLUMNS = [
    "RAW_FILE",
    "SCAN_NUMBER",
    "MODIFIED_SEQUENCE",
    "PRECURSOR_CHARGE",
    "MASS",
    "SCORE",
    "SEQUENCE",
    "num_missed_cleavages",
    "REVERSE",
]

PERCOLATOR_COLUMNS = ["PSMId", "score", "q-value", "posterior_error_prob"]


@dataclasses.dataclass
class RescoringResult:
    """Per-raw-file merged (search+pin+percolator) output dirs, consumed by the PSA stage.

    ``database_merged_dir`` is ``None`` in denovo-only mode (``config.denovo_only``) --
    there is no database branch at all, so ``psa.run`` treats every raw file as having no
    database data (see ``grovems.psa.psa_merge.run``).
    """

    database_merged_dir: Optional[Path]
    denovo_merged_dir: Path


def _build_oktoberfest_config(
    *,
    output_dir: Path,
    search_results: Path,
    search_results_type: str,
    spectra: Path,
    spectra_type: str,
    config: GrovemsConfig,
    num_threads: int,
) -> dict:
    cfg = {
        "type": "Rescoring",
        "tag": "",
        "output": str(output_dir),
        "inputs": {
            "search_results": str(search_results),
            "search_results_type": search_results_type,
            "spectra": str(spectra),
            "spectra_type": spectra_type,
        },
        "models": {
            "irt": config.irt_model,
            "intensity": config.intensity_model,
        },
        "prediction_server": config.prediction_server,
        "numThreads": num_threads,
        "fdr_estimation_method": config.fdr_estimation_method,
        "all_features": config.all_features,
        "regressionMethod": config.regression_method,
        "ssl": config.ssl,
        "massTolerance": config.mass_tolerance,
        "p_window": config.p_window,
        "unitMassTolerance": config.unit_mass_tolerance,
        "fragmentation_method": config.fragmentation_method,
    }
    # Omitted entirely (not set to null) when unset: Oktoberfest's Config.thermo_exe does
    # self.data.get("thermoExe", default_thermo()) -- an explicit null would return None
    # instead of falling through to the default, crashing downstream.
    if config.thermo_exe:
        cfg["thermoExe"] = config.thermo_exe
    return cfg


def _run_oktoberfest(config_path: Path, *, max_attempts: int = 3, retry_delay_seconds: float = 30.0) -> None:
    """Run Oktoberfest, retrying the whole job a few times on failure.

    Oktoberfest's JobPool.check_pool() treats any single worker exception -- including
    a one-off dropped Koina gRPC request, which is not a systemic outage -- as fatal for
    the entire pool and calls sys.exit(1), aborting a multi-hour run over one transient
    network blip. Retrying is cheap and safe: Oktoberfest marks each per-raw-file step
    done via .done files under proc/, so re-running run_job() skips everything already
    completed and only redoes the file(s) that hadn't finished.
    """
    logger.info("Running Oktoberfest: %s", config_path)
    for attempt in range(1, max_attempts + 1):
        try:
            oktoberfest_runner.run_job(str(config_path))
            return
        except SystemExit as exc:
            # A worker pool failure inside Oktoberfest (JobPool.check_pool) calls
            # sys.exit(1) instead of raising a normal exception -- catch it explicitly so
            # it doesn't kill the whole grovems process.
            error = RuntimeError(f"Oktoberfest exited (SystemExit({exc.code})) while running {config_path}")
            error.__cause__ = exc
        except Exception as exc:
            error = RuntimeError(f"Oktoberfest failed while running {config_path}: {exc}")
            error.__cause__ = exc

        if attempt == max_attempts:
            raise error
        logger.warning(
            "Oktoberfest attempt %d/%d failed, retrying in %.0fs: %s", attempt, max_attempts, retry_delay_seconds, error
        )
        time.sleep(retry_delay_seconds)


def _run_oktoberfest_stage(
    config: GrovemsConfig,
    config_path: Path,
    *,
    output_dir: Path,
    search_path: str,
    search_type: str,
    num_threads: int,
) -> None:
    json_config = _build_oktoberfest_config(
        output_dir=output_dir,
        search_results=Path(search_path),
        search_results_type=search_type,
        spectra=Path(config.rawdata_path),
        spectra_type=config.spectra_type,
        config=config,
        num_threads=num_threads,
    )
    config_path.write_text(json.dumps(json_config, indent=4))
    _run_oktoberfest(config_path)


def _run_percolator(config: GrovemsConfig, args: list[str], *, cwd: Path) -> None:
    if config.percolator_module:
        # `module load` is a shell function (from Lmod/Environment Modules init), not a
        # binary -- needs a real shell, not the args-list subprocess form.
        command = " ".join(shlex.quote(part) for part in [config.percolator_exe, *args])
        shell_command = f"module load {config.percolator_module} && {command}"
        logger.info("Running percolator via shell (module load %s)", config.percolator_module)
        subprocess.run(shell_command, shell=True, executable="/bin/bash", cwd=cwd, check=True)  # noqa: S602
    else:
        command = [config.percolator_exe, *args]
        logger.info("Running percolator: %s", " ".join(command))
        subprocess.run(command, cwd=cwd, check=True)


def _relink(source: Path, dest: Path) -> None:
    resolved_source = source.resolve()
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(resolved_source, target_is_directory=resolved_source.is_dir())


def _require_matching_columns(database_columns: list[str], denovo_columns: list[str]) -> None:
    """Verify the two branches' filtered rescore.tab columns match, name and order.

    De novo's Percolator rescoring is static (``--init-weights``/``--static``): it
    applies the database branch's learned weight vector positionally to de novo's
    feature columns. If the two rescore.tab files don't end up with identical,
    identically-ordered columns after filtering -- whether from
    drop_columns_database/drop_columns_denovo not actually aligning, or from
    database_oktoberfest_dir/denovo_oktoberfest_dir pointing at runs with different
    feature sets -- those weights would be silently misapplied to the wrong features.
    """
    if database_columns == denovo_columns:
        return
    database_only = [c for c in database_columns if c not in denovo_columns]
    denovo_only = [c for c in denovo_columns if c not in database_columns]
    raise ValueError(
        "database and de novo rescore.tab have different columns after filtering, but de "
        "novo's Percolator rescoring reuses the database branch's weights positionally and "
        "needs identical, identically-ordered feature columns.\n"
        f"database-only columns: {database_only}\n"
        f"de novo-only columns:  {denovo_only}\n"
        "Adjust drop_columns_database/drop_columns_denovo (or the reused Oktoberfest "
        "directories) so the surviving feature columns match."
    )


def _drop_columns(input_file: str, output_file: str, delimiter: str, columns_to_drop: list[str]) -> list[str]:
    """Read a delimited file, drop the given columns, and write the result.

    Args:
        input_file: Path to the input delimited file.
        output_file: Path to write the column-dropped file to (may be the same path,
            to filter in place -- the read completes before the write starts).
        delimiter: Field delimiter, e.g. ``"\\t"``.
        columns_to_drop: Column names to drop; names not present are ignored.

    Returns:
        The resulting column names, in order -- used to verify database/de novo feature
        parity before de novo's static Percolator rescoring (see :func:`run`).
    """
    logger.info("Dropping columns %s: %s -> %s", columns_to_drop, input_file, output_file)
    df = pd.read_csv(input_file, sep=delimiter, engine="c")
    df = df.drop(columns=columns_to_drop, errors="ignore")
    df.to_csv(output_file, sep=delimiter, index=False)
    logger.info("Wrote %d rows, %d columns to '%s'", len(df), len(df.columns), output_file)
    return list(df.columns)


def _parse_and_write_avg_weights(path: str, out_path: str, decimals: int = 4) -> None:
    """Average the per-bin weight vectors in ``path`` and write the result to ``out_path``.

    Args:
        path: Input file with repeated (header, normalized weights, raw weights) line
            triples, one triple per bin.
        out_path: Output file to write the averaged weights to.
        decimals: Decimal places to round the averages to.
    """
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    i = 0
    header: list[str] = []
    bins_norm: list[list[float]] = []
    bins_raw: list[list[float]] = []
    while i < len(lines):
        header = lines[i].split("\t")
        bins_norm.append(list(map(float, lines[i + 1].split("\t"))))
        bins_raw.append(list(map(float, lines[i + 2].split("\t"))))
        i += 3

    logger.info("Averaging %d bin(s) of weights from '%s'", len(bins_norm), path)
    avg_norm = [sum(x) / len(bins_norm) for x in zip(*bins_norm)]
    avg_raw = [sum(x) / len(bins_raw) for x in zip(*bins_raw)]

    with open(out_path, "w") as f:
        f.write("# Average weights across all bins\n")
        f.write("\t".join(header) + "\n")
        f.write("\t".join(f"{x:.{decimals}f}" for x in avg_norm) + "\n")
        f.write("\t".join(f"{x:.{decimals}f}" for x in avg_raw) + "\n")
    logger.info("Averages written to '%s'", out_path)


def _safe_raw_file(raw_file: str) -> str:
    return quote(str(raw_file), safe="")


def _extract_raw_file(spec_ids: pd.Series) -> pd.Series:
    return spec_ids.astype("string").str.extract(SPECID_RAW_PATTERN, expand=False)


def _partition_by_raw_file(
    input_path: Path, output_root: Path, *, id_column: str, sep: str, chunksize: int, label: str, usecols=None
) -> None:
    """Chunk-read ``input_path``, split rows by RAW_FILE, append to one parquet file per raw file.

    Keeps one open ``ParquetWriter`` per raw file across all chunks instead of writing a new
    small file per chunk per raw file -- for a multi-million-row rescore.tab chunked at
    PIN_CHUNKSIZE, the per-chunk version fans out into thousands of tiny files across ~90
    raw-file directories, which is expensive filesystem-metadata churn (worse on NFS).
    """
    writers: dict[str, pq.ParquetWriter] = {}
    try:
        reader = pd.read_csv(input_path, sep=sep, chunksize=chunksize, usecols=usecols)
        for chunk in reader:
            if chunk.empty:
                continue
            chunk["RAW_FILE"] = _extract_raw_file(chunk[id_column])
            chunk = chunk.loc[chunk["RAW_FILE"].notna()]
            for raw_file, part in chunk.groupby("RAW_FILE", sort=False):
                table = pa.Table.from_pandas(part, preserve_index=False)
                writer = writers.get(raw_file)
                if writer is None:
                    raw_dir = output_root / _safe_raw_file(raw_file)
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(raw_dir / f"{label}.parquet", table.schema)
                    writers[raw_file] = writer
                elif not table.schema.equals(writer.schema):
                    # Chunk-local dtype inference (e.g. an int column that only turns out to
                    # have nulls in a later chunk) can drift between chunks for the same raw
                    # file; align to the first chunk's schema rather than failing the write.
                    table = table.cast(writer.schema, safe=False)
                writer.write_table(table)
    finally:
        for writer in writers.values():
            writer.close()


def _read_raw_partition(root: Path, raw_file: str, columns: Optional[list[str]] = None) -> pd.DataFrame:
    raw_dir = root / _safe_raw_file(raw_file)
    if not raw_dir.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(raw_dir, columns=columns)


def _load_percolator_scores(percolator_by_raw: Path, raw_file: str) -> pd.Series:
    scores = _read_raw_partition(percolator_by_raw, raw_file, columns=["PSMId", "score"])
    if scores.empty:
        return pd.Series(dtype="float64")
    scores = scores.drop_duplicates("PSMId", keep="last")
    return scores.set_index("PSMId")["score"]


def _read_search_rescore(search_dir: Path, raw_file: str, keep_columns: list[str]) -> pd.DataFrame:
    path = search_dir / f"{raw_file}.rescore"
    if not path.exists():
        return pd.DataFrame(columns=keep_columns + ["SpecId"])

    df = pd.read_csv(path, sep=",", usecols=keep_columns)
    if df.empty:
        df["SpecId"] = pd.Series(dtype="object")
        return df

    specid = df["RAW_FILE"].astype(str)
    for column in ("SCAN_NUMBER", "MODIFIED_SEQUENCE", "PRECURSOR_CHARGE"):
        specid = specid.str.cat(df[column].astype(str), sep="-")
    df["SpecId"] = specid
    return df


def _merge_one_raw_file(
    raw_file: str, search_dir: Path, keep_columns: list[str], pin_by_raw: Path, percolator_by_raw: Optional[Path]
) -> pd.DataFrame:
    search = _read_search_rescore(search_dir, raw_file, keep_columns)
    pin = _read_raw_partition(pin_by_raw, raw_file).drop(columns=["RAW_FILE"], errors="ignore")
    if not pin.empty and percolator_by_raw is not None:
        pin["percolator_score"] = pin["SpecId"].map(_load_percolator_scores(percolator_by_raw, raw_file))

    if search.empty and pin.empty:
        return pd.DataFrame(columns=["SpecId", "RAW_FILE", "SCAN_NUMBER"])
    return search.merge(pin, how="outer", on="SpecId", indicator=False)


def _merge_and_partition_branch(
    *,
    branch_dir: Path,
    rescore_tab: Path,
    percolator_psms: Optional[Path],
    percolator_decoy_psms: Optional[Path],
    keep_columns: list[str],
) -> Path:
    """Merge rescore.tab (+ Percolator output, if any) + msms/*.rescore into <branch>/merged/<raw>.parquet.

    ``percolator_psms=None`` (denovo-only mode, see ``run``) skips Percolator output
    entirely -- the merged data carries no ``percolator_score`` column.

    Deletes rescore_tab and the Percolator output file(s) once merged/ is built --
    their data now lives in merged/, so keeping both would just be two copies of the
    same rows.
    """
    search_dir = branch_dir / "msms"
    merged_dir = branch_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = branch_dir / "_tmp_partitions"
    pin_by_raw = tmp_dir / "pin_by_raw"
    percolator_by_raw = tmp_dir / "percolator_by_raw"
    _partition_by_raw_file(rescore_tab, pin_by_raw, id_column="SpecId", sep="\t", chunksize=PIN_CHUNKSIZE, label="pin")
    for label, percolator_path in (("target", percolator_psms), ("decoy", percolator_decoy_psms)):
        if percolator_path is None or not percolator_path.exists():
            continue
        _partition_by_raw_file(
            percolator_path,
            percolator_by_raw,
            id_column="PSMId",
            sep="\t",
            chunksize=PERCOLATOR_CHUNKSIZE,
            label=label,
            usecols=PERCOLATOR_COLUMNS,
        )

    raw_files = sorted(path.stem for path in search_dir.glob("*.rescore"))
    logger.info("Merging %d raw file(s) into %s", len(raw_files), merged_dir)
    for raw_file in raw_files:
        merged = _merge_one_raw_file(
            raw_file, search_dir, keep_columns, pin_by_raw, percolator_by_raw if percolator_psms is not None else None
        )
        merged.to_parquet(merged_dir / f"{_safe_raw_file(raw_file)}.parquet", index=False, engine="pyarrow")

    shutil.rmtree(tmp_dir)
    logger.info(
        "Deleting %s%s; data now lives in %s",
        rescore_tab,
        " and Percolator output(s)" if percolator_psms is not None else "",
        merged_dir,
    )
    rescore_tab.unlink()
    if percolator_psms is not None:
        percolator_psms.unlink()
        if percolator_decoy_psms is not None and percolator_decoy_psms.exists():
            percolator_decoy_psms.unlink()
    return merged_dir


def run(config: GrovemsConfig, outdir: Path) -> RescoringResult:
    """Run rescoring: database Oktoberfest, then de novo Oktoberfest, then Percolator + merge for both.

    De novo's Oktoberfest run only depends on the database branch's ce_calibration/rt_model
    (relinked below), so it's run right after the database branch's Oktoberfest step instead
    of after that branch's Percolator + merge too -- nothing runs concurrently with either
    Oktoberfest pass. Once both are done, independent steps across the two branches run
    concurrently where the real data dependencies allow it: database's Percolator run
    alongside de novo's column filtering, then database's merge alongside de novo's
    Percolator + merge (de novo's Percolator only needs database's weights.csv, not its
    merge).

    With ``config.denovo_only``, there is no database branch at all: de novo's Oktoberfest
    run computes its own ce_calibration/rt_model (Oktoberfest's normal standalone
    behavior, rather than reusing a database branch's) and Percolator is skipped
    entirely -- it's only ever trained on the database branch's real target/decoy
    competition (de novo's own ``Label`` is always target), so without one there's
    nothing to train, and nothing to statically apply learned weights from either.

    Args:
        config: Pipeline configuration.
        outdir: Top-level output directory (results go under ``outdir/rescoring``).

    Returns:
        The per-branch merged/ directories the PSA stage reads from (``database_merged_dir``
        is ``None`` for ``config.denovo_only``).
    """
    rescoring_dir = outdir / "rescoring"
    rescoring_dir.mkdir(parents=True, exist_ok=True)
    num_threads = config.num_threads or os.cpu_count() or 1

    # --- database branch: Oktoberfest ---
    database_dir: Optional[Path] = None
    if not config.denovo_only:
        if config.database_oktoberfest_dir:
            database_dir = Path(config.database_oktoberfest_dir)
            logger.info("Reusing existing Oktoberfest database results: %s", database_dir)
        else:
            database_dir = rescoring_dir / "oktoberfest_database"
            _run_oktoberfest_stage(
                config,
                rescoring_dir / "rescoring_config_database.json",
                output_dir=database_dir,
                search_path=config.database_search_path,
                search_type=config.database_search_type,
                num_threads=num_threads,
            )

    # --- de novo branch: Oktoberfest (reuses database's ce_calibration/rt_model, unless denovo_only) ---
    if config.denovo_oktoberfest_dir:
        denovo_dir = Path(config.denovo_oktoberfest_dir)
        logger.info("Reusing existing Oktoberfest de novo results: %s", denovo_dir)
    else:
        denovo_dir = rescoring_dir / "oktoberfest_denovo"
        (denovo_dir / "results").mkdir(parents=True, exist_ok=True)
        if database_dir is not None:
            _relink(database_dir / "results" / "ce_calibration", denovo_dir / "results" / "ce_calibration")
            _relink(database_dir / "results" / "rt_model", denovo_dir / "results" / "rt_model")

        _run_oktoberfest_stage(
            config,
            rescoring_dir / "rescoring_config_denovo.json",
            output_dir=denovo_dir,
            search_path=config.denovo_search_path,
            search_type=config.denovo_search_type,
            num_threads=num_threads,
        )

    if config.denovo_only:
        denovo_rescore_tab = denovo_dir / "results" / "percolator" / "rescore.tab"
        logger.info("denovo_only: skipping Percolator, merging de novo's own pin + search output directly")
        _drop_columns(str(denovo_rescore_tab), str(denovo_rescore_tab), "\t", config.drop_columns_denovo.split())
        denovo_merged_dir = _merge_and_partition_branch(
            branch_dir=denovo_dir,
            rescore_tab=denovo_rescore_tab,
            percolator_psms=None,
            percolator_decoy_psms=None,
            keep_columns=DENOVO_COLUMNS,
        )
        return RescoringResult(database_merged_dir=None, denovo_merged_dir=denovo_merged_dir)

    assert database_dir is not None  # guaranteed by the denovo_only early-return above
    database_percolator_dir = database_dir / "results" / "percolator"
    database_rescore_tab = database_percolator_dir / "rescore.tab"
    database_weights = database_percolator_dir / "rescore.percolator.weights.csv"
    database_psms = database_percolator_dir / "rescore.percolator.psms.txt"
    database_decoy_psms = database_percolator_dir / "rescore.percolator.decoy.psms.txt"

    denovo_percolator_dir = denovo_dir / "results" / "percolator"
    denovo_rescore_tab = denovo_percolator_dir / "rescore.tab"

    def _drop_database_columns_and_run_percolator() -> list[str]:
        database_feature_columns = _drop_columns(
            str(database_rescore_tab), str(database_rescore_tab), "\t", config.drop_columns_database.split()
        )
        _run_percolator(
            config,
            [
                "--weights",
                database_weights.name,
                "--num-threads",
                str(config.percolator_threads),
                "--subset-max-train",
                str(config.percolator_subset_max_train),
                "--only-psms",
                "--no-terminate",
                "--verbose",
                str(config.percolator_verbose),
                "--post-processing-tdc",
                "--testFDR",
                str(config.percolator_test_fdr),
                "--trainFDR",
                str(config.percolator_train_fdr),
                "--results-psms",
                database_psms.name,
                "--decoy-results-psms",
                database_decoy_psms.name,
                database_rescore_tab.name,
            ],
            cwd=database_percolator_dir,
        )
        return database_feature_columns

    def _drop_denovo_columns() -> list[str]:
        return _drop_columns(str(denovo_rescore_tab), str(denovo_rescore_tab), "\t", config.drop_columns_denovo.split())

    # Database's Percolator run only needs its own rescore.tab (already on disk); de novo's
    # column filtering is on a different file and doesn't depend on database's Percolator at
    # all -- run them concurrently instead of one after the other.
    with ThreadPoolExecutor(max_workers=2) as pool:
        database_columns_future = pool.submit(_drop_database_columns_and_run_percolator)
        denovo_columns_future = pool.submit(_drop_denovo_columns)
        database_feature_columns = database_columns_future.result()
        denovo_feature_columns = denovo_columns_future.result()

    _require_matching_columns(database_feature_columns, denovo_feature_columns)

    def _merge_database() -> Path:
        return _merge_and_partition_branch(
            branch_dir=database_dir,
            rescore_tab=database_rescore_tab,
            percolator_psms=database_psms,
            percolator_decoy_psms=database_decoy_psms,
            keep_columns=DATABASE_COLUMNS,
        )

    def _run_denovo_percolator_and_merge() -> Path:
        denovo_avg_weights = denovo_percolator_dir / "rescore.percolator.weights.avg.csv"
        _parse_and_write_avg_weights(str(database_weights), str(denovo_avg_weights))

        denovo_psms = denovo_percolator_dir / "rescore.percolator.psms.txt"
        _run_percolator(
            config,
            [
                "--init-weights",
                denovo_avg_weights.name,
                "--static",
                "--num-threads",
                str(config.percolator_threads),
                "--only-psms",
                "--no-terminate",
                "--results-psms",
                denovo_psms.name,
                denovo_rescore_tab.name,
            ],
            cwd=denovo_percolator_dir,
        )
        return _merge_and_partition_branch(
            branch_dir=denovo_dir,
            rescore_tab=denovo_rescore_tab,
            percolator_psms=denovo_psms,
            percolator_decoy_psms=None,
            keep_columns=DENOVO_COLUMNS,
        )

    # Database's merge only needs its own Percolator output, already on disk at this point;
    # de novo's Percolator only needs database's weights.csv, also already on disk -- neither
    # depends on the other, so run database's merge concurrently with de novo's Percolator run
    # instead of merge-then-percolator.
    with ThreadPoolExecutor(max_workers=2) as pool:
        database_merge_future = pool.submit(_merge_database)
        denovo_future = pool.submit(_run_denovo_percolator_and_merge)
        database_merged_dir = database_merge_future.result()
        denovo_merged_dir = denovo_future.result()

    return RescoringResult(database_merged_dir=database_merged_dir, denovo_merged_dir=denovo_merged_dir)
