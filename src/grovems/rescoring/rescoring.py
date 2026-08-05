from __future__ import annotations

import dataclasses
import json
import logging
import os
import shlex
import subprocess
from pathlib import Path

from oktoberfest import runner as oktoberfest_runner

from ..config import GrovemsConfig
from .averageweight import parse_and_write_avg_weights
from .drop_columns import drop_columns

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RescoringResult:
    """Paths produced by :func:`run`, consumed by the PSA stage."""

    oktoberfest_database: Path
    database_config: Path
    oktoberfest_denovo: Path
    denovo_config: Path
    database_psms: Path
    database_decoy_psms: Path
    database_weights: Path
    database_filtered_pin: Path
    denovo_psms: Path
    denovo_avg_weights: Path
    denovo_filtered_pin: Path


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


def _run_oktoberfest(config_path: Path) -> None:
    logger.info("Running Oktoberfest: %s", config_path)
    try:
        oktoberfest_runner.run_job(str(config_path))
    except SystemExit as exc:
        # A worker pool failure inside Oktoberfest (JobPool.check_pool) calls
        # sys.exit(1) instead of raising a normal exception -- catch it explicitly so it
        # doesn't kill the whole grovems process.
        raise RuntimeError(f"Oktoberfest exited (SystemExit({exc.code})) while running {config_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Oktoberfest failed while running {config_path}: {exc}") from exc


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


def run(config: GrovemsConfig, outdir: Path) -> RescoringResult:
    """Run the rescoring stage: database Oktoberfest+Percolator, then de novo.

    Args:
        config: Pipeline configuration.
        outdir: Top-level output directory (results go under ``outdir/rescoring``).

    Returns:
        Paths to every rescoring output the PSA stage needs.
    """
    rescoring_dir = outdir / "rescoring"
    rescoring_dir.mkdir(parents=True, exist_ok=True)
    num_threads = config.num_threads or os.cpu_count() or 1

    # --- database: Oktoberfest ---
    database_out = rescoring_dir / "oktoberfest_database"
    database_config_path = rescoring_dir / "rescoring_config_database.json"
    database_json = _build_oktoberfest_config(
        output_dir=database_out,
        search_results=Path(config.database_search_path),
        search_results_type=config.database_search_type,
        spectra=Path(config.rawdata_path),
        spectra_type=config.spectra_type,
        config=config,
        num_threads=num_threads,
    )
    database_config_path.write_text(json.dumps(database_json, indent=4))
    _run_oktoberfest(database_config_path)

    # --- database: Percolator ---
    percolator_database_dir = rescoring_dir / "percolator_database"
    percolator_database_dir.mkdir(parents=True, exist_ok=True)
    database_filtered_pin = percolator_database_dir / "rescore.filtered.tab"
    drop_columns(
        str(database_out / "results" / "percolator" / "rescore.tab"),
        str(database_filtered_pin),
        "\t",
        config.drop_columns_database.split(),
    )
    database_psms = percolator_database_dir / "rescore.percolator.psms.txt"
    database_decoy_psms = percolator_database_dir / "rescore.percolator.decoy.psms.txt"
    database_weights = percolator_database_dir / "rescore.percolator.weights.csv"
    _run_percolator(
        config,
        [
            "--weights",
            "rescore.percolator.weights.csv",
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
            "rescore.percolator.psms.txt",
            "--decoy-results-psms",
            "rescore.percolator.decoy.psms.txt",
            "rescore.filtered.tab",
        ],
        cwd=percolator_database_dir,
    )

    # --- de novo: Oktoberfest (reuse database's ce_calibration/rt_model) ---
    denovo_out = rescoring_dir / "oktoberfest_denovo"
    (denovo_out / "results").mkdir(parents=True, exist_ok=True)
    _relink(database_out / "results" / "ce_calibration", denovo_out / "results" / "ce_calibration")
    _relink(database_out / "results" / "rt_model", denovo_out / "results" / "rt_model")

    denovo_config_path = rescoring_dir / "rescoring_config_denovo.json"
    denovo_json = _build_oktoberfest_config(
        output_dir=denovo_out,
        search_results=Path(config.denovo_search_path),
        search_results_type=config.denovo_search_type,
        spectra=Path(config.rawdata_path),
        spectra_type=config.spectra_type,
        config=config,
        num_threads=num_threads,
    )
    denovo_config_path.write_text(json.dumps(denovo_json, indent=4))
    _run_oktoberfest(denovo_config_path)

    # --- de novo: Percolator (init with averaged database weights, static) ---
    percolator_denovo_dir = rescoring_dir / "percolator_denovo"
    percolator_denovo_dir.mkdir(parents=True, exist_ok=True)
    denovo_avg_weights = percolator_denovo_dir / "rescore.percolator.weights.avg.csv"
    parse_and_write_avg_weights(str(database_weights), str(denovo_avg_weights))

    denovo_filtered_pin = percolator_denovo_dir / "rescore.filtered.tab"
    drop_columns(
        str(denovo_out / "results" / "percolator" / "rescore.tab"),
        str(denovo_filtered_pin),
        "\t",
        config.drop_columns_denovo.split(),
    )
    denovo_psms = percolator_denovo_dir / "rescore.percolator.psms.txt"
    _run_percolator(
        config,
        [
            "--init-weights",
            "rescore.percolator.weights.avg.csv",
            "--static",
            "--num-threads",
            str(config.percolator_threads),
            "--only-psms",
            "--no-terminate",
            "--results-psms",
            "rescore.percolator.psms.txt",
            "rescore.filtered.tab",
        ],
        cwd=percolator_denovo_dir,
    )

    return RescoringResult(
        oktoberfest_database=database_out,
        database_config=database_config_path,
        oktoberfest_denovo=denovo_out,
        denovo_config=denovo_config_path,
        database_psms=database_psms,
        database_decoy_psms=database_decoy_psms,
        database_weights=database_weights,
        database_filtered_pin=database_filtered_pin,
        denovo_psms=denovo_psms,
        denovo_avg_weights=denovo_avg_weights,
        denovo_filtered_pin=denovo_filtered_pin,
    )
