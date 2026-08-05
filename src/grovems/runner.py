from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from joblib import dump

from grovems import iforest as ifr
from grovems import psa as ps
from grovems import rescoring as rs
from grovems.config import GrovemsConfig
from grovems.psa import combine_results_psa

logger = logging.getLogger(__name__)


def _require(config: GrovemsConfig, field_name: str) -> None:
    if not getattr(config, field_name):
        raise ValueError(f"Missing required config value: {field_name}")


def _run_psa(config: GrovemsConfig, outdir: Path, rescoring_result: rs.RescoringResult) -> Path:
    max_workers = config.psa_max_workers or os.cpu_count() or 1

    loader = ps.ResultLoader(
        database_oktoberfest_config=rescoring_result.database_config,
        denovo_oktoberfest_config=rescoring_result.denovo_config,
        database_pin_name="rescore.filtered.tab",
        denovo_pin_name="rescore.filtered.tab",
        max_workers=max_workers,
    )
    loader.set_path("database_pin", rescoring_result.database_filtered_pin)
    loader.set_path("denovo_pin", rescoring_result.denovo_filtered_pin)
    loader.set_path("database_percolator_target", rescoring_result.database_psms)
    loader.set_path("database_percolator_decoy", rescoring_result.database_decoy_psms)
    loader.set_path("denovo_percolator_target", rescoring_result.denovo_psms)
    # denovo_percolator_decoy is left at ResultLoader's auto-derived default: the de novo
    # Percolator run is static (--static, no --decoy-results-psms), so that path won't
    # exist and reading it degrades to an empty DataFrame with a logged warning.

    psa_output_dir = outdir / "rescoring" / "psa_dataframes"
    # run_pipeline() reads these as module globals (mirrors what combine_results_psa's own
    # main() does with its CLI args) rather than taking them as parameters.
    combine_results_psa.OUTPUT_DIR = psa_output_dir
    combine_results_psa.CACHE_DIR = psa_output_dir / "_cache"
    combine_results_psa.PSA_MAX_WORKERS = max_workers

    combine_results_psa.run_pipeline(
        loader,
        max_raw_files=config.psa_max_raw_files,
        rebuild_cache=config.psa_rebuild_cache,
        overwrite_outputs=config.psa_overwrite_outputs,
    )
    return psa_output_dir / "merged_SCAN"


def _run_iforest(config: GrovemsConfig, outdir: Path, merged_scan_dir: Path) -> None:
    iforest_dir = outdir / "iforest"
    iforest_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = config.iforest_features

    logger.info("Reading merged_SCAN dataset: %s", merged_scan_dir)
    merged_df = pd.read_parquet(merged_scan_dir)
    train_set = ifr.build_training_set(merged_df)

    logger.info("Training SUOD model")
    model = ifr.build_suod_model(train_set.shape[0])
    model.fit(train_set[feature_cols])
    dump(model, iforest_dir / "SUOD_model.pkl")

    logger.info("Predicting on database-only, denovo-only, and shared PSMs")
    database_results = ifr.predict_search_set(model, merged_df, "database_only", "database", feature_cols)
    denovo_results = ifr.predict_search_set(model, merged_df, "denovo_only", "denovo", feature_cols)
    # Shared PSMs are scored on their database-side features and are not dropped: every
    # _merge == "shared" row gets a prediction here, not just the high-confidence subset
    # used to train the model.
    shared_results = ifr.predict_search_set(model, merged_df, "shared", "database", feature_cols)

    ifr.write_partitioned(database_results, iforest_dir / "iforest_database")
    ifr.write_partitioned(denovo_results, iforest_dir / "iforest_denovo")
    ifr.write_partitioned(shared_results, iforest_dir / "iforest_shared")

    common_cols = [
        col for col in database_results.columns if col in denovo_results.columns and col in shared_results.columns
    ]
    iforest_all = pd.concat(
        [database_results[common_cols], denovo_results[common_cols], shared_results[common_cols]],
        ignore_index=True,
    )
    ifr.write_partitioned(iforest_all, iforest_dir / "iforest_all")


def run(config: GrovemsConfig) -> None:
    """Run the pipeline: rescoring -> PSA -> IForest, per ``config``'s stage toggles."""
    outdir = Path(config.outdir)

    if config.run_psa and not config.run_rescoring:
        raise ValueError(
            "run_psa requires run_rescoring (PSA is chained directly from this invocation's "
            "rescoring outputs, not an external directory). Enable both."
        )

    rescoring_result = None
    if config.run_rescoring:
        _require(config, "database_search_path")
        _require(config, "denovo_search_path")
        _require(config, "rawdata_path")
        logger.info("Running rescoring stage")
        rescoring_result = rs.run(config, outdir)

    merged_scan_dir = None
    if config.run_psa:
        logger.info("Running PSA stage")
        merged_scan_dir = _run_psa(config, outdir, rescoring_result)

    if config.run_iforest:
        _require(config, "iforest_features")
        if merged_scan_dir is None:
            _require(config, "psa_merged_scan_dir")
            merged_scan_dir = Path(config.psa_merged_scan_dir)
        logger.info("Running IForest stage")
        _run_iforest(config, outdir, merged_scan_dir)

    logger.info("Done.")
