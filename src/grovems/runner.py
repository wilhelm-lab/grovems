from __future__ import annotations

import logging
from pathlib import Path

from grovems import iforest as ifr
from grovems import plotting as plot
from grovems import postprocess as pp
from grovems import psa
from grovems import rescoring as rs
from grovems.config import GrovemsConfig

logger = logging.getLogger(__name__)


def _require(config: GrovemsConfig, field_name: str) -> None:
    if not getattr(config, field_name):
        raise ValueError(f"Missing required config value: {field_name}")


def run(config: GrovemsConfig) -> None:
    """Run the pipeline: rescoring -> PSA -> IForest, per ``config``'s stage toggles.

    With ``config.denovo_only``, the database branch/Percolator/PSA are skipped entirely
    (see ``rescoring.run``/``psa.run``) and postprocess/plotting run their de novo-only
    variants instead.
    """
    outdir = Path(config.outdir)

    if config.run_psa and not config.run_rescoring:
        raise ValueError(
            "run_psa requires run_rescoring (PSA is chained directly from this invocation's "
            "rescoring outputs, not an external directory). Enable both."
        )

    if (
        config.denovo_only
        and (config.run_iforest or config.run_postprocess)
        and config.iforest_training_source != "denovo_score"
    ):
        raise ValueError(
            "denovo_only=True requires iforest_training_source='denovo_score' -- "
            "'percolator_percentile' needs a database-side Percolator score that won't exist."
        )

    rescoring_result = None
    if config.run_rescoring:
        # database_search_path/denovo_search_path/rawdata_path are only needed for
        # branches that aren't reusing an existing Oktoberfest output directory.
        if config.denovo_only:
            if not config.denovo_oktoberfest_dir:
                _require(config, "denovo_search_path")
                _require(config, "rawdata_path")
        else:
            if not config.database_oktoberfest_dir:
                _require(config, "database_search_path")
            if not config.denovo_oktoberfest_dir:
                _require(config, "denovo_search_path")
            if not (config.database_oktoberfest_dir and config.denovo_oktoberfest_dir):
                _require(config, "rawdata_path")
        logger.info("Running rescoring stage%s", " (denovo-only, no database)" if config.denovo_only else "")
        rescoring_result = rs.run(config, outdir)

    grove_forest_dir = outdir / "rescoring" / "grove_forest"
    ran_psa = False
    if config.run_psa:
        logger.info("Running PSA stage")
        psa.run(
            rescoring_result.database_merged_dir,
            rescoring_result.denovo_merged_dir,
            grove_forest_dir,
            max_raw_files=config.psa_max_raw_files,
            max_workers=config.psa_max_workers,
            overwrite_outputs=config.overwrite_outputs,
        )
        ran_psa = True

    ran_iforest = False
    if config.run_iforest:
        _require(config, "iforest_features")
        if not ran_psa:
            _require(config, "grove_forest_dir")
            grove_forest_dir = Path(config.grove_forest_dir)
        logger.info("Running IForest stage")
        ifr.run(
            grove_forest_dir,
            config.iforest_features,
            grove_forest_dir / "model",
            training_source=config.iforest_training_source,
            denovo_score_threshold=config.iforest_denovo_score_threshold,
        )
        ran_iforest = True

    ran_postprocess = False
    if config.run_postprocess:
        if not ran_iforest:
            _require(config, "grove_forest_dir")
            grove_forest_dir = Path(config.grove_forest_dir)
        logger.info("Running postprocess (ECDF/KS) stage")
        if config.denovo_only:
            pp.run_denovo_only(grove_forest_dir, config.iforest_denovo_score_threshold)
        else:
            pp.run(grove_forest_dir)
        ran_postprocess = True

    if config.run_plotting:
        if not (ran_iforest or ran_postprocess):
            _require(config, "grove_forest_dir")
            grove_forest_dir = Path(config.grove_forest_dir)
        logger.info("Running plotting stage")
        if config.denovo_only:
            plot.run_denovo_only(grove_forest_dir)
        else:
            plot.run(grove_forest_dir)

    logger.info("Done.")
