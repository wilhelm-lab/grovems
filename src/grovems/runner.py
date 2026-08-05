from __future__ import annotations

import logging
from pathlib import Path

from grovems import iforest as ifr
from grovems import psa
from grovems import rescoring as rs
from grovems.config import GrovemsConfig

logger = logging.getLogger(__name__)


def _require(config: GrovemsConfig, field_name: str) -> None:
    if not getattr(config, field_name):
        raise ValueError(f"Missing required config value: {field_name}")


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

    if config.run_iforest:
        _require(config, "iforest_features")
        if not ran_psa:
            _require(config, "grove_forest_dir")
            grove_forest_dir = Path(config.grove_forest_dir)
        logger.info("Running IForest stage")
        ifr.run(grove_forest_dir, config.iforest_features, grove_forest_dir / "model")

    logger.info("Done.")
