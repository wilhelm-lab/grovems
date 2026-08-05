from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Optional, Union

import yaml

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GrovemsConfig:
    """Pipeline configuration -- one field per key in a config YAML file.

    Load with :meth:`from_yaml`, then optionally apply ``--set key=value`` overrides
    with :meth:`apply_overrides`.
    """

    # ---- stage toggles ----
    run_rescoring: bool = True
    run_psa: bool = True
    run_iforest: bool = True
    run_postprocess: bool = True

    outdir: str = "results"

    # ---- rescoring stage: inputs ----
    database_search_path: Optional[str] = None  # dir of MSFragger/FragPipe combined search results
    denovo_search_path: Optional[str] = None  # dir of Casanovo search results
    rawdata_path: Optional[str] = None  # dir of mzML spectra
    database_search_type: str = "Msfragger"
    denovo_search_type: str = "Casanovo"
    spectra_type: str = "mzML"

    # Reuse an existing Oktoberfest output directory instead of running Oktoberfest for
    # that branch (predictions/ce_calibration/rt_model/msms/rescore.tab already there,
    # as if Oktoberfest had just finished running in that location). Used in place:
    # rescore.tab is still filtered in place and, with Percolator's output, deleted once
    # merged/ is built -- same lifecycle as a freshly-run branch. Set one, both, or
    # neither; database_search_path/denovo_search_path/rawdata_path are only required
    # for branches that aren't reused this way.
    database_oktoberfest_dir: Optional[str] = None
    denovo_oktoberfest_dir: Optional[str] = None

    # ---- rescoring stage: Oktoberfest / Prosit ----
    irt_model: str = "Prosit_2019_irt"
    intensity_model: str = "Prosit_2020_intensity_HCD"
    prediction_server: str = "koina.wilhelmlab.org:443"
    fdr_estimation_method: str = "percolator"
    all_features: bool = False
    regression_method: str = "spline"
    ssl: bool = True
    thermo_exe: Optional[str] = None
    mass_tolerance: float = 20
    p_window: float = 1.2
    unit_mass_tolerance: str = "ppm"
    fragmentation_method: str = "HCD"
    num_threads: Optional[int] = None  # defaults to os.cpu_count() if unset

    # ---- rescoring stage: column filtering before Percolator ----
    # Space-separated, matching the original conf/base.config format.
    drop_columns_database: str = (
        "lda_scores annotated_ions delta_mass_ppm log10_evalue next_score collision_energy_aligned"
    )
    drop_columns_denovo: str = "lda_scores collision_energy_aligned"

    # ---- rescoring stage: Percolator ----
    # Percolator is provided externally -- not a pyproject.toml dependency. Point one of
    # these at your install:
    #   percolator_exe    command/path to invoke, e.g. an absolute path to a local build
    #                     (default: bare 'percolator', resolved via PATH)
    #   percolator_module environment-module name (e.g. 'percolator/3.7.1') for sites
    #                     using Environment Modules / Lmod; loaded before percolator_exe
    #                     runs (requires the invoking shell to have Lmod's init sourced).
    #                     Leave blank (default) to skip module loading.
    percolator_exe: str = "percolator"
    percolator_module: str = ""
    percolator_threads: int = 3
    percolator_subset_max_train: int = 100000
    percolator_test_fdr: float = 0.01
    percolator_train_fdr: float = 0.01
    percolator_verbose: int = 3

    # ---- psa stage ----
    psa_max_raw_files: Optional[int] = None  # limit to first N raw files, for testing
    overwrite_outputs: bool = False  # overwrite existing per-RAW output files (merged/, grove_forest/)
    psa_max_workers: Optional[int] = None  # defaults to os.cpu_count() if unset

    # ---- iforest stage ----
    # SUOD feature columns to train/score on -- inlined here rather than a separate
    # config_train.yaml file.
    iforest_features: list[str] = dataclasses.field(default_factory=list)
    # Only needed when run_iforest and/or run_postprocess are true but run_psa is false
    # (standalone iforest/postprocess run); otherwise the PSA stage's own grove_forest/
    # output (or, for a postprocess-only run, the IForest stage's own output) is used
    # directly.
    grove_forest_dir: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "GrovemsConfig":
        """Load a config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "GrovemsConfig":
        """Build a config from a plain dict, rejecting unknown keys."""
        known_fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known_fields
        if unknown:
            raise ValueError(f"Unknown config key(s): {sorted(unknown)}")
        return cls(**data)

    def apply_overrides(self, overrides: list[str]) -> None:
        """Apply ``key=value`` overrides in place (as from ``--set``), YAML-typed.

        Args:
            overrides: Strings of the form ``"key=value"``; ``value`` is parsed with
                ``yaml.safe_load`` so ``true``/``false``/numbers/plain strings all come
                out the same type they would if written directly in the config file.
        """
        known_fields = {f.name for f in dataclasses.fields(self)}
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"--set expects key=value, got: {item!r}")
            key, raw_value = item.split("=", 1)
            key = key.strip()
            if key not in known_fields:
                raise ValueError(f"Unknown config key: {key!r}")
            value = yaml.safe_load(raw_value)
            logger.info("Overriding %s = %r", key, value)
            setattr(self, key, value)
