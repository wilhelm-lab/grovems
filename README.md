# grovems

A plain, installable Python package: Oktoberfest/Percolator rescoring -> PSA (peptide
similarity assignment) -> isolation-forest (IForest/SUOD) scoring, for the `denovo_fdr`
project. Converted from the Nextflow pipeline at `tools/nf_denovo_rescoring/` (itself
ported from `run-rescoring.job` / `combine_results_psa_memory_efficient.py` /
`run_suod.job`) -- see ["Why not Nextflow"](#why-not-nextflow) below.

Repo layout follows `tools/master_template`'s conventions (`docs/`, `tests/`,
`tutorials/`, `LICENSE`, `Makefile`, `pyproject.toml`, `.pre-commit-config.yaml`,
`src/<package>/`); internally, `src/grovems/` mirrors the Oktoberfest package's own
shape -- a top-level `runner.py` orchestrator plus one subpackage per stage
(`rescoring/`, `psa/`, `iforest/`), each with an `__init__.py` re-exporting its public
functions, plus a shared `utils/`.

## What it does

**`rescoring/`** (`rescoring.run()`)
1. Runs Oktoberfest (Rescoring job type, in-process via `oktoberfest.runner.run_job`) on
   the MSFragger/FragPipe search results, producing predictions, CE calibration, RT
   model, a Percolator `.tab`/pin input, and `msms/*.rescore` (Oktoberfest's own
   normalized copy of the search results, used later by the PSA stage).
2. Drops unwanted columns from `rescore.tab` and trains Percolator on the filtered pin,
   producing PSMs, decoy PSMs, and learned feature weights.
3. Runs Oktoberfest on the Casanovo de novo results, reusing the CE-calibration and
   RT-model fitted in step 1 (same instrument run) instead of refitting them.
4. Averages the weight bins from step 2, drops the de novo-specific columns, and
   re-scores the de novo pin **statically** (`--init-weights ... --static`, no
   retraining) with those averaged weights.

**`psa/`** (`combine_results_psa.run_pipeline()`)
5. Merges the database and de novo search+pin+Percolator results per RAW file: on
   `SpecId` (`merged_PSM`) and on `RAW_FILE`/`SCAN_NUMBER` (`merged_SCAN`), then runs PSA
   (`psa_classifier.PSA`) on shared scans whose database and de novo sequences disagree,
   classifying how similar/different they are. Writes `psa_dataframes/`
   (`merged_database`, `merged_denovo`, `merged_PSM`, `merged_SCAN`, `shared_scan_psa`,
   plus `row_counts.csv`/`manifest.csv`/`shared_scan_psa_summary.csv`).

**`iforest/`** (composed directly in `runner.py`, from `iforest.py`'s functions)
6. Trains a SUOD ensemble of isolation forests on a high-confidence subset of shared
   (database/de novo-agreeing) PSMs from step 5's `merged_SCAN`, then scores **every**
   database-only, de novo-only, *and* shared PSM with it -- none are dropped. Writes
   `iforest_database/`, `iforest_denovo/`, and `iforest_shared/` (each
   `RAW_FILE`-partitioned) plus a combined `iforest_all/` with all three concatenated.

## Quickstart

```bash
pip install -e .   # or: poetry install
cp assets/default_config.yaml my_config.yaml   # edit the *_search_path/rawdata_path fields

grovems run --config my_config.yaml \
  --set run_iforest=false \
  --set outdir=/path/to/output
```

Any config value can be overridden per-run with `--set key=value` (repeatable) instead
of editing the file. Full usage (SLURM submission, standalone IForest runs) is in
[`docs/notes/quickstart.rst`](docs/notes/quickstart.rst); one-time setup is in
[`docs/notes/installation.rst`](docs/notes/installation.rst).

## Cluster portability

- **Execution**: grovems is a plain sequential Python process -- no scheduler required
  to run it, and no scheduler-specific code inside the package. For SLURM, see
  [`tutorials/run_pipeline.slurm`](tutorials/run_pipeline.slurm): a single `sbatch` job
  that activates the environment and runs `grovems run`. Resource sizing is just that
  job's `#SBATCH` lines -- there's no separate per-step resource model to keep in sync.
- **Percolator**: provided externally, not a `pyproject.toml` dependency. Point
  `percolator_exe: /path/to/percolator` at a specific local install (default: bare
  `percolator`, resolved via `PATH`), or set `percolator_module: percolator/3.7.1` for
  sites that provide it as an environment module instead (loaded via `module load`
  before `percolator_exe` runs -- requires the invoking shell to have Lmod's init
  sourced).
- **ThermoRawFileParser**: `thermo_exe: /path/to/ThermoRawFileParser.exe` points
  Oktoberfest at a local install (only relevant if it has to convert raw -> mzML itself;
  left unset, it falls back to its own per-platform default path).
- **The environment itself**: `oktoberfest`/`spectrum_fundamentals`/`spectrum_io` (the
  `feature/denovo_fdr` branches) and every other Python dependency are real
  `pyproject.toml` dependencies -- one `pip install -e .` / `poetry install` sets up
  everything except Percolator.

## Why not Nextflow

The pipeline's DAG is a short, mostly-linear chain of a few coarse-grained stages, each
operating on a whole directory in one call rather than fanning out per raw file -- it was
never exploiting Nextflow's main strength (implicit parallel scatter-gather over many
independent samples). What it *was* using -- SLURM/local portability, per-step conda env
management, resume/checkpointing -- turned out to have a real ongoing cost: a Nextflow
version upgrade broke the pipeline in three separate ways in one afternoon (a banned
`check_max()` function definition in `.config` files, `--flag false` on the CLI not
coercing to a real Boolean, and `params.x = ...` reassignment inside a workflow body
being silently ignored). None of that risk applies to a plain Python package.

One capability is genuinely given up in this conversion: Nextflow's `-resume`
(content-hash-based per-step caching) has no equivalent here -- a failed run currently
restarts from the beginning rather than skipping already-completed stages.

## Key config values

See [`assets/default_config.yaml`](assets/default_config.yaml) for the full list,
defaults, and comments. Notable ones:

| Key | Default | Meaning |
|---|---|---|
| `run_rescoring` / `run_psa` / `run_iforest` | `true` / `true` / `true` | enable/disable each stage. `run_psa` requires `run_rescoring` in the same invocation. |
| `drop_columns_database` | `lda_scores annotated_ions delta_mass_ppm log10_evalue next_score collision_energy_aligned` | columns dropped before training Percolator on the database pin |
| `drop_columns_denovo` | `lda_scores collision_energy_aligned` | columns dropped before scoring the de novo pin |
| `percolator_exe` | `percolator` | command/path used to invoke percolator (provided externally); point at an absolute path for a specific local install |
| `percolator_module` | `""` | environment module to load for percolator before `percolator_exe` runs; leave unset to skip module loading |
| `thermo_exe` | `null` | path to a local `ThermoRawFileParser.exe`; leave unset to use Oktoberfest's own default |
| `num_threads` / `psa_max_workers` / `percolator_threads` | `null` (-> `os.cpu_count()`) / `null` (-> `os.cpu_count()`) / `3` | worker/thread counts per stage; match these to your `#SBATCH --cpus-per-task` on a cluster |
| `psa_max_raw_files` | `null` | limit PSA to the first N raw files, for testing |
| `iforest_features` | (see file) | SUOD feature columns to train/score on -- inlined here instead of a separate config file |
| `psa_merged_scan_dir` | `null` | only used when `run_psa: false` -- external `psa_dataframes/merged_SCAN` for a standalone IForest run |

## Files

```
pyproject.toml                          poetry package config + deps (oktoberfest/spectrum_* @ feature/denovo_fdr + pipeline deps)
src/grovems/
  __main__.py                           CLI entry point (`grovems run --config ... [--set key=value ...]`)
  runner.py                             top-level orchestration (replaces main.nf): sequences rescoring -> PSA -> IForest
  config.py                             GrovemsConfig dataclass, YAML loading, --set overrides
  utils/                                 shared logging setup
  rescoring/
    rescoring.py                         Oktoberfest + Percolator orchestration (database & de novo)
    drop_columns.py                      drop unwanted pin columns before Percolator
    averageweight.py                     average Percolator's per-bin weight vectors
  psa/
    combine_results_psa.py               PSA merge driver
    eval.py                              ResultLoader + column constants used by the PSA driver
    psa_classifier.py                    PSA classification: combines the mixins below into one PSA class
    psa_result.py                        PSAResult data container
    psa_mass_utils.py                    peptide mass calculation/equality helpers
    psa_levenshtein.py                   Levenshtein distance/tier + edit-operation tracing
    psa_event_detection.py               alignment-based event-candidate detection
    psa_event_labeling.py                event-candidate labeling and change summaries
  iforest/
    iforest.py                           train SUOD + score database-only/denovo-only/shared PSMs
assets/default_config.yaml              default config (copy, edit *_search_path/rawdata_path, run)
docs/                                    Sphinx docs (installation/quickstart/citation notes)
test.sh                                 run locally against one real proteometools raw file (symlinked fixture, no copying)
tests/integration_tests/README.md       smoke-test instructions (no bundled fixtures)
tutorials/run_pipeline.slurm            example SLURM batch script
LICENSE                                 MIT, matching the sibling wilhelm-lab tool repos
Makefile                                poetry install/lint/format/test/docs/dist targets
.pre-commit-config.yaml                 ruff + generic hooks (large files, yaml/toml checks, ...)
```

## Deliberate differences from `nf_denovo_rescoring` (beyond dropping Nextflow itself)

- **`iforest/iforest.py`, `rescoring/drop_columns.py`, `rescoring/averageweight.py`,
  `psa/eval.py`, `psa/combine_results_psa.py`, and the `psa_*.py` mixins are moved
  essentially verbatim** from `nf_denovo_rescoring/bin/`, with only their sibling imports
  rewritten to relative package imports (they only worked as flat `bin/` scripts because
  Nextflow invoked each one directly, letting Python auto-add the script's own directory
  to `sys.path`; that stops applying once they're modules inside a package).
- **`peptide_similarity_assisgnment.py` is renamed to `psa_classifier.py`**, fixing the
  original filename typo -- nothing outside the package depended on the old name.
- **`combine_results_psa.run_pipeline()` still reads `OUTPUT_DIR`/`CACHE_DIR`/
  `PSA_MAX_WORKERS` as module globals**, exactly as it did when only its own CLI `main()`
  set them. `runner.py` sets these on the module directly before calling
  `run_pipeline()`, replicating what `main()` used to do, rather than refactoring the
  function's signature.
- **Oktoberfest runs in-process** (`oktoberfest.runner.run_job()`) instead of via
  `python -m oktoberfest` subprocess calls -- it's a real dependency of this package now,
  not something invoked as an external tool per Nextflow process. Both `run_job()` calls
  are wrapped to catch `SystemExit` specifically: a worker-pool failure inside Oktoberfest
  calls `sys.exit(1)` rather than raising a normal exception, which would otherwise kill
  the whole `grovems` process silently instead of surfacing a clean error.
- **The SUOD feature list is inlined into the main config** (`iforest_features`) instead
  of a separate `config_train.yaml` file that `iforest_config` pointed at.
- **No per-stage resource/executor abstraction.** Nextflow's `conf/cluster.config` and
  the `process.resourceLimits` closure in `conf/base.config` solved a problem specific to
  Nextflow's per-process model (each step needing its own right-sized resource request);
  a single sequential Python process doesn't have that problem. `num_threads`/
  `psa_max_workers`/`percolator_threads` are plain config values instead.
