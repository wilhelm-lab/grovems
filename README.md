# grovems

A plain, installable Python package: Oktoberfest/Percolator rescoring -> PSA (peptide
similarity assignment) -> isolation-forest (IForest/SUOD) scoring, for the de novo sequencing.

![Pipeline overview: FragPipe/Casanovo search results feed PSMs into Oktoberfest feature generation, then PSA similarity grading, then isolation-forest rescoring](docs/assets/full_pipeline.png)

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

![Workflow](/cmnfs/proj/denovo_fdr/tools/grovems/docs/assets/full_pipeline.png)

## Quickstart

```bash
pip install -e .
cp assets/default_config.yaml my_config.yaml

grovems run --config my_config.yaml \
  --set run_iforest=false \
  --set outdir=/path/to/output
```

Any config value can be overridden per-run with `--set key=value` (repeatable) instead
of editing the file. Full usage (SLURM submission, standalone IForest runs) is in
[`docs/notes/quickstart.rst`](docs/notes/quickstart.rst); one-time setup is in
[`docs/notes/installation.rst`](docs/notes/installation.rst).

## Cluster portability

- **Percolator**: provided externally, not a `pyproject.toml` dependency. Point
  `percolator_exe: /path/to/percolator` at a specific local install (default: bare
  `percolator`, resolved via `PATH`), or set `percolator_module: percolator/3.7.1` for
  sites that provide it as an environment module instead (loaded via `module load`
  before `percolator_exe` runs -- requires the invoking shell to have Lmod's init
  sourced).
- **ThermoRawFileParser**: `thermo_exe: /path/to/ThermoRawFileParser.exe` points
  Oktoberfest at a local install (only relevant if it has to convert raw -> mzML itself;
  left unset, it falls back to its own per-platform default path).


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