# grovems

A plain, installable Python package: Oktoberfest/Percolator rescoring -> PSA (peptide
similarity assignment) -> isolation-forest (IForest/SUOD) scoring, for the de novo sequencing.


## What it does

**`rescoring/`** (`rescoring.run()`) -- once per branch (database, then de novo):
1. Runs Oktoberfest (Rescoring job type, in-process via `oktoberfest.runner.run_job`) on
   the MSFragger/FragPipe search results, producing predictions, CE calibration, RT
   model, and a Percolator `.tab`/pin input. The de novo branch reuses the database
   branch's CE-calibration and RT-model (same instrument run) instead of refitting them.
2. Filters `rescore.tab` **in place** (drops unwanted columns, overwrites the same
   file -- no separate filtered copy). Since de novo rescoring is static (next step)
   and applies the database branch's weights positionally, the two branches' filtered
   columns must end up identical and identically ordered -- checked explicitly
   (`_require_matching_columns`) before de novo's Percolator run, so a
   `drop_columns_database`/`drop_columns_denovo` mismatch (or reusing an existing
   Oktoberfest directory with a different feature set) fails loudly instead of silently
   misapplying weights to the wrong features.
3. Runs Percolator against that filtered `rescore.tab`. The de novo branch averages the
   database branch's learned weights first and rescores **statically** (no retraining).
4. Merges `rescore.tab` + Percolator's PSM output + that branch's own `msms/*.rescore`
   search results into `<branch>/merged/<raw_file>.parquet`, one file per raw file, then
   **deletes** `rescore.tab` and the Percolator PSM output -- their data now lives in
   `merged/`, so keeping both would just be two copies of the same rows.

**`psa/`** (`psa_merge.run()`)
5. For every raw file, outer-merges the database and de novo `merged/` data on
   `RAW_FILE`/`SCAN_NUMBER` (a plain outer merge already keeps every match when a scan
   has more than one hit on either side -- nothing is deduplicated), then runs PSA
   (`psa_classifier.PSA`) on shared scans whose sequences differ, classifying how
   similar/different they are. Writes `grove_forest/<raw_file>.parquet` -- feature and
   PSA columns together in one file (PSA columns are null for non-shared rows) -- then
   deletes the per-branch `merged/` directories the same way step 4 deleted their own
   sources.

**`iforest/`** (`iforest.run()`, composed in `runner.py`)
6. Two passes over `grove_forest/*.parquet`: pass 1 gathers only the high-confidence
   shared-PSM training candidates from every file (without holding each file's full data
   at once) to fit a SUOD ensemble of isolation forests, saved to
   `grove_forest/model/SUOD_model.pkl`; pass 2 scores **every** row of every file
   (database-only, de novo-only, *and* shared -- none are dropped) and overwrites that
   file with `ISO_labels`/`ISO_scores` columns added. End state: every
   `grove_forest/<raw_file>.parquet` carries feature, PSA, and IForest columns together,
   updated in place at each stage.

**`postprocess/`** (`postprocess.run()`)
7. Builds an empirical CDF of `ISO_scores` from the trusted shared PSMs -- the same
   `_merge == "shared"`/target/positive-database-score population IForest itself trains
   on (`iforest.trusted_shared_mask`) -- then adds a `TP_GOODNESS` column to every PSM in
   every `grove_forest/*.parquet` file: the fraction of that reference at least as
   anomalous as this PSM (1.0 = better than every reference PSM, 0.0 = worse than all of
   them). Separately, a two-sample Kolmogorov-Smirnov test compares the reference
   distribution against `database_only` and against `denovo_only` `ISO_scores`; each
   comparison's point of maximum divergence (`ks_2samp`'s `statistic_location`) is that
   comparison's natural accept/reject boundary, and averaging the two locations gives one
   cutoff used to label every `database_only`/`denovo_only` PSM `GOOD` or `BAD`. Writes
   `grove_forest/qc/ks_vs_tp.svg`, `qc/ks_vs_tp_summary.csv` (D, p-value, location per
   comparison, plus the averaged cutoff), and `qc/good_bad_database_only.csv` /
   `qc/good_bad_denovo_only.csv`.

![Pipeline overview: FragPipe/Casanovo search results feed PSMs into Oktoberfest feature generation, then PSA similarity grading, then isolation-forest rescoring](docs/assets/full_pipeline.png)

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
| `run_rescoring` / `run_psa` / `run_iforest` / `run_postprocess` | `true` / `true` / `true` / `true` | enable/disable each stage. `run_psa` requires `run_rescoring` in the same invocation; `run_postprocess` needs `ISO_scores` (from `run_iforest`, this invocation or an existing `grove_forest_dir`). |
| `database_oktoberfest_dir` / `denovo_oktoberfest_dir` | `null` / `null` | reuse an existing Oktoberfest output directory for that branch instead of running Oktoberfest (used in place); set one, both, or neither |
| `drop_columns_database` | `lda_scores annotated_ions delta_mass_ppm log10_evalue next_score collision_energy_aligned` | columns dropped before training Percolator on the database pin |
| `drop_columns_denovo` | `lda_scores collision_energy_aligned` | columns dropped before scoring the de novo pin |
| `percolator_exe` | `percolator` | command/path used to invoke percolator (provided externally); point at an absolute path for a specific local install |
| `percolator_module` | `""` | environment module to load for percolator before `percolator_exe` runs; leave unset to skip module loading |
| `thermo_exe` | `null` | path to a local `ThermoRawFileParser.exe`; leave unset to use Oktoberfest's own default |
| `num_threads` / `psa_max_workers` / `percolator_threads` | `null` (-> `os.cpu_count()`) / `null` (-> `os.cpu_count()`) / `3` | worker/thread counts per stage; match these to your `#SBATCH --cpus-per-task` on a cluster |
| `psa_max_raw_files` | `null` | limit PSA to the first N raw files, for testing |
| `iforest_features` | (see file) | SUOD feature columns to train/score on -- inlined here instead of a separate config file |
| `overwrite_outputs` | `false` | overwrite existing per-raw-file outputs (`merged/`, `grove_forest/`) instead of skipping raw files already processed |
| `grove_forest_dir` | `null` | only used when `run_psa: false` -- external `grove_forest/` directory for a standalone IForest and/or postprocess run (postprocess additionally needs it if `run_iforest: false` too, pointing at already-IForest-scored data) |