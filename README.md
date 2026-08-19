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

**`psa/`** (`psa_merge.run()`) 5. For every raw file, outer-merges the database and de novo `merged/` data on
`RAW_FILE`/`SCAN_NUMBER` (a plain outer merge already keeps every match when a scan
has more than one hit on either side -- nothing is deduplicated), then runs PSA
(`psa_classifier.PSA`) on shared scans whose sequences differ, classifying how
similar/different they are. Writes `grove_forest/results/<raw_file>.parquet` --
feature and PSA columns together in one file (PSA columns are null for non-shared
rows) -- then deletes the per-branch `merged/` directories the same way step 4
deleted their own sources.

**`iforest/`** (`iforest.run()`, composed in `runner.py`) 6. Two passes over `grove_forest/results/*.parquet`: pass 1 gathers only the
high-confidence shared-PSM training candidates from every file (without holding
each file's full data at once) to fit a SUOD ensemble of isolation forests, saved to
`grove_forest/model/SUOD_model.pkl`; pass 2 scores **every** row of every file
(database-only, de novo-only, _and_ shared -- none are dropped). Each side
(database, de novo) is scored independently on its own feature vector wherever that
side has an identification at all, so `shared` rows -- a scan with a hit from both
engines, not necessarily the same peptide -- get both an `ISO_scores_database` and an
`ISO_scores_denovo`. The legacy single-value `ISO_scores`/`ISO_labels` columns are
kept too, for existing consumers: database-side for database-only/shared rows,
de novo-side for de novo-only rows, with `ISO_scores_side` recording which of the two
independent columns that legacy value mirrors. A final pass min-max normalizes every
`ISO_scores*` column to `[0, 1]` across every file (shared min/max, so all three stay
comparable), inverted so `1.0` = good/least anomalous and `0.0` = bad/most anomalous.
End state: every `grove_forest/results/<raw_file>.parquet` carries feature, PSA, and
IForest columns together, updated in place at each stage.

**`postprocess/`** (`postprocess.run()`) 7. Builds two empirical CDFs from the trusted shared PSMs -- the same
`_merge == "shared"`/target/positive-database-score population IForest itself trains
on (`iforest.trusted_shared_mask`) -- one from their `ISO_scores_database` and one
from their `ISO_scores_denovo`, since `shared` rows are scored independently on both
sides. Adds `TP_GOODNESS_database`/`TP_GOODNESS_denovo` to every PSM in every
`grove_forest/results/*.parquet` file: the fraction of that side's reference at least
as anomalous as this PSM's score on that side (1.0 = better than every reference PSM,
0.0 = worse than all of them). Separately, a two-sample Kolmogorov-Smirnov test
compares each side's reference distribution against that side's own uncorroborated
population (`ISO_scores_database` reference vs `database_only`, `ISO_scores_denovo`
reference vs `denovo_only`); each comparison's point of maximum divergence
(`ks_2samp`'s `statistic_location`) is that side's own accept/reject cutoff -- the two
cutoffs are kept separate, not averaged, since the two sides were scored on different
feature vectors. Every PSM gets a `CALL_database`/`CALL_denovo` (`GOOD` if
`ISO_scores_<side> >= cutoff_<side>`, else `BAD`, null where that side has no
identification), plus legacy single-value `ISO_scores`/`TP_GOODNESS`/`CALL` columns
for whichever side is that row's primary identification (`ISO_scores_side`:
database-side for database-only/shared rows, denovo-side for denovo-only rows).
Writes `grove_forest/qc/ks_vs_tp.svg` (one ECDF panel per side), `qc/ks_vs_tp_summary.csv`
(D, p-value, location per comparison, plus both cutoffs), and `qc/psm_overlap.svg` (a
bar chart of PSM counts per `DETECTION_LEVEL`: `shared`/`database`/`denovo`). The
classified PSMs themselves are split by the legacy `CALL` into `grove_forest/good.csv`
and `grove_forest/bad.csv`, each carrying both sides' scores/goodness/calls, a
`DETECTION_LEVEL` column, and, per PSM, its Casanovo score (`CASANOVO_SCORE`),
database search score (`DATABASE_SCORE`), database Percolator score
(`PERCOLATOR_SCORE_DATABASE`), and, for rows with a de novo call, a comma-separated
`CASANOVO_AA_SCORE` column of that call's per-residue Casanovo confidence scores.

**`plotting/`** (`plotting.run()`, opt-in via `run_plotting: true` -- see below) 8. Extra QC plots, written to `grove_forest/qc/` alongside the postprocess ones:
`shared_venn.svg` (shared scans vs shared PSM -- how much of the scan-level overlap
between the two engines is an actual agreeing identification vs just two different
calls on the same spectrum), `levenshtein_distribution.svg` (edit distance between
database and de novo sequence on shared PSMs, from `PSA_LEVENSHTEIN` -- one
distribution, since the distance is symmetric and only defined where both sequences
exist), `peptide_length_distribution.svg` (database vs de novo, all PSMs, no
filtering), and `perc_vs_iso_database.svg`/`perc_vs_iso_denovo.svg` (joint density of
`percolator_score_<side>` vs `ISO_scores_<side>`, each against that side's own
postprocess cutoff -- database split target/decoy, de novo single-layer since it has
no decoys). Reads straight from `grove_forest/results/*.parquet` and computes its own
cutoffs read-only, so it can run standalone against an already-scored `grove_forest_dir`
even if postprocess didn't run in the same invocation.

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
of editing the file. Every stage's log output is both printed to stdout and appended to
`<outdir>/grovems.log`, and the fully-resolved config (input file + any `--set`
overrides applied) is written to `<outdir>/config.yaml` -- a permanent, self-contained
record of exactly what produced that run's output, independent of whatever `--set` flags
were used to launch it. Full usage (SLURM submission, standalone IForest runs) is in
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

| Key                                                             | Default                                                                                     | Meaning                                                                                                                                                                                                                        |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `run_rescoring` / `run_psa` / `run_iforest` / `run_postprocess` | `true` / `true` / `true` / `true`                                                           | enable/disable each stage. `run_psa` requires `run_rescoring` in the same invocation; `run_postprocess` needs `ISO_scores` (from `run_iforest`, this invocation or an existing `grove_forest_dir`).                            |
| `run_plotting`                                                  | `false`                                                                                     | opt-in extra QC plots (Venn, Levenshtein, peptide length, percolator-vs-isolation-score); needs `ISO_scores_database`/`ISO_scores_denovo` (from `run_iforest`, this invocation or an existing `grove_forest_dir`)              |
| `database_oktoberfest_dir` / `denovo_oktoberfest_dir`           | `null` / `null`                                                                             | reuse an existing Oktoberfest output directory for that branch instead of running Oktoberfest (used in place); set one, both, or neither                                                                                       |
| `drop_columns_database`                                         | `lda_scores annotated_ions delta_mass_ppm log10_evalue next_score collision_energy_aligned` | columns dropped before training Percolator on the database pin                                                                                                                                                                 |
| `drop_columns_denovo`                                           | `lda_scores collision_energy_aligned`                                                       | columns dropped before scoring the de novo pin                                                                                                                                                                                 |
| `percolator_exe`                                                | `percolator`                                                                                | command/path used to invoke percolator (provided externally); point at an absolute path for a specific local install                                                                                                           |
| `percolator_module`                                             | `""`                                                                                        | environment module to load for percolator before `percolator_exe` runs; leave unset to skip module loading                                                                                                                     |
| `thermo_exe`                                                    | `null`                                                                                      | path to a local `ThermoRawFileParser.exe`; leave unset to use Oktoberfest's own default                                                                                                                                        |
| `num_threads` / `psa_max_workers` / `percolator_threads`        | `null` (-> `os.cpu_count()`) / `null` (-> `os.cpu_count()`) / `3`                           | worker/thread counts per stage; match these to your `#SBATCH --cpus-per-task` on a cluster                                                                                                                                     |
| `psa_max_raw_files`                                             | `null`                                                                                      | limit PSA to the first N raw files, for testing                                                                                                                                                                                |
| `iforest_features`                                              | (see file)                                                                                  | SUOD feature columns to train/score on -- inlined here instead of a separate config file                                                                                                                                       |
| `overwrite_outputs`                                             | `false`                                                                                     | overwrite existing per-raw-file outputs (`merged/`, `grove_forest/`) instead of skipping raw files already processed                                                                                                           |
| `grove_forest_dir`                                              | `null`                                                                                      | only used when `run_psa: false` -- external `grove_forest/` directory for a standalone IForest and/or postprocess run (postprocess additionally needs it if `run_iforest: false` too, pointing at already-IForest-scored data) |
