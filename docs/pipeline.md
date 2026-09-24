# Pipeline stages

`grovems` runs four required stages (`rescoring` -> `psa` -> `iforest` -> `postprocess`)
plus one optional one (`plotting`), each toggleable via `run_<stage>: true/false` in the
config. Every stage reads/writes `grove_forest/results/*.parquet` (or its own
intermediate directories) in place, so later stages see the columns earlier stages added.

## 1. `rescoring/` (`rescoring.run()`)

Runs once per branch (database, then de novo):

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

## 2. `psa/` (`psa_merge.run()`)

5. For every raw file, outer-merges the database and de novo `merged/` data on
   `RAW_FILE`/`SCAN_NUMBER` (a plain outer merge already keeps every match when a scan
   has more than one hit on either side -- nothing is deduplicated), then runs PSA
   (`psa_classifier.PSA`) on shared scans whose sequences differ, classifying how
   similar/different they are. Writes `grove_forest/results/<raw_file>.parquet` --
   feature and PSA columns together in one file (PSA columns are null for non-shared
   rows) -- then deletes the per-branch `merged/` directories the same way step 4
   deleted their own sources.

## 3. `iforest/` (`iforest.run()`, composed in `runner.py`)

6. Two passes over `grove_forest/results/*.parquet`: pass 1 gathers only the
   high-confidence shared-PSM training candidates from every file (without holding
   each file's full data at once) to fit a SUOD ensemble of isolation forests, saved to
   `grove_forest/model/SUOD_model.pkl`; pass 2 scores **every** row of every file
   (database-only, de novo-only, *and* shared -- none are dropped). Each side
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

## 4. `postprocess/` (`postprocess.run()`)

7. Builds two empirical CDFs from the trusted shared PSMs -- the same
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

## 5. `plotting/` (`plotting.run()`, opt-in via `run_plotting: true`)

8. Extra QC plots, written to `grove_forest/qc/` alongside the postprocess ones:
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

![Pipeline overview: FragPipe/Casanovo search results feed PSMs into Oktoberfest feature generation, then PSA similarity grading, then isolation-forest rescoring](assets/full_pipeline.png)
