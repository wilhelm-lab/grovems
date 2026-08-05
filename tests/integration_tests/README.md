# Integration tests

No fixtures are bundled here -- the pipeline's inputs (FragPipe/MSFragger search
results, Casanovo results, mzML spectra) are large, project-specific data that lives on
shared project storage, not something to commit into this repo.

## Quickest: one real raw file, fully automated

[`../../test.sh`](../../test.sh) builds a minimal single-raw-file fixture by symlinking
the matching FragPipe/Casanovo/mzML entries straight out of the production
proteometools dataset (nothing is copied), then runs `grovems run` against it locally
with IForest disabled:

```bash
./test.sh                                                   # uses a known-good default raw file
./test.sh 01625b_GA1-TUM_first_pool_1_01_01-3xHCD-1h-R1      # or name one explicitly
```

IForest is skipped: SUOD needs a reasonably large shared-PSM training set to fit at all
(at least 2 base estimators, each needing >= 256 rows -- see `build_suod_model` in
`src/grovems/iforest/iforest.py`), and a single raw file's shared PSMs will usually be
too few. To try it anyway, or to run against a larger real input set, invoke `grovems
run` directly against a small real `database_search_path`/`denovo_search_path`/
`rawdata_path` (still small enough to run quickly, unlike a full production dataset).
