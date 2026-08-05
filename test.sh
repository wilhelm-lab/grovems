#!/bin/bash
# Quick local run of the pipeline against one proteometools raw file.
# Requires grovems already installed in your active environment (pip install -e . or
# poetry install).
set -euo pipefail

RAW_FILE="${1:-01625b_GA1-TUM_first_pool_1_01_01-3xHCD-1h-R1}"

FRAGPIPE_DIR=/cmnfs/proj/denovo_fdr/results/proteometools/results/search/fragpipe
CASANOVO_DIR=/cmnfs/proj/denovo_fdr/results/proteometools/results/search/casanovo
MZML_DIR=/cmnfs/proj/denovo_fdr/results/proteometools/raw/mzml

WORKDIR=/tmp/grovems_test
FIXTURE="${WORKDIR}/fixture"
rm -rf "${FIXTURE}"
mkdir -p "${FIXTURE}/fragpipe" "${FIXTURE}/casanovo" "${FIXTURE}/mzml"

fragpipe_src="$(find "${FRAGPIPE_DIR}" -maxdepth 3 -name "${RAW_FILE}*.pepXML" -exec dirname {} \; | sort -u | head -1)"
# File-level symlinks, not a directory one: Oktoberfest scans search_results with
# Path.rglob("*.pepXML"), which doesn't descend into a symlinked subdirectory -- a
# directory symlink here would make the pepXML files silently invisible to the scan.
find "${fragpipe_src}" -maxdepth 1 -type f -exec ln -s {} "${FIXTURE}/fragpipe/" \;
ln -s "${CASANOVO_DIR}/${RAW_FILE}".* "${FIXTURE}/casanovo/"
ln -s "${MZML_DIR}/${RAW_FILE}.mzML" "${FIXTURE}/mzml/"

cd "$(dirname "${BASH_SOURCE[0]}")"
grovems run --config assets/default_config.yaml \
    --set database_search_path="${FIXTURE}/fragpipe" \
    --set denovo_search_path="${FIXTURE}/casanovo" \
    --set rawdata_path="${FIXTURE}/mzml" \
    --set outdir="${WORKDIR}/output" \
    --set run_iforest=false
