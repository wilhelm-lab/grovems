#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate grovems

PATH=$PATH:/cmnfs/data/cluster/software/percolator/3.7.1/usr/bin

#=== Configuration ===#
TEST_PATH="/cmnfs/proj/denovo_fdr/tools/grovems/tests/integration_tests"
CONFIG="${TEST_PATH}/config.yaml"
DATABASE_SEARCH_PATH="${TEST_PATH}/data"
DENOVO_SEARCH_PATH="${TEST_PATH}/data"
RAWDATA_PATH="${TEST_PATH}/data"
OUTDIR="${TEST_PATH}/out"

grovems run --config "${CONFIG}" \
    --set database_search_path="${DATABASE_SEARCH_PATH}" \
    --set denovo_search_path="${DENOVO_SEARCH_PATH}" \
    --set rawdata_path="${RAWDATA_PATH}" \
    --set outdir="${OUTDIR}" \
    --set num_threads="10" \
    --set psa_max_workers="10" \
    --set percolator_threads="3" 