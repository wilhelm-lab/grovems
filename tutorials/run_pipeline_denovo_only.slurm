#!/bin/bash

#SBATCH -J grovems-denovo
#SBATCH --cpus-per-task=40
#SBATCH --mem=300G
#SBATCH --time=4-00:00:00
#SBATCH --partition=shared-cpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

#=== Configuration ===#
PIPELINE_DIR="/cmnfs/proj/denovo_fdr/tools/grovems"
CONFIG="${PIPELINE_DIR}/assets/default_config.yaml"

DENOVO_SEARCH_PATH="/cmnfs/proj/denovo_fdr/results/proteometools/results/search/casanovo"
RAWDATA_PATH="/cmnfs/proj/denovo_fdr/results/proteometools/raw/mzml"
OUTDIR="/cmnfs/proj/denovo_fdr/results/proteometools/results/run.$(date +%d%m%Y).grovems.${SLURM_JOB_ID}"

#=== Environment -- edit to match how you installed grovems (see docs/notes/installation.rst) ===#
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grovems

# conda's own activate/deactivate hooks (e.g. deactivate-gxx_linux-64.sh, which backs up
# CC/CXX env vars) aren't written to be `set -u` safe, so strict mode is only enabled
# after conda activation completes, not before.
set -euo pipefail

#=== START ===#
echo "=== Starting grovems (denovo-only, no database) at $(date) ==="
#NOTE: "--set" overrides config parameter and should be "key=value"
# denovo_only=true skips the database branch entirely (no Oktoberfest/Percolator on the
# database side, no PSA) and requires iforest_training_source=denovo_score, since there's
# no database-side Percolator score to build a percolator_percentile training set from.
grovems run --config "${CONFIG}" \
    --set denovo_only=true \
    --set denovo_search_path="${DENOVO_SEARCH_PATH}" \
    --set rawdata_path="${RAWDATA_PATH}" \
    --set outdir="${OUTDIR}" \
    --set iforest_training_source=denovo_score \
    --set iforest_denovo_score_threshold=0.9 \
    --set num_threads="${SLURM_CPUS_PER_TASK}" \
    --set psa_max_workers="${SLURM_CPUS_PER_TASK}" \
    --set percolator_threads="${SLURM_CPUS_PER_TASK}"

echo "=== grovems finished at $(date) ==="
