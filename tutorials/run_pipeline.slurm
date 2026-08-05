#!/bin/bash

#SBATCH -J grovems
#SBATCH --cpus-per-task=40
#SBATCH --mem=300G
#SBATCH --time=4-00:00:00
#SBATCH --partition=shared-cpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# grovems runs as a single sequential process (no per-step job submission like the old
# Nextflow pipeline) -- the #SBATCH lines above are the pipeline's actual resource
# request, not just a small driver job. Size cpus/mem/time to your data volume, and
# match num_threads/psa_max_workers/percolator_threads in your config YAML to
# --cpus-per-task.

#=== Configuration -- edit these for your run ===#
PIPELINE_DIR="/cmnfs/proj/denovo_fdr/tools/grovems"
CONFIG="${PIPELINE_DIR}/assets/default_config.yaml"

DATABASE_SEARCH_PATH="/cmnfs/proj/denovo_fdr/results/proteometools/results/search/fragpipe"
DENOVO_SEARCH_PATH="/cmnfs/proj/denovo_fdr/results/proteometools/results/search/casanovo"
RAWDATA_PATH="/cmnfs/proj/denovo_fdr/results/proteometools/raw/mzml"
OUTDIR="/cmnfs/proj/denovo_fdr/results/proteometools/results/run.$(date +%d%m%Y).grovems"

#=== Environment -- edit to match how you installed grovems (see docs/notes/installation.md) ===#
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grovems

#=== START ===#
echo "=== Starting grovems at $(date) ==="

grovems run --config "${CONFIG}" \
    --set database_search_path="${DATABASE_SEARCH_PATH}" \
    --set denovo_search_path="${DENOVO_SEARCH_PATH}" \
    --set rawdata_path="${RAWDATA_PATH}" \
    --set outdir="${OUTDIR}" \
    --set num_threads="${SLURM_CPUS_PER_TASK}" \
    --set psa_max_workers="${SLURM_CPUS_PER_TASK}" \
    --set percolator_threads="${SLURM_CPUS_PER_TASK}"

echo "=== grovems finished at $(date) ==="
