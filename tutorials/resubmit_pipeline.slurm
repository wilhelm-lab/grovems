#!/bin/bash

#SBATCH -J grovems
#SBATCH --cpus-per-task=40
#SBATCH --mem=300G
#SBATCH --time=4-00:00:00
#SBATCH --partition=shared-cpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

#=== Resubmit an existing grovems run, unchanged ===#
# Usage: sbatch resubmit_pipeline.slurm /path/to/existing/outdir
#
# Every grovems run writes its fully-resolved config to <outdir>/config.yaml (input
# file + any --set overrides applied, including outdir itself -- see README). This
# reruns that exact config, so already-finished work is skipped rather than redone
# (Oktoberfest's per-file .done markers, PSA/IForest's overwrite_outputs: false)
# instead of requiring the original --set flags to be reconstructed by hand.
if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch resubmit_pipeline.slurm /path/to/existing/outdir" >&2
    exit 1
fi

OUTDIR="$1"
CONFIG="${OUTDIR}/config.yaml"

if [[ ! -f "${CONFIG}" ]]; then
    echo "No config.yaml found at ${CONFIG} -- is this a grovems output directory?" >&2
    exit 1
fi

#=== Environment -- edit to match how you installed grovems (see docs/notes/installation.rst) ===#
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grovems

# conda's own activate/deactivate hooks (e.g. deactivate-gxx_linux-64.sh, which backs up
# CC/CXX env vars) aren't written to be `set -u` safe, so strict mode is only enabled
# after conda activation completes, not before.
set -euo pipefail

#=== START ===#
echo "=== Resubmitting grovems at $(date), reusing ${CONFIG} ==="
#NOTE: thread counts are re-pinned to this job's own SLURM allocation, in case it
# differs from the original submission's; every other value comes from CONFIG as-is.
grovems run --config "${CONFIG}" \
    --set num_threads="${SLURM_CPUS_PER_TASK}" \
    --set psa_max_workers="${SLURM_CPUS_PER_TASK}" \
    --set percolator_threads="${SLURM_CPUS_PER_TASK}"

echo "=== grovems finished at $(date) ==="
