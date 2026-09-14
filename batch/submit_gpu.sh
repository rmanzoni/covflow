#!/bin/bash
#
# Submit CovFlow GPU training as a Slurm array.
#
# Usage:
#   ./batch/submit_gpu.sh
#   ./batch/submit_gpu.sh configs/train_gpu.conf
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/../configs/train_gpu.conf"
CONFIG="${1:-${COVFLOW_CONFIG:-${DEFAULT_CONFIG}}}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: configuration file not found:"
    echo "  ${CONFIG}"
    exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG}"

: "${COVFLOW_REPO:?COVFLOW_REPO is not set in the config}"
: "${COVFLOW_DATA:?COVFLOW_DATA is not set in the config}"
: "${COVFLOW_MC:?COVFLOW_MC is not set in the config}"
: "${COVFLOW_OUT_BASE:?COVFLOW_OUT_BASE is not set in the config}"

# One or more seeds can be specified here without touching train_gpu.sh.
#
# Example:
#   COVFLOW_SEEDS=(0 1 2 3 4)
#
COVFLOW_SEEDS=("${COVFLOW_SEEDS[@]:-0}")

mkdir -p "${COVFLOW_OUT_BASE}"

SEED_FILE="${COVFLOW_OUT_BASE}/seed_list.txt"
printf '%s\n' "${COVFLOW_SEEDS[@]}" > "${SEED_FILE}"

export COVFLOW_CONFIG="${CONFIG}"
export COVFLOW_SEED_FILE="${SEED_FILE}"

NSEEDS="${#COVFLOW_SEEDS[@]}"

echo "============================================================"
echo "Submitting CovFlow GPU training"
echo "============================================================"
echo "config : ${CONFIG}"
echo "repo   : ${COVFLOW_REPO}"
echo "data   : ${COVFLOW_DATA}"
echo "mc     : ${COVFLOW_MC}"
echo "output : ${COVFLOW_OUT_BASE}"
echo "seeds  : ${COVFLOW_SEEDS[*]}"
echo "============================================================"

JOB_ID=$(
    sbatch \
        --parsable \
        --export=ALL \
        --array="0-$((NSEEDS - 1))" \
        "${SCRIPT_DIR}/train_gpu.sh" \
        "${CONFIG}"
)

echo
echo "Submitted Slurm array: ${JOB_ID}"
echo
echo "Monitor:"
echo "  squeue -j ${JOB_ID}"
echo
echo "Accounting:"
echo "  sacct -j ${JOB_ID} --format='JobID,State,Elapsed,MaxRSS,ReqMem,NodeList'"
