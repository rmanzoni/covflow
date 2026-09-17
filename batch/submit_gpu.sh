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
shift || true

# Anything after the config used to be read and discarded without a word, so
#   ./batch/submit_gpu.sh <config> --time=12:00:00
# submitted happily with the default wall clock and truncated the training two
# hours in. Extra sbatch options go through COVFLOW_SBATCH_ARGS; say so rather
# than swallow them.
if [[ $# -gt 0 ]]; then
    echo "ERROR: unexpected argument(s) after the config file: $*"
    echo
    echo "sbatch options are not read from the command line here. Use:"
    echo "  COVFLOW_SBATCH_ARGS=\"$*\" ${BASH_SOURCE[0]} ${CONFIG}"
    echo "or set COVFLOW_TIME / COVFLOW_MEM in ${CONFIG}."
    exit 2
fi

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

# Wall clock and memory, overridable per config. The defaults live here rather
# than as #SBATCH directives in train_gpu.sh so that there is exactly one place
# to change them, and so the echo below can report the real values.
#
# For reference, on the T3 `gpu` partition: MaxTime=7-00:00:00,
# DefaultTime=1-00:00:00, and the `normal` QOS sets no MaxWall. Nothing caps you
# below a week -- but a larger request is scheduled later, so raise these
# against a measured Elapsed/MaxRSS rather than on principle.
COVFLOW_TIME="${COVFLOW_TIME:-5:00:00}"
COVFLOW_MEM="${COVFLOW_MEM:-64G}"

mkdir -p "${COVFLOW_OUT_BASE}"

SEED_FILE="${COVFLOW_OUT_BASE}/seed_list.txt"
printf '%s\n' "${COVFLOW_SEEDS[@]}" > "${SEED_FILE}"

# One run directory per array task, created HERE and not inside the job.
# Slurm resolves --output/--error at submission time and will not create a
# missing directory, so the task would die before printing anything.
#
# The directory is named after the array INDEX, not the seed: Slurm cannot
# substitute the seed into the path (it does not know it -- train_gpu.sh reads
# it from seed_list.txt at run time). A seed_<n> symlink is left alongside for
# navigation, and each run records its own seed in seed.txt.
for i in "${!COVFLOW_SEEDS[@]}"; do
    mkdir -p "${COVFLOW_OUT_BASE}/task_${i}"
    ln -sfn "task_${i}" "${COVFLOW_OUT_BASE}/seed_${COVFLOW_SEEDS[$i]}"
done

export COVFLOW_CONFIG="${CONFIG}"
export COVFLOW_SEED_FILE="${SEED_FILE}"

NSEEDS="${#COVFLOW_SEEDS[@]}"

# Extra sbatch options, e.g. COVFLOW_SBATCH_ARGS="--gres=gpu:2".
# These are appended LAST on the sbatch command line, so they override both the
# #SBATCH directives inside train_gpu.sh and the --time/--mem set above. For
# wall clock and memory alone, prefer COVFLOW_TIME / COVFLOW_MEM.
SBATCH_EXTRA=()
if [[ -n "${COVFLOW_SBATCH_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    SBATCH_EXTRA=(${COVFLOW_SBATCH_ARGS})
fi

echo "============================================================"
echo "Submitting CovFlow GPU training"
echo "============================================================"
echo "config : ${CONFIG}"
echo "repo   : ${COVFLOW_REPO}"
echo "data   : ${COVFLOW_DATA}"
echo "mc     : ${COVFLOW_MC}"
echo "output : ${COVFLOW_OUT_BASE}"
echo "seeds  : ${COVFLOW_SEEDS[*]}"
echo "time   : ${COVFLOW_TIME}"
echo "mem    : ${COVFLOW_MEM}"
echo "sbatch : ${COVFLOW_SBATCH_ARGS:-<none>}"
echo "logs   : ${COVFLOW_OUT_BASE}/task_<n>/covflow-<jobid>_<n>.{out,err}"
echo "============================================================"

JOB_ID=$(
    sbatch \
        --parsable \
        --export=ALL \
        --mem="${COVFLOW_MEM}" \
        --array="0-$((NSEEDS - 1))" \
        --time="${COVFLOW_TIME}" \
        --output="${COVFLOW_OUT_BASE}/task_%a/covflow-%A_%a.out" \
        --error="${COVFLOW_OUT_BASE}/task_%a/covflow-%A_%a.err" \
        ${SBATCH_EXTRA[@]+"${SBATCH_EXTRA[@]}"} \
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
