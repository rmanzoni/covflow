#!/bin/bash
#
# CovFlow GPU training on PSI Tier-3
#
#SBATCH --job-name=covflow
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#
# There is deliberately no --time and no --mem here.
#
# submit_gpu.sh always passes both on the sbatch command line, from
# COVFLOW_TIME / COVFLOW_MEM in the config. A directive here would be dead on
# that path and live on the direct `sbatch batch/train_gpu.sh` path -- which is
# exactly how a stale --time=02:00:00 truncated a training that had asked for
# more, silently, because sbatch reports no conflict. With the lines gone, a
# direct submission inherits the partition defaults (DefaultTime=1-00:00:00 on
# `gpu`), which is a safe thing to inherit.
#
# cpus-per-task stays small on purpose: a wide reservation cannot be backfilled
# into the gaps between other jobs, so on a busy partition the queue wait
# dominates the run.

set -euo pipefail

# ------------------------------------------------------------
# Locate repository and configuration
# ------------------------------------------------------------

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
: "${COVFLOW_CONDA_ENV:?COVFLOW_CONDA_ENV is not set in the config}"
: "${COVFLOW_DATA:?COVFLOW_DATA is not set in the config}"
: "${COVFLOW_MC:?COVFLOW_MC is not set in the config}"
: "${COVFLOW_OUT_BASE:?COVFLOW_OUT_BASE is not set in the config}"

# ------------------------------------------------------------
# Job information
# ------------------------------------------------------------

echo "============================================================"
echo "CovFlow GPU training"
echo "============================================================"
echo "date       : $(date)"
echo "host       : ${HOSTNAME}"
echo "job        : ${SLURM_JOB_ID}"
echo "array task : ${SLURM_ARRAY_TASK_ID:-none}"
echo "config     : ${CONFIG}"
echo "CUDA       : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

# ------------------------------------------------------------
# Local scratch
# ------------------------------------------------------------

JOB_SCRATCH="/scratch/${USER}/${SLURM_JOB_ID}"
mkdir -p "${JOB_SCRATCH}"
export TMPDIR="${JOB_SCRATCH}"

# ------------------------------------------------------------
# Conda
# ------------------------------------------------------------

if [[ -n "${COVFLOW_CONDA_BASE:-}" ]]; then
    CONDA_BASE="${COVFLOW_CONDA_BASE}"
elif [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="${HOME}/miniconda3"
elif [[ -f "${HOME}/mambaforge/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="${HOME}/mambaforge"
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="${HOME}/miniforge3"
elif [[ -f "/work/${USER}/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="/work/${USER}/miniconda3"
elif [[ -f "/work/${USER}/miniforge3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="/work/${USER}/miniforge3"
else
    echo "ERROR: Cannot find conda.sh."
    echo "Set COVFLOW_CONDA_BASE in the configuration file."
    exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${COVFLOW_CONDA_ENV}"

echo
echo "Conda:"
echo "  base   : ${CONDA_BASE}"
echo "  env    : ${COVFLOW_CONDA_ENV}"
echo "  python : $(which python)"
python --version

# ------------------------------------------------------------
# CUDA sanity check
# ------------------------------------------------------------

echo
echo "PyTorch/CUDA:"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available; refusing to run on CPU.")

print("GPU:", torch.cuda.get_device_name(0))
print("GPU capability:", torch.cuda.get_device_capability(0))

# Actually execute a small GPU operation. This catches cases where
# CUDA is visible but the installed PyTorch has no suitable kernel.
x = torch.randn(256, 256, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("GPU computation: OK")
PY

# ------------------------------------------------------------
# Repository
# ------------------------------------------------------------

cd "${COVFLOW_REPO}"

echo
echo "Repository:"
echo "  path : $(pwd)"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "  git  : $(git rev-parse --short HEAD)"
fi

# ------------------------------------------------------------
# Seed handling
# ------------------------------------------------------------

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [[ -n "${COVFLOW_SEED_FILE:-}" && -f "${COVFLOW_SEED_FILE}" ]]; then
    SEED="$(sed -n "$((TASK_ID + 1))p" "${COVFLOW_SEED_FILE}")"
    : "${SEED:?Could not read seed for array task ${TASK_ID}}"
else
    SEED="${COVFLOW_SEED:-0}"
fi

# Named after the array index, so that submit_gpu.sh can point Slurm's
# --output/--error at it at submission time. See the comment there.
OUT="${COVFLOW_OUT_BASE}/task_${TASK_ID}"
mkdir -p "${OUT}"

# Make the run directory self-contained: the seed that produced it, the seed
# list it was drawn from, and the exact config that was sourced. Without the
# config copy, a later edit to configs/*.conf silently rewrites the history of
# every run that used it.
echo "${SEED}" > "${OUT}/seed.txt"
if [[ -n "${COVFLOW_SEED_FILE:-}" && -f "${COVFLOW_SEED_FILE}" ]]; then
    cp -f "${COVFLOW_SEED_FILE}" "${OUT}/seed_list.txt"
fi
cp -f "${CONFIG}" "${OUT}/$(basename "${CONFIG}")"

# ------------------------------------------------------------
# What did Slurm actually grant?
# ------------------------------------------------------------
#
# One scontrol call, three answers: where the .out is going, how much memory
# this job may use, and how long it may run.
#
# This exists because none of the three is visible from inside the job
# otherwise, and getting any of them wrong fails EXPENSIVELY and LATE: a
# too-small --mem dies in the OOM killer partway through the read, a too-small
# --time dies hours in with nothing written. Both have now happened. Finding
# out in the first second costs one scontrol call.
#
# Slurm reports the memory request as MinMemoryNode (from --mem) or
# MinMemoryCPU (from --mem-per-cpu), never both, and 0 for "no request at all".

JOBINFO=""
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    JOBINFO="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | tr ' ' '\n')"
fi

_jobfield() { sed -n "s/^$1=//p" <<< "${JOBINFO}" | head -1; }

# Slurm memory strings: 64G, 8192M, 0. A bare number means MB.
_mem_mb() {
    local v="$1" num unit
    [[ -z "${v}" || "${v}" == "0" ]] && { echo ""; return; }
    unit="${v: -1}"
    num="${v%[A-Za-z]}"
    num="${num%.*}"
    case "${unit}" in
        K|k) echo $(( num / 1024 )) ;;
        M|m) echo "${num}" ;;
        G|g) echo $(( num * 1024 )) ;;
        T|t) echo $(( num * 1024 * 1024 )) ;;
        *)   echo "${v%.*}" ;;
    esac
}

STDOUT_PATH="$(_jobfield StdOut)"
TIME_LIMIT="$(_jobfield TimeLimit)"
MEM_NODE="$(_mem_mb "$(_jobfield MinMemoryNode)")"
MEM_CPU="$(_mem_mb "$(_jobfield MinMemoryCPU)")"
NUM_CPUS="$(_jobfield NumCPUs)"

MEM_MB=""
MEM_SRC=""
if [[ -n "${MEM_NODE}" ]]; then
    MEM_MB="${MEM_NODE}"
    MEM_SRC="--mem"
elif [[ -n "${MEM_CPU}" ]]; then
    MEM_MB=$(( MEM_CPU * ${NUM_CPUS:-1} ))
    MEM_SRC="--mem-per-cpu x ${NUM_CPUS:-1} cpus"
fi

echo
echo "Slurm grant:"
echo "  stdout : ${STDOUT_PATH:-<unknown>}"
echo "  time   : ${TIME_LIMIT:-<unknown>}"
if [[ -n "${MEM_MB}" ]]; then
    echo "  memory : ${MEM_MB} MB  (${MEM_SRC})"
else
    echo "  memory : NO EXPLICIT REQUEST"
fi

if [[ -n "${STDOUT_PATH}" && "$(dirname "${STDOUT_PATH}")" != "${OUT}" ]]; then
    echo
    echo ">>>> NOTE: the .out/.err are NOT inside ${OUT}."
    echo ">>>>       Submit through batch/submit_gpu.sh to keep them next to"
    echo ">>>>       the pdfs. The python printout is in ${OUT}/train.log"
    echo ">>>>       either way."
fi

# ------------------------------------------------------------
# Refuse to start on a memory grant that cannot work
# ------------------------------------------------------------
#
# COVFLOW_MIN_MEM is a floor in MB, not a request. Set it to 0 to disable the
# check -- but prefer fixing the submission, since the failure it prevents is
# an OOM kill with a truncated log and no artefacts.
#
# "No explicit request" is treated as a failure rather than as unlimited even
# where the partition says DefMemPerNode=UNLIMITED: what the partition defaults
# to and what the cgroup on the node will actually allow are different
# questions, and a job that never asked has no answer to the second one.

COVFLOW_MIN_MEM="${COVFLOW_MIN_MEM:-16384}"

if [[ "${COVFLOW_MIN_MEM}" != "0" ]]; then
    if [[ -z "${MEM_MB}" ]]; then
        echo
        echo "ERROR: this job made no memory request."
        echo
        echo "  A bare 'sbatch batch/train_gpu.sh' requests no memory: the"
        echo "  --mem here comes from submit_gpu.sh, which is also what creates"
        echo "  the array and the per-task output directory. Job 213068 failed"
        echo "  exactly this way, in the OOM killer, 11 minutes in."
        echo
        echo "  Submit with:"
        echo "    ./batch/submit_gpu.sh ${CONFIG}"
        echo "  or, to keep the direct path:"
        echo "    sbatch --mem=64G --time=8:00:00 batch/train_gpu.sh ${CONFIG}"
        echo
        echo "  Set COVFLOW_MIN_MEM=0 to bypass this check."
        exit 3
    fi
    if (( MEM_MB < COVFLOW_MIN_MEM )); then
        echo
        echo "ERROR: memory grant of ${MEM_MB} MB is below COVFLOW_MIN_MEM"
        echo "       (${COVFLOW_MIN_MEM} MB)."
        echo
        echo "  Raise it in the config (COVFLOW_MEM) or lower the floor"
        echo "  (COVFLOW_MIN_MEM) if you know this run is small. A full-sample"
        echo "  load has been measured at ~20 GB resident."
        exit 3
    fi
fi

# ------------------------------------------------------------
# Build command
# ------------------------------------------------------------

CMD=(
    python train_covflow.py
    --data "${COVFLOW_DATA}"
    --mc "${COVFLOW_MC}"
    --tree "${COVFLOW_TREE}"
    --cov-prefix "${COVFLOW_COV_PREFIX}"
    --context ${COVFLOW_CONTEXT}
    --log-pt "${COVFLOW_LOG_PT}"
    --epochs "${COVFLOW_EPOCHS}"
    --batch-size "${COVFLOW_BATCH_SIZE}"
    --lr "${COVFLOW_LR}"
    --transforms "${COVFLOW_TRANSFORMS}"
    --hidden ${COVFLOW_HIDDEN}
    --bins "${COVFLOW_BINS}"
    --seed "${SEED}"
    --device cuda
    --out "${OUT}"
)

if [[ -n "${COVFLOW_DATA_WEIGHT:-}" ]]; then
    CMD+=(--data-weight-branch "${COVFLOW_DATA_WEIGHT}")
fi

if [[ -n "${COVFLOW_MC_WEIGHT:-}" ]]; then
    CMD+=(--mc-weight-branch "${COVFLOW_MC_WEIGHT}")
fi

# Additional train_covflow.py options from the config.
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo
echo "============================================================"
echo "Training configuration"
echo "============================================================"
echo "config     : ${CONFIG}"
echo "data       : ${COVFLOW_DATA}"
echo "mc         : ${COVFLOW_MC}"
echo "tree       : ${COVFLOW_TREE}"
echo "cov-prefix : ${COVFLOW_COV_PREFIX}"
echo "context    : ${COVFLOW_CONTEXT}"
echo "log-pt     : ${COVFLOW_LOG_PT}"
echo "epochs     : ${COVFLOW_EPOCHS}"
echo "batch size : ${COVFLOW_BATCH_SIZE}"
echo "lr         : ${COVFLOW_LR}"
echo "transforms : ${COVFLOW_TRANSFORMS}"
echo "hidden     : ${COVFLOW_HIDDEN}"
echo "bins       : ${COVFLOW_BINS}"
echo "seed       : ${SEED}"
echo "output     : ${OUT}"
echo "============================================================"

echo
echo "Command:"
printf ' %q' "${CMD[@]}"
echo
echo

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

time "${CMD[@]}"

echo
echo "============================================================"
echo "Training finished successfully"
echo "output : ${OUT}"
echo "date   : $(date)"
echo "============================================================"

rm -rf "${JOB_SCRATCH}"
