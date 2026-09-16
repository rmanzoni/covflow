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
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#
# cpus-per-task, mem and time above are deliberately small. A day-long, 32 GB,
# 5-core reservation cannot be backfilled into the gaps between other jobs, so
# on a busy partition the queue wait dominates the run. Raise them when a job
# actually hits a limit, not before, and do it from the config rather than by
# editing this file:
#
#   COVFLOW_SBATCH_ARGS="--time=12:00:00 --mem=32G"
#
# sbatch command-line options override the #SBATCH directives above.

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

# Where is Slurm actually writing this? Only submit_gpu.sh points it into the
# run directory; a bare `sbatch batch/train_gpu.sh` leaves it in the submission
# cwd, because #SBATCH directives cannot expand shell variables.
if command -v scontrol >/dev/null 2>&1; then
    STDOUT_PATH="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
                   | tr ' ' '\n' | sed -n 's/^StdOut=//p' | head -1)"
    if [[ -n "${STDOUT_PATH}" ]]; then
        echo
        echo "slurm stdout : ${STDOUT_PATH}"
        if [[ "$(dirname "${STDOUT_PATH}")" != "${OUT}" ]]; then
            echo ">>>> NOTE: that is NOT inside ${OUT}."
            echo ">>>>       Submit through batch/submit_gpu.sh to keep .out/.err"
            echo ">>>>       next to the pdfs. The python printout is in"
            echo ">>>>       ${OUT}/train.log either way."
        fi
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
