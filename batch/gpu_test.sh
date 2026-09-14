#!/bin/bash
#SBATCH --job-name=covflow_test
#SBATCH --account=gpu_gres
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=covflow_test-%j.out
#SBATCH --error=covflow_test-%j.err

set -euo pipefail

CONDA_ENV="${COVFLOW_CONDA_ENV:-covflow}"

if [[ -n "${CONDA_EXE:-}" ]]; then
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
    echo "Set COVFLOW_CONDA_BASE to your Conda installation."
    exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "HOST: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "CONDA_ENV: ${CONDA_ENV}"
echo "PYTHON: $(which python)"

nvidia-smi

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)

assert torch.cuda.is_available(), "CUDA is not available"

x = torch.randn(5000, 5000, device="cuda")
y = x @ x
torch.cuda.synchronize()

print("GPU computation OK")
print("GPU:", torch.cuda.get_device_name(0))
PY
