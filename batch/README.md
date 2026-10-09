# covflow PSI Tier-3 GPU batch files — Conda version

These scripts assume that the Python environment is a Conda environment
called `covflow`.

## Files

- `batch/train_gpu.sh` — Slurm job script for one training.
- `batch/submit_gpu.sh` — submit several trainings as a Slurm job array.
- `batch/gpu_test.sh` — short CUDA smoke test.

## Repository

Recommended location:

    /work/$USER/covflow

The scripts default to this location, but it can be changed with:

    export COVFLOW_REPO=/path/to/covflow

## Conda

The scripts automatically try to find a standard Conda installation in:

    $HOME/miniconda3
    $HOME/mambaforge
    $HOME/miniforge3
    /work/$USER/miniconda3
    /work/$USER/miniforge3

The environment defaults to:

    covflow

To use another environment:

    export COVFLOW_CONDA_ENV=myenv

If Conda is installed somewhere else, set:

    export COVFLOW_CONDA_BASE=/path/to/conda

and add the following line near the beginning of the scripts if needed:

    source "${COVFLOW_CONDA_BASE}/etc/profile.d/conda.sh"

## GPU test

First test the environment and GPU:

    sbatch batch/gpu_test.sh

This uses the short `qgpu` partition.

## Training

Set the input ROOT files:

    export COVFLOW_DATA='/pnfs/.../data*.root'
    export COVFLOW_MC='/pnfs/.../mc*.root'

Set output:

    export COVFLOW_OUT_BASE='/work/'$USER'/covflow-runs/test'

Run one seed:

    export COVFLOW_SEEDS='0'
    ./batch/submit_gpu.sh

Run several independent trainings:

    export COVFLOW_SEEDS='0 1 2 3 4 5 6 7'
    ./batch/submit_gpu.sh

Each seed becomes one Slurm array task and requests one GPU.

## Training parameters

These can be overridden before submission:

    export COVFLOW_EPOCHS=200
    export COVFLOW_BATCH_SIZE=4096
    export COVFLOW_LR=1e-3
    export COVFLOW_TRANSFORMS=6
    export COVFLOW_HIDDEN='256 256'
    export COVFLOW_BINS=16

Then:

    ./batch/submit_gpu.sh

The training script explicitly uses:

    --device cuda

and aborts if PyTorch cannot see the GPU, preventing an accidental CPU run.

## Run 3: one training per epoch and per muon

`configs/run3_epochs.py` holds the MC <-> data map (one epoch = the data eras
matched to one MC campaign, as in the PPD summary tables; 2024, 2025 and 2026
are separate epochs that share Summer24 MC) and the training settings shared by
every job. `batch/submit_run3_epochs.py` turns it into one Slurm array per
(epoch, muon).

On the UI, with the `covflow` env active and a proxy on a shared filesystem:

    voms-proxy-init --voms cms --valid 192:00 --out $HOME/.x509up_u$(id -u)
    export X509_USER_PROXY=$HOME/.x509up_u$(id -u)

    python batch/submit_run3_epochs.py --dry-run      # checks, configs, no sbatch
    python batch/submit_run3_epochs.py                # all 7 epochs x mu1, mu2
    python batch/submit_run3_epochs.py --only 2024 --muons mu2

Before anything is submitted it checks that every data file in the input
directory belongs to exactly one epoch, that every file has every branch the
job will read, and that each MC pileup-weight branch is filled for the year it
serves. Each job gets a self-contained `job.conf` in
`OUT_ROOT/TAG/<epoch>/<mu>/`, copied again into its `task_<n>` directory, and
the submissions are recorded in `OUT_ROOT/TAG/submitted.json`.

The worker nodes do not mount /pnfs. With `COVFLOW_STAGE=1` (set in every
generated config) `train_gpu.sh` copies the inputs to `/scratch` with xrdcp
through `COVFLOW_SE_HOST`, using a private copy of `X509_USER_PROXY`, after
checking the free space against `COVFLOW_STAGE_BYTES`. The scratch directory is
removed when the job exits, successfully or not.

`COVFLOW_DATA` / `COVFLOW_MC` can now be bash arrays of paths as well as a
single path or glob, so the hand-written configs keep working unchanged.
