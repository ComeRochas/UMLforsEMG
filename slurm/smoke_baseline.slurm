#!/bin/bash
# -----------------------------------------------------------------------------
# smoke_baseline.slurm
#
# End-to-end 1-epoch smoke run of the BASELINE pipeline on L40S.  Validates
# that the EMG cache loads, forward/backward complete, val WER is computed,
# and a checkpoint is written — before burning H100 queue priority.
#
# Requires slurm/precompute_caches.slurm to have completed first (EMG cache).
# -----------------------------------------------------------------------------
#SBATCH --job-name=ssl_smoke_bl
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/cr4206/logs/%j.out
#SBATCH --error=/scratch/cr4206/logs/%j.err

set -euo pipefail

JOB_START_TS=$(date +%s)
echo "[job-start] ts=$(date '+%F %T %Z') job_id=${SLURM_JOB_ID:-na} host=$(hostname)"
echo "[job-meta] partition=${SLURM_JOB_PARTITION:-na} nodelist=${SLURM_JOB_NODELIST:-na} cpus=${SLURM_CPUS_PER_TASK:-na}"

heartbeat() {
    while true; do
        echo "[heartbeat] ts=$(date '+%F %T %Z') job_id=${SLURM_JOB_ID:-na} still-running"
        sleep 120
    done
}
heartbeat &
HEARTBEAT_PID=$!

cleanup() {
    rc=$?
    kill "$HEARTBEAT_PID" 2>/dev/null || true
    dur=$(( $(date +%s) - JOB_START_TS ))
    echo "[job-end] ts=$(date '+%F %T %Z') job_id=${SLURM_JOB_ID:-na} exit_code=${rc} duration_s=${dur}"
    exit "$rc"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
export SCRATCH=/scratch/cr4206

mkdir -p ${SCRATCH}/logs \
         ${SCRATCH}/checkpoints/smoke_baseline \
         ${SCRATCH}/.cache/huggingface \
         ${SCRATCH}/.cache/torch \
         ${SCRATCH}/.cache/wandb \
         ${SCRATCH}/.cache/pip

export HF_HOME=${SCRATCH}/.cache/huggingface
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/huggingface
export TORCH_HOME=${SCRATCH}/.cache/torch
export WANDB_DIR=${SCRATCH}/.cache/wandb
export PIP_CACHE_DIR=${SCRATCH}/.cache/pip
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
for p in \
    ${SCRATCH}/data/emg_cache/train.pt \
    ${SCRATCH}/data/emg_cache/dev.pt ; do
    if [ ! -e "$p" ]; then
        echo "[preflight] MISSING: $p"
        echo "[preflight] run slurm/precompute_caches.slurm first."
        exit 1
    fi
done
echo "[preflight] EMG caches present."

# ---------------------------------------------------------------------------
# Warm page cache
# ---------------------------------------------------------------------------
echo "[warm] touching caches to warm page cache"
cat ${SCRATCH}/data/emg_cache/train.pt > /dev/null &
cat ${SCRATCH}/data/emg_cache/dev.pt   > /dev/null &
wait
echo "[warm] done"

# ---------------------------------------------------------------------------
# Conda
# ---------------------------------------------------------------------------
module purge
module load anaconda3/2025.06
CONDA_ENV_DIR=${SCRATCH}/envs/silent_speech_uml
echo "[env] python=$(conda run -p ${CONDA_ENV_DIR} python -V 2>&1)"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd $HOME/silent_speech_uml
echo "[run] launching baseline smoke run (1 epoch)"
conda run --no-capture-output -p ${CONDA_ENV_DIR} python -u \
    src/train_baseline.py --config configs/baseline_smoke.yaml
