#!/bin/bash
# -----------------------------------------------------------------------------
# precompute_caches.slurm
#
# One-shot job that prepares two caches so training jobs (esp. H100) never
# pay these costs at runtime:
#
#   1. EMG .pt cache       — CPU-only DSP over Gaddy dataset (~8 min / split)
#   2. LibriSpeech features — frozen wav2vec2-base forward pass, stored as
#                             one .pt per utterance (~45 min for train-clean-100)
#
# Runs on L40S because (a) queue time is shortest, (b) wav2vec2-base forward
# is tiny — H100/H200 would be wasted queue priority, and (c) the EMG phase
# uses the CPU cores allocated here, not the GPU.
# -----------------------------------------------------------------------------
#SBATCH --job-name=ssl_precompute
#SBATCH --account=torch_pr_39_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
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
export EMG_DATA_DIR=${SCRATCH}/data/emg_data/emg_data
export EMG_CACHE_DIR=${SCRATCH}/data/emg_cache
export LIBRISPEECH_DIR=${SCRATCH}/data/librispeech
export LIBRISPEECH_FEATURES_DIR=${SCRATCH}/data/librispeech_features

mkdir -p ${SCRATCH}/logs \
         ${EMG_CACHE_DIR} \
         ${LIBRISPEECH_FEATURES_DIR} \
         ${SCRATCH}/.cache/huggingface \
         ${SCRATCH}/.cache/torch \
         ${SCRATCH}/.cache/pip

# ---------------------------------------------------------------------------
# Redirect caches away from $HOME
# ---------------------------------------------------------------------------
export HF_HOME=${SCRATCH}/.cache/huggingface
export TRANSFORMERS_CACHE=${SCRATCH}/.cache/huggingface
export TORCH_HOME=${SCRATCH}/.cache/torch
export PIP_CACHE_DIR=${SCRATCH}/.cache/pip
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Conda
# ---------------------------------------------------------------------------
module purge
module load anaconda3/2025.06

CONDA_ENV_DIR=${SCRATCH}/envs/silent_speech_uml
if [ ! -d "${CONDA_ENV_DIR}" ]; then
    echo "[setup] Creating conda env at ${CONDA_ENV_DIR}"
    conda env create --prefix ${CONDA_ENV_DIR} \
        --file $HOME/silent_speech_uml/environment.yml
fi
echo "[env] python=$(conda run -p ${CONDA_ENV_DIR} python -V 2>&1)"

cd $HOME/silent_speech_uml

# ---------------------------------------------------------------------------
# Phase 1: EMG cache (CPU only — GPU sits idle, that's fine)
# ---------------------------------------------------------------------------
echo "[phase1] ===== building EMG cache ====="
conda run --no-capture-output -p ${CONDA_ENV_DIR} python -u \
    src/precompute_emg_cache.py \
    --emg_data_dir  ${EMG_DATA_DIR} \
    --emg_cache_dir ${EMG_CACHE_DIR} \
    --splits train dev \
    --num_workers 0

# ---------------------------------------------------------------------------
# Phase 2: LibriSpeech wav2vec2 features (uses the L40S)
# ---------------------------------------------------------------------------
echo "[phase2] ===== building LibriSpeech features ====="
conda run --no-capture-output -p ${CONDA_ENV_DIR} python -u \
    src/precompute_librispeech_features.py \
    --librispeech_dir ${LIBRISPEECH_DIR} \
    --features_dir    ${LIBRISPEECH_FEATURES_DIR} \
    --splits train-clean-100 \
    --batch_size 16

echo "[done] caches ready:"
echo "  EMG:          ${EMG_CACHE_DIR}"
echo "  LibriSpeech:  ${LIBRISPEECH_FEATURES_DIR}"
