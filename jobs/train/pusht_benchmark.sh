#!/bin/bash
#SBATCH --job-name=pusht_benchmark
#SBATCH --partition=lrz-hgx-h100-94x4    # H100 partition
#SBATCH --gres=gpu:1                     # Request 1 H100 GPU
#SBATCH --cpus-per-task=16               # 16 CPU cores for fast data loading
#SBATCH --mem=64G                        # 64GB RAM
#SBATCH --time=48:00:00                  # 48 hour limit for full benchmark
#SBATCH --output=logs/benchmark-%j.out   # Save logs here
#SBATCH --chdir=$PROJECT_ROOT  # Set working directory

# PushT Benchmark: BFN vs Diffusion Policy
#
# This script runs the full benchmark comparing Bayesian Flow Networks and Diffusion policies
# on the PushT task with multiple seeds.
#
# BFN (Bayesian Flow Network):
# - Training: predict x from noisy mu, weighted BFN loss
# - Inference: Bayesian posterior updates
#
# Usage (interactive):
#   ./scripts/run_pusht_benchmark.sh [--seeds "42 43 44"] [--epochs 300]
#
# Usage (SLURM):
#   sbatch scripts/run_pusht_benchmark.sh

set -e

# ================== Environment Setup ==================
# First, ensure we're in the project directory (handles both SLURM and interactive)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
elif [ -f "$(dirname "$0")/../scripts/train_workspace.py" ]; then
    cd "$(dirname "$0")/.."
fi
echo "Working directory: $(pwd)"

# Activate conda environment
source activate robodiff

# Debug info
echo "Running on host: $(hostname)"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
fi

# Set environment variables
export WANDB_API_KEY='d47c0bfd84b46e8364f863f142e4fa03c425500e'
export PYTHONPATH=.
export HYDRA_FULL_ERROR=1

# ================== Disk Space Management ==================
# Use home directory for outputs (LRZ setup)
# Your home: $PROJECT_ROOT
SCRATCH_DIR="${SCRATCH_DIR:-$HOME}"
OUTPUT_DIR="${SCRATCH_DIR}/condBFNPol/outputs"
mkdir -p "$OUTPUT_DIR" || { echo "ERROR: Cannot create $OUTPUT_DIR"; exit 1; }

echo "Using output directory: $OUTPUT_DIR"

# Set wandb to use same location
export WANDB_DIR="${SCRATCH_DIR}/condBFNPol/wandb"
export WANDB_CACHE_DIR="${SCRATCH_DIR}/condBFNPol/wandb_cache"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" 2>/dev/null || true

# Clean up old wandb runs from home (optional, uncomment if needed)
# rm -rf ~/condBFNPol/wandb/run-*

# ================== Configuration ==================
# Default values
SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-300}"
DEVICE="${DEVICE:-cuda:0}"
PROJECT="${PROJECT:-pusht-benchmark}"
ENTITY="${ENTITY:-aleyna-research}"
NUM_WORKERS="${NUM_WORKERS:-8}"  # Reduced to avoid memory issues

# Data path - auto-detect based on hostname or set explicitly
if [ -z "$DATA_PATH" ]; then
    if [[ "$(hostname)" == *"lrz"* ]] || [[ "$(hostname)" == *"gpu"* ]]; then
        # Cluster path
        DATA_PATH="$PROJECT_ROOT/data/pusht/pusht_cchi_v7_replay.zarr"
    else
        # Local path
        DATA_PATH="data/pusht/pusht_cchi_v7_replay.zarr"
    fi
fi

# Parse command line arguments (for interactive use)
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --entity)
            ENTITY="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "======================================"
echo "PushT Benchmark: BFN vs Diffusion"
echo "======================================"
echo "Seeds: $SEEDS"
echo "Epochs: $EPOCHS"
echo "Device: $DEVICE"
echo "WandB Project: $PROJECT"
echo "WandB Entity: $ENTITY"
echo "Data Path: $DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Num Workers: $NUM_WORKERS"
echo "======================================"

# Verify we're in the project directory (already set by SLURM --chdir or earlier cd)
if [ ! -f "scripts/train_workspace.py" ]; then
    echo "ERROR: Cannot find scripts/train_workspace.py in $(pwd)"
    echo "Please run from condBFNPol project root"
    exit 1
fi
echo "Project root confirmed: $(pwd)"

# ================== Run Benchmark ==================
for SEED in $SEEDS; do
    echo ""
    echo "======================================"
    echo "Running seed $SEED"
    echo "======================================"
    
    # Train BFN (Bayesian Flow Network)
    echo ""
    echo "--- Training BFN (seed=$SEED) ---"
    python scripts/train_workspace.py \
        --config-name=benchmark_bfn_pusht \
        hydra.run.dir="${OUTPUT_DIR}/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_bfn_seed${SEED}" \
        task.dataset.zarr_path="$DATA_PATH" \
        training.seed=$SEED \
        training.device=$DEVICE \
        training.num_epochs=$EPOCHS \
        training.checkpoint_every=100 \
        checkpoint.save_last_ckpt=false \
        checkpoint.topk.k=2 \
        dataloader.num_workers=$NUM_WORKERS \
        val_dataloader.num_workers=$NUM_WORKERS \
        logging.project=$PROJECT \
        logging.mode=online \
        +logging.entity=$ENTITY
    
    # Train Diffusion
    echo ""
    echo "--- Training Diffusion (seed=$SEED) ---"
    python scripts/train_workspace.py \
        --config-name=benchmark_diffusion_pusht \
        hydra.run.dir="${OUTPUT_DIR}/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_diffusion_seed${SEED}" \
        task.dataset.zarr_path="$DATA_PATH" \
        training.seed=$SEED \
        training.device=$DEVICE \
        training.num_epochs=$EPOCHS \
        training.checkpoint_every=100 \
        checkpoint.save_last_ckpt=false \
        checkpoint.topk.k=2 \
        dataloader.num_workers=$NUM_WORKERS \
        val_dataloader.num_workers=$NUM_WORKERS \
        logging.project=$PROJECT \
        logging.mode=online \
        +logging.entity=$ENTITY
done

echo ""
echo "======================================"
echo "Benchmark complete!"
echo "======================================"
echo "Results logged to WandB project: $PROJECT"
echo "View at: https://wandb.ai/$ENTITY/$PROJECT"
