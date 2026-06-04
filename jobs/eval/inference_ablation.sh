#!/bin/bash
#SBATCH --job-name=inference_ablation
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=GPU_PARTITION
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ==============================================================================
# Inference Steps Ablation Study
# ==============================================================================
# Evaluates BFN and Diffusion policies at various inference step counts.
#
# Usage:
#   sbatch jobs/eval/inference_ablation.sh
# ==============================================================================

set -e

PROJECT_DIR="$PROJECT_ROOT"
cd $PROJECT_DIR

echo "========================================"
echo "Inference Steps Ablation - $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Activate environment
export PATH="$HOME/.conda/envs/bfn/bin:$PATH"
source activate robodiff

# GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Create output directories
mkdir -p logs
mkdir -p results/ablation

# Run ablation
python scripts/analysis/run_inference_steps_ablation.py \
    --all-seeds \
    --checkpoint-dir outputs/2025.12.25 \
    --bfn-steps "5,10,15,20,25,30,50" \
    --diffusion-steps "10,20,30,40,50,75,100" \
    --n-envs 50 \
    --device cuda \
    --output-dir results/ablation

echo "========================================"
echo "Complete - $(date)"
echo "Results: results/ablation/"
echo "========================================"
