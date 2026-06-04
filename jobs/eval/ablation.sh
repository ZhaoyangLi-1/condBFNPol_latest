#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --output=logs/ablation-%j.out
#SBATCH --error=logs/ablation-%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ==============================================================================
# Ablation Study: Inference Steps Analysis
# ==============================================================================
# Evaluates BFN and Diffusion policies at varying inference step counts.
#
# Usage:
#   sbatch jobs/eval/ablation.sh
# ==============================================================================

set -e

PROJECT_DIR="$PROJECT_ROOT"
cd $PROJECT_DIR

echo "========================================"
echo "Ablation Study - $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Activate environment
source ~/.bashrc
source activate robodiff

# GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Run ablation
python scripts/analysis/comprehensive_ablation_study.py \
    --checkpoint-dir outputs/2025.12.25 \
    --n-envs 50 \
    --device cuda \
    --output-dir results/ablation

echo "========================================"
echo "Complete - $(date)"
echo "========================================"
