#!/bin/bash
# =============================================================================
# SLURM Job Script: BFN vs Diffusion on Hybrid Action Spaces
# 
# This script runs experiments to test the hypothesis that BFN-Policy's native
# discrete action handling (Dirichlet-Categorical) outperforms Diffusion Policy's
# continuous relaxation on tasks with prominent discrete actions (gripper).
#
# Tasks: Lift (RoboMimic)
# Action space: 7D continuous (arm) + 1D binary discrete (gripper)
#
# Usage:
#   sbatch jobs/train/hybrid_discrete.sh
#   OR for local testing:
#   bash jobs/train/hybrid_discrete.sh --local
# =============================================================================

#SBATCH --job-name=bfn_hybrid_discrete
#SBATCH --output=logs/hybrid_%j.out
#SBATCH --error=logs/hybrid_%j.err
#SBATCH --time=47:00:00
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# =============================================================================
# Setup
# =============================================================================
set -e

# Configuration
PROJECT_DIR="$PROJECT_ROOT"
TASK="${TASK:-lift}"           # lift, can, or square
METHOD="${METHOD:-bfn_hybrid}" # bfn_hybrid or diffusion
SEED="${SEED:-42}"
NUM_EPOCHS="${NUM_EPOCHS:-600}"

echo "========================================"
echo "Hybrid Discrete Experiment - $(date)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Task: ${TASK}"
echo "Method: ${METHOD}"
echo "Seed: ${SEED}"
echo ""

# Navigate to project root
cd $PROJECT_DIR

# Activate conda environment for SLURM (compatible with older conda 4.3.x)
# Add anaconda to PATH first
export PATH="$HOME/.conda/envs/bfn/bin:$PATH"

# Use source activate for older conda
source activate robodiff || {
    echo "ERROR: Could not activate conda environment 'robodiff'"
    echo "Available environments:"
    conda env list
    exit 1
}

echo "Activated conda environment: robodiff"
echo "Python: $(which python)"
python --version

# Check disk usage
echo "=== Disk Usage Check ==="
du -sh $PROJECT_DIR 2>/dev/null || echo "Could not check disk usage"
df -h $PROJECT_ROOT | tail -1
echo ""

# Check GPU
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "No GPU info"
echo ""

# Set environment variables
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export HYDRA_FULL_ERROR=1

# Create logs directory
mkdir -p logs

# Set task config name (RoboMimic format)
TASK_CONFIG="${TASK}_image_abs"

# =============================================================================
# Run Experiment
# =============================================================================

echo "=========================================="
echo "Running: $METHOD on $TASK (seed=$SEED)"
echo "=========================================="

# Create output directory
OUTPUT_DIR="outputs/hybrid_$(date +%Y.%m.%d)/${METHOD}_${TASK}_seed${SEED}"
mkdir -p $OUTPUT_DIR

if [ "$METHOD" == "bfn_hybrid" ]; then
    echo "Training BFN with native discrete handling..."
    
    # Use the hybrid BFN config
    python scripts/train_workspace.py \
        --config-name=train_bfn_hybrid_lift \
        task@_global_=${TASK_CONFIG} \
        training.seed=${SEED} \
        training.num_epochs=${NUM_EPOCHS} \
        hydra.run.dir=${OUTPUT_DIR}

elif [ "$METHOD" == "diffusion" ]; then
    echo "Training Diffusion with continuous relaxation..."
    
    # Standard diffusion config (treats gripper as continuous)
    python scripts/train_workspace.py \
        --config-name=train_diffusion_robomimic_image \
        task@_global_=${TASK_CONFIG} \
        training.seed=${SEED} \
        training.num_epochs=${NUM_EPOCHS} \
        hydra.run.dir=${OUTPUT_DIR}

elif [ "$METHOD" == "bfn_continuous" ]; then
    echo "Training BFN with continuous-only (baseline)..."
    
    # BFN without hybrid handling for comparison
    python scripts/train_workspace.py \
        --config-name=train_bfn_robomimic_image \
        task@_global_=${TASK_CONFIG} \
        training.seed=${SEED} \
        training.num_epochs=${NUM_EPOCHS} \
        hydra.run.dir=${OUTPUT_DIR}
else
    echo "ERROR: Unknown method: $METHOD"
    echo "Valid methods: bfn_hybrid, diffusion, bfn_continuous"
    exit 1
fi

# Check disk usage after training
echo ""
echo "=== Final Disk Usage ==="
du -sh $PROJECT_DIR 2>/dev/null || echo "Could not check disk usage"
du -sh $OUTPUT_DIR 2>/dev/null || echo "Could not check output dir"

echo ""
echo "========================================"
echo "Experiment complete - $(date)"
echo "========================================"
echo ""
echo "To run all configurations:"
echo "  bash jobs/train/submit_all_hybrid.sh"
echo ""
echo "Analyze results:"
echo "  python scripts/analysis/analyze_hybrid_experiments.py --task lift"
