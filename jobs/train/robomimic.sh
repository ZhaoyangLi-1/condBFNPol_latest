#!/bin/bash
#SBATCH --job-name=robomimic
#SBATCH --output=logs/robomimic-%j.out
#SBATCH --error=logs/robomimic-%j.out
#SBATCH --time=48:00:00
#SBATCH --partition=GPU_PARTITION
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ==============================================================================
# Robomimic Experiments: BFN vs Diffusion
# ==============================================================================
# This script trains both BFN and Diffusion policies on Robomimic tasks
# with disk-aware settings to avoid quota issues.
# ==============================================================================

set -e  # Exit on error

# Configuration
PROJECT_DIR="$PROJECT_ROOT"
TASK="${TASK:-lift}"           # Default task: lift (can override with --export=TASK=can)
METHOD="${METHOD:-bfn}"        # Default method: bfn (can override with --export=METHOD=diffusion)
SEED="${SEED:-42}"             # Default seed
NUM_EPOCHS="${NUM_EPOCHS:-3050}"  # Full training

echo "========================================"
echo "Robomimic Training - $(date)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Task: ${TASK}"
echo "Method: ${METHOD}"
echo "Seed: ${SEED}"
echo ""

# Check disk usage before starting
echo "=== Disk Usage Check ==="
du -sh $PROJECT_DIR 2>/dev/null || echo "Could not check disk usage"
df -h $PROJECT_ROOT | tail -1
echo ""

# Change to project directory
cd $PROJECT_DIR

# Activate conda environment
source ~/.bashrc
source activate robodiff

# Check GPU
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Set task config name
TASK_CONFIG="${TASK}_image_abs"

# Create output directory with cleanup-friendly naming
OUTPUT_DIR="outputs/robomimic_$(date +%Y.%m.%d)/${METHOD}_${TASK}_seed${SEED}"
mkdir -p $OUTPUT_DIR

echo "=== Starting Training ==="
echo "Config: train_${METHOD}_robomimic_image"
echo "Task: ${TASK_CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Run training
if [ "$METHOD" == "bfn" ]; then
    python scripts/train_workspace.py \
        --config-name=train_bfn_robomimic_image \
        task@_global_=${TASK_CONFIG} \
        training.seed=${SEED} \
        training.num_epochs=${NUM_EPOCHS} \
        hydra.run.dir=${OUTPUT_DIR}
elif [ "$METHOD" == "diffusion" ]; then
    python scripts/train_workspace.py \
        --config-name=train_diffusion_robomimic_image \
        task@_global_=${TASK_CONFIG} \
        training.seed=${SEED} \
        training.num_epochs=${NUM_EPOCHS} \
        hydra.run.dir=${OUTPUT_DIR}
else
    echo "ERROR: Unknown method: $METHOD"
    exit 1
fi

# Check disk usage after training
echo ""
echo "=== Final Disk Usage ==="
du -sh $PROJECT_DIR 2>/dev/null || echo "Could not check disk usage"
du -sh $OUTPUT_DIR 2>/dev/null || echo "Could not check output dir"

echo ""
echo "========================================"
echo "Training complete - $(date)"
echo "========================================"

