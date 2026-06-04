#!/bin/bash
# ==============================================================================
# Submit All Robomimic Experiments
# ==============================================================================
# Usage: ./jobs/submit_all_robomimic.sh [--dry-run]
#
# This script submits jobs for all combinations of:
#   - Methods: bfn, diffusion
#   - Tasks: lift, can, square
#   - Seeds: 42, 43, 44
#
# Estimated disk usage per run: ~2-5 GB (checkpoints + logs)
# Total for all experiments: ~50-100 GB
# ==============================================================================

DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No jobs will be submitted ==="
    echo ""
fi

# Configuration
METHODS="bfn diffusion"
TASKS="lift"  # Start with lift only; add "can square" when datasets are ready
SEEDS="42 43 44"

# Count total jobs
TOTAL_JOBS=0
for method in $METHODS; do
    for task in $TASKS; do
        for seed in $SEEDS; do
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
        done
    done
done

echo "========================================"
echo "Robomimic Experiment Submission"
echo "========================================"
echo "Methods: $METHODS"
echo "Tasks: $TASKS"
echo "Seeds: $SEEDS"
echo "Total jobs: $TOTAL_JOBS"
echo ""

# Check if datasets exist
echo "=== Checking Datasets ==="
for task in $TASKS; do
    DATASET_PATH="data/robomimic/datasets/${task}/ph/image_abs.hdf5"
    if [ -f "$DATASET_PATH" ]; then
        echo "✓ $task dataset found"
    else
        echo "✗ $task dataset MISSING: $DATASET_PATH"
        echo "  Download with:"
        echo "  wget https://diffusion-policy.cs.columbia.edu/data/training/robomimic/${task}/ph/image_abs.hdf5 -O $DATASET_PATH"
        echo ""
    fi
done
echo ""

# Confirm before submitting
if [ "$DRY_RUN" == "false" ]; then
    read -p "Submit $TOTAL_JOBS jobs? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Submit jobs
echo ""
echo "=== Submitting Jobs ==="
JOB_IDS=""
for method in $METHODS; do
    for task in $TASKS; do
        for seed in $SEEDS; do
            JOB_NAME="${method}_${task}_s${seed}"
            
            if [ "$DRY_RUN" == "true" ]; then
                echo "[DRY RUN] Would submit: $JOB_NAME"
            else
                # Submit job with environment variables
                JOB_ID=$(sbatch \
                    --job-name=$JOB_NAME \
                    --export=ALL,METHOD=$method,TASK=$task,SEED=$seed \
                    jobs/train/robomimic.sh | awk '{print $4}')
                
                echo "Submitted: $JOB_NAME (Job ID: $JOB_ID)"
                JOB_IDS="$JOB_IDS $JOB_ID"
            fi
        done
    done
done

echo ""
echo "========================================"
if [ "$DRY_RUN" == "false" ]; then
    echo "Submitted job IDs:$JOB_IDS"
    echo ""
    echo "Monitor with:"
    echo "  squeue --me"
    echo "  tail -f logs/robomimic-JOBID.out"
else
    echo "Dry run complete. No jobs submitted."
fi
echo "========================================"

