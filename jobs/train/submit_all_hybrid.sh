#!/bin/bash
# ==============================================================================
# Submit All Hybrid Discrete Experiments
# ==============================================================================
# Compares BFN-Policy (native discrete) vs Diffusion (continuous relaxation)
#
# Usage:
#   bash jobs/train/submit_all_hybrid.sh
#   bash jobs/train/submit_all_hybrid.sh --dry-run
# ==============================================================================

PROJECT_DIR="$PROJECT_ROOT"
JOB_SCRIPT="jobs/train/hybrid_discrete.sh"

TASKS=("lift")
METHODS=("bfn_hybrid" "diffusion")
SEEDS=(42 43 44)

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true && echo "=== DRY RUN ==="

cd $PROJECT_DIR

echo "=============================================="
echo "Hybrid Discrete Experiments"
echo "Tasks: ${TASKS[*]}"
echo "Methods: ${METHODS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total: $((${#TASKS[@]} * ${#METHODS[@]} * ${#SEEDS[@]})) jobs"
echo "=============================================="

for TASK in "${TASKS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            JOB_NAME="${METHOD}_${TASK}_s${SEED}"
            
            if [[ "$DRY_RUN" == "true" ]]; then
                echo "[DRY] sbatch --export=TASK=$TASK,METHOD=$METHOD,SEED=$SEED $JOB_SCRIPT"
            else
                sbatch --export=TASK=$TASK,METHOD=$METHOD,SEED=$SEED \
                    --job-name=$JOB_NAME $JOB_SCRIPT
            fi
            sleep 0.5
        done
    done
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Analyze: python scripts/analysis/analyze_hybrid_experiments.py --task lift"
