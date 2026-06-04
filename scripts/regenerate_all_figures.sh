#!/bin/bash
# ==============================================================================
# Regenerate All Publication Figures
# ==============================================================================
# Usage:
#   bash scripts/regenerate_all_figures.sh
#   bash scripts/regenerate_all_figures.sh --quick   # Skip slow figures
# ==============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "======================================================================"
echo "  PUBLICATION FIGURE GENERATOR"
echo "  Output: figures/publication/"
echo "======================================================================"
echo ""

# Activate conda
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate robodiff 2>/dev/null || true
fi

echo "Python: $(which python)"
echo ""

# Create output directories
mkdir -p figures/publication
mkdir -p figures/tables

# Quick mode
QUICK_MODE=false
[[ "$1" == "--quick" ]] && QUICK_MODE=true && echo -e "${YELLOW}Quick mode enabled${NC}"

# Helper function
run_script() {
    local script=$1
    local args=$2
    local name=$(basename "$script" .py)
    
    echo ">>> $name"
    if python "$script" $args 2>&1 | tail -5; then
        echo -e "${GREEN}  ✓ Done${NC}"
    else
        echo -e "${RED}  ✗ Failed${NC}"
    fi
}

echo "======================================================================"
echo "  GENERATING FIGURES"
echo "======================================================================"

# 1. Ablation figures
if [[ -f "results/ablation/ablation_results.json" ]]; then
    run_script "scripts/figures/plot_ablation_results.py" \
        "--results-file results/ablation/ablation_results.json --output-dir figures/publication"
fi

# 2. Setup figures
run_script "scripts/figures/generate_setup_figures.py" ""

# 3. Prediction visualizations
run_script "scripts/figures/generate_prediction_visualizations.py" ""

# 4. Training stability
run_script "scripts/figures/generate_training_stability_figure.py" ""

# 5. Multimodal figures
run_script "scripts/figures/generate_multimodal_figures.py" ""

# 6. Publication figures (slow)
if [[ "$QUICK_MODE" == false ]] && [[ -d "outputs/2025.12.25" ]]; then
    run_script "scripts/figures/generate_publication_figures.py" \
        "--checkpoint-dir outputs/2025.12.25 --output-dir figures/publication"
    
    run_script "scripts/figures/plot_training_loss.py" \
        "--base-dir outputs/2025.12.25 --output-dir figures/publication --all-plots"
fi

# 7. Time vs performance (thesis specific)
run_script "scripts/figures/plot_time_vs_performance.py" ""

echo ""
echo "======================================================================"
echo "  DONE - Output: figures/publication/"
echo "======================================================================"
echo "  PDFs: $(find figures/publication -name '*.pdf' 2>/dev/null | wc -l)"
echo "  PNGs: $(find figures/publication -name '*.png' 2>/dev/null | wc -l)"
echo "======================================================================"
