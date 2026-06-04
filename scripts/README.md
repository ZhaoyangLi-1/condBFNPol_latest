# Scripts

Training, analysis, and visualization.

<br>

## Training

```bash
python scripts/train_workspace.py --config-name=benchmark_bfn_pusht
```

<br>

## Analysis

```bash
python scripts/analysis/comprehensive_ablation_study.py \
    --checkpoint-dir outputs/2025.12.25 \
    --output-dir results/ablation
```

<br>

## Figures

```bash
bash scripts/regenerate_all_figures.sh
```

<br>

## Structure

```
scripts/
├── train_workspace.py          Entry point
├── regenerate_all_figures.sh   Generate all figures
│
├── figures/                    Visualization
│   ├── plot_*.py
│   └── generate_*.py
│
├── analysis/                   Experiment analysis
│   ├── comprehensive_ablation_study.py
│   └── analyze_hybrid_experiments.py
│
└── utils/
    └── colors.py               Anthropic palette
```
