# Jobs

SLURM scripts for cluster execution.

<br>

## Single Job

```bash
sbatch jobs/train/pusht_benchmark.sh
```

<br>

## Batch Submission

```bash
bash jobs/train/submit_all_hybrid.sh
```

<br>

## Monitor

```bash
squeue -u $USER
tail -f logs/*.out
```

<br>

## Structure

```
jobs/
├── train/
│   ├── pusht_benchmark.sh
│   ├── robomimic.sh
│   ├── hybrid_discrete.sh
│   └── submit_all_*.sh
│
└── eval/
    ├── ablation.sh
    └── inference_ablation.sh
```
