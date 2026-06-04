# Config

Hydra configuration files.

<br>

## Usage

```bash
python scripts/train_workspace.py --config-name=benchmark_bfn_pusht
```

Override parameters:

```bash
python scripts/train_workspace.py --config-name=benchmark_bfn_pusht \
    training.num_epochs=100 \
    training.seed=42
```

<br>

## Files

| Config | Method | Task |
|:--|:--|:--|
| `benchmark_bfn_pusht` | BFN | Push-T |
| `benchmark_diffusion_pusht` | Diffusion | Push-T |
| `train_bfn_robomimic_image` | BFN | RoboMimic |
| `train_diffusion_robomimic_image` | Diffusion | RoboMimic |
| `train_bfn_hybrid_lift` | BFN Hybrid | Lift |
