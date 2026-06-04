# PushT-xarm Real-Robot Training Hyperparameters

Reference document for BFN-Hybrid and DDPM training runs on the real-robot PushT-xarm
datasets (`borueihuang/pusht_xarm` and `borueihuang/pusht_xarm_merged`).
Trained May 2026; checkpoints published at `Abha2001/bfn_pusht_xarm_top`.

## Dataset

Source: HuggingFace `borueihuang/pusht_xarm` (raw) and `borueihuang/pusht_xarm_merged`
(consecutive same-direction segments merged into single actions).

|                                | Merged | Original |
| ------------------------------ | ------ | -------- |
| Episodes                       | 144    | 144      |
| Frames                         | 7,835  | 9,229    |
| FPS                            | 30     | 30       |
| Train / val split              | 130 / 14 | 130 / 14 |
| Train sequences (horizon=16)   | 6,190  | ~7,430   |
| `action.direction` range       | {0..7} | {0..7}   |
| `action.distance` range        | [0, 50] | [0, 62.5] |

### Observation space

| Key       | Shape          | Type | Notes                                |
| --------- | -------------- | ---- | ------------------------------------ |
| `camera_0` | 3 × 224 × 224 | RGB  | Top-down view of T-block + goal      |
| `camera_1` | 3 × 224 × 224 | RGB  | Side view of robot arm (not used in published ckpts; used in side-cam trainings) |

### Action space

8 discrete push directions × 1 continuous distance.

- **BFN encoding**: `[direction_class, distance]`, shape `[2]`
- **DDPM encoding**: `[one_hot(8), distance]`, shape `[9]`

## Shared Hyperparameters

### Horizons

| Parameter            | Value |
| -------------------- | ----- |
| horizon              | 16    |
| n_obs_steps          | 2     |
| n_action_steps       | 8     |
| n_latency_steps      | 0     |
| obs_as_global_cond   | true  |
| past_action_visible  | false |

### Vision encoder (robomimic)

| Parameter                  | Value             |
| -------------------------- | ----------------- |
| Architecture               | ResNet18 + SpatialSoftmax |
| `crop_shape`               | [216, 216]        |
| `obs_encoder_group_norm`   | true              |
| `eval_fixed_crop`          | true              |
| Output feature dim (per cam) | 64              |
| Global cond dim (1 cam × 2 obs) | 128          |
| Vision params              | ~11 M             |

### Conditional U-Net (1D)

| Parameter                | Value               |
| ------------------------ | ------------------- |
| `diffusion_step_embed_dim` | 128                |
| `down_dims`              | [256, 512, 1024]    |
| `kernel_size`            | 5                   |
| `n_groups`               | 8                   |
| `cond_predict_scale`     | true                |
| U-Net params             | ~65 M               |

### Optimizer

| Parameter           | Value         |
| ------------------- | ------------- |
| Optimizer           | AdamW         |
| Learning rate       | 1.0e-4        |
| betas               | [0.95, 0.999] |
| eps                 | 1.0e-8        |
| weight_decay        | 1.0e-6        |
| lr_scheduler        | cosine        |
| lr_warmup_steps     | 500           |

### Training

| Parameter                      | Value |
| ------------------------------ | ----- |
| num_epochs                     | 200   |
| batch_size                     | 32    |
| gradient_accumulate_every      | 1     |
| use_ema                        | true  |
| EMA inv_gamma                  | 1.0   |
| EMA power                      | 0.75  |
| EMA max_value                  | 0.9999 |
| val_ratio                      | 0.10  |
| Validation cadence             | every epoch |

### Checkpointing (known bug for re-runs)

```yaml
checkpoint:
  topk:
    monitor_key: train_loss      # SHOULD BE val_loss — fix for future runs
    mode: min
    k: 1
  save_last_ckpt: true
```

The current configs select checkpoints by `train_loss` (always decreasing → keeps the
overfit endpoint). Future runs should monitor `val_loss` instead.

## BFN-specific

| Parameter             | Value |
| --------------------- | ----- |
| `num_discrete_actions` | 8    |
| `continuous_param_dim` | 1    |
| `sigma_1` (continuous BFN) | 0.001 |
| `beta_1` (categorical accuracy schedule) | 0.2 |
| `n_timesteps` (inference) | 20 (also reported at 10) |

Policy class: `policies.bfn_hybrid_image_policy.BFNHybridImagePolicy`

## DDPM-specific

| Parameter              | Value |
| ---------------------- | ----- |
| `num_train_timesteps`  | 100   |
| `num_inference_steps`  | 100   |
| `beta_start`           | 1e-4  |
| `beta_end`             | 2e-2  |
| `beta_schedule`        | squaredcos_cap_v2 |
| `variance_type`        | fixed_small |
| `clip_sample`          | true  |
| `prediction_type`      | epsilon |

Policy class: `diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy`

## Compute

Single V100 16 GB GPU per run.

| Run         | Wall clock | Throughput |
| ----------- | ---------- | ---------- |
| BFN merged  | 2 h 15 m   | ~6 it/s    |
| DDPM merged | 2 h 16 m   | ~6 it/s    |
| BFN orig    | 2 h 46 m   | ~6 it/s    |
| DDPM orig   | 2 h 42 m   | ~6 it/s    |

## Final loss summary

| Run          | Train final | Val best (epoch) | Val final |
| ------------ | ----------- | ---------------- | --------- |
| BFN merged   | 0.043       | 1.72 (epoch 3)   | 5.22      |
| BFN orig     | 0.023       | 1.50 (epoch 5)   | 5.61      |
| DDPM merged  | 0.037       | 0.077 (epoch 94) | 0.11      |
| DDPM orig    | 0.028       | 0.070 (epoch 75) | 0.14      |

**Observation:** BFN val loss peaks very early (epoch 3–5) and then climbs ~3× by epoch 200
— heavy overfitting on 144-episode datasets. DDPM overfits more gently but also peaks
mid-training. Future runs should early-stop or save by `val_loss`.

## Reproducing

Source configs:

- `config/train_bfn_pusht_xarm_top.yaml`
- `config/train_bfn_pusht_xarm_orig_top.yaml`
- `config/train_ddpm_pusht_xarm_top.yaml`
- `config/train_ddpm_pusht_xarm_orig_top.yaml`

Resolved Hydra configs (the exact configs used at training):

- `outputs/2026.05.11/21.21.05_train_bfn_pusht_xarm_top/.hydra/config.yaml`
- `outputs/2026.05.12/11.20.46_train_bfn_pusht_xarm_orig_top/.hydra/config.yaml`
- `outputs/2026.05.12/08.29.06_train_ddpm_pusht_xarm_top/.hydra/config.yaml`
- `outputs/2026.05.12/11.20.45_train_ddpm_pusht_xarm_orig_top/.hydra/config.yaml`

Launch command (from repo root, inside conda env `bfn`):

```bash
python train.py --config-name=train_bfn_pusht_xarm_top
```
