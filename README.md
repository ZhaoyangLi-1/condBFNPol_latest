<h1 align="center">BFN-Policy</h1>

<p align="center">
  <strong>Bayesian Flow Networks for Imitation Learning in Hybrid Action Spaces</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#setup">Setup</a> ·
  <a href="#data">Data</a> ·
  <a href="#training">Training</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#reproducing-the-paper">Reproducing the Paper</a> ·
  <a href="#videos">Videos</a>
</p>

---

## Overview

**BFN-Policy** is the first imitation-learning policy built on Bayesian Flow Networks
(BFNs). Instead of reversing a Gaussian noising process in *data* space (Diffusion
Policy), it performs iterative Bayesian inference in *belief* space via unified
Gaussian and categorical conjugate updates, so every intermediate distribution is a
valid posterior and discrete–continuous dependencies are preserved without relaxation
or rounding. The same property lets it vary its inference budget at test time without
distillation or retraining.

We evaluate on eight PAMDP simulation benchmarks (Platform, Goal, Hard Goal, Catch
Point, Hard Move with n∈{4,6,8,10}), a continuous-control Push-T benchmark, and two
real-world tasks (hybrid Push-T on an xArm and continuous Push-T on a WidowX).

## Setup

```bash
# 1. Create the conda environment
conda create -n bfn python=3.10 -y && conda activate bfn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Clone the vendored external repositories (PAMDP envs + consistency policy)
mkdir -p _external && cd _external
git clone https://github.com/ScarletXiaoyu/DLPA.git
git clone https://github.com/Aaditya-Prasad/consistency-policy.git
git clone https://github.com/TJU-DRL-LAB/self-supervised-rl.git
cd ..

# 4. Add the project + diffusion-policy to PYTHONPATH
export PYTHONPATH="$PWD:$PWD/_external/consistency-policy/diffusion_policy:$PYTHONPATH"
```

Tested on Linux + Python 3.10 + CUDA 11.3 (PyTorch 1.13+). A single 16GB GPU is
sufficient for training each PAMDP policy; sim/real Push-T uses 24GB.

## Data

### Simulated Push-T (continuous, Diffusion-Policy benchmark)
```bash
git lfs install
git clone https://huggingface.co/datasets/cadene/pusht_raw pusht_raw
cd pusht_raw && git lfs pull && cd ..
```

### PAMDP demonstrations
Each PAMDP env has its own collector that runs the DLPA RL expert and saves
trajectories:
```bash
python scripts/collect_pamdp_demos.py --env hard_move_n4 --n_episodes 500
python scripts/collect_pamdp_demos.py --env goal         --n_episodes 500
python scripts/collect_pamdp_demos.py --env hard_goal    --n_episodes 500
python scripts/collect_pamdp_demos.py --env catch_point  --n_episodes 500
python scripts/collect_pamdp_demos.py --env platform     --n_episodes 500
```
Demos land under `data/<env>/`. The number of demos kept per task is set in the
config (see `config/train_*_<env>.yaml`).

### Real-world Push-T (hybrid xArm)
Demonstrations are collected via teleoperation. Pre-collected `.zarr` shards land
under `data/pusht_real_hybrid/`.

## Training

All policies share the same Conditional 1D U-Net backbone and observation encoders,
differing only in the generative formulation.

```bash
# BFN-Policy
python scripts/train_workspace.py --config-name=train_bfn_<env>
# DDPM (DDIM re-uses this checkpoint at inference)
python scripts/train_workspace.py --config-name=train_ddpm_<env>
# Consistency Policy (distilled from DDPM)
python scripts/train_workspace.py --config-name=train_consistency_<env>
```

`<env>` ∈ `{hard_move_n4, hard_move_n6, hard_move_n8, hard_move_n10, goal, hard_goal,
catch_point, platform, pusht}`.

SLURM job templates for cluster training are in `jobs/`; adapt the `#SBATCH` headers
to your cluster (partition, account, GPU type, memory).

## Evaluation

Per-env evaluation scripts run 50 rollouts and report success rate, cumulative
reward, and per-call inference time:

```bash
# PAMDP envs
python scripts/eval_hard_move.py   --ckpt <ckpt> --policy bfn --n_actuators 4
python scripts/eval_goal.py        --ckpt <ckpt> --policy bfn
python scripts/eval_hard_goal.py   --ckpt <ckpt> --policy bfn
python scripts/eval_catch_point.py --ckpt <ckpt> --policy bfn
python scripts/eval_platform.py    --ckpt <ckpt> --policy bfn

# Simulated Push-T (continuous)
python scripts/eval_bfn_pusht_steps.py
```

`--policy` is one of `bfn | ddpm | ddim | consistency1 | consistency3`. DDIM uses the
DDPM checkpoint with the DDIM sampler at inference time.

## Reproducing the Paper

### Q1 — PAMDP results (Table 2)
Train one checkpoint per (env, policy) using the configs above, then run all eight
eval scripts and aggregate. Numbers reported as mean ± 95% CI over 50 episodes.

### Q2 — Continuous Push-T (Table 3)
Train BFN, DDPM, and Consistency on `pusht` and evaluate; DDIM evaluates the DDPM
checkpoint with 10 steps.

### Q3 — Real-world hybrid xArm (Table 4)
Requires the xArm hardware setup described in the paper appendix. Pre-collected
demonstrations are needed; the same training/eval scripts apply with
`--config-name=train_bfn_pusht_real`.

### NFE ablation (Fig. 6)
A single BFN checkpoint is evaluated at any inference-step count without retraining:
```bash
sbatch jobs/nfe_sweep.job          # one BFN ckpt per env, n ∈ {1,2,4,8,10,20,30,40,50}
python scripts/plot_nfe_sweep.py   # produces figures/nfe_sweep.pdf
```

### Rollout videos (Fig. 1 / supplementary)
Generate 4-up grid videos (BFN-20, DDIM-10, DDPM-100, Cons-1 on the same env seed):
```bash
sbatch jobs/render_hardmove_all.job   # hard_move n=6,8,10
sbatch jobs/render_catchpoint.job
sbatch jobs/render_rest.job           # platform, goal, hard_goal
```
The combined renderer is `scripts/render_rollouts.py`.

## Videos

Per-environment 4-up grid videos are in `videos/`:

| File | Outcome |
|---|---|
| `hard_move_n4.mp4`  | all four policies reach the target |
| `hard_move_n6.mp4`  | BFN + DDPM succeed; DDIM and Cons-1 fail |
| `hard_move_n8.mp4`  | only BFN succeeds |
| `hard_move_n10.mp4` | only BFN succeeds |
| `catch_point.mp4`   | BFN/DDIM/DDPM catch the target; Cons-1 fails |
| `goal.mp4`          | BFN scores; baselines miss |
| `hard_goal.mp4`     | BFN/DDIM/DDPM score; Cons-1 misses |
| `platform.mp4`      | BFN/DDIM/DDPM cross all platforms; Cons-1 falls |
| `xArm.mp4`  | Real Push-T with hybrid action space using xArm |
| `widowx.mp4`  | Real Push-T with the continuous action space using WidowX |

## Repository Layout

```
policies/        BFN, diffusion, and consistency policy classes
networks/        Conditional 1D U-Net + observation encoders
workspaces/      Hydra training workspaces (one per policy class)
dataset/         Per-env demonstration datasets
environments/    PAMDP env wrappers (Hard Move, Goal, Hard Goal, Catch Point, Platform)
env_runners/     Online rollout runners for evaluation
scripts/         Training, evaluation, plotting, video-recording scripts
config/          Hydra configs (train_<policy>_<env>.yaml)
jobs/            SLURM job templates
appendix/        LaTeX appendices (math derivations, hyperparameters)
videos/          Per-env 4-up rollout videos
_external/       Vendored DLPA, gym-platform, gym-goal, consistency-policy (not tracked)
```

## License

Code released under MIT. The vendored repositories in `_external/` retain their
original licenses.