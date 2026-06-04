"""One-shot inference timing for all PAMDP policies × envs.

For each (env, method) it loads the latest checkpoint, builds a dummy obs of the
right shape, runs 5 warmups + 20 timed predict_action calls with cuda.synchronize,
and prints mean ± std (and saves to JSON for CI computation).
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from policies.bfn_hybrid_lowdim_policy import BFNHybridLowdimPolicy
from policies.diffusion_lowdim_policy import DiffusionLowdimPolicy
from policies.consistency_lowdim_policy import ConsistencyLowdimPolicy
from policies.edm_lowdim_policy import EDMLowdimPolicy


# Obs dim per env (matches eval_*.py configs)
ENV_OBS_DIM = {
    "hard_move_n4": 4, "hard_move_n6": 4, "hard_move_n8": 4, "hard_move_n10": 4,
    "hard_goal": 17, "goal": 17, "platform": 9, "catch_point": 6,
}
# n_actuators (Hard Move only); other envs use a fixed continuous_param_dim
ENV_N_ACTUATORS = {"hard_move_n4": 4, "hard_move_n6": 6, "hard_move_n8": 8, "hard_move_n10": 10}


def find_ckpt(env: str, pol: str) -> str:
    """Find latest non-132 checkpoint for env/policy."""
    pattern = f"outputs/*/*_train_{pol}_{env}_{env}/checkpoints/latest.ckpt"
    matches = [m for m in sorted(glob.glob(str(PROJECT_ROOT / pattern)), reverse=True) if "_132" not in m]
    return matches[0] if matches else None


def build_policy(env: str, pol: str, ckpt_path: str, device: str):
    obs_dim = ENV_OBS_DIM[env]
    # Hard Move uses one-hot direction encoding (2^n choices) + 1 continuous param
    if env.startswith("hard_move"):
        n_act = ENV_N_ACTUATORS[env]
        num_discrete = 2 ** n_act
        action_dim_cont = num_discrete + 1
        cont_param = 1
    elif env == "hard_goal":
        num_discrete, cont_param = 11, 2
        action_dim_cont = num_discrete + cont_param
    elif env == "catch_point":
        num_discrete, cont_param = 2, 1
        action_dim_cont = num_discrete + cont_param
    elif env == "goal":
        num_discrete, cont_param = 3, 2
        action_dim_cont = num_discrete + cont_param
    elif env == "platform":
        num_discrete, cont_param = 3, 1
        action_dim_cont = num_discrete + cont_param
    else:
        raise ValueError(env)

    H = 16
    if pol == "bfn":
        policy = BFNHybridLowdimPolicy(obs_dim=obs_dim, horizon=H, n_obs_steps=2, n_action_steps=8,
                                       num_discrete_actions=num_discrete, continuous_param_dim=cont_param,
                                       sigma_1=0.001, beta_1=0.2, n_timesteps=20)
    elif pol == "bfn10":
        policy = BFNHybridLowdimPolicy(obs_dim=obs_dim, horizon=H, n_obs_steps=2, n_action_steps=8,
                                       num_discrete_actions=num_discrete, continuous_param_dim=cont_param,
                                       sigma_1=0.001, beta_1=0.2, n_timesteps=10)
    elif pol == "ddpm":
        policy = DiffusionLowdimPolicy(obs_dim=obs_dim, action_dim=action_dim_cont, horizon=H,
                                       n_obs_steps=2, n_action_steps=8, scheduler_type="ddpm",
                                       num_train_timesteps=100, num_inference_steps=100)
    elif pol == "ddim":
        policy = DiffusionLowdimPolicy(obs_dim=obs_dim, action_dim=action_dim_cont, horizon=H,
                                       n_obs_steps=2, n_action_steps=8, scheduler_type="ddim",
                                       num_train_timesteps=100, num_inference_steps=10)
    elif pol == "edm":
        policy = EDMLowdimPolicy(obs_dim=obs_dim, action_dim=action_dim_cont, horizon=H,
                                 n_obs_steps=2, n_action_steps=8, sigma_min=0.002, sigma_max=80.0,
                                 rho=7.0, num_inference_steps=40, solver="heun",
                                 P_mean=-1.2, P_std=1.2, delta=-1.0)
    elif pol == "cons1":
        policy = ConsistencyLowdimPolicy(obs_dim=obs_dim, action_dim=action_dim_cont, horizon=H,
                                         n_obs_steps=2, n_action_steps=8, num_train_timesteps=100,
                                         num_inference_steps=1, sigma_min=0.002, sigma_max=80.0,
                                         rho=7.0, ctm_weight=1.0, dsm_weight=1.0, delta=0.0,
                                         teacher_path=None)
    elif pol == "cons3":
        policy = ConsistencyLowdimPolicy(obs_dim=obs_dim, action_dim=action_dim_cont, horizon=H,
                                         n_obs_steps=2, n_action_steps=8, num_train_timesteps=100,
                                         num_inference_steps=3, sigma_min=0.002, sigma_max=80.0,
                                         rho=7.0, ctm_weight=1.0, dsm_weight=1.0, delta=0.0,
                                         teacher_path=None)
    else:
        raise ValueError(pol)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["state_dicts"]["model"] if "state_dicts" in ckpt else ckpt
    policy.load_state_dict(state)
    policy.to(device).eval()
    return policy


def time_policy(policy, obs_dim: int, device: str, n_warmup: int = 5, n_runs: int = 20):
    obs_dict = {"state": torch.zeros((1, 2, obs_dim), device=device)}
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = policy.predict_action(obs_dict)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = policy.predict_action(obs_dict)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1000
    return dict(mean_ms=float(arr.mean()), std_ms=float(arr.std(ddof=1)),
                ci95_ms=float(1.96 * arr.std(ddof=1) / np.sqrt(n_runs)),
                raw=arr.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/inference_timing_pamdp.json")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    envs = ["hard_move_n4", "hard_move_n6", "hard_move_n8", "hard_move_n10",
            "hard_goal", "goal", "platform", "catch_point"]
    methods = ["bfn", "bfn10", "ddpm", "ddim", "edm", "cons1", "cons3"]

    # ddim and bfn10 reuse ddpm/bfn checkpoints respectively; cons3 reuses cons1
    method_to_ckpt_pol = {"bfn": "bfn", "bfn10": "bfn", "ddpm": "ddpm",
                          "ddim": "ddpm", "edm": "edm", "cons1": "consistency",
                          "cons3": "consistency"}

    results = {}
    for env in envs:
        results[env] = {}
        for m in methods:
            ckpt_pol = method_to_ckpt_pol[m]
            ckpt = find_ckpt(env, ckpt_pol)
            if not ckpt:
                print(f"[skip] {env}/{m}: no ckpt for {ckpt_pol}", flush=True)
                continue
            try:
                policy = build_policy(env, m, ckpt, device)
                r = time_policy(policy, ENV_OBS_DIM[env], device)
                print(f"{env}/{m}: {r['mean_ms']:7.2f} ± {r['ci95_ms']:.2f} ms (95% CI, N=20)", flush=True)
                results[env][m] = r
                del policy
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except Exception as e:
                print(f"[error] {env}/{m}: {e}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
