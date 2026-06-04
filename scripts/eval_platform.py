"""Evaluate BFN, DDPM, DDIM, EDM, Consistency on Platform environment."""

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
print("Starting Platform eval...", flush=True)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from policies.bfn_hybrid_lowdim_policy import BFNHybridLowdimPolicy
from policies.diffusion_lowdim_policy import DiffusionLowdimPolicy
try:
    from policies.consistency_lowdim_policy import ConsistencyLowdimPolicy
except ImportError:
    ConsistencyLowdimPolicy = None
try:
    from policies.edm_lowdim_policy import EDMLowdimPolicy
except ImportError:
    EDMLowdimPolicy = None

from environments.platform_env import PlatformEnv

NUM_DISCRETE = 3
ACTION_DIM_CONT = 4


def load_policy(checkpoint_path, policy_type, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint['state_dicts']['model'] if 'state_dicts' in checkpoint else checkpoint

    if policy_type == "bfn":
        policy = BFNHybridLowdimPolicy(
            obs_dim=9, horizon=16, n_obs_steps=2, n_action_steps=8,
            num_discrete_actions=3, continuous_param_dim=1,
            sigma_1=0.001, beta_1=0.2, n_timesteps=20,
        )
    elif policy_type == "bfn10":
        policy = BFNHybridLowdimPolicy(
            obs_dim=9, horizon=16, n_obs_steps=2, n_action_steps=8,
            num_discrete_actions=3, continuous_param_dim=1,
            sigma_1=0.001, beta_1=0.2, n_timesteps=10,
        )
    elif policy_type == "edm":
        policy = EDMLowdimPolicy(
            obs_dim=9, action_dim=4, horizon=16, n_obs_steps=2, n_action_steps=8,
            sigma_min=0.002, sigma_max=80.0, rho=7.0,
            num_inference_steps=40, solver="heun", P_mean=-1.2, P_std=1.2, delta=-1.0,
        )
    elif policy_type == "ddim":
        policy = DiffusionLowdimPolicy(
            obs_dim=9, action_dim=4, horizon=16, n_obs_steps=2, n_action_steps=8,
            scheduler_type="ddim", num_train_timesteps=100, num_inference_steps=10,
        )
    elif policy_type == "consistency":
        policy = ConsistencyLowdimPolicy(
            obs_dim=9, action_dim=4, horizon=16, n_obs_steps=2, n_action_steps=8,
            num_train_timesteps=100, num_inference_steps=1,
            sigma_min=0.002, sigma_max=80.0, rho=7.0,
            ctm_weight=1.0, dsm_weight=1.0, delta=0.0, teacher_path=None,
        )
    elif policy_type == "consistency3":
        policy = ConsistencyLowdimPolicy(
            obs_dim=9, action_dim=4, horizon=16, n_obs_steps=2, n_action_steps=8,
            num_train_timesteps=100, num_inference_steps=3,
            sigma_min=0.002, sigma_max=80.0, rho=7.0,
            ctm_weight=1.0, dsm_weight=1.0, delta=0.0, teacher_path=None,
        )
    else:  # ddpm
        policy = DiffusionLowdimPolicy(
            obs_dim=9, action_dim=4, horizon=16, n_obs_steps=2, n_action_steps=8,
            scheduler_type="ddpm", num_train_timesteps=100, num_inference_steps=100,
        )

    policy.load_state_dict(model_state)
    policy.to(device)
    policy.eval()
    return policy


def get_action_bfn(policy, obs_history):
    with torch.no_grad():
        result = policy.predict_action({'state': obs_history})
        action = result['action'][0, 0].cpu().numpy()
    k = int(np.round(action[0]).clip(0, NUM_DISCRETE - 1))
    return {"k": k, "x_k": [action[1]]}


def get_action_continuous(policy, obs_history):
    with torch.no_grad():
        result = policy.predict_action({'state': obs_history})
        action = result['action'][0, 0].cpu().numpy()
    one_hot = action[:NUM_DISCRETE]
    k = int(np.argmax(one_hot).clip(0, NUM_DISCRETE - 1))
    return {"k": k, "x_k": [action[NUM_DISCRETE]]}


def evaluate(policy, policy_type, n_episodes=50, device="cuda"):
    env = PlatformEnv()
    if policy_type in ["bfn", "bfn10"]:
        get_action = get_action_bfn
    else:
        get_action = get_action_continuous

    rewards, successes, inference_times = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        obs_np = np.array(obs, dtype=np.float32)
        obs_history = torch.from_numpy(np.stack([obs_np, obs_np], axis=0)).unsqueeze(0).to(device)
        total_reward, done, steps = 0, False, 0

        while not done and steps < env.max_steps:
            t0 = time.time()
            action = get_action(policy, obs_history)
            inference_times.append((time.time() - t0) * 1000)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
            obs_np = np.array(obs, dtype=np.float32)
            obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(0).to(device)
            obs_history = torch.cat([obs_history[:, 1:], obs_tensor], dim=1)

        rewards.append(total_reward)
        successes.append(float(total_reward > 0.9))
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: reward={total_reward:.3f}", flush=True)

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes) * 100),
        "mean_inference_ms": float(np.mean(inference_times)),
        "std_inference_ms": float(np.std(inference_times)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfn_ckpt", type=str, default=None)
    parser.add_argument("--bfn10_ckpt", type=str, default=None)
    parser.add_argument("--ddpm_ckpt", type=str, default=None)
    parser.add_argument("--ddim_ckpt", type=str, default=None)
    parser.add_argument("--edm_ckpt", type=str, default=None)
    parser.add_argument("--consistency_ckpt", type=str, default=None)
    parser.add_argument("--consistency3_ckpt", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results = {}
    eval_specs = [
        ("bfn", args.bfn_ckpt, "BFN (20)"),
        ("bfn10", args.bfn10_ckpt, "BFN-10"),
        ("ddpm", args.ddpm_ckpt, "DDPM (100)"),
        ("ddim", args.ddim_ckpt, "DDIM (10)"),
        ("edm", args.edm_ckpt, "EDM (40)"),
        ("consistency", args.consistency_ckpt, "Consistency (1)"),
        ("consistency3", args.consistency3_ckpt, "Consistency-3"),
    ]

    for ptype, ckpt, label in eval_specs:
        if not ckpt:
            continue
        print(f"\n=== Evaluating {label} ===")
        policy = load_policy(ckpt, ptype, args.device)
        results[ptype] = evaluate(policy, ptype, args.n_episodes, args.device)
        r = results[ptype]
        print(f"\n{label}: {r['success_rate']:.1f}% / {r['mean_reward']:.3f} ± {r['std_reward']:.3f} / {r['mean_inference_ms']:.1f} ms")

    if len(results) >= 2:
        print(f"\n=== Platform Comparison ===")
        for label, ptype in [("BFN(20)", "bfn"), ("BFN-10", "bfn10"), ("DDPM(100)", "ddpm"),
                              ("DDIM(10)", "ddim"), ("EDM(40)", "edm"),
                              ("Cons(1)", "consistency"), ("Cons-3", "consistency3")]:
            if ptype in results:
                r = results[ptype]
                print(f"  {label:<12} {r['success_rate']:>6.1f}%  {r['mean_reward']:>7.3f}  {r['mean_inference_ms']:>7.1f} ms")


if __name__ == "__main__":
    main()
