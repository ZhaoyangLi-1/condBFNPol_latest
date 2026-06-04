"""Evaluate BFN, DDPM, DDIM, Consistency policies on Hard Goal environment."""

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
print("Starting Hard Goal eval script...", flush=True)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

print("Importing policy modules...", flush=True)
from policies.bfn_hybrid_lowdim_policy import BFNHybridLowdimPolicy
from policies.diffusion_lowdim_policy import DiffusionLowdimPolicy
try:
    from policies.consistency_lowdim_policy import ConsistencyLowdimPolicy
    HAS_CONSISTENCY = True
except ImportError:
    HAS_CONSISTENCY = False
try:
    from policies.edm_lowdim_policy import EDMLowdimPolicy
    HAS_EDM = True
except ImportError:
    HAS_EDM = False

print("Loading environment...", flush=True)
from environments.hard_goal import HardGoalEnv
print("Environment loaded.", flush=True)


NUM_DISCRETE = 11
ACTION_DIM_CONT = NUM_DISCRETE + 2  # one-hot + 2 padded params = 13


def load_policy(checkpoint_path: str, policy_type: str, device: str = "cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'state_dicts' in checkpoint:
        model_state = checkpoint['state_dicts'].get('model', {})
    else:
        model_state = checkpoint

    if policy_type == "bfn":
        policy = BFNHybridLowdimPolicy(
            obs_dim=17,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            num_discrete_actions=NUM_DISCRETE,
            continuous_param_dim=2,
            sigma_1=0.001,
            beta_1=0.2,
            n_timesteps=20,
        )
    elif policy_type == "bfn10":
        policy = BFNHybridLowdimPolicy(
            obs_dim=17,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            num_discrete_actions=NUM_DISCRETE,
            continuous_param_dim=2,
            sigma_1=0.001,
            beta_1=0.2,
            n_timesteps=10,
        )
    elif policy_type == "edm":
        policy = EDMLowdimPolicy(
            obs_dim=17, action_dim=ACTION_DIM_CONT, horizon=16,
            n_obs_steps=2, n_action_steps=8,
            sigma_min=0.002, sigma_max=80.0, rho=7.0,
            num_inference_steps=40, solver="heun",
            P_mean=-1.2, P_std=1.2, delta=-1.0,
        )
    elif policy_type == "ddim":
        policy = DiffusionLowdimPolicy(
            obs_dim=17,
            action_dim=ACTION_DIM_CONT,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            scheduler_type="ddim",
            num_train_timesteps=100,
            num_inference_steps=10,
        )
    elif policy_type == "consistency":
        policy = ConsistencyLowdimPolicy(
            obs_dim=17,
            action_dim=ACTION_DIM_CONT,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            num_train_timesteps=100,
            num_inference_steps=1,
            sigma_min=0.002,
            sigma_max=80.0,
            rho=7.0,
            ctm_weight=1.0,
            dsm_weight=1.0,
            delta=0.0,
            teacher_path=None,
        )
    elif policy_type == "consistency3":
        policy = ConsistencyLowdimPolicy(
            obs_dim=17,
            action_dim=ACTION_DIM_CONT,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            num_train_timesteps=100,
            num_inference_steps=3,
            sigma_min=0.002,
            sigma_max=80.0,
            rho=7.0,
            ctm_weight=1.0,
            dsm_weight=1.0,
            delta=0.0,
            teacher_path=None,
        )
    else:  # ddpm
        policy = DiffusionLowdimPolicy(
            obs_dim=17,
            action_dim=ACTION_DIM_CONT,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            scheduler_type="ddpm",
            num_train_timesteps=100,
            num_inference_steps=100,
        )

    policy.load_state_dict(model_state)
    policy.to(device)
    policy.eval()
    return policy


def get_action_bfn(policy, obs_history, device):
    with torch.no_grad():
        obs_input = {'state': obs_history}
        result = policy.predict_action(obs_input)
        action = result['action'][0, 0].cpu().numpy()
    k = int(np.round(action[0]).clip(0, NUM_DISCRETE - 1))
    params = action[1:3]  # 2 continuous params
    return {"k": k, "x_k": params}


def get_action_continuous(policy, obs_history, device):
    with torch.no_grad():
        obs_input = {'state': obs_history}
        result = policy.predict_action(obs_input)
        action = result['action'][0, 0].cpu().numpy()
    one_hot = action[:NUM_DISCRETE]
    k = int(np.argmax(one_hot).clip(0, NUM_DISCRETE - 1))
    params = action[NUM_DISCRETE:NUM_DISCRETE + 2]
    return {"k": k, "x_k": params}


def evaluate_policy(policy, policy_type: str, n_episodes: int = 50, device: str = "cuda"):
    env = HardGoalEnv()

    if policy_type in ["bfn", "bfn10"]:
        get_action = get_action_bfn
    else:
        get_action = get_action_continuous

    rewards = []
    successes = []
    inference_times = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        obs_np = np.array(obs, dtype=np.float32)
        obs_hist = np.stack([obs_np, obs_np], axis=0)
        obs_history = torch.from_numpy(obs_hist).unsqueeze(0).to(device)

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < env.max_steps:
            t0 = time.time()
            action = get_action(policy, obs_history, device)
            inference_times.append((time.time() - t0) * 1000)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

            obs_np = np.array(obs, dtype=np.float32)
            obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(0).to(device)
            obs_history = torch.cat([obs_history[:, 1:], obs_tensor], dim=1)

        rewards.append(total_reward)
        successes.append(float(total_reward >= 40))  # 40+ = scored goal

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: reward={total_reward:.2f}, success={successes[-1]}", flush=True)

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
        ("bfn", args.bfn_ckpt, "BFN (20 steps)"),
        ("bfn10", args.bfn10_ckpt, "BFN-10 (10 steps)"),
        ("ddpm", args.ddpm_ckpt, "DDPM (100 steps)"),
        ("ddim", args.ddim_ckpt, "DDIM (10 steps)"),
        ("edm", args.edm_ckpt, "EDM (40 steps)"),
        ("consistency", args.consistency_ckpt, "Consistency (1 step)"),
        ("consistency3", args.consistency3_ckpt, "Consistency-3"),
    ]

    for ptype, ckpt, label in eval_specs:
        if not ckpt:
            continue
        print(f"\n{'='*60}\nEvaluating {label}\n{'='*60}")
        print(f"Checkpoint: {ckpt}")
        policy = load_policy(ckpt, ptype, args.device)
        results[ptype] = evaluate_policy(policy, ptype, args.n_episodes, args.device)
        r = results[ptype]
        print(f"\n{label} Results:")
        print(f"  Success Rate:    {r['success_rate']:.1f}%")
        print(f"  Mean Reward:     {r['mean_reward']:.2f} ± {r['std_reward']:.2f}")
        print(f"  Inference Time:  {r['mean_inference_ms']:.1f} ± {r['std_inference_ms']:.1f} ms")

    if len(results) >= 2:
        print(f"\n{'='*70}")
        print(f"Comparison Summary (Hard Goal, 11 actions)")
        print(f"{'='*70}")
        print(f"{'Method':<20} {'Success%':<12} {'Reward':<18} {'Inference (ms)':<15}")
        print(f"{'-'*70}")
        labels_map = {
            "bfn": "BFN (20)", "bfn10": "BFN-10",
            "ddpm": "DDPM (100)", "ddim": "DDIM (10)",
            "consistency": "Consistency (1)", "consistency3": "Consistency-3",
        }
        for ptype in ["bfn", "bfn10", "ddpm", "ddim", "consistency", "consistency3"]:
            if ptype in results:
                r = results[ptype]
                print(f"{labels_map[ptype]:<20} {r['success_rate']:<12.1f} "
                      f"{r['mean_reward']:>7.2f} ± {r['std_reward']:<7.2f} "
                      f"{r['mean_inference_ms']:<15.1f}")


if __name__ == "__main__":
    main()
