"""Evaluate BFN, DDPM, DDIM, Consistency policies on Hard Move environment."""

import argparse
import sys
import time
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
print("Starting Hard Move eval script...", flush=True)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

# Import policy modules BEFORE environment (which adds DLPA path with conflicting utils.py)
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
from environments.hard_move import HardMoveEnv
print(f"Environment loaded.", flush=True)


def load_policy(checkpoint_path: str, policy_type: str, n_actuators: int, device: str = "cuda"):
    """Load a trained policy from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'state_dicts' in checkpoint:
        model_state = checkpoint['state_dicts'].get('model', {})
    else:
        model_state = checkpoint

    num_discrete = 2 ** n_actuators
    action_dim_continuous = num_discrete + 1  # one-hot + 1 cont

    if policy_type == "bfn":
        policy = BFNHybridLowdimPolicy(
            obs_dim=4,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            num_discrete_actions=num_discrete,
            continuous_param_dim=1,
            sigma_1=0.001,
            beta_1=0.2,
            n_timesteps=20,
        )
    elif policy_type == "bfn10":
        policy = BFNHybridLowdimPolicy(
            obs_dim=4,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            num_discrete_actions=num_discrete,
            continuous_param_dim=1,
            sigma_1=0.001,
            beta_1=0.2,
            n_timesteps=10,
        )
    elif policy_type == "edm":
        policy = EDMLowdimPolicy(
            obs_dim=4,
            action_dim=action_dim_continuous,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            sigma_min=0.002,
            sigma_max=80.0,
            rho=7.0,
            num_inference_steps=40,
            solver="heun",
            P_mean=-1.2,
            P_std=1.2,
            delta=-1.0,
        )
    elif policy_type == "ddim":
        policy = DiffusionLowdimPolicy(
            obs_dim=4,
            action_dim=action_dim_continuous,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            scheduler_type="ddim",
            num_train_timesteps=100,
            num_inference_steps=10,
        )
    elif policy_type == "consistency":
        policy = ConsistencyLowdimPolicy(
            obs_dim=4,
            action_dim=action_dim_continuous,
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
        # Same trained model as consistency, but uses 3 inference steps
        policy = ConsistencyLowdimPolicy(
            obs_dim=4,
            action_dim=action_dim_continuous,
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
            obs_dim=4,
            action_dim=action_dim_continuous,
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


def get_action_bfn(policy, obs_history, num_discrete, device):
    """Get action from BFN policy (hybrid format)."""
    with torch.no_grad():
        obs_input = {'state': obs_history}
        result = policy.predict_action(obs_input)
        action = result['action'][0, 0].cpu().numpy()

    k = int(np.round(action[0]).clip(0, num_discrete - 1))
    x_k = action[1]
    return {"k": k, "x_k": [x_k]}


def get_action_continuous(policy, obs_history, num_discrete, device):
    """Get action from DDPM/DDIM/Consistency (one-hot format)."""
    with torch.no_grad():
        obs_input = {'state': obs_history}
        result = policy.predict_action(obs_input)
        action = result['action'][0, 0].cpu().numpy()

    one_hot = action[:num_discrete]
    k = int(np.argmax(one_hot).clip(0, num_discrete - 1))
    x_k = action[num_discrete]
    return {"k": k, "x_k": [x_k]}


def evaluate_policy(policy, policy_type: str, n_actuators: int, n_episodes: int = 50,
                    device: str = "cuda"):
    """Evaluate a policy on Hard Move."""
    env = HardMoveEnv(n_actuators=n_actuators)
    num_discrete = 2 ** n_actuators

    if policy_type in ["bfn", "bfn10"]:
        get_action = lambda p, o, d: get_action_bfn(p, o, num_discrete, d)
    else:
        get_action = lambda p, o, d: get_action_continuous(p, o, num_discrete, d)

    rewards = []
    successes = []
    lengths = []
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
        successes.append(float(total_reward > 0))
        lengths.append(steps)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: reward={total_reward:.2f}, success={successes[-1]}", flush=True)

    env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes) * 100),
        "mean_length": float(np.mean(lengths)),
        "mean_inference_ms": float(np.mean(inference_times)),
        "std_inference_ms": float(np.std(inference_times)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_actuators", type=int, required=True)
    parser.add_argument("--bfn_ckpt", type=str, default=None)
    parser.add_argument("--bfn10_ckpt", type=str, default=None)
    parser.add_argument("--ddpm_ckpt", type=str, default=None)
    parser.add_argument("--ddim_ckpt", type=str, default=None)
    parser.add_argument("--edm_ckpt", type=str, default=None,
                        help="Evaluate EDM directly (40-step)")
    parser.add_argument("--consistency_ckpt", type=str, default=None)
    parser.add_argument("--consistency3_ckpt", type=str, default=None,
                        help="Same Consistency checkpoint, but uses 3-step inference")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results = {}
    eval_specs = [
        ("bfn", args.bfn_ckpt, "BFN (20 steps)"),
        ("bfn10", args.bfn10_ckpt, "BFN-10 (10 steps)"),
        ("ddpm", args.ddpm_ckpt, "DDPM (100 steps)"),
        ("ddim", args.ddim_ckpt, "DDIM (10 steps, uses DDPM weights)"),
        ("edm", args.edm_ckpt, "EDM (40 steps, teacher)"),
        ("consistency", args.consistency_ckpt, "Consistency (1 step)"),
        ("consistency3", args.consistency3_ckpt, "Consistency-3 (3 steps)"),
    ]

    for ptype, ckpt, label in eval_specs:
        if not ckpt:
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating {label}")
        print(f"{'='*60}")
        print(f"Checkpoint: {ckpt}")

        policy = load_policy(ckpt, ptype, args.n_actuators, args.device)
        results[ptype] = evaluate_policy(policy, ptype, args.n_actuators,
                                          args.n_episodes, args.device)

        r = results[ptype]
        print(f"\n{label} Results:")
        print(f"  Success Rate:    {r['success_rate']:.1f}%")
        print(f"  Mean Reward:     {r['mean_reward']:.2f} ± {r['std_reward']:.2f}")
        print(f"  Inference Time:  {r['mean_inference_ms']:.1f} ± {r['std_inference_ms']:.1f} ms")

    # Comparison summary
    if len(results) >= 2:
        print(f"\n{'='*70}")
        print(f"Comparison Summary (n={args.n_actuators}, {2**args.n_actuators} actions)")
        print(f"{'='*70}")
        print(f"{'Method':<20} {'Success%':<12} {'Reward':<18} {'Inference (ms)':<15}")
        print(f"{'-'*70}")
        labels_map = {
            "bfn": "BFN (20)",
            "bfn10": "BFN-10",
            "ddpm": "DDPM (100)",
            "ddim": "DDIM (10)",
            "consistency": "Consistency (1)",
            "consistency3": "Consistency-3 (3)",
        }
        for ptype in ["bfn", "bfn10", "ddpm", "ddim", "consistency", "consistency3"]:
            if ptype in results:
                r = results[ptype]
                print(f"{labels_map[ptype]:<20} {r['success_rate']:<12.1f} "
                      f"{r['mean_reward']:>7.2f} ± {r['std_reward']:<7.2f} "
                      f"{r['mean_inference_ms']:<15.1f}")


if __name__ == "__main__":
    main()
