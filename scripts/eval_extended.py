"""Evaluate BFN and DDPM policies on Extended Lunar Lander (16 actions)."""

import argparse
import sys
import time
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
print("Starting eval script...", flush=True)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

print("Imports done, loading environment...", flush=True)
from environments.lunar_lander_extended import ExtendedHybridLunarLander, NUM_DISCRETE
print(f"Environment loaded. NUM_DISCRETE={NUM_DISCRETE}", flush=True)


def load_policy(checkpoint_path: str, policy_type: str, device: str = "cuda"):
    """Load a trained policy from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model state
    if 'state_dicts' in checkpoint:
        model_state = checkpoint['state_dicts'].get('model', {})
    else:
        model_state = checkpoint

    if policy_type == "bfn":
        from policies.bfn_hybrid_lowdim_policy import BFNHybridLowdimPolicy
        policy = BFNHybridLowdimPolicy(
            obs_dim=8,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            num_discrete_actions=16,
            continuous_param_dim=1,
            sigma_1=0.001,
            beta_1=0.2,
            n_timesteps=20,
        )
    elif policy_type == "bfn10":
        # BFN with 10 steps instead of 20 (like DDIM uses 10 instead of 100)
        from policies.bfn_hybrid_lowdim_policy import BFNHybridLowdimPolicy
        policy = BFNHybridLowdimPolicy(
            obs_dim=8,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            num_discrete_actions=16,
            continuous_param_dim=1,
            sigma_1=0.001,
            beta_1=0.2,
            n_timesteps=10,  # 10 steps instead of 20
        )
    elif policy_type == "ddim":
        # DDIM uses same weights as DDPM but with DDIM scheduler and fewer steps
        from policies.diffusion_lowdim_policy import DiffusionLowdimPolicy
        policy = DiffusionLowdimPolicy(
            obs_dim=8,
            action_dim=17,  # 16 one-hot + 1 continuous
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            scheduler_type="ddim",
            num_train_timesteps=100,
            num_inference_steps=10,  # DDIM uses only 10 steps
        )
    else:  # ddpm
        from policies.diffusion_lowdim_policy import DiffusionLowdimPolicy
        policy = DiffusionLowdimPolicy(
            obs_dim=8,
            action_dim=17,  # 16 one-hot + 1 continuous
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            scheduler_type="ddpm",
            num_train_timesteps=100,
            num_inference_steps=100,
        )

    # Load weights - model_state contains {'obs_encoder', 'model', 'normalizer'}
    policy.load_state_dict(model_state)

    policy.to(device)
    policy.eval()
    return policy


def get_action_bfn(policy, obs_history, device):
    """Get action from BFN policy."""
    with torch.no_grad():
        # Pass state directly - normalizer expects 'state' key
        obs_input = {'state': obs_history}
        result = policy.predict_action(obs_input)
        action = result['action'][0, 0].cpu().numpy()

    # BFN format: [discrete_class, continuous_param]
    k = int(np.round(action[0]).clip(0, NUM_DISCRETE - 1))
    x_k = np.array([action[1]])
    return {"k": k, "x_k": x_k}


def get_action_ddpm(policy, obs_history, device):
    """Get action from DDPM policy."""
    with torch.no_grad():
        # Pass state directly - normalizer expects 'state' key
        obs_input = {'state': obs_history}
        result = policy.predict_action(obs_input)
        action = result['action'][0, 0].cpu().numpy()

    # DDPM format: [16 one-hot, continuous_param]
    one_hot = action[:NUM_DISCRETE]
    k = int(np.argmax(one_hot).clip(0, NUM_DISCRETE - 1))
    x_k = np.array([action[NUM_DISCRETE]])
    return {"k": k, "x_k": x_k}


def get_action_ddim(policy, obs_history, device):
    """Get action from DDIM policy (same format as DDPM)."""
    with torch.no_grad():
        obs_input = {'state': obs_history}
        result = policy.predict_action(obs_input)
        action = result['action'][0, 0].cpu().numpy()

    # DDIM uses same one-hot format as DDPM
    one_hot = action[:NUM_DISCRETE]
    k = int(np.argmax(one_hot).clip(0, NUM_DISCRETE - 1))
    x_k = np.array([action[NUM_DISCRETE]])
    return {"k": k, "x_k": x_k}


def evaluate_policy(policy, policy_type: str, n_episodes: int = 50, device: str = "cuda"):
    """Evaluate a policy on Extended Lunar Lander."""
    env = ExtendedHybridLunarLander(use_image_obs=False)

    if policy_type in ["bfn", "bfn10"]:
        get_action = get_action_bfn
    elif policy_type == "ddim":
        get_action = get_action_ddim
    else:
        get_action = get_action_ddpm

    rewards = []
    successes = []
    lengths = []
    inference_times = []
    action_counts = np.zeros(NUM_DISCRETE)

    for ep in range(n_episodes):
        obs, _ = env.reset()

        # Initialize observation history
        obs_np = np.array(obs, dtype=np.float32)
        obs_hist = np.stack([obs_np, obs_np], axis=0)  # [2, 8]
        obs_history = torch.from_numpy(obs_hist).unsqueeze(0).to(device)  # [1, 2, 8]

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 500:
            # Time inference
            t0 = time.time()
            action = get_action(policy, obs_history, device)
            inference_times.append((time.time() - t0) * 1000)  # ms

            # Track action distribution
            action_counts[action["k"]] += 1

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

            # Update observation history
            obs_np = np.array(obs, dtype=np.float32)
            obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(0).to(device)
            obs_history = torch.cat([obs_history[:, 1:], obs_tensor], dim=1)

        rewards.append(total_reward)
        successes.append(float(terminated and total_reward > 0))
        lengths.append(steps)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: reward={total_reward:.1f}, success={successes[-1]}")

    env.close()

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "success_rate": np.mean(successes) * 100,
        "mean_length": np.mean(lengths),
        "mean_inference_ms": np.mean(inference_times),
        "std_inference_ms": np.std(inference_times),
        "action_distribution": action_counts / action_counts.sum(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfn_ckpt", type=str, default=None)
    parser.add_argument("--bfn10_ckpt", type=str, default=None, help="Uses BFN checkpoint with 10 steps instead of 20")
    parser.add_argument("--ddpm_ckpt", type=str, default=None)
    parser.add_argument("--ddim_ckpt", type=str, default=None, help="Uses DDPM checkpoint with DDIM scheduler")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results = {}

    if args.bfn_ckpt:
        print(f"\n{'='*60}")
        print("Evaluating BFN Policy")
        print(f"{'='*60}")
        print(f"Checkpoint: {args.bfn_ckpt}")

        policy = load_policy(args.bfn_ckpt, "bfn", args.device)
        results["bfn"] = evaluate_policy(policy, "bfn", args.n_episodes, args.device)

        print(f"\nBFN Results:")
        print(f"  Success Rate: {results['bfn']['success_rate']:.1f}%")
        print(f"  Mean Reward: {results['bfn']['mean_reward']:.1f} ± {results['bfn']['std_reward']:.1f}")
        print(f"  Inference Time: {results['bfn']['mean_inference_ms']:.1f} ± {results['bfn']['std_inference_ms']:.1f} ms")

    if args.bfn10_ckpt:
        print(f"\n{'='*60}")
        print("Evaluating BFN-10 Policy (10 steps instead of 20)")
        print(f"{'='*60}")
        print(f"Checkpoint: {args.bfn10_ckpt}")

        policy = load_policy(args.bfn10_ckpt, "bfn10", args.device)
        results["bfn10"] = evaluate_policy(policy, "bfn10", args.n_episodes, args.device)

        print(f"\nBFN-10 Results:")
        print(f"  Success Rate: {results['bfn10']['success_rate']:.1f}%")
        print(f"  Mean Reward: {results['bfn10']['mean_reward']:.1f} ± {results['bfn10']['std_reward']:.1f}")
        print(f"  Inference Time: {results['bfn10']['mean_inference_ms']:.1f} ± {results['bfn10']['std_inference_ms']:.1f} ms")

    if args.ddpm_ckpt:
        print(f"\n{'='*60}")
        print("Evaluating DDPM Policy")
        print(f"{'='*60}")
        print(f"Checkpoint: {args.ddpm_ckpt}")

        policy = load_policy(args.ddpm_ckpt, "ddpm", args.device)
        results["ddpm"] = evaluate_policy(policy, "ddpm", args.n_episodes, args.device)

        print(f"\nDDPM Results:")
        print(f"  Success Rate: {results['ddpm']['success_rate']:.1f}%")
        print(f"  Mean Reward: {results['ddpm']['mean_reward']:.1f} ± {results['ddpm']['std_reward']:.1f}")
        print(f"  Inference Time: {results['ddpm']['mean_inference_ms']:.1f} ± {results['ddpm']['std_inference_ms']:.1f} ms")

    if args.ddim_ckpt:
        print(f"\n{'='*60}")
        print("Evaluating DDIM Policy (DDPM weights + DDIM scheduler)")
        print(f"{'='*60}")
        print(f"Checkpoint: {args.ddim_ckpt}")
        print(f"Using 10 inference steps (vs DDPM's 100)")

        policy = load_policy(args.ddim_ckpt, "ddim", args.device)
        results["ddim"] = evaluate_policy(policy, "ddim", args.n_episodes, args.device)

        print(f"\nDDIM Results:")
        print(f"  Success Rate: {results['ddim']['success_rate']:.1f}%")
        print(f"  Mean Reward: {results['ddim']['mean_reward']:.1f} ± {results['ddim']['std_reward']:.1f}")
        print(f"  Inference Time: {results['ddim']['mean_inference_ms']:.1f} ± {results['ddim']['std_inference_ms']:.1f} ms")

    # Comparison
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print("Comparison Summary")
        print(f"{'='*60}")
        print(f"{'Method':<10} {'Success%':<12} {'Reward':<15} {'Inference (ms)':<15}")
        print(f"{'-'*52}")
        for name in ["bfn", "bfn10", "ddpm", "ddim"]:
            if name in results:
                label = name.upper() if name != "bfn10" else "BFN-10"
                print(f"{label:<10} {results[name]['success_rate']:<12.1f} {results[name]['mean_reward']:<15.1f} {results[name]['mean_inference_ms']:<15.1f}")

        # Speedup comparisons
        if "bfn" in results and "ddpm" in results:
            speedup = results['ddpm']['mean_inference_ms'] / results['bfn']['mean_inference_ms']
            print(f"\nBFN-20 is {speedup:.1f}x faster than DDPM")
        if "bfn10" in results and "ddpm" in results:
            speedup = results['ddpm']['mean_inference_ms'] / results['bfn10']['mean_inference_ms']
            print(f"BFN-10 is {speedup:.1f}x faster than DDPM")
        if "ddim" in results and "ddpm" in results:
            speedup = results['ddpm']['mean_inference_ms'] / results['ddim']['mean_inference_ms']
            print(f"DDIM is {speedup:.1f}x faster than DDPM")
        if "bfn10" in results and "ddim" in results:
            speedup = results['ddim']['mean_inference_ms'] / results['bfn10']['mean_inference_ms']
            print(f"BFN-10 vs DDIM: BFN-10 is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")


if __name__ == "__main__":
    main()
