"""
Collect demonstration data from trained SAC expert on Extended Lunar Lander.

Saves data in zarr format compatible with BFN/DDPM training.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import zarr
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.lunar_lander_extended import (
    ExtendedHybridLunarLander,
    NUM_DISCRETE,
    MAX_PARAM_DIM,
)
from scripts.train_hybrid_sac_extended import Policy, HybridActionLayout

# Optional video saving
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def load_expert(checkpoint_path: str, device: str = "cuda"):
    """Load trained SAC expert."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get action layout from checkpoint or create from env
    if 'action_layout' in checkpoint:
        al_dict = checkpoint['action_layout']
        # Convert dict to HybridActionLayout dataclass
        action_layout = HybridActionLayout(
            num_discrete=al_dict['num_discrete'],
            param_dims=al_dict['param_dims'],
            branch_slices=al_dict['branch_slices'],
            total_param_dim=al_dict['total_param_dim'],
            max_param_dim=al_dict['max_param_dim'],
            action_names=al_dict['action_names'],
        )
    else:
        env = ExtendedHybridLunarLander(use_image_obs=False)
        action_spec = env.get_action_spec()
        action_layout = HybridActionLayout.from_env_spec(action_spec)
        env.close()

    # Create policy with same architecture as training
    obs_dim = 8  # Lunar Lander state dim
    policy = Policy(
        obs_dim=obs_dim,
        action_layout=action_layout,
        hidden_dims=[256, 256],
        activation="relu",
        weights_init="orthogonal",
        bias_init="zeros",
        log_std_min=-5.0,
        log_std_max=2.0,
    ).to(device)

    # Load weights - checkpoint uses 'policy' key
    policy.load_state_dict(checkpoint['policy'])
    policy.eval()

    print(f"Loaded SAC expert from: {checkpoint_path}")
    if 'best_eval_reward' in checkpoint:
        print(f"  Best eval reward: {checkpoint['best_eval_reward']:.1f}")

    return policy, action_layout


def collect_episode(env, policy, action_layout, device: str = "cuda", render: bool = False):
    """Collect a single episode."""
    obs, _ = env.reset()

    observations = [obs.copy()]
    discrete_actions = []
    continuous_actions = []
    rewards = []
    frames = [] if render else None

    done = False
    total_reward = 0

    while not done:
        if render:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            # Get action from policy
            out = policy.forward(obs_tensor)
            h = out["h"]
            logits_d = out["logits_d"]

            # Sample discrete action (deterministic = argmax)
            d_act = logits_d.argmax(dim=-1).item()

            # Get continuous parameters for chosen action
            dim = action_layout.param_dims[d_act]
            if dim > 0:
                mean = torch.tanh(policy.branch_means[d_act](h))
                c_act = mean.squeeze(0).cpu().numpy()
            else:
                c_act = np.array([0.0])  # Placeholder for COAST

        # Store action
        discrete_actions.append(d_act)
        continuous_actions.append(c_act.copy())

        # Step environment
        env_action = {"k": d_act, "x_k": c_act}
        obs, reward, terminated, truncated, info = env.step(env_action)
        done = terminated or truncated

        observations.append(obs.copy())
        rewards.append(reward)
        total_reward += reward

    success = terminated and total_reward > 0

    result = {
        'observations': np.array(observations[:-1]),  # Exclude final obs
        'discrete_actions': np.array(discrete_actions),
        'continuous_actions': np.array(continuous_actions),
        'rewards': np.array(rewards),
        'total_reward': total_reward,
        'success': success,
        'length': len(rewards),
    }

    if render and frames:
        result['frames'] = frames

    return result


def save_video(frames, path, fps=30):
    """Save frames as video."""
    if not HAS_IMAGEIO:
        print("Warning: imageio not available, skipping video save")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(str(path), frames, fps=fps)
    print(f"  Saved video: {path}")


def create_zarr_dataset(episodes, output_path: str):
    """Create zarr dataset from collected episodes."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute episode boundaries
    episode_ends = []
    current_idx = 0
    for ep in episodes:
        current_idx += ep['length']
        episode_ends.append(current_idx)

    # Concatenate all data
    all_obs = np.concatenate([ep['observations'] for ep in episodes], axis=0)
    all_discrete = np.concatenate([ep['discrete_actions'] for ep in episodes], axis=0)
    all_continuous = np.concatenate([ep['continuous_actions'] for ep in episodes], axis=0)

    # Create hybrid action format: [discrete_class, continuous_params...]
    # For BFN: action_dim = 1 + MAX_PARAM_DIM = 2
    all_actions = np.concatenate([
        all_discrete[:, None].astype(np.float32),
        all_continuous.astype(np.float32)
    ], axis=1)

    print(f"\nDataset statistics:")
    print(f"  Total steps: {len(all_obs)}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Obs shape: {all_obs.shape}")
    print(f"  Action shape: {all_actions.shape}")
    print(f"  Discrete actions: {NUM_DISCRETE}")

    # Action distribution
    action_counts = np.bincount(all_discrete, minlength=NUM_DISCRETE)
    print(f"\n  Action distribution:")
    for k in range(NUM_DISCRETE):
        pct = 100 * action_counts[k] / len(all_discrete)
        print(f"    k={k:2d}: {action_counts[k]:5d} ({pct:5.1f}%)")

    # Create zarr store
    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=store, overwrite=True)

    # Create data group
    data = root.create_group('data')

    # Store arrays
    data.create_dataset('state', data=all_obs.astype(np.float32), chunks=(1000, all_obs.shape[1]))
    data.create_dataset('action', data=all_actions.astype(np.float32), chunks=(1000, all_actions.shape[1]))

    # Also store separate discrete/continuous for flexibility
    data.create_dataset('discrete_action', data=all_discrete.astype(np.int64), chunks=(1000,))
    data.create_dataset('continuous_action', data=all_continuous.astype(np.float32), chunks=(1000, all_continuous.shape[1]))

    # Store metadata
    meta = root.create_group('meta')
    meta.create_dataset('episode_ends', data=np.array(episode_ends, dtype=np.int64))

    # Store episode stats
    rewards = [ep['total_reward'] for ep in episodes]
    successes = [float(ep['success']) for ep in episodes]

    root.attrs['num_episodes'] = len(episodes)
    root.attrs['num_steps'] = len(all_obs)
    root.attrs['num_discrete'] = NUM_DISCRETE
    root.attrs['max_param_dim'] = MAX_PARAM_DIM
    root.attrs['action_dim'] = 1 + MAX_PARAM_DIM
    root.attrs['obs_dim'] = all_obs.shape[1]
    root.attrs['mean_reward'] = float(np.mean(rewards))
    root.attrs['std_reward'] = float(np.std(rewards))
    root.attrs['success_rate'] = float(np.mean(successes))

    print(f"\nSaved to: {output_path}")
    print(f"  Mean reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Success rate: {100*np.mean(successes):.1f}%")

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to SAC checkpoint')
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--output', type=str, default='data/lunar_lander_extended/replay.zarr')
    parser.add_argument('--video_dir', type=str, default='outputs/demo_videos')
    parser.add_argument('--video_every', type=int, default=50, help='Save video every N episodes')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_reward', type=float, default=0, help='Filter episodes below this reward')
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load expert
    policy, action_layout = load_expert(args.checkpoint, device)

    # Create environment (render_mode is set internally to rgb_array)
    env = ExtendedHybridLunarLander(use_image_obs=False)

    # Collect episodes
    print(f"\nCollecting {args.num_episodes} episodes...")
    episodes = []
    rewards = []
    successes = []
    video_count = 0

    pbar = tqdm(total=args.num_episodes, desc="Collecting")

    while len(episodes) < args.num_episodes:
        # Render video periodically
        render = (len(episodes) % args.video_every == 0) and HAS_IMAGEIO

        episode = collect_episode(env, policy, action_layout, device, render=render)

        # Filter by minimum reward if specified
        if episode['total_reward'] >= args.min_reward:
            episodes.append(episode)
            rewards.append(episode['total_reward'])
            successes.append(episode['success'])

            # Save video if rendered
            if render and 'frames' in episode and episode['frames']:
                video_path = Path(args.video_dir) / f"demo_ep{len(episodes):04d}_r{episode['total_reward']:.0f}.mp4"
                save_video(episode['frames'], video_path)
                video_count += 1

            pbar.update(1)

            if len(episodes) % 100 == 0:
                pbar.set_postfix({
                    'reward': f"{np.mean(rewards[-100:]):.1f}",
                    'success': f"{100*np.mean(successes[-100:]):.0f}%"
                })

    pbar.close()

    print(f"\nCollection complete!")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Mean reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Success rate: {100*np.mean(successes):.1f}%")
    print(f"  Videos saved: {video_count}")

    # Create zarr dataset
    create_zarr_dataset(episodes, args.output)

    env.close()


if __name__ == '__main__':
    main()
