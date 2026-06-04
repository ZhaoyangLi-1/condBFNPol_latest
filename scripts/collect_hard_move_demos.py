"""Collect demonstrations from trained DLPA expert on Hard Move environment.

Uses DLPA's full MPPI planning to generate high-quality expert demonstrations.
"""

import argparse
import numpy as np
import torch
import zarr
from pathlib import Path
import sys

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
DLPA_PATH = PROJECT_ROOT / "_external" / "DLPA"
sys.path.insert(0, str(DLPA_PATH))

from DLPA import Trainer
import utils as u


def make_dlpa_args(action_n_dim: int, checkpoint_path: str = None, seed: int = 0):
    """Create args namespace matching DLPA's expected format."""
    n_dim = action_n_dim  # Capture in local var for class closure
    seed_val = seed

    class Args:
        # Environment
        env = "simple_move_4_direction_v1-v0"
        seed = seed_val
        action_n_dim = n_dim

        # Training (not used for inference but required)
        max_timesteps = 100000
        eval_freq = 2000
        eval_eposides = 50
        num_updates = 25
        seed_steps = 50
        layers = 64

        # MPC parameters (key for action quality)
        mpc_horizon = 5
        mpc_gamma = 0.99
        mpc_popsize = 1000
        mpc_num_elites = 100
        mpc_patrical = 1
        mpc_init_mean = 0.
        mpc_init_var = 1.
        mpc_epsilon = 0.001
        mpc_alpha = 0.1
        mpc_max_iters = 1e3

        # Buffer
        max_buffer_size = 1e6
        episode_length = 25
        mixture_coef = 0.05
        min_std = 0.05
        cem_iter = 6
        mpc_temperature = 0.5
        td_lr = 3e-4
        rho = 0.5
        grad_clip_norm = 10
        consistency_coef = 2
        reward_coef = 0.5
        contin_coef = 0.5
        value_coef = 0.1
        per_alpha = 0.6
        per_beta = 0.4
        batch_size = 64

        # Model type
        model_type = "concat"
        save_dir = "demo_collection"
        visualise = 0
        save_points = 0

    args = Args()
    return args


def visualize_episode(states, ep_idx: int, viz_dir: Path, n_actuators: int, episode_reward: float):
    """Save a 2D plot showing agent trajectory and target."""
    import matplotlib.pyplot as plt

    states = np.array(states)
    # State format: [vel_x, vel_y, target_dx, target_dy]
    # Reconstruct agent positions: target_dx = target_x - agent_x
    # We don't know absolute pos, but can show relative
    target_relative = states[:, 2:4]  # relative position to target

    # Agent moved from (rel_x, rel_y) to (0, 0) at target
    # Agent positions relative to start: -target_relative + initial_target_relative
    initial_target = target_relative[0]
    agent_pos = initial_target - target_relative  # agent position in start frame

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(agent_pos[:, 0], agent_pos[:, 1], 'b-o', markersize=4, label='Agent path')
    ax.plot(0, 0, 'g*', markersize=20, label='Start')
    ax.plot(initial_target[0], initial_target[1], 'r*', markersize=20, label='Target')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Episode {ep_idx} (n={n_actuators}): reward={episode_reward:.2f}, steps={len(states)-1}')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    save_path = viz_dir / f'episode_{ep_idx:03d}.png'
    plt.savefig(save_path, dpi=80, bbox_inches='tight')
    plt.close()


def collect_demonstrations(
    n_actuators: int,
    checkpoint_path: str,
    n_episodes: int = 500,
    output_path: str = None,
    verbose: bool = True,
    n_visualizations: int = 10,
    seed: int = 0,
):
    """Collect demonstrations from trained DLPA model using MPPI planning."""

    # Set up args
    args = make_dlpa_args(action_n_dim=n_actuators, seed=seed)

    # Create trainer (this loads env and creates world model)
    print(f"Creating DLPA trainer for n={n_actuators}...")
    trainer = Trainer(args)

    # Load trained checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    trainer.model.load_state_dict(
        torch.load(checkpoint_path, map_location=trainer.args.device)
    )
    trainer.model.eval()
    trainer.model_target.load_state_dict(trainer.model.state_dict())

    # Set output path
    if output_path is None:
        output_path = f"data/hard_move_n{n_actuators}/replay.zarr"

    # Set up visualization directory
    viz_dir = Path(output_path).parent / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    print(f"Visualizations will be saved to: {viz_dir}")

    # Collect data
    all_states = []
    all_k = []  # discrete action index
    all_x = []  # continuous parameter
    all_rewards = []
    episode_ends = []

    total_steps = 0
    success_count = 0
    episode_returns = []

    for ep in range(n_episodes):
        state = trainer.reset()
        episode_states = [state.copy()]
        episode_k = []
        episode_x = []
        episode_rewards = []

        terminal = False
        t = 0
        total_reward = 0.

        # First action
        with torch.no_grad():
            act, act_param = trainer.plan(state, eval_mode=True, t0=True, step=0, local_step=t)
            action = trainer.pad_action(act, act_param)

        while not terminal:
            t += 1
            state, reward, terminal = trainer.act(action, t, pre_state=state)

            # Record (s, a, r) - use action that led to this state
            # action format from pad_action: [[7, accel, k, n_dim]]
            # We want to save k and the continuous param
            # act is the discrete index, act_param is continuous params
            episode_k.append(int(act))
            # Get the continuous param for this action
            if hasattr(act_param, 'cpu'):
                ap = act_param.cpu().numpy().flatten()
            else:
                ap = np.array(act_param).flatten()
            # For Hard Move, each action has 1 continuous param
            episode_x.append(float(ap[int(act)]) if len(ap) > int(act) else float(ap[0]))
            episode_rewards.append(float(reward))
            episode_states.append(state.copy())

            total_reward += reward

            if not terminal:
                with torch.no_grad():
                    act, act_param = trainer.plan(state, eval_mode=True, t0=False, step=0, local_step=t)
                    action = trainer.pad_action(act, act_param)

        # Record episode (exclude final state since action[N] -> state[N+1])
        all_states.extend(episode_states[:-1])
        all_k.extend(episode_k)
        all_x.extend(episode_x)
        all_rewards.extend(episode_rewards)
        total_steps += len(episode_k)
        episode_ends.append(total_steps)
        episode_returns.append(total_reward)

        # Success: total reward indicates reaching target
        if total_reward > 0:
            success_count += 1

        # Save visualization for first n_visualizations episodes
        if ep < n_visualizations:
            visualize_episode(episode_states, ep, viz_dir, n_actuators, total_reward)

        if verbose and (ep + 1) % 25 == 0:
            recent = episode_returns[-25:]
            print(f"Episode {ep+1}/{n_episodes}: "
                  f"recent_mean_reward={np.mean(recent):.2f}, "
                  f"recent_success={sum(1 for r in recent if r > 0)/len(recent)*100:.1f}%, "
                  f"total_steps={total_steps}")

    # Save to zarr
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_path}...")
    root = zarr.open(str(output_path), mode='w')

    data = root.create_group('data')
    data.create_dataset('state', data=np.array(all_states, dtype=np.float32))
    data.create_dataset('action_k', data=np.array(all_k, dtype=np.int32))
    data.create_dataset('action_x', data=np.array(all_x, dtype=np.float32))
    data.create_dataset('reward', data=np.array(all_rewards, dtype=np.float32))

    meta = root.create_group('meta')
    meta.create_dataset('episode_ends', data=np.array(episode_ends, dtype=np.int64))

    print(f"\n{'='*60}")
    print(f"Demonstration Collection Complete (n={n_actuators})")
    print(f"{'='*60}")
    print(f"Total episodes:  {n_episodes}")
    print(f"Total steps:     {total_steps}")
    print(f"Mean reward:     {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Success rate:    {success_count / n_episodes * 100:.1f}%")
    print(f"Saved to:        {output_path}")

    return {
        'n_episodes': n_episodes,
        'total_steps': total_steps,
        'success_rate': success_count / n_episodes * 100,
        'mean_reward': np.mean(episode_returns),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_actuators", type=int, required=True,
                        help="Number of actuators (2^n discrete actions)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained DLPA world_model_*.pth")
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_visualizations", type=int, default=10,
                        help="Number of episodes to save as visualizations")
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    collect_demonstrations(
        n_actuators=args.n_actuators,
        checkpoint_path=args.checkpoint,
        n_episodes=args.n_episodes,
        output_path=args.output,
        n_visualizations=args.n_visualizations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
