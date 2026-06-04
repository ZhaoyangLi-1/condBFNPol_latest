"""Collect demonstrations from trained DLPA expert on Catch Point environment."""

import argparse
import numpy as np
import torch
import zarr
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
DLPA_PATH = PROJECT_ROOT / "_external" / "DLPA"
sys.path.insert(0, str(DLPA_PATH))

from DLPA import Trainer


def make_dlpa_args():
    class Args:
        env = "simple_catch-v0"
        seed = 0
        max_timesteps = 200000
        eval_freq = 5000
        eval_eposides = 50
        num_updates = 25
        seed_steps = 50
        layers = 64

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

        max_buffer_size = 1e6
        episode_length = 50
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

        model_type = "concat"
        save_dir = "demo_collection"
        visualise = 0
        save_points = 0

    return Args()


def collect_demonstrations(checkpoint_path: str, n_episodes: int = 500,
                            output_path: str = None, verbose: bool = True):
    args = make_dlpa_args()
    print(f"Creating DLPA trainer for Catch Point...")
    trainer = Trainer(args)

    print(f"Loading checkpoint from {checkpoint_path}")
    trainer.model.load_state_dict(torch.load(checkpoint_path, map_location=trainer.args.device))
    trainer.model.eval()
    trainer.model_target.load_state_dict(trainer.model.state_dict())

    if output_path is None:
        output_path = "data/catch_point/replay.zarr"

    all_states = []
    all_k = []
    all_x = []
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

        with torch.no_grad():
            act, act_param = trainer.plan(state, eval_mode=True, t0=True, step=0, local_step=t)
            action = trainer.pad_action(act, act_param)

        while not terminal:
            t += 1
            state, reward, terminal = trainer.act(action, t, pre_state=state)

            episode_k.append(int(act))
            ap = act_param.cpu().numpy().flatten() if hasattr(act_param, 'cpu') else np.array(act_param).flatten()
            episode_x.append(float(ap[0]) if len(ap) > 0 else 0.0)
            episode_rewards.append(float(reward))
            episode_states.append(state.copy())

            total_reward += reward

            if not terminal:
                with torch.no_grad():
                    act, act_param = trainer.plan(state, eval_mode=True, t0=False, step=0, local_step=t)
                    action = trainer.pad_action(act, act_param)

        all_states.extend(episode_states[:-1])
        all_k.extend(episode_k)
        all_x.extend(episode_x)
        all_rewards.extend(episode_rewards)
        total_steps += len(episode_k)
        episode_ends.append(total_steps)
        episode_returns.append(total_reward)

        # Catch Point: success = caught the target (positive reward typically means caught)
        if total_reward > 0:
            success_count += 1

        if verbose and (ep + 1) % 25 == 0:
            recent = episode_returns[-25:]
            print(f"Episode {ep+1}/{n_episodes}: "
                  f"recent_mean={np.mean(recent):.2f}, "
                  f"recent_success={sum(1 for r in recent if r > 0)/len(recent)*100:.1f}%, "
                  f"total_steps={total_steps}")

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
    print(f"Demo Collection Complete (Catch Point)")
    print(f"{'='*60}")
    print(f"Total episodes:  {n_episodes}")
    print(f"Total steps:     {total_steps}")
    print(f"Mean reward:     {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Success rate:    {success_count / n_episodes * 100:.1f}%")
    print(f"Saved to:        {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    collect_demonstrations(checkpoint_path=args.checkpoint, n_episodes=args.n_episodes,
                            output_path=args.output)


if __name__ == "__main__":
    main()
