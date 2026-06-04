"""Filter Hard Goal demonstrations to keep only successful episodes (reward >= 40)."""

import argparse
import numpy as np
import zarr
from pathlib import Path


def filter_demos(input_path: str, output_path: str = None, success_threshold: float = 40.0):
    """Filter zarr replay buffer to keep only successful episodes.

    For Hard Goal, success = reward >= 40 (goal scored).
    """
    root = zarr.open(str(input_path), mode='r')
    states = root['data/state'][:]
    actions_k = root['data/action_k'][:]
    actions_x = root['data/action_x'][:]
    rewards = root['data/reward'][:]
    episode_ends = root['meta/episode_ends'][:]

    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    n_total = len(episode_ends)
    successful_eps = []
    for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
        ep_return = rewards[start:end].sum()
        if ep_return >= success_threshold:
            successful_eps.append(i)

    print(f"Total episodes: {n_total}")
    print(f"Successful episodes: {len(successful_eps)} ({len(successful_eps)/n_total*100:.1f}%)")

    filtered_states = []
    filtered_k = []
    filtered_x = []
    filtered_rewards = []
    filtered_ends = []
    total_steps = 0

    for ep_idx in successful_eps:
        start = episode_starts[ep_idx]
        end = episode_ends[ep_idx]
        ep_len = end - start

        filtered_states.append(states[start:end])
        filtered_k.append(actions_k[start:end])
        filtered_x.append(actions_x[start:end])
        filtered_rewards.append(rewards[start:end])
        total_steps += ep_len
        filtered_ends.append(total_steps)

    filtered_states = np.concatenate(filtered_states, axis=0)
    filtered_k = np.concatenate(filtered_k, axis=0)
    filtered_x = np.concatenate(filtered_x, axis=0)
    filtered_rewards = np.concatenate(filtered_rewards, axis=0)
    filtered_ends = np.array(filtered_ends, dtype=np.int64)

    if output_path is None:
        ip = Path(input_path)
        output_path = ip.parent / f"{ip.stem}_filtered.zarr"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving filtered demos to {output_path}...")
    out_root = zarr.open(str(output_path), mode='w')
    data = out_root.create_group('data')
    data.create_dataset('state', data=filtered_states)
    data.create_dataset('action_k', data=filtered_k)
    data.create_dataset('action_x', data=filtered_x)
    data.create_dataset('reward', data=filtered_rewards)

    meta = out_root.create_group('meta')
    meta.create_dataset('episode_ends', data=filtered_ends)

    print(f"\n{'='*60}")
    print(f"Filter Complete")
    print(f"{'='*60}")
    print(f"Episodes:    {len(successful_eps)}/{n_total} ({len(successful_eps)/n_total*100:.1f}%)")
    print(f"Steps:       {total_steps}/{len(states)} ({total_steps/len(states)*100:.1f}%)")
    starts = np.concatenate([[0], filtered_ends[:-1]])
    rets = np.array([filtered_rewards[s:e].sum() for s, e in zip(starts, filtered_ends)])
    lens = filtered_ends - starts
    print(f"Mean return: {rets.mean():.2f} ± {rets.std():.2f}")
    print(f"Mean length: {lens.mean():.1f} steps")
    print(f"Saved to:    {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=40.0,
                        help="Episode return threshold for success (default 40)")
    args = parser.parse_args()
    filter_demos(args.input, args.output, args.threshold)


if __name__ == "__main__":
    main()
