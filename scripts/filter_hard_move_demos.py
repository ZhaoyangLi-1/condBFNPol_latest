"""Filter Hard Move demonstrations to keep only successful episodes."""

import argparse
import numpy as np
import zarr
from pathlib import Path


def filter_demos(input_path: str, output_path: str = None, success_threshold: float = 0.0):
    """Filter zarr replay buffer to keep only successful episodes.

    Args:
        input_path: Path to input zarr (full demos)
        output_path: Path to output zarr (filtered). If None, uses input_path with _filtered suffix
        success_threshold: Episode return threshold for success (default 0.0)
    """
    # Load full demos
    root = zarr.open(str(input_path), mode='r')
    states = root['data/state'][:]
    actions_k = root['data/action_k'][:]
    actions_x = root['data/action_x'][:]
    rewards = root['data/reward'][:]
    episode_ends = root['meta/episode_ends'][:]

    # Compute episode boundaries
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    # Identify successful episodes
    n_total = len(episode_ends)
    successful_eps = []
    for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
        ep_return = rewards[start:end].sum()
        if ep_return > success_threshold:
            successful_eps.append(i)

    print(f"Total episodes: {n_total}")
    print(f"Successful episodes: {len(successful_eps)} ({len(successful_eps)/n_total*100:.1f}%)")

    # Build filtered arrays
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

    # Set output path
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = input_path_obj.parent / f"{input_path_obj.stem}_filtered.zarr"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save filtered demos
    print(f"\nSaving filtered demos to {output_path}...")
    out_root = zarr.open(str(output_path), mode='w')
    data = out_root.create_group('data')
    data.create_dataset('state', data=filtered_states)
    data.create_dataset('action_k', data=filtered_k)
    data.create_dataset('action_x', data=filtered_x)
    data.create_dataset('reward', data=filtered_rewards)

    meta = out_root.create_group('meta')
    meta.create_dataset('episode_ends', data=filtered_ends)

    # Stats
    print(f"\n{'='*60}")
    print(f"Filter Complete")
    print(f"{'='*60}")
    print(f"Episodes:    {len(successful_eps)}/{n_total} ({len(successful_eps)/n_total*100:.1f}%)")
    print(f"Steps:       {total_steps}/{len(states)} ({total_steps/len(states)*100:.1f}%)")
    print(f"Mean return: {np.array([filtered_rewards[s:e].sum() for s, e in zip(np.concatenate([[0], filtered_ends[:-1]]), filtered_ends)]).mean():.2f}")
    print(f"Mean length: {(filtered_ends - np.concatenate([[0], filtered_ends[:-1]])).mean():.1f} steps")
    print(f"Saved to:    {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input zarr (full demos)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output zarr (filtered)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Episode return threshold for success")
    args = parser.parse_args()

    filter_demos(args.input, args.output, args.threshold)


if __name__ == "__main__":
    main()
