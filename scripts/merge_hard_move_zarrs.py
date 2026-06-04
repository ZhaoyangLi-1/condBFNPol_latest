"""Merge multiple hard_move zarr replay buffers into one."""
import argparse
from pathlib import Path

import numpy as np
import zarr


def merge_zarrs(input_paths, output_path):
    all_states, all_k, all_x, all_rewards, episode_ends = [], [], [], [], []
    step_offset = 0
    total_eps = 0

    for p in input_paths:
        root = zarr.open(str(p), mode='r')
        states = root['data/state'][:]
        ks = root['data/action_k'][:]
        xs = root['data/action_x'][:]
        rs = root['data/reward'][:]
        ends = root['meta/episode_ends'][:]

        all_states.append(states)
        all_k.append(ks)
        all_x.append(xs)
        all_rewards.append(rs)
        episode_ends.append(ends + step_offset)

        step_offset += len(states)
        total_eps += len(ends)
        print(f"  {p}: {len(ends)} episodes, {len(states)} steps")

    states = np.concatenate(all_states, axis=0)
    ks = np.concatenate(all_k, axis=0)
    xs = np.concatenate(all_x, axis=0)
    rs = np.concatenate(all_rewards, axis=0)
    ends = np.concatenate(episode_ends, axis=0)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(out), mode='w')
    data = root.create_group('data')
    data.create_dataset('state', data=states.astype(np.float32))
    data.create_dataset('action_k', data=ks.astype(np.int32))
    data.create_dataset('action_x', data=xs.astype(np.float32))
    data.create_dataset('reward', data=rs.astype(np.float32))
    meta = root.create_group('meta')
    meta.create_dataset('episode_ends', data=ends.astype(np.int64))

    print(f"\nMerged: {total_eps} episodes, {len(states)} steps -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    merge_zarrs(args.inputs, args.output)
