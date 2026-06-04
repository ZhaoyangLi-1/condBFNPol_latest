"""Convert a LeRobot-format dataset (parquet + mp4) into a zarr replay buffer
with image observations and hybrid actions (discrete direction + continuous distance).
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm


def decode_video(path: Path, n_frames_expected: int = None) -> np.ndarray:
    """Decode all frames of an mp4 into a (T, H, W, 3) uint8 array."""
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    frames = []
    for frame in container.decode(stream):
        img = frame.to_ndarray(format="rgb24")
        frames.append(img)
    container.close()
    arr = np.stack(frames, axis=0)
    if n_frames_expected is not None and len(arr) != n_frames_expected:
        print(f"  WARNING {path.name}: decoded {len(arr)} frames, expected {n_frames_expected}")
    return arr


def convert(src_dir: Path, dst_path: Path):
    src_dir = Path(src_dir)
    dst_path = Path(dst_path)

    info = json.load(open(src_dir / "meta/info.json"))
    print(f"Dataset: {info['total_episodes']} episodes, {info['total_frames']} frames, fps={info['fps']}")

    data_df = pd.read_parquet(src_dir / "data/chunk-000/file-000.parquet")
    print(f"Loaded data parquet: {len(data_df)} rows")

    print("Decoding cam0 video...")
    cam0 = decode_video(src_dir / "videos/observation.images.cam0/chunk-000/file-000.mp4",
                        n_frames_expected=info["total_frames"])
    print(f"  cam0 shape: {cam0.shape}")

    print("Decoding cam1 video...")
    cam1 = decode_video(src_dir / "videos/observation.images.cam1/chunk-000/file-000.mp4",
                        n_frames_expected=info["total_frames"])
    print(f"  cam1 shape: {cam1.shape}")

    data_df = data_df.sort_values("index").reset_index(drop=True)
    assert len(data_df) == len(cam0) == len(cam1), \
        f"length mismatch: parquet={len(data_df)}, cam0={len(cam0)}, cam1={len(cam1)}"

    action_direction = data_df["action.direction"].to_numpy().astype(np.int64)
    action_distance = data_df["action.distance"].to_numpy().astype(np.float32)
    episode_index = data_df["episode_index"].to_numpy().astype(np.int64)

    # Build episode_ends (cumulative frame counts)
    unique_eps, counts = np.unique(episode_index, return_counts=True)
    sort_idx = np.argsort(unique_eps)
    counts = counts[sort_idx]
    episode_ends = np.cumsum(counts).astype(np.int64)
    print(f"Episodes: {len(unique_eps)}, total frames: {episode_ends[-1]}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing zarr: {dst_path}")
    root = zarr.open(str(dst_path), mode="w")
    data = root.create_group("data")

    chunk_img = (64, cam0.shape[1], cam0.shape[2], 3)
    data.create_dataset("camera_0", data=cam0, chunks=chunk_img,
                        compressor=zarr.Blosc(cname="zstd", clevel=3))
    data.create_dataset("camera_1", data=cam1, chunks=chunk_img,
                        compressor=zarr.Blosc(cname="zstd", clevel=3))
    data.create_dataset("action_direction", data=action_direction, chunks=(1024,))
    data.create_dataset("action_distance", data=action_distance, chunks=(1024,))

    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=episode_ends, chunks=(len(episode_ends),))
    meta.attrs["fps"] = info["fps"]
    meta.attrs["num_discrete_actions"] = 8
    meta.attrs["image_shape"] = list(cam0.shape[1:])

    print(f"Done. Stats: dist range=[{action_distance.min():.3f}, {action_distance.max():.3f}], "
          f"dir unique={sorted(np.unique(action_direction).tolist())}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="LeRobot dataset dir")
    p.add_argument("--dst", required=True, help="Output zarr path")
    args = p.parse_args()
    convert(Path(args.src), Path(args.dst))
