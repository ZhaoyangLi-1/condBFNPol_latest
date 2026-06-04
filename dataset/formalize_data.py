from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Tuple

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable


TRAJ_PATTERN = re.compile(r"^traj\d+$")


class Dummy(SimpleNamespace):
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["state"] = state


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return Dummy


def find_traj_dirs(root: Path) -> List[Path]:
    trajs = [p for p in root.iterdir() if p.is_dir() and TRAJ_PATTERN.match(p.name)]
    trajs.sort(key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    return trajs


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_policy_out(path: Path):
    with path.open("rb") as f:
        return SafeUnpickler(f).load()


def transform_to_pose(T: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 transform(s) to 6D pose (x, y, z, rotvec).
    T: (..., 4, 4)
    Returns: (..., 6)
    """
    T = np.asarray(T)
    pos = T[..., :3, 3]
    rot = R.from_matrix(T[..., :3, :3]).as_rotvec()
    return np.concatenate([pos, rot], axis=-1)


def compute_pose_vel(T: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """
    Compute 6D pose velocity from transforms and timestamps.
    Returns shape (N, 6) with last velocity repeated.
    """
    T = np.asarray(T)
    timestamps = np.asarray(timestamps)
    n = T.shape[0]
    if n < 2:
        return np.zeros((n, 6), dtype=np.float64)

    pos = T[:, :3, 3]
    dt = np.diff(timestamps)
    # avoid divide by zero
    dt_safe = np.where(dt == 0, np.nan, dt)

    dpos = (pos[1:] - pos[:-1]) / dt_safe[:, None]

    rot_prev = R.from_matrix(T[:-1, :3, :3])
    rot_next = R.from_matrix(T[1:, :3, :3])
    rot_rel = rot_prev.inv() * rot_next
    ang_vel = rot_rel.as_rotvec() / dt_safe[:, None]

    dpos = np.nan_to_num(dpos)
    ang_vel = np.nan_to_num(ang_vel)

    vel = np.concatenate([dpos, ang_vel], axis=1)
    # repeat last velocity to match length
    vel = np.vstack([vel, vel[-1]])
    return vel.astype(np.float64)


def estimate_hz(timestamps: np.ndarray) -> float:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.size < 2:
        return float("nan")
    diffs = np.diff(timestamps)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return float("nan")
    dt = np.median(diffs)
    if dt <= 0 or not np.isfinite(dt):
        return float("nan")
    return 1.0 / dt


def make_downsample_indices(length: int, factor: int) -> np.ndarray:
    if length <= 0:
        return np.zeros((0,), dtype=np.int64)
    if factor <= 1:
        return np.arange(length, dtype=np.int64)
    return np.arange(0, length, factor, dtype=np.int64)


def extract_action(policy_out, target_len: int) -> np.ndarray:
    """
    Build action array of shape (target_len, 6).
    Prefer new_robot_transform if available; otherwise use actions[:6].
    """
    if policy_out is None:
        return np.zeros((target_len, 6), dtype=np.float64)

    if isinstance(policy_out, list) and len(policy_out) > 0:
        sample = policy_out[0]
        if isinstance(sample, dict) and "new_robot_transform" in sample:
            Ts = np.stack([p["new_robot_transform"] for p in policy_out], axis=0)
            act = transform_to_pose(Ts)
        elif isinstance(sample, dict) and "actions" in sample:
            act = np.stack([p["actions"][:6] for p in policy_out], axis=0)
        else:
            raise ValueError("policy_out format not recognized")
    else:
        raise ValueError("policy_out format not recognized")

    if act.shape[0] == target_len:
        return act.astype(np.float64)
    if act.shape[0] == target_len - 1:
        act = np.vstack([act, act[-1]])
        return act.astype(np.float64)

    # fallback: truncate or pad with last
    if act.shape[0] > target_len:
        return act[:target_len].astype(np.float64)
    if act.shape[0] < target_len:
        pad = np.repeat(act[-1][None, :], target_len - act.shape[0], axis=0)
        return np.vstack([act, pad]).astype(np.float64)

    return act.astype(np.float64)


def detect_image_ext(img_dir: Path) -> str:
    for ext in ("jpg", "png", "jpeg"):
        if (img_dir / f"im_0.{ext}").exists():
            return ext
    # fallback: infer from any file
    for p in img_dir.iterdir():
        if p.is_file() and p.name.startswith("im_"):
            return p.suffix.lstrip(".")
    raise FileNotFoundError(f"No images found in {img_dir}")


def make_video_from_images(
    img_dir: Path,
    out_path: Path,
    fps: int,
    frame_stride: int = 1,
    input_fps: float | None = None,
) -> None:
    ext = detect_image_ext(img_dir)
    pattern = str(img_dir / f"im_%d.{ext}")
    src_fps = fps
    if frame_stride > 1:
        if input_fps is not None and np.isfinite(input_fps):
            src_fps = input_fps
        else:
            src_fps = fps * frame_stride
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(src_fps),
        "-start_number",
        "0",
        "-i",
        pattern,
    ]
    if frame_stride > 1:
        vf = f"select='not(mod(n\\,{frame_stride}))',setpts=N*{frame_stride}/FRAME_RATE/TB"
        cmd += ["-vf", vf, "-r", str(fps)]
    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "22",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def build_videos(
    traj_dirs: List[Path],
    out_videos_dir: Path,
    fps: int,
    num_videos: int | None,
    duplicate_with_symlink: bool,
    frame_stride: int = 1,
    input_fps: float | None = None,
) -> None:
    for ep_idx, traj_dir in enumerate(tqdm(traj_dirs, desc="Videos", unit="traj")):
        ep_out = out_videos_dir / str(ep_idx)
        ep_out.mkdir(parents=True, exist_ok=True)

        img_dirs = sorted(
            [p for p in traj_dir.iterdir() if p.is_dir() and p.name.startswith("images")],
            key=lambda p: int(re.findall(r"\d+", p.name)[0]) if re.findall(r"\d+", p.name) else 0,
        )

        for cam_idx, img_dir in enumerate(img_dirs):
            out_path = ep_out / f"{cam_idx}.mp4"
            if out_path.exists():
                continue
            make_video_from_images(
                img_dir,
                out_path,
                fps,
                frame_stride=frame_stride,
                input_fps=input_fps,
            )

        # If fewer cameras than requested, duplicate camera 0 (or last available)
        if num_videos is not None and len(img_dirs) < num_videos:
            if len(img_dirs) == 0:
                raise RuntimeError(f"No image directories found in {traj_dir}")
            src_cam = 0
            src_path = ep_out / f"{src_cam}.mp4"
            for cam_idx in range(len(img_dirs), num_videos):
                dst_path = ep_out / f"{cam_idx}.mp4"
                if dst_path.exists():
                    continue
                if duplicate_with_symlink:
                    os.symlink(src_path.name, dst_path)
                else:
                    shutil.copyfile(src_path, dst_path)


def parse_default_output(src_root: Path) -> Path:
    metadata_path = src_root / "collection_metadata.json"
    date_tag = "custom"
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text())
            date_time = meta.get("date_time", "").strip("_")
            date_tag = date_time.split("_")[0].replace("-", "") or date_tag
        except Exception:
            pass
    # default to sibling pusht_real folder
    return Path("$PROJECT_ROOT/data/pusht_real_raw") / f"real_pusht_{date_tag}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Formalize BFN pusht_real data to diffusion-policy format")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("$PROJECT_ROOT/data/pusht_real_raw"),
        help="Source BFN pusht_real directory",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Output directory (defaults based on collection_metadata.json)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if exists")
    parser.add_argument("--no-videos", action="store_true", help="Skip video generation")
    parser.add_argument("--fps", type=int, default=None, help="Video FPS (default: match data rate after downsampling)")
    parser.add_argument(
        "--num-videos",
        type=int,
        default=None,
        help="Number of video files per episode (duplicate if fewer cameras). Default: use available cameras.",
    )
    parser.add_argument(
        "--no-symlink",
        action="store_true",
        help="Duplicate videos by copying instead of symlink",
    )
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=None,
        help="Zarr chunk length (time dimension). Defaults to max episode length.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Downsample factor (keep every Nth frame).",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=None,
        help="Target sampling rate in Hz. Overrides --downsample by computing a factor from timestamps (first trajectory).",
    )
    parser.add_argument(
        "--check-existing-hz",
        action="store_true",
        help="If output exists, report its estimated Hz and exit.",
    )

    args = parser.parse_args()
    if args.downsample < 1:
        raise ValueError("--downsample must be >= 1")
    if args.target_hz is not None and args.target_hz <= 0:
        raise ValueError("--target-hz must be > 0")
    if args.target_hz is not None and args.downsample != 1:
        raise ValueError("Use either --target-hz or --downsample, not both")

    src_root: Path = args.src
    if args.dst is None:
        dst_root = parse_default_output(src_root)
    else:
        dst_root = args.dst

    if dst_root.exists():
        replay_path = dst_root / "replay_buffer.zarr"
        existing_hz = None
        if replay_path.exists():
            try:
                root = zarr.open(str(replay_path), mode="r")
                existing_hz = estimate_hz(root["data"]["timestamp"][:])
            except Exception:
                existing_hz = None
        if args.check_existing_hz:
            msg = f"Existing output: {dst_root}"
            if existing_hz is not None and np.isfinite(existing_hz):
                msg += f" (estimated {existing_hz:.2f} Hz)"
            print(msg)
            return
        if args.overwrite:
            shutil.rmtree(dst_root)
        else:
            hz_msg = ""
            if existing_hz is not None and np.isfinite(existing_hz):
                hz_msg = f" (estimated {existing_hz:.2f} Hz)"
            raise FileExistsError(f"Output directory exists: {dst_root}{hz_msg}")

    traj_dirs = find_traj_dirs(src_root)
    if not traj_dirs:
        raise FileNotFoundError(f"No traj directories found under {src_root}")

    all_timestamp = []
    all_joint = []
    all_joint_vel = []
    all_eef_pose = []
    all_eef_pose_vel = []
    all_stage = []
    all_action = []
    episode_ends = []

    total = 0
    resolved_downsample = None
    resolved_src_hz = None
    resolved_eff_hz = None

    for traj_dir in tqdm(traj_dirs, desc="Trajs", unit="traj"):
        obs_path = traj_dir / "obs_dict.pkl"
        pol_path = traj_dir / "policy_out.pkl"

        obs = load_pickle(obs_path)
        policy_out = load_policy_out(pol_path) if pol_path.exists() else None

        timestamps_full = np.asarray(obs["time_stamp"], dtype=np.float64)
        qpos = np.asarray(obs["qpos"], dtype=np.float64)
        qvel = np.asarray(obs["qvel"], dtype=np.float64)
        stage = np.asarray(obs.get("task_stage", np.zeros(len(timestamps_full))), dtype=np.int64)
        if stage.ndim == 0:
            stage = np.full(len(timestamps_full), int(stage), dtype=np.int64)
        eef_T = np.asarray(obs["eef_transform"], dtype=np.float64)

        if len(timestamps_full) != len(eef_T):
            raise ValueError(
                f"Length mismatch in {traj_dir}: timestamps {len(timestamps_full)} vs eef_transform {len(eef_T)}"
            )
        if len(qpos) != len(timestamps_full) or len(qvel) != len(timestamps_full) or len(stage) != len(timestamps_full):
            raise ValueError(
                f"Length mismatch in {traj_dir}: "
                f"timestamps {len(timestamps_full)} qpos {len(qpos)} qvel {len(qvel)} stage {len(stage)}"
            )

        if resolved_src_hz is None:
            resolved_src_hz = estimate_hz(timestamps_full)

        if resolved_downsample is None:
            if args.target_hz is not None:
                if resolved_src_hz is None or not np.isfinite(resolved_src_hz):
                    raise ValueError("Unable to estimate source Hz for --target-hz")
                resolved_downsample = max(1, int(round(resolved_src_hz / args.target_hz)))
                resolved_eff_hz = resolved_src_hz / resolved_downsample
                if resolved_downsample == 1:
                    print(
                        f"Target Hz {args.target_hz:.2f} ~ source {resolved_src_hz:.2f} Hz, no downsampling needed."
                    )
                else:
                    print(
                        f"Target Hz {args.target_hz:.2f} -> downsample {resolved_downsample} "
                        f"(source {resolved_src_hz:.2f} Hz -> {resolved_eff_hz:.2f} Hz)"
                    )
            else:
                resolved_downsample = args.downsample
                if resolved_src_hz is not None and np.isfinite(resolved_src_hz):
                    resolved_eff_hz = resolved_src_hz / resolved_downsample
                    if resolved_downsample > 1:
                        print(
                            f"Downsample {resolved_downsample}x "
                            f"(source {resolved_src_hz:.2f} Hz -> {resolved_eff_hz:.2f} Hz)"
                        )

        action_full = extract_action(policy_out, target_len=len(timestamps_full))
        idx = make_downsample_indices(len(timestamps_full), resolved_downsample)
        timestamps = timestamps_full[idx]
        qpos = qpos[idx]
        qvel = qvel[idx]
        stage = stage[idx]
        eef_T = eef_T[idx]
        action = action_full[idx]

        eef_pose = transform_to_pose(eef_T)
        eef_pose_vel = compute_pose_vel(eef_T, timestamps)

        all_timestamp.append(timestamps)
        all_joint.append(qpos)
        all_joint_vel.append(qvel)
        all_eef_pose.append(eef_pose)
        all_eef_pose_vel.append(eef_pose_vel)
        all_stage.append(stage)
        all_action.append(action)

        total += len(timestamps)
        episode_ends.append(total)

    # concatenate
    timestamp = np.concatenate(all_timestamp, axis=0)
    robot_joint = np.concatenate(all_joint, axis=0)
    robot_joint_vel = np.concatenate(all_joint_vel, axis=0)
    robot_eef_pose = np.concatenate(all_eef_pose, axis=0)
    robot_eef_pose_vel = np.concatenate(all_eef_pose_vel, axis=0)
    stage = np.concatenate(all_stage, axis=0)
    action = np.concatenate(all_action, axis=0)

    # create zarr
    replay_path = dst_root / "replay_buffer.zarr"
    replay_path.parent.mkdir(parents=True, exist_ok=True)

    compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=2)
    max_ep_len = int(np.max(np.diff([0] + episode_ends)))
    chunk_len = args.chunk_len or max_ep_len
    chunk_len = int(max(1, min(chunk_len, len(timestamp))))

    root = zarr.open(str(replay_path), mode="w")
    data_grp = root.create_group("data")
    meta_grp = root.create_group("meta")

    data_grp.create_dataset("timestamp", data=timestamp, chunks=(chunk_len,), compressor=compressor)
    data_grp.create_dataset("robot_joint", data=robot_joint, chunks=(chunk_len, 6), compressor=compressor)
    data_grp.create_dataset("robot_joint_vel", data=robot_joint_vel, chunks=(chunk_len, 6), compressor=compressor)
    data_grp.create_dataset("robot_eef_pose", data=robot_eef_pose, chunks=(chunk_len, 6), compressor=compressor)
    data_grp.create_dataset("robot_eef_pose_vel", data=robot_eef_pose_vel, chunks=(chunk_len, 6), compressor=compressor)
    data_grp.create_dataset("stage", data=stage, chunks=(chunk_len,), compressor=compressor)
    data_grp.create_dataset("action", data=action, chunks=(chunk_len, 6), compressor=compressor)

    meta_grp.create_dataset(
        "episode_ends",
        data=np.asarray(episode_ends, dtype=np.int64),
        chunks=(len(episode_ends),),
        compressor=None,
    )

    # videos
    if not args.no_videos:
        video_fps = args.fps
        if video_fps is None:
            if resolved_eff_hz is not None and np.isfinite(resolved_eff_hz):
                video_fps = int(round(resolved_eff_hz))
            elif resolved_src_hz is not None and np.isfinite(resolved_src_hz):
                video_fps = int(round(resolved_src_hz))
            else:
                video_fps = 30
        videos_dir = dst_root / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        build_videos(
            traj_dirs,
            videos_dir,
            fps=video_fps,
            num_videos=args.num_videos,
            duplicate_with_symlink=not args.no_symlink,
            frame_stride=resolved_downsample,
            input_fps=resolved_src_hz,
        )

    print(f"Done. Output written to {dst_root}")


if __name__ == "__main__":
    main()
    # python $PROJECT_ROOT/data/formalize_data.py --src $PROJECT_ROOT/data/pusht_real_raw --dst $PROJECT_ROOT/data/pusht_real_10hz --overwrite --target-hz 10