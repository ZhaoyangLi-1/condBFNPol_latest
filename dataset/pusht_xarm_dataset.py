"""Dataset for real-robot PushT data (LeRobot-format -> zarr conversion).

Supports:
- Top-only camera (camera_0) or top+side (camera_0 + camera_1)
- Hybrid action format (BFN): [direction_class, distance] -> shape (T, 2)
- One-hot action format (Diffusion): [one_hot(direction, 8), distance] -> shape (T, 9)
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import zarr

from diffusion_policy.common.normalize_util import (
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
    get_identity_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask


NUM_DISCRETE = 8

# 8 push directions — matches operator's gym_pusht/bfn_policy.py (y-up math convention).
# Verified empirically: integrating actions with this convention starting from INIT=[50,50]
# keeps eef_pose inside [0, 512] for >94% of frames (vs ~12% with y-down).
# 0=E, 1=SE, 2=S, 3=SW, 4=W, 5=NW, 6=N, 7=NE (joystick labels; vectors are y-up).
DIR_VECTORS = np.array([
    [1.0, 0.0], [1.0, -1.0], [0.0, -1.0], [-1.0, -1.0],
    [-1.0, 0.0], [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0],
], dtype=np.float32)
DIR_VECTORS[[1, 3, 5, 7]] /= np.sqrt(2)  # unit-norm diagonals

# Initial EEF position in pixel coords (512x512 workspace, image-style origin top-left).
INIT_EEF_POS = np.array([50.0, 50.0], dtype=np.float32)
WORKSPACE_BOUNDS = (0.0, 512.0)


def compute_eef_trajectory(directions: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Integrate (direction, distance) actions starting from INIT_EEF_POS.

    Returns: array of shape (T+1, 2). pos[t] is the EEF position observed at time t
    (i.e. BEFORE action[t] is applied).
    """
    n = len(directions)
    pos = np.zeros((n + 1, 2), dtype=np.float32)
    pos[0] = INIT_EEF_POS
    for t in range(n):
        delta = DIR_VECTORS[int(directions[t])] * float(distances[t])
        pos[t + 1] = np.clip(pos[t] + delta, WORKSPACE_BOUNDS[0], WORKSPACE_BOUNDS[1])
    return pos


def _sequence_indices(episode_ends: np.ndarray, horizon: int, pad_before: int, pad_after: int):
    """Yield (ep_idx, start_in_ep, end_in_ep, global_start, global_end) slices."""
    indices = []
    starts = np.concatenate([[0], episode_ends[:-1]])
    for ep_idx, (s, e) in enumerate(zip(starts, episode_ends)):
        ep_len = e - s
        for t in range(-pad_before, ep_len - horizon + 1 + pad_after):
            start = max(0, t)
            end = min(ep_len, t + horizon)
            indices.append((ep_idx, s + start, s + end, t, t + horizon, ep_len))
    return indices


class PushTXArmDataset(BaseImageDataset):
    """Real-robot PushT dataset with hybrid (BFN) or one-hot (diffusion) action format."""

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        n_obs_steps: int = 2,
        seed: int = 42,
        val_ratio: float = 0.1,
        cameras: List[str] = ("camera_0",),
        action_mode: str = "hybrid",
        max_distance: float = 50.0,
        use_eef_pose: bool = False,
    ):
        super().__init__()
        assert action_mode in ("hybrid", "onehot"), action_mode
        self.use_eef_pose = use_eef_pose
        self._eef_cache: Dict[int, np.ndarray] = {}
        assert all(c in ("camera_0", "camera_1") for c in cameras), cameras

        self.zarr_path = zarr_path
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.cameras = list(cameras)
        self.action_mode = action_mode
        self.max_distance = float(max_distance)

        root = zarr.open(zarr_path, mode="r")
        episode_ends = root["meta/episode_ends"][:]
        n_episodes = len(episode_ends)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_episodes)
        n_val = max(1, int(n_episodes * val_ratio))
        self.val_episodes = set(perm[:n_val].tolist())
        self.train_episodes = set(perm[n_val:].tolist())

        self.episode_ends = episode_ends
        self._all_indices = _sequence_indices(episode_ends, horizon, pad_before, pad_after)
        self._train_indices = [
            i for i in self._all_indices if i[0] in self.train_episodes
        ]
        self._val_indices = [
            i for i in self._all_indices if i[0] in self.val_episodes
        ]

        self._root = None  # lazy open per-process
        self._mode = "train"

        print(f"PushTXArmDataset:")
        print(f"  zarr: {zarr_path}")
        print(f"  episodes: {n_episodes} (train={len(self.train_episodes)}, val={len(self.val_episodes)})")
        print(f"  train sequences: {len(self._train_indices)}, val: {len(self._val_indices)}")
        print(f"  cameras: {cameras}, action_mode: {action_mode}")

    def _open(self):
        if self._root is None:
            self._root = zarr.open(self.zarr_path, mode="r")
        return self._root

    def __len__(self):
        return len(self._train_indices) if self._mode == "train" else len(self._val_indices)

    def get_validation_dataset(self):
        # Return a shallow copy switched to val
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__.update(self.__dict__)
        clone._mode = "val"
        clone._root = None
        return clone

    def _get_episode_eef(self, ep_idx: int) -> np.ndarray:
        """Return the full eef trajectory for episode ep_idx, shape (ep_len, 2).

        Computed once per episode and cached.
        """
        if ep_idx in self._eef_cache:
            return self._eef_cache[ep_idx]
        root = self._open()
        starts = np.concatenate([[0], self.episode_ends[:-1]])
        gs, ge = int(starts[ep_idx]), int(self.episode_ends[ep_idx])
        directions = root["data/action_direction"][gs:ge]
        distances = root["data/action_distance"][gs:ge]
        # compute_eef_trajectory returns (ep_len+1, 2) — pose BEFORE each action + final.
        # We want pose AT each frame (which is the pose BEFORE that frame's action).
        traj = compute_eef_trajectory(directions, distances)[:-1]
        self._eef_cache[ep_idx] = traj
        return traj

    def _build_horizon_window(self, ep_idx, global_start, global_end, t_start, t_end, ep_len):
        """Pad with first/last frame if needed to fill horizon."""
        root = self._open()
        T = self.horizon
        n_actual = global_end - global_start
        before_pad = max(0, -t_start)  # pad with first frame
        after_pad = max(0, t_end - ep_len)  # pad with last frame

        # Image observations
        imgs = {}
        for cam in self.cameras:
            block = root[f"data/{cam}"][global_start:global_end]
            if before_pad:
                block = np.concatenate([np.repeat(block[:1], before_pad, axis=0), block], axis=0)
            if after_pad:
                block = np.concatenate([block, np.repeat(block[-1:], after_pad, axis=0)], axis=0)
            # HWC -> CHW float in [0,1]
            block = block.astype(np.float32) / 255.0
            block = np.transpose(block, (0, 3, 1, 2))
            imgs[cam] = block

        # Actions
        direction = root["data/action_direction"][global_start:global_end]
        distance = root["data/action_distance"][global_start:global_end]
        if before_pad:
            direction = np.concatenate([np.repeat(direction[:1], before_pad), direction])
            distance = np.concatenate([np.repeat(distance[:1], before_pad), distance])
        if after_pad:
            # End-of-episode pad: keep frame replay for images, but zero the action.
            # distance=0 = "no push", regardless of direction class — model learns
            # to stop moving at episode end instead of repeating the last push.
            direction = np.concatenate([direction, np.zeros(after_pad, dtype=direction.dtype)])
            distance = np.concatenate([distance, np.zeros(after_pad, dtype=distance.dtype)])

        if self.action_mode == "hybrid":
            # [T, 2]: discrete class index, continuous distance
            action = np.stack([direction.astype(np.float32), distance.astype(np.float32)], axis=-1)
        else:
            # [T, 9]: one-hot(8) + distance
            onehot = np.zeros((T, NUM_DISCRETE), dtype=np.float32)
            onehot[np.arange(T), direction.astype(np.int64)] = 1.0
            action = np.concatenate([onehot, distance.astype(np.float32)[:, None]], axis=-1)

        # EEF pose (low-dim) — only computed when requested
        eef_pose = None
        if self.use_eef_pose:
            ep_eef = self._get_episode_eef(ep_idx)  # (ep_len, 2)
            ep_starts = np.concatenate([[0], self.episode_ends[:-1]])
            real_start = global_start - int(ep_starts[ep_idx])
            real_end = global_end - int(ep_starts[ep_idx])
            block = ep_eef[real_start:real_end]
            if before_pad:
                block = np.concatenate([np.repeat(block[:1], before_pad, axis=0), block], axis=0)
            if after_pad:
                block = np.concatenate([block, np.repeat(block[-1:], after_pad, axis=0)], axis=0)
            eef_pose = block.astype(np.float32)

        return imgs, action, eef_pose

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        indices = self._train_indices if self._mode == "train" else self._val_indices
        ep_idx, gs, ge, ts, te, ep_len = indices[idx]
        imgs, action, eef_pose = self._build_horizon_window(ep_idx, gs, ge, ts, te, ep_len)

        obs = {cam: torch.from_numpy(imgs[cam]) for cam in self.cameras}
        if eef_pose is not None:
            obs["robot_eef_pose"] = torch.from_numpy(eef_pose)
        # Convention: limit obs to n_obs_steps from the start of the horizon window for conditioning
        obs = {k: v[: self.n_obs_steps] for k, v in obs.items()}
        action_t = torch.from_numpy(action)

        return {"obs": obs, "action": action_t}

    # ==================== Normalizer ====================
    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        """Build a normalizer matching action and image specs."""
        normalizer = LinearNormalizer()
        # Action normalizer: identity for direction (already in {0..7}), range for distance
        # We build a per-dim stat
        root = self._open()
        direction = root["data/action_direction"][:]
        distance = root["data/action_distance"][:]

        if self.action_mode == "hybrid":
            # [2]: identity-like for direction, range for distance
            # Use limits style: stat.min/max per dim
            mins = np.array([0.0, float(distance.min())], dtype=np.float32)
            maxs = np.array([float(NUM_DISCRETE - 1), float(distance.max())], dtype=np.float32)
        else:
            # [9]: one-hot stays in [0,1]; distance in its own range
            mins = np.zeros(NUM_DISCRETE + 1, dtype=np.float32)
            maxs = np.ones(NUM_DISCRETE + 1, dtype=np.float32)
            mins[-1] = float(distance.min())
            maxs[-1] = float(distance.max())

        # Build stat dict expected by get_range_normalizer_from_stat
        action_stat = {
            "min": torch.from_numpy(mins),
            "max": torch.from_numpy(maxs),
        }
        normalizer["action"] = get_range_normalizer_from_stat(action_stat)

        # Image normalizer: scale [0,1] image to whatever the encoder expects
        for cam in self.cameras:
            normalizer[cam] = get_image_range_normalizer()

        # EEF pose normalizer: range over the workspace bounds
        if self.use_eef_pose:
            eef_stat = {
                "min": torch.tensor([WORKSPACE_BOUNDS[0], WORKSPACE_BOUNDS[0]], dtype=torch.float32),
                "max": torch.tensor([WORKSPACE_BOUNDS[1], WORKSPACE_BOUNDS[1]], dtype=torch.float32),
            }
            normalizer["robot_eef_pose"] = get_range_normalizer_from_stat(eef_stat)

        return normalizer
