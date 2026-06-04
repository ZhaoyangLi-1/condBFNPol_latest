"""
Hard Move Dataset for BFN/Diffusion Policy training.

This dataset handles the hybrid action space of the Hard Move environment:
- Discrete action (k): Integer in {0, 1, ..., 2^n - 1} (16, 64, or 256 actions)
- Continuous parameter (x_k): Scalar in [-1, 1] (1D per action)

Two action encoding modes are supported:
1. "hybrid" - Native hybrid: discrete as class index, continuous as value
2. "continuous" - All actions as continuous: one-hot discrete + continuous param

The zarr format from collect_hard_move_demos.py:
- data/state: [N, 4]
- data/action_k: [N] (discrete action index)
- data/action_x: [N] (continuous parameter)
- data/reward: [N]
- meta/episode_ends: [num_episodes]
"""

from typing import Dict, Optional
import torch
import numpy as np
import copy
import zarr

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sampler import get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset


class HardMoveReplayBuffer:
    """Replay buffer for Hard Move data stored in zarr format."""

    def __init__(self, zarr_path: str, num_discrete: int):
        self.root = zarr.open(str(zarr_path), mode='r')
        self.data = self.root['data']
        self.num_discrete = num_discrete
        self.max_param_dim = 1

        if 'meta' in self.root and 'episode_ends' in self.root['meta']:
            self.episode_ends = np.array(self.root['meta']['episode_ends'])
        else:
            total_len = len(np.array(self.data['state']))
            self.episode_ends = np.array([total_len])

        self.n_episodes = len(self.episode_ends)
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])

        print(f"HardMoveReplayBuffer:")
        print(f"  Episodes: {self.n_episodes}")
        print(f"  Total steps: {len(self)}")
        print(f"  Discrete actions: {self.num_discrete}")

    def __len__(self):
        return int(self.episode_ends[-1]) if len(self.episode_ends) > 0 else 0

    def get_episode(self, idx: int) -> Dict[str, np.ndarray]:
        start = int(self.episode_starts[idx])
        end = int(self.episode_ends[idx])
        return {
            'state': np.array(self.data['state'][start:end]),
            'action_k': np.array(self.data['action_k'][start:end]),
            'action_x': np.array(self.data['action_x'][start:end]),
        }

    def get_episode_length(self, idx: int) -> int:
        return int(self.episode_ends[idx]) - int(self.episode_starts[idx])


class HardMoveSequenceSampler:
    """Sequence sampler for Hard Move data."""

    def __init__(
        self,
        replay_buffer: HardMoveReplayBuffer,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        episode_mask: Optional[np.ndarray] = None,
    ):
        self.replay_buffer = replay_buffer
        self.sequence_length = sequence_length
        self.pad_before = pad_before
        self.pad_after = pad_after

        if episode_mask is None:
            episode_mask = np.ones(replay_buffer.n_episodes, dtype=bool)
        self.episode_mask = episode_mask

        self.indices = []
        for ep_idx in range(replay_buffer.n_episodes):
            if not episode_mask[ep_idx]:
                continue
            ep_len = replay_buffer.get_episode_length(ep_idx)
            for start in range(ep_len):
                self.indices.append((ep_idx, start))

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx: int) -> Dict[str, np.ndarray]:
        ep_idx, start_idx = self.indices[idx]
        episode = self.replay_buffer.get_episode(ep_idx)
        ep_len = len(episode['state'])

        result = {}
        for key, data in episode.items():
            seq = []
            for i in range(start_idx - self.pad_before,
                          start_idx + self.sequence_length - self.pad_before):
                if i < 0:
                    seq.append(data[0])
                elif i >= ep_len:
                    seq.append(data[-1])
                else:
                    seq.append(data[i])
            result[key] = np.stack(seq, axis=0)

        return result


class HardMoveDataset(BaseLowdimDataset):
    """Hard Move dataset for BFN/Diffusion Policy training."""

    def __init__(
        self,
        zarr_path: str,
        num_discrete: int = 16,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        action_mode: str = "hybrid",  # "hybrid" or "continuous"
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: Optional[int] = None,
    ):
        super().__init__()

        self.zarr_path = zarr_path
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.action_mode = action_mode
        self.num_discrete = num_discrete
        self.max_param_dim = 1

        self.replay_buffer = HardMoveReplayBuffer(zarr_path, num_discrete)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed
        )

        self.train_mask = train_mask
        self.sampler = HardMoveSequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

        if action_mode == "hybrid":
            self.discrete_dim = 1
            self.continuous_dim = self.max_param_dim
            self.action_dim = self.discrete_dim + self.continuous_dim  # 2
        else:
            self.action_dim = self.num_discrete + self.max_param_dim

        print(f"HardMoveDataset initialized:")
        print(f"  Episodes: {self.replay_buffer.n_episodes}")
        print(f"  Train episodes: {train_mask.sum()}")
        print(f"  Action mode: {action_mode}")
        print(f"  Discrete actions: {self.num_discrete}")
        print(f"  Total action dim: {self.action_dim}")

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = HardMoveSequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        all_data = self._get_all_data()
        normalizer = LinearNormalizer()
        normalizer.fit(data=all_data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def _get_all_data(self) -> Dict[str, torch.Tensor]:
        all_obs = []
        all_actions = []
        for ep_idx in range(self.replay_buffer.n_episodes):
            episode = self.replay_buffer.get_episode(ep_idx)
            obs, action = self._process_episode(episode)
            all_obs.append(obs)
            all_actions.append(action)
        return {
            'state': torch.from_numpy(np.concatenate(all_obs, axis=0)),
            'action': torch.from_numpy(np.concatenate(all_actions, axis=0))
        }

    def _process_episode(self, episode: Dict[str, np.ndarray]):
        state = episode['state'].astype(np.float32)
        discrete_k = episode['action_k'].astype(np.int64)
        continuous_x = episode['action_x'].astype(np.float32)

        if continuous_x.ndim == 1:
            continuous_x = continuous_x[:, None]

        if self.action_mode == "hybrid":
            action = np.concatenate([
                discrete_k[:, None].astype(np.float32),
                continuous_x
            ], axis=-1)
        else:
            one_hot = np.zeros((len(discrete_k), self.num_discrete), dtype=np.float32)
            one_hot[np.arange(len(discrete_k)), discrete_k] = 1.0
            action = np.concatenate([one_hot, continuous_x], axis=-1)

        return state, action

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        state = sample['state'].astype(np.float32)
        discrete_k = sample['action_k'].astype(np.int64)
        continuous_x = sample['action_x'].astype(np.float32)

        if continuous_x.ndim == 1:
            continuous_x = continuous_x[:, None]

        if self.action_mode == "hybrid":
            action = np.concatenate([
                discrete_k[:, None].astype(np.float32),
                continuous_x
            ], axis=-1)
        else:
            one_hot = np.zeros((len(discrete_k), self.num_discrete), dtype=np.float32)
            one_hot[np.arange(len(discrete_k)), discrete_k] = 1.0
            action = np.concatenate([one_hot, continuous_x], axis=-1)

        data = {
            'obs': {'state': state},
            'action': action
        }
        return dict_apply(data, torch.from_numpy)

    def get_all_actions(self) -> torch.Tensor:
        all_actions = []
        for ep_idx in range(self.replay_buffer.n_episodes):
            episode = self.replay_buffer.get_episode(ep_idx)
            _, action = self._process_episode(episode)
            all_actions.append(action)
        return torch.from_numpy(np.concatenate(all_actions, axis=0))


class HardMoveHybridDataset(HardMoveDataset):
    """Convenience class for hybrid action mode."""
    def __init__(self, zarr_path: str, **kwargs):
        kwargs['action_mode'] = 'hybrid'
        super().__init__(zarr_path, **kwargs)


class HardMoveContinuousDataset(HardMoveDataset):
    """Convenience class for continuous action mode (one-hot)."""
    def __init__(self, zarr_path: str, **kwargs):
        kwargs['action_mode'] = 'continuous'
        super().__init__(zarr_path, **kwargs)
