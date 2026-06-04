"""Goal Dataset for BFN/Diffusion Policy training.

Goal-v0: 3 discrete actions, variable continuous params (par_size=[2, 1, 1]).

Padded to max=2 continuous params:
- act 0 (KICK_TO): uses 2D params
- act 1 (SHOOT_LEFT): uses 1D, 2nd is padded
- act 2 (SHOOT_RIGHT): uses 1D, 2nd is padded

zarr format:
- data/state: [N, 17]
- data/action_k: [N]
- data/action_x: [N, 2]
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


NUM_DISCRETE = 3
MAX_PARAM_DIM = 2


class GoalReplayBuffer:
    def __init__(self, zarr_path: str):
        self.root = zarr.open(str(zarr_path), mode='r')
        self.data = self.root['data']
        self.num_discrete = NUM_DISCRETE
        self.max_param_dim = MAX_PARAM_DIM

        if 'meta' in self.root and 'episode_ends' in self.root['meta']:
            self.episode_ends = np.array(self.root['meta']['episode_ends'])
        else:
            total_len = len(np.array(self.data['state']))
            self.episode_ends = np.array([total_len])

        self.n_episodes = len(self.episode_ends)
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])
        print(f"GoalReplayBuffer: Episodes={self.n_episodes}, Steps={len(self)}")

    def __len__(self):
        return int(self.episode_ends[-1]) if len(self.episode_ends) > 0 else 0

    def get_episode(self, idx: int):
        start = int(self.episode_starts[idx])
        end = int(self.episode_ends[idx])
        return {
            'state': np.array(self.data['state'][start:end]),
            'action_k': np.array(self.data['action_k'][start:end]),
            'action_x': np.array(self.data['action_x'][start:end]),
        }

    def get_episode_length(self, idx: int):
        return int(self.episode_ends[idx]) - int(self.episode_starts[idx])


class GoalSequenceSampler:
    def __init__(self, replay_buffer, sequence_length, pad_before=0, pad_after=0,
                 episode_mask=None):
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

    def sample_sequence(self, idx: int):
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


class GoalDataset(BaseLowdimDataset):
    def __init__(self, zarr_path: str, horizon: int = 16, pad_before: int = 1,
                 pad_after: int = 7, action_mode: str = "hybrid",
                 seed: int = 42, val_ratio: float = 0.02,
                 max_train_episodes: Optional[int] = None):
        super().__init__()
        self.zarr_path = zarr_path
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.action_mode = action_mode
        self.num_discrete = NUM_DISCRETE
        self.max_param_dim = MAX_PARAM_DIM

        self.replay_buffer = GoalReplayBuffer(zarr_path)
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes,
                                 val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.train_mask = train_mask
        self.sampler = GoalSequenceSampler(
            replay_buffer=self.replay_buffer, sequence_length=horizon,
            pad_before=pad_before, pad_after=pad_after, episode_mask=train_mask,
        )

        if action_mode == "hybrid":
            self.action_dim = 1 + self.max_param_dim  # [k, p1, p2] = 3
        else:
            self.action_dim = self.num_discrete + self.max_param_dim  # [one_hot, p1, p2] = 5

        print(f"GoalDataset: episodes={self.replay_buffer.n_episodes}, "
              f"action_dim={self.action_dim}, action_mode={action_mode}")

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = GoalSequenceSampler(
            replay_buffer=self.replay_buffer, sequence_length=self.horizon,
            pad_before=self.pad_before, pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        all_data = self._get_all_data()
        normalizer = LinearNormalizer()
        normalizer.fit(data=all_data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def _get_all_data(self):
        all_obs, all_actions = [], []
        for ep_idx in range(self.replay_buffer.n_episodes):
            episode = self.replay_buffer.get_episode(ep_idx)
            obs, action = self._process_episode(episode)
            all_obs.append(obs)
            all_actions.append(action)
        return {
            'state': torch.from_numpy(np.concatenate(all_obs, axis=0)),
            'action': torch.from_numpy(np.concatenate(all_actions, axis=0)),
        }

    def _process_episode(self, episode):
        state = episode['state'].astype(np.float32)
        discrete_k = episode['action_k'].astype(np.int64)
        params = episode['action_x'].astype(np.float32)
        if params.ndim == 1:
            params = params[:, None]
        if params.shape[-1] < self.max_param_dim:
            pad = np.zeros((params.shape[0], self.max_param_dim - params.shape[-1]), dtype=np.float32)
            params = np.concatenate([params, pad], axis=-1)

        if self.action_mode == "hybrid":
            action = np.concatenate([discrete_k[:, None].astype(np.float32), params], axis=-1)
        else:
            one_hot = np.zeros((len(discrete_k), self.num_discrete), dtype=np.float32)
            one_hot[np.arange(len(discrete_k)), discrete_k] = 1.0
            action = np.concatenate([one_hot, params], axis=-1)
        return state, action

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int):
        sample = self.sampler.sample_sequence(idx)
        state = sample['state'].astype(np.float32)
        discrete_k = sample['action_k'].astype(np.int64)
        params = sample['action_x'].astype(np.float32)
        if params.ndim == 1:
            params = params[:, None]
        if params.shape[-1] < self.max_param_dim:
            pad = np.zeros((params.shape[0], self.max_param_dim - params.shape[-1]), dtype=np.float32)
            params = np.concatenate([params, pad], axis=-1)

        if self.action_mode == "hybrid":
            action = np.concatenate([discrete_k[:, None].astype(np.float32), params], axis=-1)
        else:
            one_hot = np.zeros((len(discrete_k), self.num_discrete), dtype=np.float32)
            one_hot[np.arange(len(discrete_k)), discrete_k] = 1.0
            action = np.concatenate([one_hot, params], axis=-1)

        data = {'obs': {'state': state}, 'action': action}
        return dict_apply(data, torch.from_numpy)


class GoalHybridDataset(GoalDataset):
    def __init__(self, zarr_path: str, **kwargs):
        kwargs['action_mode'] = 'hybrid'
        super().__init__(zarr_path, **kwargs)


class GoalContinuousDataset(GoalDataset):
    def __init__(self, zarr_path: str, **kwargs):
        kwargs['action_mode'] = 'continuous'
        super().__init__(zarr_path, **kwargs)
