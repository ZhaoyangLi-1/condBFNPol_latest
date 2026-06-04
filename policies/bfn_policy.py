"""Conditional Bayesian Flow Network (BFN) policy implementation.

This module provides a policy wrapper for Continuous Bayesian Flow Networks.
It handles the interface between the environment (observations) and the
generative model (BFN), managing conditioning, sampling, and training loops.
"""

from __future__ import annotations

import logging
import contextlib
from typing import Any, Callable, Dict, NamedTuple, Optional, Union

import torch
import torch.nn as nn
import einops

from policies.base import BasePolicy
from networks.conditional_bfn import ContinuousBFN
from networks.base import BFNetwork
from utils.bfn_utils import default

# Try importing from local utils first, fallback to diffusion_policy
try:
    from utils.normalizer import LinearNormalizer
except ImportError:
    from diffusion_policy.model.common.normalizer import LinearNormalizer

log = logging.getLogger(__name__)

__all__ = ["BFNPolicy", "BFNConfig"]


class BFNConfig(NamedTuple):
    """Configuration for Bayesian Flow Network hyperparameters."""

    sigma_1: float = 0.001
    n_timesteps: int = 20
    cond_scale: Optional[float] = None
    rescaled_phi: Optional[float] = None
    deterministic_seed: int = 42


class BFNPolicy(BasePolicy):
    """Policy wrapper for Continuous Bayesian Flow Networks with Action Chunking."""

    def __init__(
        self,
        *,
        action_space: Any,
        bfn: Optional[ContinuousBFN] = None,
        network: Optional[BFNetwork] = None,
        action_dim: Optional[int] = None,
        n_action_steps: int = 1,
        obs_encoder: Optional[
            Callable[[Union[torch.Tensor, Dict]], torch.Tensor]
        ] = None,
        cond_from_obs: Optional[
            Callable[[Union[torch.Tensor, Dict]], torch.Tensor]
        ] = None,
        config: Optional[BFNConfig] = None,
        sigma_1: float = 0.001,
        n_timesteps: int = 20,
        cond_scale: Optional[float] = None,
        rescaled_phi: Optional[float] = None,
        deterministic_seed: int = 42,
        device: str = "cpu",
        dtype: str = "float32",
        clip_actions: bool = True,
    ):
        super().__init__(
            action_space=action_space,
            device=device,
            dtype=dtype,
            clip_actions=clip_actions,
        )

        self.config = config or BFNConfig(
            sigma_1=sigma_1,
            n_timesteps=n_timesteps,
            cond_scale=cond_scale,
            rescaled_phi=rescaled_phi,
            deterministic_seed=deterministic_seed,
        )

        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.flat_dim = action_dim * n_action_steps

        self.normalizer = LinearNormalizer()

        if bfn is None:
            if network is None:
                raise ValueError("Must provide either `ContinuousBFN` or `network`.")
            if action_dim is None:
                raise ValueError("`action_dim` is required.")

            bfn = ContinuousBFN(
                dim=self.flat_dim, net=network, device_str=device, dtype_str=dtype
            )

        self.bfn = bfn
        self.obs_encoder = obs_encoder or cond_from_obs

    def _encode_obs(self, obs: Union[torch.Tensor, Dict[str, Any]]) -> torch.Tensor:
        if isinstance(obs, dict):
            obs = {
                k: v.to(device=self.device, dtype=self.dtype)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in obs.items()
            }
        else:
            obs = obs.to(device=self.device, dtype=self.dtype)

        if self.obs_encoder is not None:
            return self.obs_encoder(obs)

        if isinstance(obs, dict):
            raise ValueError("Cannot flatten dict obs. Provide obs_encoder.")
        return obs.view(obs.size(0), -1)

    def _sample_actions(self, cond, *, cond_scale, rescaled_phi) -> torch.Tensor:
        actions_flat = self.bfn.sample(
            n_samples=1,
            sigma_1=self.config.sigma_1,
            n_timesteps=self.config.n_timesteps,
            cond=cond,
            cond_scale=cond_scale,
            rescaled_phi=rescaled_phi,
        )
        return actions_flat.squeeze(1)

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Loads normalizer statistics from a fitted instance."""
        self.normalizer.load_state_dict(normalizer.state_dict())
        log.info(
            f"BFNPolicy normalizer loaded. Params: {len(self.normalizer.params_dict)}"
        )

    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, Any]],
        *,
        deterministic: bool = False,
        cond_scale: Optional[float] = None,
        rescaled_phi: Optional[float] = None,
    ) -> torch.Tensor:
        # 1. Normalize (Handling both Dict and Tensor inputs)
        if len(self.normalizer.params_dict) > 0:
            if isinstance(obs, dict):
                obs = self.normalizer["obs"].normalize(obs)
            else:
                # Try normalizing assuming 'obs' key exists, else fallback to default
                try:
                    obs = self.normalizer["obs"].normalize(obs)
                except (KeyError, AttributeError):
                    # If normalizer has no 'obs' key, it might be a single-field normalizer
                    obs = self.normalizer.normalize(obs)

        cond = self._encode_obs(obs)
        c_scale = default(cond_scale, self.config.cond_scale)
        r_phi = default(rescaled_phi, self.config.rescaled_phi)

        rng_context = contextlib.nullcontext()
        if deterministic:
            if self.device.type == "cuda":
                rng_context = torch.random.fork_rng(devices=[self.device])
            else:
                rng_context = torch.random.fork_rng()

        with rng_context:
            if deterministic:
                torch.manual_seed(self.config.deterministic_seed)
            actions_flat = self._sample_actions(
                cond, cond_scale=c_scale, rescaled_phi=r_phi
            )

        B = actions_flat.shape[0]
        actions = actions_flat.view(B, self.n_action_steps, self.action_dim)

        # Unnormalize
        if len(self.normalizer.params_dict) > 0:
            actions = self.normalizer["action"].unnormalize(actions)

        return actions

    def compute_loss(self, batch: Any) -> torch.Tensor:
        # 1. Unpack Batch into Dict
        # This is the critical fix: Convert list/tuple to dict BEFORE normalizing
        if isinstance(batch, (list, tuple)):
            data = {"obs": batch[0], "action": batch[1]}
        elif isinstance(batch, dict):
            data = batch
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        # 2. Normalize
        if len(self.normalizer.params_dict) > 0:
            data = self.normalizer.normalize(data)
        else:
            # Only log once to avoid spamming
            if not getattr(self, "_warned_norm", False):
                log.warning(
                    "BFNPolicy normalizer is empty! Training on raw data (unstable)."
                )
                self._warned_norm = True

        # 3. Extract normalized tensors
        obs_t = self._to_tensor(data["obs"])
        actions_t = self._to_tensor(data["action"])

        # Flatten [B, T, D] -> [B, T*D] for BFN ingestion
        if actions_t.ndim == 3:
            B, T, D = actions_t.shape
            actions_t = actions_t.reshape(B, -1)

        cond = self._encode_obs(obs_t)

        # 4. Compute Loss
        loss = self.bfn.loss(
            actions_t,
            cond=cond,
            sigma_1=self.config.sigma_1,
            cond_scale=self.config.cond_scale,
            rescaled_phi=self.config.rescaled_phi,
        )

        if loss.dim() > 0:
            loss = loss.mean()

        return loss

    def to(self, *args, **kwargs):
        """Moves policy components to the specified device."""
        if isinstance(self.bfn, nn.Module):
            self.bfn.to(*args, **kwargs)
        elif hasattr(self.bfn, "net") and isinstance(self.bfn.net, nn.Module):
            self.bfn.net.to(*args, **kwargs)
        if self.obs_encoder and isinstance(self.obs_encoder, nn.Module):
            self.obs_encoder.to(*args, **kwargs)

        device = kwargs.get("device", None)
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device = arg
                break
        if device is not None:
            self._device = torch.device(device)

        return self
