"""Diffusion Policy for Low-Dimensional Observations.

This module implements Diffusion Policy (DDPM/DDIM) for environments with
low-dimensional state observations (no images).

For hybrid action spaces, discrete actions are encoded as one-hot vectors
and concatenated with continuous parameters.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.common.pytorch_util import dict_apply

from policies.base import BasePolicy

__all__ = ["DiffusionLowdimPolicy"]


class DiffusionLowdimPolicy(BasePolicy):
    """Diffusion Policy for low-dimensional observations.

    For Lunar Lander with continuous encoding:
    - Observation: 8D state vector
    - Action: 6D (4D one-hot discrete + 2D continuous params)

    Args:
        obs_dim: Dimension of observation space
        action_dim: Dimension of action space
        horizon: Prediction horizon
        n_action_steps: Number of action steps to execute
        n_obs_steps: Number of observation steps for conditioning
        num_train_timesteps: Number of diffusion timesteps for training
        num_inference_steps: Number of diffusion timesteps for inference
        scheduler_type: "ddpm" or "ddim"
    """

    def __init__(
        self,
        obs_dim: int = 8,
        action_dim: int = 6,  # 4 one-hot + 2 continuous
        horizon: int = 16,
        n_action_steps: int = 8,
        n_obs_steps: int = 2,
        # Diffusion config
        num_train_timesteps: int = 100,
        num_inference_steps: int = 100,
        scheduler_type: str = "ddpm",  # "ddpm" or "ddim"
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        # Network config
        obs_encoder_dims: tuple = (256, 256),
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        # Base policy args
        device: str = "cpu",
        dtype: str = "float32",
        clip_actions: bool = True,
        **kwargs,
    ):
        super().__init__(
            action_space=None,
            device=device,
            dtype=dtype,
            clip_actions=clip_actions,
        )

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.scheduler_type = scheduler_type

        # Build observation encoder (MLP)
        encoder_layers = []
        in_dim = obs_dim * n_obs_steps
        for hidden_dim in obs_encoder_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        self.obs_encoder = nn.Sequential(*encoder_layers)
        obs_feature_dim = obs_encoder_dims[-1]
        global_cond_dim = obs_feature_dim

        # Build U-Net
        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        # Build noise scheduler
        if scheduler_type == "ddpm":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule=beta_schedule,
                clip_sample=clip_sample,
                prediction_type=prediction_type,
            )
        elif scheduler_type == "ddim":
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule=beta_schedule,
                clip_sample=clip_sample,
                prediction_type=prediction_type,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        self.normalizer = LinearNormalizer()

        print(f"Diffusion Lowdim Policy initialized:")
        print(f"  Scheduler: {scheduler_type.upper()}")
        print(f"  Obs dim: {obs_dim} x {n_obs_steps} obs steps")
        print(f"  Action dim: {action_dim}")
        print(f"  Train timesteps: {num_train_timesteps}")
        print(f"  Inference timesteps: {num_inference_steps}")
        print(f"  U-Net params: {sum(p.numel() for p in self.model.parameters()):.2e}")

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set the normalizer."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        *,
        deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for inference."""
        if isinstance(obs, torch.Tensor):
            obs_dict = {'obs': {'state': obs}}
        else:
            obs_dict = obs

        result = self.predict_action(obs_dict)
        return result['action']

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Predict actions using diffusion sampling."""
        # Normalize observations
        nobs = self.normalizer.normalize(obs_dict)

        # Get state observations
        if 'obs' in nobs and isinstance(nobs['obs'], dict):
            state = nobs['obs']['state']
        elif 'state' in nobs:
            state = nobs['state']
        else:
            state = nobs['obs']

        B = state.shape[0]
        device = state.device
        dtype = state.dtype

        # Encode observations
        obs_flat = state[:, :self.n_obs_steps].reshape(B, -1)
        cond = self.obs_encoder(obs_flat)

        # Sample using diffusion
        naction = self._sample_diffusion(B, cond, device, dtype)

        # Extract action steps
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = naction[:, start:end]

        # Unnormalize
        action_unnorm = self.normalizer['action'].unnormalize(action)

        return {
            'action': action_unnorm,
            'action_pred': naction
        }

    @torch.no_grad()
    def _sample_diffusion(
        self,
        batch_size: int,
        cond: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample from diffusion model."""
        # Start from random noise
        sample = torch.randn(
            (batch_size, self.horizon, self.action_dim),
            device=device,
            dtype=dtype
        )

        # Set inference timesteps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        # Iterative denoising
        for t in self.noise_scheduler.timesteps:
            timesteps = torch.full(
                (batch_size,), t, device=device, dtype=torch.long
            )

            # Predict noise
            noise_pred = self.model(
                sample=sample,
                timestep=timesteps,
                global_cond=cond,
            )

            # Denoise step
            sample = self.noise_scheduler.step(
                noise_pred, t, sample
            ).prev_sample

        return sample.clamp(-1.0, 1.0)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute diffusion loss (noise prediction MSE)."""
        # Normalize inputs
        nobs = self.normalizer.normalize(batch['obs'])
        naction = self.normalizer['action'].normalize(batch['action'])

        # Get state observations
        if isinstance(nobs, dict) and 'state' in nobs:
            state = nobs['state']
        elif isinstance(nobs, dict) and 'obs' in nobs:
            obs_inner = nobs['obs']
            state = obs_inner['state'] if isinstance(obs_inner, dict) else obs_inner
        else:
            state = nobs

        B = naction.shape[0]
        device = naction.device

        # Encode observations
        obs_flat = state[:, :self.n_obs_steps].reshape(B, -1)
        cond = self.obs_encoder(obs_flat)

        # Sample noise
        noise = torch.randn_like(naction)

        # Sample timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps,
            (B,), device=device, dtype=torch.long
        )

        # Add noise to actions
        noisy_action = self.noise_scheduler.add_noise(
            naction, noise, timesteps
        )

        # Predict noise
        noise_pred = self.model(
            sample=noisy_action,
            timestep=timesteps,
            global_cond=cond,
        )

        # Compute loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss

    def state_dict(self):
        return {
            'obs_encoder': self.obs_encoder.state_dict(),
            'model': self.model.state_dict(),
            'normalizer': self.normalizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.obs_encoder.load_state_dict(state_dict['obs_encoder'])
        self.model.load_state_dict(state_dict['model'])
        if 'normalizer' in state_dict:
            self.normalizer.load_state_dict(state_dict['normalizer'])

    def set_actions(self, action: torch.Tensor):
        pass

    def reset(self):
        pass
