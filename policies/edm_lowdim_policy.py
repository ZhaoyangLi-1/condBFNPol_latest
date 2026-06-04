"""EDM (Elucidating Diffusion Models) Policy for Low-Dimensional Observations.

This module implements an EDM-style diffusion policy for environments with
low-dimensional state observations (no images). It serves as the teacher
model for Consistency Policy distillation.

Reference: Karras et al. "Elucidating the Design Space of Diffusion-Based
           Generative Models" (NeurIPS 2022)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.common.pytorch_util import dict_apply

from policies.base import BasePolicy

logger = logging.getLogger(__name__)

__all__ = ["EDMLowdimPolicy"]


def huber_loss(x: torch.Tensor, y: torch.Tensor, delta: float = -1.0) -> torch.Tensor:
    """Huber loss with auto-computed delta."""
    if delta < 0:
        # Auto-compute delta based on data scale
        delta = 0.5 * (x.abs().mean() + y.abs().mean() + 1e-8)

    diff = x - y
    abs_diff = diff.abs()
    return torch.where(
        abs_diff < delta,
        0.5 * diff ** 2,
        delta * (abs_diff - 0.5 * delta)
    ).mean()


class EDMScheduler:
    """EDM noise schedule and sampling."""

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        num_steps: int = 40,
        solver: str = "heun",
        P_mean: float = -1.2,
        P_std: float = 1.2,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.num_steps = num_steps
        self.solver = solver
        self.P_mean = P_mean
        self.P_std = P_std

    def get_sigmas(self, num_steps: Optional[int] = None, device: torch.device = None):
        """Get sigma schedule for sampling."""
        num_steps = num_steps or self.num_steps
        step_indices = torch.arange(num_steps, device=device)

        sigma_max_pow = self.sigma_max ** (1 / self.rho)
        sigma_min_pow = self.sigma_min ** (1 / self.rho)

        t = step_indices / max(num_steps - 1, 1)
        sigmas = (sigma_max_pow + t * (sigma_min_pow - sigma_max_pow)) ** self.rho

        # Append 0 for final step
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])
        return sigmas

    def sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample sigma for training (log-normal)."""
        log_sigma = torch.randn(batch_size, device=device) * self.P_std + self.P_mean
        return log_sigma.exp()


class EDMLowdimPolicy(BasePolicy):
    """EDM Diffusion Policy for low-dimensional observations.

    Serves as teacher model for Consistency Policy distillation.

    Args:
        obs_dim: Dimension of observation space
        action_dim: Dimension of action space
        horizon: Prediction horizon
        n_action_steps: Number of action steps to execute
        n_obs_steps: Number of observation steps for conditioning
        sigma_min: Minimum sigma for noise schedule
        sigma_max: Maximum sigma for noise schedule
        rho: Schedule parameter
        num_inference_steps: Number of sampling steps
        delta: Huber loss delta (-1 for auto)
    """

    def __init__(
        self,
        obs_dim: int = 8,
        action_dim: int = 6,  # 4 one-hot + 2 continuous
        horizon: int = 16,
        n_action_steps: int = 8,
        n_obs_steps: int = 2,
        # EDM scheduler config
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        num_inference_steps: int = 40,
        solver: str = "heun",
        P_mean: float = -1.2,
        P_std: float = 1.2,
        # Loss config
        delta: float = -1.0,  # Auto-compute
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
        self.delta = delta

        # Build scheduler
        self.scheduler = EDMScheduler(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            num_steps=num_inference_steps,
            solver=solver,
            P_mean=P_mean,
            P_std=P_std,
        )
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

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

        self.normalizer = LinearNormalizer()

        print(f"EDM Lowdim Policy initialized:")
        print(f"  Obs dim: {obs_dim} x {n_obs_steps} obs steps")
        print(f"  Action dim: {action_dim}")
        print(f"  Inference steps: {num_inference_steps}")
        print(f"  Sigma range: [{sigma_min}, {sigma_max}]")
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
        """Predict actions using EDM sampling."""
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

        # Sample using EDM
        naction = self._sample_edm(B, cond, device, dtype)

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
    def _sample_edm(
        self,
        batch_size: int,
        cond: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample using EDM with Heun's method."""
        sigmas = self.scheduler.get_sigmas(device=device)

        # Start from noise scaled by sigma_max
        x = torch.randn(
            (batch_size, self.horizon, self.action_dim),
            device=device, dtype=dtype
        ) * sigmas[0]

        # Iterative denoising
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Get denoised estimate
            denoised = self._denoise(x, sigma, cond)

            # Euler step
            d = (x - denoised) / sigma
            x_next = x + d * (sigma_next - sigma)

            # Heun's method (2nd order)
            if self.scheduler.solver == "heun" and sigma_next > 0:
                denoised_next = self._denoise(x_next, sigma_next, cond)
                d_next = (x_next - denoised_next) / sigma_next
                d_avg = 0.5 * (d + d_next)
                x_next = x + d_avg * (sigma_next - sigma)

            x = x_next

        return x.clamp(-1.0, 1.0)

    def _denoise(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Apply denoising with EDM preconditioning."""
        B = x.shape[0]

        # Ensure sigma is properly shaped
        if sigma.dim() == 0:
            sigma = sigma.expand(B)
        sigma = sigma.view(B, 1, 1)

        # EDM preconditioning
        c_skip = 1.0 / (sigma ** 2 + 1)
        c_out = sigma / (sigma ** 2 + 1).sqrt()
        c_in = 1.0 / (sigma ** 2 + 1).sqrt()

        # Convert sigma to timestep
        timestep = (sigma.view(B) * 1000).long().clamp(0, 999)

        # Network forward
        out = self.model(
            sample=x * c_in,
            timestep=timestep,
            global_cond=cond,
        )

        # Apply skip connection
        denoised = c_skip * x + c_out * out
        return denoised

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute EDM denoising loss."""
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

        # Sample sigma (log-normal)
        sigma = self.scheduler.sample_sigma(B, device)
        sigma = sigma.view(B, 1, 1)

        # Add noise
        noise = torch.randn_like(naction)
        noisy_action = naction + noise * sigma

        # Get denoised prediction
        denoised = self._denoise(noisy_action, sigma, cond)

        # EDM loss weight: lambda(sigma) = 1 / (sigma^2 + sigma_data^2)
        # Using sigma_data = 0.5 for normalized data in [-1, 1]
        sigma_data = 0.5
        weight = 1.0 / (sigma ** 2 + sigma_data ** 2)

        # Weighted MSE loss (weight applied correctly outside the norm)
        loss = (weight * (denoised - naction) ** 2).mean()
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
