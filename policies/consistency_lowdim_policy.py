"""Consistency Policy for Low-Dimensional Observations.

This module implements Consistency Policy for environments with low-dimensional
state observations (no images). It uses consistency distillation from a
diffusion teacher for fast 1-step or few-step inference.

Reference: "Consistency Policy: Accelerated Visuomotor Policies via
           Consistency Distillation" (RSS 2024)
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.common.pytorch_util import dict_apply

from policies.base import BasePolicy

logger = logging.getLogger(__name__)

__all__ = ["ConsistencyLowdimPolicy"]


def huber_loss(x: torch.Tensor, y: torch.Tensor, delta: float = 0.0) -> torch.Tensor:
    """Huber loss with configurable delta."""
    if delta <= 0:
        return F.mse_loss(x, y, reduction='mean')

    diff = torch.abs(x - y)
    return torch.where(
        diff < delta,
        0.5 * diff ** 2,
        delta * (diff - 0.5 * delta)
    ).mean()


class ConsistencyScheduler:
    """Simple consistency scheduler based on EDM."""

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        num_train_timesteps: int = 100,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.num_train_timesteps = num_train_timesteps

        # Build sigma schedule
        step_indices = torch.arange(num_train_timesteps)
        sigma = self._sigma_schedule(step_indices, num_train_timesteps)
        self.sigmas = sigma

    def _sigma_schedule(self, step: torch.Tensor, num_steps: int) -> torch.Tensor:
        """Karras sigma schedule."""
        rho = self.rho
        sigma_max_pow = self.sigma_max ** (1 / rho)
        sigma_min_pow = self.sigma_min ** (1 / rho)
        step = step.float()
        t = step / max(num_steps - 1, 1)
        sigma = (sigma_max_pow + t * (sigma_min_pow - sigma_max_pow)) ** rho
        return sigma

    def get_sigma_pair(self, t_idx: int) -> Tuple[float, float]:
        """Get sigma pair for teacher training."""
        sigma_t = self.sigmas[t_idx].item()
        sigma_tp1 = self.sigmas[min(t_idx + 1, len(self.sigmas) - 1)].item()
        return sigma_t, sigma_tp1


class ConsistencyLowdimPolicy(BasePolicy):
    """Consistency Policy for low-dimensional observations.

    For Lunar Lander with continuous encoding:
    - Observation: 8D state vector
    - Action: 6D (4D one-hot discrete + 2D continuous params)

    Uses consistency distillation from diffusion teacher for fast inference.

    Args:
        obs_dim: Dimension of observation space
        action_dim: Dimension of action space
        horizon: Prediction horizon
        n_action_steps: Number of action steps to execute
        n_obs_steps: Number of observation steps for conditioning
        num_train_timesteps: Number of timesteps for training
        num_inference_steps: Number of inference steps (1 or 3)
        sigma_min: Minimum sigma for noise schedule
        sigma_max: Maximum sigma for noise schedule
        teacher_path: Path to pretrained diffusion teacher checkpoint
    """

    def __init__(
        self,
        obs_dim: int = 8,
        action_dim: int = 6,  # 4 one-hot + 2 continuous
        horizon: int = 16,
        n_action_steps: int = 8,
        n_obs_steps: int = 2,
        # Scheduler config
        num_train_timesteps: int = 100,
        num_inference_steps: int = 1,  # 1 for fast, 3 for better quality
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        # Loss config
        ctm_weight: float = 1.0,
        dsm_weight: float = 1.0,
        delta: float = 0.0,  # Huber loss delta
        # Teacher
        teacher_path: Optional[str] = None,
        # Network config
        obs_encoder_dims: tuple = (256, 256),
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        # EMA
        initial_ema_decay: float = 0.9,
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
        self.num_inference_steps = num_inference_steps
        self.teacher_path = teacher_path

        # Loss weights
        self.ctm_weight = ctm_weight
        self.dsm_weight = dsm_weight
        self.delta = delta

        # EMA decay
        self.ema_decay = initial_ema_decay

        # Build scheduler
        self.scheduler = ConsistencyScheduler(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            num_train_timesteps=num_train_timesteps,
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

        # Build student U-Net
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

        # Build EMA model (target network)
        self.ema_model = copy.deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad = False

        # Build teacher model (frozen, loaded from checkpoint)
        self.teacher_model = None
        if teacher_path is not None:
            self._load_teacher(teacher_path, obs_encoder_dims, global_cond_dim,
                              diffusion_step_embed_dim, down_dims, kernel_size,
                              n_groups, cond_predict_scale)

        self.normalizer = LinearNormalizer()

        print(f"Consistency Lowdim Policy initialized:")
        print(f"  Obs dim: {obs_dim} x {n_obs_steps} obs steps")
        print(f"  Action dim: {action_dim}")
        print(f"  Inference steps: {num_inference_steps}")
        print(f"  Teacher loaded: {teacher_path is not None}")
        print(f"  U-Net params: {sum(p.numel() for p in self.model.parameters()):.2e}")

    def _load_teacher(
        self,
        teacher_path: str,
        obs_encoder_dims: tuple,
        global_cond_dim: int,
        diffusion_step_embed_dim: int,
        down_dims: tuple,
        kernel_size: int,
        n_groups: int,
        cond_predict_scale: bool,
    ):
        """Load pretrained teacher model."""
        # Build teacher network
        self.teacher_model = ConditionalUnet1D(
            input_dim=self.action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        # Load checkpoint
        checkpoint = torch.load(teacher_path, map_location='cpu', weights_only=False)
        if 'state_dicts' in checkpoint and 'model' in checkpoint['state_dicts']:
            model_state = checkpoint['state_dicts']['model']
            # The checkpoint structure is: model_state['model'] contains U-Net weights
            if isinstance(model_state, dict) and 'model' in model_state:
                # Nested structure: state_dicts -> model -> model (U-Net)
                self.teacher_model.load_state_dict(model_state['model'])
            else:
                # Flat structure with 'model.' prefix
                state_dict = {}
                for k, v in model_state.items():
                    if k.startswith('model.'):
                        state_dict[k[6:]] = v
                self.teacher_model.load_state_dict(state_dict)
        else:
            self.teacher_model.load_state_dict(checkpoint)

        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        logger.info(f"Loaded teacher from {teacher_path}")

    def update_ema(self):
        """Update EMA model parameters."""
        with torch.no_grad():
            for p_ema, p_model in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                p_ema.data.mul_(self.ema_decay).add_(
                    p_model.data, alpha=1 - self.ema_decay
                )

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
        """Predict actions using consistency sampling."""
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

        # Sample using consistency
        naction = self._sample_consistency(B, cond, device, dtype)

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
    def _sample_consistency(
        self,
        batch_size: int,
        cond: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample using consistency model (1-step or multi-step chaining)."""
        # Start from random noise scaled by sigma_max
        sample = torch.randn(
            (batch_size, self.horizon, self.action_dim),
            device=device, dtype=dtype
        ) * self.sigma_max

        if self.num_inference_steps == 1:
            # Single-step inference (fastest)
            timestep = torch.full(
                (batch_size,), self.sigma_max, device=device, dtype=dtype
            )
            sample = self._consistency_step(sample, timestep, cond, use_ema=True)
        else:
            # Multi-step chaining
            sigmas = self._get_chaining_sigmas(self.num_inference_steps, device)
            for i, sigma in enumerate(sigmas):
                timestep = torch.full((batch_size,), sigma, device=device, dtype=dtype)
                sample = self._consistency_step(sample, timestep, cond, use_ema=True)

                # Add noise for next step (except last)
                if i < len(sigmas) - 1:
                    next_sigma = sigmas[i + 1]
                    noise = torch.randn_like(sample) * next_sigma
                    sample = sample + noise

        return sample.clamp(-1.0, 1.0)

    def _get_chaining_sigmas(
        self, num_steps: int, device: torch.device
    ) -> List[float]:
        """Get sigma values for multi-step chaining."""
        if num_steps == 1:
            return [self.sigma_max]
        elif num_steps == 3:
            # Common 3-step chaining schedule
            return [self.sigma_max, 1.0, 0.1]
        else:
            # Linear spacing in log space
            import numpy as np
            sigmas = np.geomspace(self.sigma_max, self.sigma_min, num_steps)
            return list(sigmas)

    def _consistency_step(
        self,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        cond: torch.Tensor,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Single consistency model forward pass."""
        model = self.ema_model if use_ema else self.model

        # Precondition (skip connection)
        c_skip = 1.0 / (sigma.view(-1, 1, 1) ** 2 + 1)
        c_out = sigma.view(-1, 1, 1) / (sigma.view(-1, 1, 1) ** 2 + 1).sqrt()
        c_in = 1.0 / (sigma.view(-1, 1, 1) ** 2 + 1).sqrt()

        # Forward pass
        # Convert sigma to timestep format expected by U-Net
        timestep = (sigma * 1000).long().clamp(0, 999)

        out = model(
            sample=sample * c_in,
            timestep=timestep,
            global_cond=cond,
        )

        # Apply preconditioning
        denoised = c_skip * sample + c_out * out
        return denoised

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute consistency loss (CTM + DSM)."""
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
        dtype = naction.dtype

        # Encode observations
        obs_flat = state[:, :self.n_obs_steps].reshape(B, -1)
        cond = self.obs_encoder(obs_flat)

        total_loss = 0.0

        # CTM Loss: consistency between student and teacher (or EMA)
        if self.ctm_weight > 0:
            ctm_loss = self._compute_ctm_loss(naction, cond, device, dtype)
            total_loss = total_loss + self.ctm_weight * ctm_loss

        # DSM Loss: denoising score matching
        if self.dsm_weight > 0:
            dsm_loss = self._compute_dsm_loss(naction, cond, device, dtype)
            total_loss = total_loss + self.dsm_weight * dsm_loss

        return total_loss

    def _compute_ctm_loss(
        self,
        action: torch.Tensor,
        cond: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute CTM loss for consistency training."""
        B = action.shape[0]
        num_timesteps = self.scheduler.num_train_timesteps

        # Sample random timestep
        t_idx = torch.randint(0, num_timesteps - 1, (B,), device=device)

        # Get sigma pair
        sigmas = self.scheduler.sigmas.to(device)
        sigma_t = sigmas[t_idx]
        sigma_tp1 = sigmas[torch.clamp(t_idx + 1, max=num_timesteps - 1)]

        # Add noise at sigma_t
        noise = torch.randn_like(action)
        noisy_action = action + noise * sigma_t.view(B, 1, 1)

        # Student prediction at t
        student_out = self._consistency_step(
            noisy_action, sigma_t, cond, use_ema=False
        )

        # Target: EMA prediction (or teacher if available)
        with torch.no_grad():
            if self.teacher_model is not None:
                # One-step denoising with teacher
                teacher_denoised = self._teacher_denoise(
                    noisy_action, sigma_t, sigma_tp1, cond
                )
                target_out = self._consistency_step(
                    teacher_denoised, sigma_tp1, cond, use_ema=True
                )
            else:
                # Self-consistency with EMA
                target_out = self._consistency_step(
                    noisy_action, sigma_t, cond, use_ema=True
                )

        # Huber loss
        loss = huber_loss(student_out, target_out, self.delta)
        return loss

    def _compute_dsm_loss(
        self,
        action: torch.Tensor,
        cond: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute denoising score matching loss."""
        B = action.shape[0]

        # Sample random sigma (log-uniform)
        log_sigma = (
            torch.rand(B, device=device) *
            (math.log(self.sigma_max) - math.log(self.sigma_min)) +
            math.log(self.sigma_min)
        )
        sigma = log_sigma.exp()

        # Add noise
        noise = torch.randn_like(action)
        noisy_action = action + noise * sigma.view(B, 1, 1)

        # Predict denoised
        denoised = self._consistency_step(
            noisy_action, sigma, cond, use_ema=False
        )

        # DSM loss: predict clean action
        loss = huber_loss(denoised, action, self.delta)
        return loss

    def _teacher_denoise(
        self,
        noisy_action: torch.Tensor,
        sigma_t: torch.Tensor,
        sigma_tp1: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """One-step denoising with teacher model."""
        if self.teacher_model is None:
            return noisy_action

        B = noisy_action.shape[0]

        # Precondition
        c_skip = 1.0 / (sigma_t.view(-1, 1, 1) ** 2 + 1)
        c_out = sigma_t.view(-1, 1, 1) / (sigma_t.view(-1, 1, 1) ** 2 + 1).sqrt()
        c_in = 1.0 / (sigma_t.view(-1, 1, 1) ** 2 + 1).sqrt()

        timestep = (sigma_t * 1000).long().clamp(0, 999)

        out = self.teacher_model(
            sample=noisy_action * c_in,
            timestep=timestep,
            global_cond=cond,
        )

        denoised = c_skip * noisy_action + c_out * out

        # Interpolate toward target sigma
        alpha = sigma_tp1 / sigma_t
        noise = torch.randn_like(noisy_action)
        result = alpha.view(-1, 1, 1) * denoised + (1 - alpha.view(-1, 1, 1)) * noise * sigma_tp1.view(-1, 1, 1)

        return result

    def state_dict(self):
        return {
            'obs_encoder': self.obs_encoder.state_dict(),
            'model': self.model.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'normalizer': self.normalizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.obs_encoder.load_state_dict(state_dict['obs_encoder'])
        self.model.load_state_dict(state_dict['model'])
        if 'ema_model' in state_dict:
            self.ema_model.load_state_dict(state_dict['ema_model'])
        if 'normalizer' in state_dict:
            self.normalizer.load_state_dict(state_dict['normalizer'])

    def set_actions(self, action: torch.Tensor):
        pass

    def reset(self):
        pass
