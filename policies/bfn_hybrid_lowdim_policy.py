"""BFN Policy with Native Hybrid Action Space Support for Low-Dim Observations.

This module implements BFN-Policy for environments with low-dimensional state
observations (no images) and hybrid discrete-continuous action spaces.

Designed for Lunar Lander environment:
- Observation: 8D state vector
- Action: 4 discrete actions + 2D continuous parameters
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.common.pytorch_util import dict_apply

from policies.base import BasePolicy
from networks.base import BFNetwork

__all__ = ["BFNHybridLowdimPolicy"]


class HybridUnetWrapper(BFNetwork):
    """U-Net wrapper for hybrid continuous-discrete action spaces (low-dim obs)."""

    def __init__(
        self,
        model: ConditionalUnet1D,
        horizon: int,
        continuous_dim: int,
        discrete_configs: List[Tuple[int, int]],  # [(dim_idx, n_classes), ...]
        cond_dim: int,
    ):
        super().__init__(is_conditional_model=True)
        self.model = model
        self.horizon = horizon
        self.continuous_dim = continuous_dim
        self.discrete_configs = discrete_configs
        self.cond_dim = cond_dim
        self.cond_is_discrete = False

        # Total action dimension
        self.action_dim = continuous_dim + len(discrete_configs)

        # Output dimension includes logits for discrete
        self.output_dim = continuous_dim + sum(nc for _, nc in discrete_configs)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass with hybrid input/output."""
        B = x.shape[0]

        input_per_step = self.continuous_dim + sum(nc for _, nc in self.discrete_configs)
        x_reshaped = x.view(B, self.horizon, input_per_step)

        if t.dim() == 0:
            t = t.expand(B)

        timesteps = (1.0 - t) * 999.0

        out = self.model(
            sample=x_reshaped,
            timestep=timesteps,
            global_cond=cond,
        )

        out = out.reshape(B, -1)
        return out


class BFNHybridLowdimPolicy(BasePolicy):
    """BFN Policy with native hybrid action space support for low-dim observations.

    For Lunar Lander:
    - 8D state observations
    - 4 discrete actions (COAST, MAIN_ENGINE, LEFT_BOOST, RIGHT_BOOST)
    - 2D continuous parameters (intensity, duration)

    Args:
        obs_dim: Dimension of observation space (8 for lunar lander)
        horizon: Prediction horizon
        n_action_steps: Number of action steps to execute
        n_obs_steps: Number of observation steps for conditioning
        num_discrete_actions: Number of discrete action choices (4 for lunar lander)
        continuous_param_dim: Dimension of continuous parameters (2 for lunar lander)
        sigma_1: BFN continuous noise schedule parameter
        beta_1: BFN discrete accuracy schedule parameter
        n_timesteps: Number of BFN sampling steps
    """

    def __init__(
        self,
        obs_dim: int = 8,
        horizon: int = 16,
        n_action_steps: int = 8,
        n_obs_steps: int = 2,
        # Hybrid action config
        num_discrete_actions: int = 4,
        continuous_param_dim: int = 2,
        # BFN config
        sigma_1: float = 0.001,
        beta_1: float = 0.2,
        n_timesteps: int = 20,
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

        # Store dimensions
        self.obs_dim = obs_dim
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.continuous_dim = continuous_param_dim
        self.num_discrete_actions = num_discrete_actions

        # Discrete config: single discrete action with num_discrete_actions classes
        # Index 0 is the discrete action dimension
        discrete_configs = [(0, num_discrete_actions)]
        self.discrete_configs = discrete_configs
        self.discrete_action_indices = {0}

        # Total action dim for output
        self.total_action_dim = 1 + continuous_param_dim  # discrete class + continuous

        # BFN parameters
        self.sigma_1 = sigma_1
        self.beta_1 = beta_1
        self.n_timesteps = n_timesteps

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

        # U-Net input/output dimension
        # Input: continuous values + softmax probs for discrete
        unet_dim = continuous_param_dim + num_discrete_actions

        unet_model = ConditionalUnet1D(
            input_dim=unet_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        # Wrap U-Net
        self.unet_wrapper = HybridUnetWrapper(
            model=unet_model,
            horizon=horizon,
            continuous_dim=continuous_param_dim,
            discrete_configs=discrete_configs,
            cond_dim=global_cond_dim,
        )

        self.model = unet_model
        self.normalizer = LinearNormalizer()

        print(f"BFN Hybrid Lowdim Policy initialized:")
        print(f"  Obs dim: {obs_dim} x {n_obs_steps} obs steps")
        print(f"  Discrete actions: {num_discrete_actions}")
        print(f"  Continuous params: {continuous_param_dim}")
        print(f"  U-Net params: {sum(p.numel() for p in self.model.parameters()):.2e}")
        print(f"  Encoder params: {sum(p.numel() for p in self.obs_encoder.parameters()):.2e}")

    # ==================== Normalizer ====================

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set the normalizer."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ==================== Forward ====================

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

    # ==================== Inference ====================

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Predict actions using hybrid BFN sampling."""
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
        T = self.horizon
        device = state.device
        dtype = state.dtype

        # Flatten obs for encoder: [B, n_obs_steps, obs_dim] -> [B, n_obs_steps * obs_dim]
        obs_flat = state[:, :self.n_obs_steps].reshape(B, -1)
        cond = self.obs_encoder(obs_flat)

        # Sample using hybrid BFN
        naction = self._sample_hybrid_bfn(B, T, cond, device, dtype)

        # Extract action steps
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = naction[:, start:end]

        # Unnormalize ONLY continuous params, NOT discrete class
        # action format: [discrete_class, continuous_0, continuous_1]
        # discrete_class comes from argmax and should NOT be unnormalized
        action_unnorm = action.clone()
        # Only unnormalize continuous params (indices 1:)
        if action.shape[-1] > 1:
            # Create a dummy tensor with same shape to unnormalize
            # We need to handle the normalizer which expects full action shape
            full_action_for_unnorm = action.clone()
            full_unnorm = self.normalizer['action'].unnormalize(full_action_for_unnorm)
            # Keep discrete as-is (already 0,1,2,3 from argmax), only take continuous from unnorm
            action_unnorm[:, :, 1:] = full_unnorm[:, :, 1:]
            # Discrete class stays as the raw predicted class index (0, 1, 2, or 3)

        return {
            'action': action_unnorm,
            'action_pred': naction
        }

    @torch.no_grad()
    def _sample_hybrid_bfn(
        self,
        batch_size: int,
        horizon: int,
        cond: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample from hybrid BFN with separate continuous/discrete handling."""
        n_steps = self.n_timesteps
        sigma_1 = self.sigma_1
        beta_1 = self.beta_1

        cont_dim = self.continuous_dim
        disc_configs = self.discrete_configs

        # Initialize beliefs
        mu_cont = torch.zeros(batch_size, horizon, cont_dim, device=device, dtype=dtype)
        rho_cont = 1.0

        theta_list = []
        for _, n_classes in disc_configs:
            theta = torch.full(
                (batch_size, horizon, n_classes),
                1.0 / n_classes,
                device=device, dtype=dtype
            )
            theta_list.append(theta)

        # Iterative refinement
        for i in range(1, n_steps + 1):
            t_val = (i - 1) / n_steps
            t_batch = torch.full((batch_size,), t_val, device=device, dtype=dtype)

            # Build network input
            if len(theta_list) > 0:
                theta_concat = torch.cat(theta_list, dim=-1)
                net_input = torch.cat([mu_cont, theta_concat], dim=-1)
            else:
                net_input = mu_cont

            net_input_flat = net_input.reshape(batch_size, -1)
            out_flat = self.unet_wrapper(net_input_flat, t_batch, cond=cond)
            out = out_flat.reshape(batch_size, horizon, -1)

            # Continuous update
            x_cont_pred = out[:, :, :cont_dim]
            alpha_cont = (sigma_1 ** (-2.0 * i / n_steps)) * (1.0 - sigma_1 ** (2.0 / n_steps))
            sender_std = 1.0 / (alpha_cont ** 0.5 + 1e-8)
            y_cont = x_cont_pred + sender_std * torch.randn_like(x_cont_pred)
            new_rho = rho_cont + alpha_cont
            mu_cont = (rho_cont * mu_cont + alpha_cont * y_cont) / new_rho
            rho_cont = new_rho

            # Discrete update
            alpha_disc = beta_1 * (2 * i - 1) / (n_steps ** 2)
            offset = cont_dim
            new_theta_list = []

            for j, (_, n_classes) in enumerate(disc_configs):
                logits = out[:, :, offset:offset + n_classes]
                probs = torch.softmax(logits, dim=-1)

                probs_flat = probs.reshape(-1, n_classes)
                k_samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
                k_samples = k_samples.reshape(batch_size, horizon)

                e_k = F.one_hot(k_samples, num_classes=n_classes).float()
                y_mean = alpha_disc * (n_classes * e_k - 1)
                y_std = (alpha_disc * n_classes + 1e-8) ** 0.5
                y_disc = y_mean + y_std * torch.randn_like(y_mean)

                log_theta = torch.log(theta_list[j] + 1e-8)
                theta_new = torch.softmax(log_theta + y_disc, dim=-1)
                new_theta_list.append(theta_new)
                offset += n_classes

            theta_list = new_theta_list

        # Final prediction
        t_final = torch.ones(batch_size, device=device, dtype=dtype)

        if len(theta_list) > 0:
            theta_concat = torch.cat(theta_list, dim=-1)
            net_input = torch.cat([mu_cont, theta_concat], dim=-1)
        else:
            net_input = mu_cont

        net_input_flat = net_input.reshape(batch_size, -1)
        out_final = self.unet_wrapper(net_input_flat, t_final, cond=cond)
        out_final = out_final.reshape(batch_size, horizon, -1)

        # Extract final predictions
        x_cont_final = out_final[:, :, :cont_dim].clamp(-1.0, 1.0)

        # Discrete: argmax of final logits
        disc_values = []
        offset = cont_dim
        for j, (_, n_classes) in enumerate(disc_configs):
            logits = out_final[:, :, offset:offset + n_classes]
            class_idx = logits.argmax(dim=-1).float()  # [B, T]
            disc_values.append(class_idx.unsqueeze(-1))
            offset += n_classes

        # Build output action: [discrete_idx, continuous_params]
        if len(disc_values) > 0:
            disc_tensor = torch.cat(disc_values, dim=-1)  # [B, T, 1]
            naction = torch.cat([disc_tensor, x_cont_final], dim=-1)
        else:
            naction = x_cont_final

        return naction

    # ==================== Training ====================

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute hybrid BFN loss."""
        # Normalize observations
        nobs = self.normalizer.normalize(batch['obs'])

        # Get state observations
        if isinstance(nobs, dict) and 'state' in nobs:
            state = nobs['state']
        elif isinstance(nobs, dict) and 'obs' in nobs:
            obs_inner = nobs['obs']
            state = obs_inner['state'] if isinstance(obs_inner, dict) else obs_inner
        else:
            state = nobs

        # For hybrid actions: DON'T normalize discrete class, only normalize continuous
        # action format: [discrete_class (0-3), continuous_params (2D)]
        raw_action = batch['action']
        discrete_k = raw_action[:, :, 0].long()  # [B, T] - use RAW discrete class

        # Normalize only continuous params
        continuous_raw = raw_action[:, :, 1:]  # [B, T, cont_dim]
        # Create full action tensor for normalizer, then extract continuous
        naction = self.normalizer['action'].normalize(raw_action)
        continuous_x = naction[:, :, 1:]  # Normalized continuous params

        B = raw_action.shape[0]
        T = self.horizon
        device = raw_action.device
        dtype = raw_action.dtype

        # Encode observations
        obs_flat = state[:, :self.n_obs_steps].reshape(B, -1)
        cond = self.obs_encoder(obs_flat)

        # Sample time uniformly
        t = torch.rand(B, device=device, dtype=dtype)
        t = t.clamp(min=1e-5, max=1.0 - 1e-5)
        t_expanded = t.view(B, 1, 1)

        # Continuous: Sample noisy mean
        gamma = 1.0 - (self.sigma_1 ** (2.0 * t_expanded))
        var = gamma * (1.0 - gamma)
        std = (var + 1e-8).sqrt()
        mu_cont = gamma * continuous_x + std * torch.randn_like(continuous_x)

        # Discrete: Sample theta
        beta = self.beta_1 * t_expanded.pow(2.0)

        theta_list = []
        disc_targets = []

        for j, (_, n_classes) in enumerate(self.discrete_configs):
            disc_class = discrete_k.clamp(0, n_classes - 1)
            disc_targets.append(disc_class)

            e_x = F.one_hot(disc_class, num_classes=n_classes).float()
            mean = beta * (n_classes * e_x - 1)
            std_disc = (beta * n_classes + 1e-8).sqrt()
            y_samples = mean + std_disc * torch.randn_like(mean)

            theta = torch.softmax(y_samples, dim=-1)
            theta_list.append(theta)

        # Build network input
        if len(theta_list) > 0:
            theta_concat = torch.cat(theta_list, dim=-1)
            net_input = torch.cat([mu_cont, theta_concat], dim=-1)
        else:
            net_input = mu_cont

        net_input_flat = net_input.reshape(B, -1)

        # Network forward
        out_flat = self.unet_wrapper(net_input_flat, t, cond=cond)
        out = out_flat.reshape(B, T, -1)

        # Compute losses
        # Continuous loss: weighted MSE
        x_cont_pred = out[:, :, :self.continuous_dim]
        cont_loss = (gamma * (continuous_x - x_cont_pred).pow(2.0)).mean()

        # Discrete loss: cross-entropy
        disc_loss = 0.0
        offset = self.continuous_dim
        for j, (_, n_classes) in enumerate(self.discrete_configs):
            logits = out[:, :, offset:offset + n_classes]
            target = disc_targets[j]

            logits_flat = logits.reshape(-1, n_classes)
            target_flat = target.reshape(-1)

            disc_loss = disc_loss + F.cross_entropy(logits_flat, target_flat)
            offset += n_classes

        total_loss = cont_loss + disc_loss
        return total_loss

    # ==================== State Dict ====================

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
