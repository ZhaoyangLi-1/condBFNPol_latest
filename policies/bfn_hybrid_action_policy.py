"""BFN Policy with Native Hybrid Action Space Support.

This module implements BFN-Policy with explicit handling of discrete actions
(e.g., gripper open/close) using Dirichlet-Categorical conjugacy, rather than
treating them as continuous values.

Key advantages over Diffusion Policy for hybrid action spaces:
1. Native categorical support - no relaxation/rounding artifacts
2. End-to-end differentiable training via continuous message passing
3. Semantically meaningful discrete uncertainty during inference
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
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.common.robomimic_config_util import get_robomimic_config

from policies.base import BasePolicy
from networks.base import BFNetwork

# Try to import robomimic components
try:
    from robomimic.algo import algo_factory
    from robomimic.algo.algo import PolicyAlgo
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.models.base_nets as rmbn
    import diffusion_policy.model.vision.crop_randomizer as dmvc
    HAS_ROBOMIMIC = True
except ImportError:
    HAS_ROBOMIMIC = False

__all__ = ["BFNHybridActionPolicy"]


class HybridUnetWrapper(BFNetwork):
    """U-Net wrapper for hybrid continuous-discrete action spaces.
    
    The network outputs:
    - continuous_dim values for arm joint predictions
    - n_classes logits for each discrete dimension (gripper)
    """
    
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
        
        # Total action dimension (continuous + discrete)
        self.action_dim = continuous_dim + len(discrete_configs)
        
        # Output dimension includes logits for discrete
        self.output_dim = continuous_dim + sum(nc for _, nc in discrete_configs)
    
    def forward(
        self,
        x: torch.Tensor,  # [B, horizon * (cont_dim + sum(n_classes))]
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass with hybrid input/output."""
        B = x.shape[0]
        
        # For input to U-Net, we need [B, horizon, action_dim]
        # x contains continuous values + softmax probs for discrete
        # Reshape appropriately
        input_per_step = self.continuous_dim + sum(nc for _, nc in self.discrete_configs)
        x_reshaped = x.view(B, self.horizon, input_per_step)
        
        # U-Net processes [B, T, D]
        if t.dim() == 0:
            t = t.expand(B)
        
        timesteps = (1.0 - t) * 999.0
        
        out = self.model(
            sample=x_reshaped,
            timestep=timesteps,
            global_cond=cond,
        )
        
        # Output shape: [B, horizon, output_dim]
        out = out.reshape(B, -1)
        
        return out


class BFNHybridActionPolicy(BasePolicy):
    """BFN Policy with native hybrid action space support.
    
    This policy properly handles:
    - Continuous actions (arm joints): Gaussian-Gaussian conjugacy
    - Discrete actions (gripper): Dirichlet-Categorical conjugacy
    
    For RoboMimic tasks like Lift, Can, Square:
    - 7D continuous arm actions
    - 1D binary gripper (open=0, close=1)
    """

    def __init__(
        self,
        shape_meta: dict,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        # Hybrid action config
        discrete_action_indices: List[int] = None,  # e.g., [7] for gripper at index 7
        discrete_action_classes: List[int] = None,  # e.g., [2] for binary gripper
        # BFN config
        sigma_1: float = 0.001,  # Continuous noise schedule
        beta_1: float = 0.2,     # Discrete accuracy schedule
        n_timesteps: int = 20,
        # Vision encoder config
        crop_shape: tuple = (76, 76),
        obs_encoder_group_norm: bool = False,
        eval_fixed_crop: bool = False,
        # U-Net config
        diffusion_step_embed_dim: int = 256,
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
        
        # Parse action dimensions
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1, "Action must be 1D"
        total_action_dim = action_shape[0]
        
        # Default: last dimension is binary gripper (common in RoboMimic)
        if discrete_action_indices is None:
            discrete_action_indices = [total_action_dim - 1]  # Last dim
        if discrete_action_classes is None:
            discrete_action_classes = [2]  # Binary
        
        assert len(discrete_action_indices) == len(discrete_action_classes)
        
        # Compute continuous dimension
        continuous_dim = total_action_dim - len(discrete_action_indices)
        
        # Build discrete configs: [(dim_idx, n_classes), ...]
        discrete_configs = list(zip(discrete_action_indices, discrete_action_classes))
        
        self.continuous_dim = continuous_dim
        self.discrete_configs = discrete_configs
        self.total_action_dim = total_action_dim
        self.discrete_action_indices = set(discrete_action_indices)
        
        # Parse observation shape meta
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                obs_config['rgb'].append(key)
            elif obs_type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")
        
        # Build observation encoder
        if HAS_ROBOMIMIC:
            obs_encoder = self._build_robomimic_encoder(
                obs_config=obs_config,
                obs_key_shapes=obs_key_shapes,
                action_dim=total_action_dim,
                crop_shape=crop_shape,
                obs_encoder_group_norm=obs_encoder_group_norm,
                eval_fixed_crop=eval_fixed_crop,
            )
        else:
            obs_encoder = self._build_simple_encoder(obs_shape_meta)
        
        # Compute network dimensions
        obs_feature_dim = obs_encoder.output_shape()[0] if HAS_ROBOMIMIC else 512
        global_cond_dim = obs_feature_dim * n_obs_steps
        
        # U-Net input/output dimension for hybrid actions
        # Input: continuous values + softmax probs for discrete
        # Output: continuous values + logits for discrete
        unet_dim = continuous_dim + sum(discrete_action_classes)
        
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
        unet_wrapper = HybridUnetWrapper(
            model=unet_model,
            horizon=horizon,
            continuous_dim=continuous_dim,
            discrete_configs=discrete_configs,
            cond_dim=global_cond_dim,
        )
        
        # Store components
        self.obs_encoder = obs_encoder
        self.model = unet_model
        self.unet_wrapper = unet_wrapper
        self.normalizer = LinearNormalizer()
        
        # Store config
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.sigma_1 = sigma_1
        self.beta_1 = beta_1
        self.n_timesteps = n_timesteps
        self.kwargs = kwargs
        
        print(f"BFN Hybrid Policy initialized:")
        print(f"  Continuous dims: {continuous_dim} (indices 0-{continuous_dim-1})")
        print(f"  Discrete dims: {discrete_configs}")
        print(f"  U-Net params: {sum(p.numel() for p in self.model.parameters()):.2e}")
        print(f"  Vision params: {sum(p.numel() for p in self.obs_encoder.parameters()):.2e}")
    
    def _build_robomimic_encoder(
        self,
        obs_config: dict,
        obs_key_shapes: dict,
        action_dim: int,
        crop_shape: tuple,
        obs_encoder_group_norm: bool,
        eval_fixed_crop: bool,
    ) -> nn.Module:
        """Build observation encoder using robomimic."""
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph'
        )
        
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw
        
        ObsUtils.initialize_obs_utils_with_config(config)
        
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device='cpu',
        )
        
        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features
                )
            )
        
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )
        
        return obs_encoder
    
    def _build_simple_encoder(self, obs_shape_meta: dict) -> nn.Module:
        """Build simple MLP encoder as fallback."""
        total_dim = 0
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            if attr.get('type', 'low_dim') == 'low_dim':
                total_dim += shape[0]
            else:
                total_dim += 512
        
        return nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
    
    # ==================== Normalizer ====================
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set the normalizer."""
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    # ==================== Action Space Utilities ====================
    
    def _split_actions(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split actions into continuous and discrete parts.
        
        Args:
            actions: [B, T, D] full action tensor
            
        Returns:
            continuous: [B, T, D_cont] continuous actions
            discrete: [B, T, N_disc] discrete action indices (as floats)
        """
        B, T, D = actions.shape
        
        # Build index masks
        cont_indices = [i for i in range(D) if i not in self.discrete_action_indices]
        disc_indices = sorted(self.discrete_action_indices)
        
        continuous = actions[:, :, cont_indices]
        discrete = actions[:, :, disc_indices]
        
        return continuous, discrete
    
    def _merge_actions(
        self,
        continuous: torch.Tensor,
        discrete: torch.Tensor,
    ) -> torch.Tensor:
        """Merge continuous and discrete actions back to full action tensor.
        
        Args:
            continuous: [B, T, D_cont] continuous actions
            discrete: [B, T, N_disc] discrete actions (as floats in [0, 1] or class indices)
            
        Returns:
            actions: [B, T, D] full action tensor
        """
        B, T, _ = continuous.shape
        D = self.total_action_dim
        
        actions = torch.zeros(B, T, D, device=continuous.device, dtype=continuous.dtype)
        
        cont_indices = [i for i in range(D) if i not in self.discrete_action_indices]
        disc_indices = sorted(self.discrete_action_indices)
        
        actions[:, :, cont_indices] = continuous
        actions[:, :, disc_indices] = discrete
        
        return actions
    
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
            obs_dict = {'obs': obs}
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
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        
        # Encode observations
        this_nobs = dict_apply(
            nobs, 
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        cond = nobs_features.reshape(B, -1)
        device = cond.device
        dtype = cond.dtype
        
        # Sample using hybrid BFN
        naction = self._sample_hybrid_bfn(B, T, cond, device, dtype)

        # Extract action steps
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = naction[:, start:end]

        # Unnormalize ONLY continuous params, NOT discrete class
        # action format: [discrete_class, continuous_params...]
        # discrete_class comes from argmax and should NOT be unnormalized
        action_unnorm = action.clone()

        # Only unnormalize continuous params (not discrete dimensions)
        # Create full tensor for unnormalization, then copy back continuous only
        full_unnorm = self.normalizer['action'].unnormalize(action)

        # Get continuous indices (all indices that are NOT discrete)
        cont_indices = [i for i in range(action.shape[-1]) if i not in self.discrete_action_indices]
        for idx in cont_indices:
            action_unnorm[:, :, idx] = full_unnorm[:, :, idx]
        # Discrete indices stay as raw class values (0, 1, 2, 3, etc.)

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
        """Sample from hybrid BFN with separate continuous/discrete handling.
        
        This implements the core BFN sampling with:
        - Gaussian-Gaussian conjugacy for continuous actions
        - Dirichlet-Categorical conjugacy for discrete actions
        """
        n_steps = self.n_timesteps
        sigma_1 = self.sigma_1
        beta_1 = self.beta_1
        
        cont_dim = self.continuous_dim
        disc_configs = self.discrete_configs
        
        # Compute dimensions
        total_input_dim = cont_dim + sum(nc for _, nc in disc_configs)
        
        # ============ Initialize Beliefs ============
        # Continuous: mu=0, rho=1 (uninformative Gaussian prior)
        mu_cont = torch.zeros(batch_size, horizon, cont_dim, device=device, dtype=dtype)
        rho_cont = 1.0
        
        # Discrete: uniform theta (1/K for each class)
        theta_list = []
        for _, n_classes in disc_configs:
            theta = torch.full(
                (batch_size, horizon, n_classes), 
                1.0 / n_classes, 
                device=device, dtype=dtype
            )
            theta_list.append(theta)
        
        # ============ Iterative Refinement ============
        for i in range(1, n_steps + 1):
            t_val = (i - 1) / n_steps
            t_batch = torch.full((batch_size,), t_val, device=device, dtype=dtype)
            
            # Build network input: [B, T, cont_dim + sum(n_classes)]
            if len(theta_list) > 0:
                theta_concat = torch.cat(theta_list, dim=-1)  # [B, T, sum(n_classes)]
                net_input = torch.cat([mu_cont, theta_concat], dim=-1)
            else:
                net_input = mu_cont
            
            # Flatten for network: [B, T * input_dim]
            net_input_flat = net_input.reshape(batch_size, -1)
            
            # Network prediction
            out_flat = self.unet_wrapper(net_input_flat, t_batch, cond=cond)
            out = out_flat.reshape(batch_size, horizon, -1)
            
            # ============ Continuous Update ============
            x_cont_pred = out[:, :, :cont_dim]
            
            # Precision increment: alpha = sigma_1^(-2i/n) * (1 - sigma_1^(2/n))
            alpha_cont = (sigma_1 ** (-2.0 * i / n_steps)) * (1.0 - sigma_1 ** (2.0 / n_steps))
            
            # Sender: y ~ N(x_pred, 1/sqrt(alpha))
            sender_std = 1.0 / (alpha_cont ** 0.5 + 1e-8)
            y_cont = x_cont_pred + sender_std * torch.randn_like(x_cont_pred)
            
            # Bayesian update: mu = (rho * mu + alpha * y) / (rho + alpha)
            new_rho = rho_cont + alpha_cont
            mu_cont = (rho_cont * mu_cont + alpha_cont * y_cont) / new_rho
            rho_cont = new_rho
            
            # ============ Discrete Update ============
            # Discrete alpha: beta_1 * (2i - 1) / n^2
            alpha_disc = beta_1 * (2 * i - 1) / (n_steps ** 2)
            
            offset = cont_dim
            new_theta_list = []
            for j, (_, n_classes) in enumerate(disc_configs):
                logits = out[:, :, offset:offset + n_classes]
                probs = torch.softmax(logits, dim=-1)
                
                # Sample k from predicted distribution (per timestep)
                probs_flat = probs.reshape(-1, n_classes)
                k_samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
                k_samples = k_samples.reshape(batch_size, horizon)
                
                # One-hot encode
                e_k = F.one_hot(k_samples, num_classes=n_classes).float()
                
                # Sender: y ~ N(alpha * (K*e_k - 1), sqrt(alpha*K))
                y_mean = alpha_disc * (n_classes * e_k - 1)
                y_std = (alpha_disc * n_classes + 1e-8) ** 0.5
                y_disc = y_mean + y_std * torch.randn_like(y_mean)
                
                # Dirichlet update: theta' = softmax(log(theta) + y)
                log_theta = torch.log(theta_list[j] + 1e-8)
                theta_new = torch.softmax(log_theta + y_disc, dim=-1)
                new_theta_list.append(theta_new)
                
                offset += n_classes
            
            theta_list = new_theta_list
        
        # ============ Final Prediction ============
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

        # Discrete: argmax of final logits to get class index
        disc_values = []
        offset = cont_dim
        for j, (_, n_classes) in enumerate(disc_configs):
            logits = out_final[:, :, offset:offset + n_classes]
            class_idx = logits.argmax(dim=-1).float()  # [B, T] - raw class index
            disc_values.append(class_idx.unsqueeze(-1))
            offset += n_classes

        # Build output action: [discrete_idx, continuous_params]
        # For Lunar Lander: [class (0-3), param_0, param_1]
        if len(disc_values) > 0:
            disc_tensor = torch.cat(disc_values, dim=-1)  # [B, T, N_disc]
            naction = self._merge_actions(x_cont_final, disc_tensor)
        else:
            naction = x_cont_final

        return naction
    
    # ==================== Training ====================
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute hybrid BFN loss.

        - Continuous: Weighted MSE (gamma * ||x - x_pred||^2)
        - Discrete: Cross-entropy loss

        IMPORTANT: For hybrid actions like [discrete_class, param_0, param_1],
        we do NOT normalize the discrete class dimension, only the continuous params.
        """
        # Normalize observations
        nobs = self.normalizer.normalize(batch['obs'])

        # For hybrid actions: DON'T normalize discrete class, only continuous
        raw_action = batch['action']
        naction = self.normalizer['action'].normalize(raw_action)

        B = raw_action.shape[0]
        T = self.horizon
        device = raw_action.device
        dtype = raw_action.dtype

        # Encode observations
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        cond = nobs_features.reshape(B, -1)

        # Split actions - extract discrete from RAW action (not normalized)
        _, action_disc_raw = self._split_actions(raw_action)  # Raw discrete class indices
        action_cont, _ = self._split_actions(naction)  # Normalized continuous params

        # Sample time uniformly
        t = torch.rand(B, device=device, dtype=dtype)
        t = t.clamp(min=1e-5, max=1.0 - 1e-5)
        t_expanded = t.view(B, 1, 1)

        # ============ Continuous: Sample noisy mean ============
        gamma = 1.0 - (self.sigma_1 ** (2.0 * t_expanded))
        var = gamma * (1.0 - gamma)
        std = (var + 1e-8).sqrt()
        mu_cont = gamma * action_cont + std * torch.randn_like(action_cont)

        # ============ Discrete: Sample theta ============
        beta = self.beta_1 * t_expanded.pow(2.0)

        theta_list = []
        disc_targets = []
        for j, (dim_idx, n_classes) in enumerate(self.discrete_configs):
            # Get discrete action as class index directly from RAW action
            # No need to unnormalize - we use the raw discrete class values
            disc_class = action_disc_raw[:, :, j].long().clamp(0, n_classes - 1)
            disc_targets.append(disc_class)
            
            # One-hot encode
            e_x = F.one_hot(disc_class, num_classes=n_classes).float()
            
            # Sample y: N(beta * (K*e_x - 1), sqrt(beta*K))
            mean = beta * (n_classes * e_x - 1)
            std_disc = (beta * n_classes + 1e-8).sqrt()
            y_samples = mean + std_disc * torch.randn_like(mean)
            
            # theta = softmax(y)
            theta = torch.softmax(y_samples, dim=-1)
            theta_list.append(theta)
        
        # ============ Build network input ============
        if len(theta_list) > 0:
            theta_concat = torch.cat(theta_list, dim=-1)  # [B, T, sum(n_classes)]
            net_input = torch.cat([mu_cont, theta_concat], dim=-1)
        else:
            net_input = mu_cont
        
        net_input_flat = net_input.reshape(B, -1)
        
        # ============ Network forward ============
        out_flat = self.unet_wrapper(net_input_flat, t, cond=cond)
        out = out_flat.reshape(B, T, -1)
        
        # ============ Compute losses ============
        # Continuous loss: weighted MSE
        # gamma shape: (B, 1, 1), action_cont shape: (B, T, D)
        x_cont_pred = out[:, :, :self.continuous_dim]
        cont_loss = (gamma * (action_cont - x_cont_pred).pow(2.0)).mean()
        
        # Discrete loss: cross-entropy for each discrete dim
        disc_loss = 0.0
        offset = self.continuous_dim
        for j, (_, n_classes) in enumerate(self.discrete_configs):
            logits = out[:, :, offset:offset + n_classes]  # [B, T, K]
            target = disc_targets[j]  # [B, T]
            
            # Reshape for cross_entropy
            logits_flat = logits.reshape(-1, n_classes)
            target_flat = target.reshape(-1)
            
            disc_loss = disc_loss + F.cross_entropy(logits_flat, target_flat)
            offset += n_classes
        
        # Combined loss (equal weighting by default)
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

