"""
BFN U-Net Hybrid Image Policy.

This module implements a Bayesian Flow Network policy for visual robot control tasks.
It uses the same architecture as DiffusionUnetHybridImagePolicy but with BFN sampling.

The policy uses:
- A vision encoder (e.g., ResNet) to extract features from RGB images
- A conditional 1D U-Net as the backbone network
- Continuous BFN for generative modeling
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.common.robomimic_config_util import get_robomimic_config

from policies.base import BasePolicy
from networks.conditional_bfn import ContinuousBFN
from networks.base import BFNetwork

# =========================
# Robomimic optional imports (version-compatible)
# =========================
try:
    from robomimic.algo import algo_factory
    from robomimic.algo.algo import PolicyAlgo
    import robomimic.utils.obs_utils as ObsUtils

    # Keep module handles (optional, useful for debugging)
    import robomimic.models.obs_core as rm_obs_core
    import robomimic.models.base_nets as rm_base_nets

    # Robustly locate CropRandomizer across robomimic versions
    try:
        # some versions export CropRandomizer here
        from robomimic.models.obs_core import CropRandomizer as RM_CropRandomizer
    except Exception:
        try:
            # older versions may export it here
            from robomimic.models.base_nets import CropRandomizer as RM_CropRandomizer
        except Exception:
            RM_CropRandomizer = None

    import diffusion_policy.model.vision.crop_randomizer as dmvc
    HAS_ROBOMIMIC = True
except ImportError:
    HAS_ROBOMIMIC = False
    RM_CropRandomizer = None
    rm_obs_core = None
    rm_base_nets = None
    ObsUtils = None
    algo_factory = None
    PolicyAlgo = None
    dmvc = None

__all__ = ["BFNUnetHybridImagePolicy"]


class UnetBFNWrapper(BFNetwork):
    """Wrapper to make ConditionalUnet1D compatible with ContinuousBFN.

    The BFN expects a network that takes (x, t, cond) and outputs the same shape as x.
    ConditionalUnet1D expects (sample, timestep, global_cond) and outputs the same shape.
    """

    def __init__(
        self,
        model: ConditionalUnet1D,
        horizon: int,
        action_dim: int,
        cond_dim: int,
    ):
        super().__init__(is_conditional_model=True)
        self.model = model
        self.horizon = horizon
        self.action_dim = action_dim
        # Required by ContinuousBFN for conditional models
        self.cond_dim = cond_dim
        self.cond_is_discrete = False  # We use continuous conditioning

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass through the U-Net.

        Args:
            x: Noisy actions [B, horizon * action_dim]
            t: Timesteps [B] or scalar, values in [0, 1]
            cond: Global conditioning [B, cond_dim]

        Returns:
            Output [B, horizon * action_dim]
        """
        B = x.shape[0]

        # Reshape from [B, horizon * action_dim] to [B, horizon, action_dim]
        x = x.view(B, self.horizon, self.action_dim)  # [B, horizon, action_dim]

        # Convert BFN time [0, 1] to diffusion timesteps
        # BFN: t=0 is noisy, t=1 is clean
        # Diffusion: timestep=0 is clean, timestep=999 is noisy
        # So we need to reverse: timestep = (1 - t) * 999
        if t.dim() == 0:
            t = t.expand(B)

        timesteps = (1.0 - t) * 999.0

        out = self.model(
            sample=x,
            timestep=timesteps,
            global_cond=cond,
        )

        out = out.reshape(B, -1)  # [B, horizon * action_dim]
        return out


class BFNUnetHybridImagePolicy(BasePolicy):
    """BFN-based policy for image observations using U-Net backbone."""

    def __init__(
        self,
        shape_meta: dict,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        # BFN config
        sigma_1: float = 0.001,
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
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1, "Action must be 1D"
        action_dim = action_shape[0]

        super().__init__(
            action_space=None,
            device=device,
            dtype=dtype,
            clip_actions=clip_actions,
        )

        # Parse observation shape meta
        obs_shape_meta = shape_meta["obs"]
        obs_config = {
            "low_dim": [],
            "rgb": [],
            "depth": [],
            "scan": [],
        }
        obs_key_shapes = {}

        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                obs_config["rgb"].append(key)
            elif obs_type == "low_dim":
                obs_config["low_dim"].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        # Build observation encoder
        if HAS_ROBOMIMIC:
            obs_encoder = self._build_robomimic_encoder(
                obs_config=obs_config,
                obs_key_shapes=obs_key_shapes,
                action_dim=action_dim,
                crop_shape=crop_shape,
                obs_encoder_group_norm=obs_encoder_group_norm,
                eval_fixed_crop=eval_fixed_crop,
            )
        else:
            obs_encoder = self._build_simple_encoder(obs_shape_meta)

        obs_feature_dim = obs_encoder.output_shape()[0] if HAS_ROBOMIMIC else 512
        global_cond_dim = obs_feature_dim * n_obs_steps

        unet_model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        unet_wrapper = UnetBFNWrapper(
            model=unet_model,
            horizon=horizon,
            action_dim=action_dim,
            cond_dim=global_cond_dim,
        )

        bfn = ContinuousBFN(
            dim=horizon * action_dim,
            net=unet_wrapper,
            device_str=device,
            dtype_str=dtype,
        )

        self.obs_encoder = obs_encoder
        self.model = unet_model
        self.unet_wrapper = unet_wrapper
        self.bfn = bfn
        self.normalizer = LinearNormalizer()

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.sigma_1 = sigma_1
        self.n_timesteps = n_timesteps
        self.kwargs = kwargs

        print(f"BFN U-Net params: {sum(p.numel() for p in self.model.parameters()):.2e}")
        print(f"Vision params: {sum(p.numel() for p in self.obs_encoder.parameters()):.2e}")

    def _build_robomimic_encoder(
        self,
        obs_config: dict,
        obs_key_shapes: dict,
        action_dim: int,
        crop_shape: tuple,
        obs_encoder_group_norm: bool,
        eval_fixed_crop: bool,
    ) -> nn.Module:
        """Build observation encoder using robomimic via algo_factory."""
        config = get_robomimic_config(
            algo_name="bc_rnn",
            hdf5_type="image",
            task_name="square",
            dataset_type="ph",
        )

        with config.unlocked():
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)

        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device="cpu",
        )

        obs_encoder = policy.nets["policy"].nets["encoder"].nets["obs"]

        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=max(1, x.num_features // 16),
                    num_channels=x.num_features,
                ),
            )

        # ---- FIX: make eval_fixed_crop robust to robomimic version/layout ----
        if eval_fixed_crop and (RM_CropRandomizer is not None):
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, RM_CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=getattr(x, "num_crops", 1),
                    pos_enc=getattr(x, "pos_enc", False),
                ),
            )
        elif eval_fixed_crop and (RM_CropRandomizer is None):
            print(
                "[WARN] eval_fixed_crop=True but could not locate robomimic CropRandomizer; skip replacing."
            )

        return obs_encoder

    def _build_simple_encoder(self, obs_shape_meta: dict) -> nn.Module:
        """Build a simple MLP encoder as fallback."""
        total_dim = 0
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            if attr.get("type", "low_dim") == "low_dim":
                total_dim += shape[0]
            else:
                total_dim += 512

        return nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    # ==================== Normalizer Methods ====================

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set the normalizer for inputs/outputs."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ==================== Forward (required by BasePolicy) ====================

    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        *,
        deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for inference."""
        if isinstance(obs, torch.Tensor):
            obs_dict = {"obs": obs}
        else:
            obs_dict = obs

        result = self.predict_action(obs_dict)
        return result["action"]

    # ==================== Inference ====================

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict action from observation using BFN sampling."""
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim

        this_nobs = dict_apply(
            nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)

        cond = nobs_features.reshape(B, -1)
        device = cond.device
        dtype = cond.dtype

        naction = self._sample_bfn(B, T * Da, cond, device, dtype)
        naction = naction.reshape(B, T, Da)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = naction[:, start:end]

        action = self.normalizer["action"].unnormalize(action)

        return {
            "action": action,
            "action_pred": naction,
        }

    @torch.no_grad()
    def _sample_bfn(
        self,
        batch_size: int,
        dim: int,
        cond: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample using true BFN Bayesian posterior updates."""
        n_steps = self.n_timesteps
        sigma_1 = self.sigma_1

        mu = torch.zeros(batch_size, dim, device=device, dtype=dtype)
        rho = 1.0

        for i in range(1, n_steps + 1):
            t_val = (i - 1) / n_steps
            t_batch = torch.full((batch_size,), t_val, device=device, dtype=dtype)

            x_pred = self.unet_wrapper(mu, t_batch, cond=cond)

            alpha_i = (sigma_1 ** (-2.0 * i / n_steps)) * (1.0 - sigma_1 ** (2.0 / n_steps))
            sender_std = 1.0 / (alpha_i**0.5)
            y = x_pred + sender_std * torch.randn_like(x_pred)

            new_rho = rho + alpha_i
            mu = (rho * mu + alpha_i * y) / new_rho
            rho = new_rho

        t_final = torch.ones(batch_size, device=device, dtype=dtype)
        x_final = self.unet_wrapper(mu, t_final, cond=cond)

        return x_final.clamp(-1.0, 1.0)

    # ==================== Training ====================

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute BFN loss with simple MSE (stable training)."""
        nobs = self.normalizer.normalize(batch["obs"])
        naction = self.normalizer["action"].normalize(batch["action"])

        B = naction.shape[0]
        device = naction.device
        dtype = naction.dtype

        this_nobs = dict_apply(
            nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        cond = nobs_features.reshape(B, -1)

        x = naction.reshape(B, -1)

        t = torch.rand(B, device=device, dtype=dtype)
        t = t.clamp(min=1e-5, max=1.0 - 1e-5)

        sigma_1 = self.sigma_1
        gamma = 1.0 - (sigma_1 ** (2.0 * t))
        gamma_expanded = gamma.unsqueeze(-1)

        var = gamma_expanded * (1.0 - gamma_expanded)
        std = (var + 1e-8).sqrt()
        mu = gamma_expanded * x + std * torch.randn_like(x)

        x_pred = self.unet_wrapper(mu, t, cond=cond)

        loss = F.mse_loss(x_pred, x)
        return loss

    def set_actions(self, action: torch.Tensor):
        """Set actions for inpainting (not used with BFN)."""
        pass

    def reset(self):
        """Reset policy state between episodes."""
        pass

    # ==================== State Dict ====================

    def state_dict(self):
        """Get state dict including all components."""
        return {
            "obs_encoder": self.obs_encoder.state_dict(),
            "model": self.model.state_dict(),
            "normalizer": self.normalizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state dict for all components."""
        self.obs_encoder.load_state_dict(state_dict["obs_encoder"])
        self.model.load_state_dict(state_dict["model"])
        if "normalizer" in state_dict:
            self.normalizer.load_state_dict(state_dict["normalizer"])
