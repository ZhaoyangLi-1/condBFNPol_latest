"""EDM (Elucidating Diffusion Models) Policy for Image-based Robot Control.

This module implements an EDM-style diffusion policy that serves as the teacher
for Consistency Policy distillation. It uses the Karras et al. noise schedule
and Heun's 2nd-order ODE solver for high-quality action generation.

Adapted from: https://github.com/Aaditya-Prasad/consistency-policy
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from policies.base import BasePolicy
from networks.edm_scheduler import EDMScheduler, huber_loss
from networks.unet import Unet

# Try importing vision encoder from diffusion_policy
try:
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
    from diffusion_policy.common.robomimic_config_util import get_robomimic_config
    from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
    from robomimic.algo import algo_factory
    from robomimic.algo.algo import PolicyAlgo
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.models.base_nets as rmbn
    import diffusion_policy.model.vision.crop_randomizer as dmvc

    # Robustly locate CropRandomizer across robomimic versions
    try:
        from robomimic.models.obs_core import CropRandomizer as RM_CropRandomizer
    except Exception:
        try:
            from robomimic.models.base_nets import CropRandomizer as RM_CropRandomizer
        except Exception:
            RM_CropRandomizer = None

    HAS_ROBOMIMIC = True
except ImportError:
    HAS_ROBOMIMIC = False
    RM_CropRandomizer = None
    from diffusion_policy.model.common.normalizer import LinearNormalizer

logger = logging.getLogger(__name__)

__all__ = ["EDMImagePolicy"]


class EDMImagePolicy(BasePolicy):
    """EDM Diffusion Policy for image-based observations.

    This policy uses the EDM framework (Karras et al.) for diffusion-based
    action generation. It serves as the teacher model for Consistency Policy
    distillation.

    Args:
        shape_meta: Dictionary containing observation and action shapes.
        horizon: Total planning horizon (number of timesteps).
        n_action_steps: Number of action steps to predict.
        n_obs_steps: Number of observation steps to condition on.
        noise_scheduler: EDMScheduler instance (or config dict).
        crop_shape: Image crop size for augmentation.
        diffusion_step_embed_dim: Dimension of diffusion time embedding.
        down_dims: U-Net channel dimensions.
        kernel_size: Convolution kernel size.
        n_groups: GroupNorm groups.
        cond_predict_scale: Use FiLM scale+shift conditioning.
        obs_encoder_group_norm: Use GroupNorm in vision encoder.
        eval_fixed_crop: Use center crop during evaluation.
        delta: Huber loss delta parameter.
        obs_as_global_cond: Use observation as global conditioning.
        action_space: Gym action space (for clipping).
        device: Device string.
        dtype: Data type string.
        clip_actions: Whether to clip actions.
    """

    def __init__(
        self,
        shape_meta: dict,
        horizon: int = 16,
        n_action_steps: int = 8,
        n_obs_steps: int = 2,
        # Scheduler
        noise_scheduler: Optional[Union[EDMScheduler, dict]] = None,
        # Vision encoder
        crop_shape: Tuple[int, int] = (76, 76),
        obs_encoder_group_norm: bool = False,
        eval_fixed_crop: bool = False,
        obs_as_global_cond: bool = True,
        # U-Net architecture
        diffusion_step_embed_dim: int = 128,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        # Training
        delta: float = 0.0,
        # Base policy args
        action_space: Any = None,
        device: str = "cpu",
        dtype: str = "float32",
        clip_actions: bool = True,
        **kwargs,
    ):
        super().__init__(
            action_space=action_space,
            device=device,
            dtype=dtype,
            clip_actions=clip_actions,
        )

        # Parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]

        # Build observation config
        obs_config = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
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

        # Build vision encoder using robomimic
        if HAS_ROBOMIMIC:
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
                        num_groups=x.num_features // 16, num_channels=x.num_features
                    ),
                )

            if eval_fixed_crop and RM_CropRandomizer is not None:
                replace_submodules(
                    root_module=obs_encoder,
                    predicate=lambda x: isinstance(x, RM_CropRandomizer),
                    func=lambda x: dmvc.CropRandomizer(
                        input_shape=x.input_shape,
                        crop_height=x.crop_height,
                        crop_width=x.crop_width,
                        num_crops=x.num_crops,
                        pos_enc=x.pos_enc,
                    ),
                )
            elif eval_fixed_crop and RM_CropRandomizer is None:
                logger.warning(
                    "eval_fixed_crop=True but could not locate robomimic CropRandomizer; skipping."
                )

            obs_feature_dim = obs_encoder.output_shape()[0]
        else:
            # Fallback: simple MLP encoder for low-dim observations
            obs_feature_dim = sum(
                sum(s) if isinstance(s, (list, tuple)) else s
                for s in obs_key_shapes.values()
            )
            obs_encoder = nn.Identity()
            logger.warning(
                "robomimic not available, using identity obs encoder. "
                "Image observations will not work correctly."
            )

        # Create diffusion model (U-Net)
        input_dim = action_dim
        global_cond_dim = None
        if obs_as_global_cond:
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = Unet(
            input_dim=input_dim,
            cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        # Create noise scheduler
        if noise_scheduler is None:
            noise_scheduler = EDMScheduler()
        elif isinstance(noise_scheduler, (dict, DictConfig)):
            noise_scheduler = EDMScheduler(**noise_scheduler)

        # Store components
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.delta = delta

        # Mask generator (for inpainting mode, not used with global_cond)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )

        logger.info(
            f"EDMImagePolicy: diffusion_params={sum(p.numel() for p in self.model.parameters())}, "
            f"vision_params={sum(p.numel() for p in self.obs_encoder.parameters())}"
        )

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Load normalizer from fitted instance."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ==================== INFERENCE ====================

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample trajectory using EDM ODE solver.

        Args:
            condition_data: Conditioning data [B, T, D].
            condition_mask: Boolean mask for conditioning [B, T, D].
            local_cond: Local conditioning [B, T, D_local].
            global_cond: Global conditioning [B, D_global].
            generator: Random number generator.

        Returns:
            Sampled trajectory [B, T, D].
        """
        scheduler = self.noise_scheduler

        # Sample initial position
        trajectory = scheduler.sample_initial_position(condition_data, generator=generator)

        # ODE integration
        timesteps = torch.arange(0, scheduler.bins, device=condition_data.device)

        for b, next_b in zip(timesteps[:-1], timesteps[1:]):
            trajectory[condition_mask] = condition_data[condition_mask]

            t = scheduler.timesteps_to_times(b)
            next_t = scheduler.timesteps_to_times(next_b)

            denoise = lambda traj, time: self.model(
                traj, time, cond=global_cond, local_cond=local_cond
            )

            trajectory = scheduler.step(denoise, trajectory, t, next_t, clamp=True)

        # Final conditioning enforcement
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Predict action sequence from observations.

        Args:
            obs_dict: Dictionary with observation tensors.

        Returns:
            Dictionary with 'action' and 'action_pred' tensors.
        """
        assert "past_action" not in obs_dict  # Not implemented

        # Normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim

        device = self.device
        dtype = self.dtype

        # Build conditioning
        local_cond = None
        global_cond = None

        if self.obs_as_global_cond:
            # Global conditioning from observations
            this_nobs = dict_apply(
                nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)

            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # Inpainting mode
            this_nobs = dict_apply(
                nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, self.n_obs_steps, -1)

            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, : self.n_obs_steps, Da:] = nobs_features
            cond_mask[:, : self.n_obs_steps, Da:] = True

        # Run sampling
        nsample = self.conditional_sample(
            cond_data, cond_mask, local_cond=local_cond, global_cond=global_cond
        )

        # Unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # Extract action window
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {"action": action, "action_pred": action_pred}

    def forward(
        self, obs: Any, *, deterministic: bool = False, **kwargs: Any
    ) -> torch.Tensor:
        """Generate actions from observations.

        Args:
            obs: Observation tensor or dictionary.
            deterministic: Not used (EDM sampling is deterministic given same noise).

        Returns:
            Action tensor [B, n_action_steps, action_dim] or [n_action_steps, action_dim].
        """
        input_obs = obs if isinstance(obs, dict) else {"obs": obs}

        # Ensure tensor
        for key in input_obs:
            if isinstance(input_obs[key], torch.Tensor):
                input_obs[key] = input_obs[key].to(self.device, self.dtype)
            else:
                input_obs[key] = torch.as_tensor(
                    input_obs[key], device=self.device, dtype=self.dtype
                )

        result = self.predict_action(input_obs)
        action = result["action"]

        if action.shape[0] == 1:
            action = action[0]
        return action

    def reset(self) -> None:
        """Reset policy state between episodes."""
        pass

    # ==================== TRAINING ====================

    def compute_loss(self, batch: Any) -> torch.Tensor:
        """Compute EDM training loss.

        Args:
            batch: Batch dictionary with 'obs' and 'action'.

        Returns:
            Scalar loss tensor.
        """
        assert "valid_mask" not in batch

        # Normalize
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # Build conditioning
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            this_nobs = dict_apply(
                nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # Generate mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample times
        times, _ = self.noise_scheduler.sample_times(trajectory)

        # Add noise
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, times)

        # Compute loss mask
        loss_mask = ~condition_mask

        # Apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict clean trajectory
        denoise = lambda traj, t: self.model(
            traj, t, cond=global_cond, local_cond=local_cond
        )
        pred = self.noise_scheduler.calc_out(denoise, noisy_trajectory, times, clamp=False)

        # Get loss weights (karras weighting per original implementation)
        weights = self.noise_scheduler.get_weights(times, None, "karras")

        # Compute loss
        target = trajectory
        loss = huber_loss(pred, target, delta=self.delta, weights=weights)

        return loss

    def denoise_step(
        self,
        noisy_traj: torch.Tensor,
        t: torch.Tensor,
        next_t: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None,
        local_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform a single denoising step (for teacher sampling in CTM).

        Args:
            noisy_traj: Noisy trajectory [B, T, D].
            t: Current time [B] or scalar.
            next_t: Next time [B] or scalar.
            global_cond: Global conditioning.
            local_cond: Local conditioning.

        Returns:
            Denoised trajectory at next_t.
        """
        denoise = lambda traj, time: self.model(
            traj, time, cond=global_cond, local_cond=local_cond
        )
        return self.noise_scheduler.step(denoise, noisy_traj, t, next_t, clamp=False)
