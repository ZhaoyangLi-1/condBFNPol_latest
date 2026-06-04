"""Consistency Policy for Fast Robot Control.

This module implements Consistency Policy, which distills a pretrained EDM
teacher into a fast 1-step or few-step action generator. Uses CTM (Consistency
Training Model) distillation with combined CTM + DSM losses.

Reference: "Consistency Policy: Accelerated Visuomotor Policies via
           Consistency Distillation" (RSS 2024)
Adapted from: https://github.com/Aaditya-Prasad/consistency-policy
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from policies.base import BasePolicy
from networks.edm_scheduler import CTMScheduler, huber_loss
from networks.consistency_unet import ConsistencyUnet1D
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

__all__ = ["ConsistencyImagePolicy"]


def state_dict_to_model(state_dict: dict, pattern: str = r"model\.") -> dict:
    """Extract model state dict from checkpoint."""
    import re

    new_state_dict = {}
    prefix = re.compile(pattern)
    prefix_len = len(pattern.replace("\\", "").replace(".", "")) + 1  # +1 for the dot

    for k, v in state_dict["state_dicts"]["model"].items():
        if re.match(prefix, k):
            new_k = k[prefix_len:]
            new_state_dict[new_k] = v

    return new_state_dict


def state_dict_to_obs_encoder(state_dict: dict) -> dict:
    """Extract obs_encoder state dict from checkpoint."""
    new_state_dict = {}
    prefix = "obs_encoder."
    prefix_len = len(prefix)

    for k, v in state_dict["state_dicts"]["model"].items():
        if k.startswith(prefix):
            new_k = k[prefix_len:]
            new_state_dict[new_k] = v

    return new_state_dict


class ConsistencyImagePolicy(BasePolicy):
    """Consistency Policy for fast image-based robot control.

    This policy is distilled from a pretrained EDM teacher using CTM
    (Consistency Training Model) distillation. It can generate actions
    in 1 step (fastest) or use multi-step chaining for better quality.

    Args:
        shape_meta: Dictionary containing observation and action shapes.
        horizon: Total planning horizon (number of timesteps).
        n_action_steps: Number of action steps to predict.
        n_obs_steps: Number of observation steps to condition on.
        noise_scheduler: CTMScheduler instance (or config dict).
        num_inference_steps: Number of inference steps (1 for fastest).
        chaining_times: Timesteps for multi-step chaining.
        teacher_path: Path to pretrained EDM teacher checkpoint.
        losses: Loss configuration dict (ctm_weight, dsm_weight).
        dsm_weights: DSM loss weighting scheme.
        dropout_rate: Dropout rate for consistency training.
        delta: Huber loss delta parameter.
        crop_shape: Image crop size for augmentation.
        diffusion_step_embed_dim: Dimension of diffusion time embedding.
        down_dims: U-Net channel dimensions.
        kernel_size: Convolution kernel size.
        n_groups: GroupNorm groups.
        cond_predict_scale: Use FiLM scale+shift conditioning.
        obs_encoder_group_norm: Use GroupNorm in vision encoder.
        eval_fixed_crop: Use center crop during evaluation.
        obs_as_global_cond: Use observation as global conditioning.
        initial_ema_decay: EMA decay rate for student model.
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
        noise_scheduler: Optional[Union[CTMScheduler, dict]] = None,
        # Inference
        num_inference_steps: int = 1,
        chaining_times: Optional[List[Union[str, int]]] = None,
        # Teacher
        teacher_path: Optional[str] = None,
        inference_mode: bool = False,
        # Loss configuration
        losses: Optional[Dict[str, float]] = None,
        dsm_weights: str = "none",
        # Training
        dropout_rate: float = 0.2,
        delta: float = 0.0,
        initial_ema_decay: float = 0.9,
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

        # Default loss configuration
        if losses is None:
            losses = {"ctm": 1.0, "dsm": 0.5}

        # Default chaining times (for 3-step inference)
        if chaining_times is None:
            chaining_times = ["D", 27, 54]

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
            obs_feature_dim = sum(
                sum(s) if isinstance(s, (list, tuple)) else s
                for s in obs_key_shapes.values()
            )
            obs_encoder = nn.Identity()
            logger.warning(
                "robomimic not available, using identity obs encoder."
            )

        # Create student model (ConsistencyUnet1D with stop-time)
        input_dim = action_dim
        global_cond_dim = None
        if obs_as_global_cond:
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConsistencyUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            dropout_rate=dropout_rate,
        )

        # Create EMA model for consistency training
        model_ema = copy.deepcopy(model)
        model_ema.requires_grad_(False)

        # Prepare dropout generators
        model.prepare_drop_generators()
        model_ema.prepare_drop_generators()

        # Create teacher model (standard Unet, loaded from checkpoint)
        teacher = Unet(
            input_dim=input_dim,
            cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        # Load teacher weights if provided
        if not inference_mode and teacher_path is not None:
            checkpoint = torch.load(teacher_path, map_location="cpu")
            # Load teacher Unet weights
            unet_state_dict = state_dict_to_model(checkpoint)
            teacher.load_state_dict(unet_state_dict)
            teacher.eval()
            teacher.requires_grad_(False)
            # Load obs_encoder weights from teacher (critical for matching embeddings)
            obs_encoder_state_dict = state_dict_to_obs_encoder(checkpoint)
            obs_encoder.load_state_dict(obs_encoder_state_dict)
            logger.info(f"Loaded teacher Unet and obs_encoder from: {teacher_path}")

        # Create noise scheduler
        if noise_scheduler is None:
            noise_scheduler = CTMScheduler(ode_steps_max=1)
        elif isinstance(noise_scheduler, (dict, DictConfig)):
            noise_scheduler = CTMScheduler(**noise_scheduler)

        # Store components
        self.obs_encoder = obs_encoder
        self.model = model
        self.model_ema = model_ema
        self.teacher = teacher
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond

        # Training parameters
        self.losses = losses
        self.dsm_weights = dsm_weights
        self.delta = delta
        self.ema_decay = initial_ema_decay

        # Inference parameters
        self.num_inference_steps = num_inference_steps
        self.chaining_times = chaining_times
        self.chain = False  # Enable via enable_chaining()

        # Mask generator
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )

        logger.info(
            f"ConsistencyImagePolicy: student_params={sum(p.numel() for p in self.model.parameters())}, "
            f"teacher_params={sum(p.numel() for p in self.teacher.parameters())}, "
            f"vision_params={sum(p.numel() for p in self.obs_encoder.parameters())}"
        )
        logger.info(f"Using losses: {self.losses}")

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Load normalizer from fitted instance."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def drop_teacher(self) -> None:
        """Remove teacher model to save memory during inference."""
        self.teacher = None

    def enable_chaining(self) -> None:
        """Enable multi-step chaining for inference."""
        if self.chaining_times is not None:
            self.chain = True
            logger.info(f"Chaining enabled with times: {self.chaining_times}")
        else:
            raise ValueError("Chaining times not set")

    def disable_chaining(self) -> None:
        """Disable multi-step chaining."""
        self.chain = False

    # ==================== FORWARD PASS ====================

    def _forward(
        self,
        model: nn.Module,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        stop_time: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        clamp: bool = False,
    ) -> torch.Tensor:
        """Forward pass through CTM model with stop-time conditioning."""
        denoise = lambda x, t, s: model(
            x, t, s, local_cond=local_cond, global_cond=global_cond
        )
        return self.noise_scheduler.ctm_calc_out(
            denoise, sample, timestep, stop_time, clamp=clamp
        )

    # ==================== INFERENCE ====================

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample trajectory using Consistency Model.

        Single-step: Directly denoise from time_max to time_min.
        Multi-step: Use chaining for better quality.
        """
        scheduler = self.noise_scheduler

        # Sample initial position (reduced variance)
        trajectory = scheduler.sample_initial_position(condition_data, generator=generator)

        t = torch.tensor([scheduler.time_max], device=condition_data.device)
        s = torch.tensor([scheduler.time_min], device=condition_data.device)

        # Apply conditioning
        trajectory[condition_mask] = condition_data[condition_mask]

        # Single-step generation: directly to time 0
        out = self._forward(
            self.model,
            trajectory,
            t,
            s,
            local_cond=local_cond,
            global_cond=global_cond,
            clamp=True,
        )

        out[condition_mask] = condition_data[condition_mask]

        if not self.chain:
            return out

        # Multi-step chaining
        for chain_t in self.chaining_times[1:]:
            t = torch.tensor([float(chain_t)], device=condition_data.device)

            if self.chaining_times[0] == "C":
                # Convert from bin index to time
                t = scheduler.timesteps_to_times(t)

            s = torch.tensor([scheduler.time_min], device=condition_data.device)

            # Re-noise to intermediate time
            trajectory = scheduler.add_noise(out, t)

            # Denoise to time 0
            out = self._forward(
                self.model,
                trajectory,
                t,
                s,
                local_cond=local_cond,
                global_cond=global_cond,
                clamp=True,
            )

        return out

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Predict action sequence from observations."""
        assert "past_action" not in obs_dict

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
            this_nobs = dict_apply(
                nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)

            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
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
        """Generate actions from observations."""
        input_obs = obs if isinstance(obs, dict) else {"obs": obs}

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

    def compute_loss(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Compute CTM + DSM training loss.

        Returns a dictionary of losses for logging.
        """
        total_loss = {}

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

        # ==================== CTM LOSS ====================
        if "ctm" in self.losses:
            # Sample t, s, u (as bin indices)
            t, s, u = self.noise_scheduler.sample_times_ctm(trajectory)

            times = self.noise_scheduler.timesteps_to_times(t)
            stops = self.noise_scheduler.timesteps_to_times(s)
            u_times = self.noise_scheduler.timesteps_to_times(u)

            # Add noise at time t
            noise_traj = self.noise_scheduler.add_noise(trajectory, times)

            # Use teacher to denoise from t to u
            denoise = lambda x, t: self.teacher(
                x, t, cond=global_cond, local_cond=local_cond
            )

            u_noise_traj = noise_traj
            distances = u - t
            max_d = torch.max(distances)

            for d in range(self.noise_scheduler.ode_steps_max):
                ct = torch.stack(
                    [
                        (t_i + d).clamp(int(t_i.item()), int(u_i.item()))
                        for t_i, u_i in zip(t, u)
                    ]
                )
                nt = torch.stack(
                    [
                        (t_i + d + 1).clamp(int(t_i.item()), int(u_i.item()))
                        for t_i, u_i in zip(t, u)
                    ]
                )

                current_times = self.noise_scheduler.timesteps_to_times(ct)
                next_times = self.noise_scheduler.timesteps_to_times(nt)

                u_noise_traj = self.noise_scheduler.step(
                    denoise, u_noise_traj, current_times, next_times, clamp=False
                )

            # Student: t -> s
            pred = self._forward(
                self.model,
                noise_traj,
                times,
                stops,
                local_cond=local_cond,
                global_cond=global_cond,
            )

            # EMA Student: u -> s
            target = self._forward(
                self.model_ema,
                u_noise_traj,
                u_times,
                stops,
                local_cond=local_cond,
                global_cond=global_cond,
            )

            # Both back to time 0
            start = torch.tensor(
                [self.noise_scheduler.time_min], device=trajectory.device
            ).expand(times.shape)

            pred = self._forward(
                self.model_ema,
                pred,
                stops,
                start,
                local_cond=local_cond,
                global_cond=global_cond,
            )

            target = self._forward(
                self.model_ema,
                target,
                stops,
                start,
                local_cond=local_cond,
                global_cond=global_cond,
            )

            loss = huber_loss(pred, target, delta=self.delta, weights=None)
            total_loss["ctm"] = loss * self.losses["ctm"]

        # ==================== DSM LOSS ====================
        if "dsm" in self.losses:
            # Sample times for DSM
            times, _ = self.noise_scheduler.sample_times(
                trajectory, time_sampler="ctm_dsm"
            )
            weights = self.noise_scheduler.get_weights(times, None, self.dsm_weights)

            # Add noise
            noisy_trajectory = self.noise_scheduler.add_noise(trajectory, times)

            # Predict clean trajectory (stop at time_min)
            stop = torch.tensor(
                [self.noise_scheduler.time_min], device=trajectory.device
            ).expand(times.shape)

            pred = self._forward(
                self.model,
                noisy_trajectory,
                times,
                stop,
                local_cond=local_cond,
                global_cond=global_cond,
                clamp=False,
            )

            target = trajectory

            loss = huber_loss(pred, target, delta=self.delta, weights=weights)
            total_loss["dsm"] = loss * self.losses["dsm"]

        return total_loss

    @torch.no_grad()
    def ema_update(self) -> None:
        """Update EMA model parameters."""
        param = [p.data for p in self.model.parameters()]
        param_ema = [p.data for p in self.model_ema.parameters()]

        torch._foreach_mul_(param_ema, self.ema_decay)
        torch._foreach_add_(param_ema, param, alpha=1 - self.ema_decay)
