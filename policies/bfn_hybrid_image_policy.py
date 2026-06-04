"""BFN Hybrid Image Policy: image observations + categorical-discrete + continuous action heads.

This is the policy for real-robot PushT with hybrid action space:
- Discrete: 8 push directions
- Continuous: push distance
- Observation: one or more RGB cameras (cam0 top, cam1 side)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.common.pytorch_util import dict_apply

from policies.base import BasePolicy
from networks.base import BFNetwork

try:
    import robomimic.models.obs_core as rmbn
    import diffusion_policy.model.vision.crop_randomizer as dmvc
    from diffusion_policy.common.pytorch_util import replace_submodules
    import robomimic.utils.obs_utils as ObsUtils
    from robomimic.config import config_factory
    from robomimic.algo import algo_factory, PolicyAlgo
    from diffusion_policy.common.robomimic_config_util import get_robomimic_config
    HAS_ROBOMIMIC = True
except ImportError:
    HAS_ROBOMIMIC = False


__all__ = ["BFNHybridImagePolicy"]


class HybridUnetWrapper(BFNetwork):
    def __init__(self, model, horizon, continuous_dim, discrete_configs, cond_dim):
        super().__init__(is_conditional_model=True)
        self.model = model
        self.horizon = horizon
        self.continuous_dim = continuous_dim
        self.discrete_configs = discrete_configs
        self.cond_dim = cond_dim
        self.cond_is_discrete = False
        total_disc = sum(n for _, n in discrete_configs)
        self.input_dim = continuous_dim + total_disc

    def forward(self, x, t, cond=None):
        B = x.shape[0]
        x = x.view(B, self.horizon, self.input_dim)
        out = self.model(x, t, global_cond=cond)
        return out.reshape(B, -1)


class BFNHybridImagePolicy(BasePolicy):
    """BFN policy with image observations + hybrid (categorical + continuous) action head."""

    def __init__(
        self,
        shape_meta: dict,
        horizon: int = 16,
        n_action_steps: int = 8,
        n_obs_steps: int = 2,
        num_discrete_actions: int = 8,
        continuous_param_dim: int = 1,
        sigma_1: float = 0.001,
        beta_1: float = 0.2,
        n_timesteps: int = 20,
        crop_shape: tuple = (216, 216),
        obs_encoder_group_norm: bool = True,
        eval_fixed_crop: bool = True,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        device: str = "cpu",
        dtype: str = "float32",
        clip_actions: bool = True,
        **kwargs,
    ):
        super().__init__(action_space=None, device=device, dtype=dtype, clip_actions=clip_actions)

        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_discrete_actions = num_discrete_actions
        self.continuous_dim = continuous_param_dim
        self.discrete_configs = [(0, num_discrete_actions)]
        self.discrete_action_indices = {0}
        self.total_action_dim = 1 + continuous_param_dim
        self.sigma_1 = sigma_1
        self.beta_1 = beta_1
        self.n_timesteps = n_timesteps

        # Parse shape_meta for image obs keys
        obs_shape_meta = shape_meta["obs"]
        obs_config = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
        obs_key_shapes = {}
        self.rgb_keys: List[str] = []
        for key, attr in obs_shape_meta.items():
            obs_key_shapes[key] = list(attr["shape"])
            t = attr.get("type", "low_dim")
            if t == "rgb":
                obs_config["rgb"].append(key)
                self.rgb_keys.append(key)
            elif t == "low_dim":
                obs_config["low_dim"].append(key)
            else:
                raise ValueError(f"Unsupported obs type: {t}")
        assert HAS_ROBOMIMIC, "robomimic required for image policy"

        self.obs_encoder = self._build_robomimic_encoder(
            obs_config, obs_key_shapes, crop_shape, obs_encoder_group_norm, eval_fixed_crop
        )
        obs_feature_dim = self.obs_encoder.output_shape()[0]
        global_cond_dim = obs_feature_dim * n_obs_steps

        # U-Net input/output dim = continuous + discrete-logits
        unet_dim = continuous_param_dim + num_discrete_actions

        self.model = ConditionalUnet1D(
            input_dim=unet_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.unet_wrapper = HybridUnetWrapper(
            model=self.model,
            horizon=horizon,
            continuous_dim=continuous_param_dim,
            discrete_configs=self.discrete_configs,
            cond_dim=global_cond_dim,
        )
        self.normalizer = LinearNormalizer()
        self.global_cond_dim = global_cond_dim

        print(f"BFN Hybrid Image Policy:")
        print(f"  cameras: {self.rgb_keys}")
        print(f"  discrete: {num_discrete_actions}, continuous: {continuous_param_dim}")
        print(f"  obs_feature_dim: {obs_feature_dim}, global_cond_dim: {global_cond_dim}")
        print(f"  U-Net params: {sum(p.numel() for p in self.model.parameters()):.2e}")
        print(f"  Vision params: {sum(p.numel() for p in self.obs_encoder.parameters()):.2e}")

    def _build_robomimic_encoder(self, obs_config, obs_key_shapes, crop_shape, group_norm, eval_fixed_crop):
        config = get_robomimic_config(algo_name="bc_rnn", hdf5_type="image", task_name="square", dataset_type="ph")
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
            algo_name=config.algo_name, config=config,
            obs_key_shapes=obs_key_shapes, ac_dim=1 + self.num_discrete_actions, device="cpu",
        )
        obs_encoder = policy.nets["policy"].nets["encoder"].nets["obs"]
        if group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape, crop_height=x.crop_height,
                    crop_width=x.crop_width, num_crops=x.num_crops, pos_enc=x.pos_enc,
                ),
            )
        return obs_encoder

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _encode_obs(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode obs dict to [B, global_cond_dim]."""
        B = nobs[self.rgb_keys[0]].shape[0]
        To = self.n_obs_steps
        # Stack across time: build dict of [B*To, C, H, W]
        flat = {}
        for k, v in nobs.items():
            v_t = v[:, :To]  # [B, To, C, H, W]
            flat[k] = v_t.reshape(B * To, *v_t.shape[2:])
        feats = self.obs_encoder(flat)  # [B*To, feat_dim]
        feats = feats.reshape(B, To, -1)
        return feats.reshape(B, -1)

    def forward(self, obs, *, deterministic: bool = False, **kwargs):
        if isinstance(obs, torch.Tensor):
            obs = {"obs": obs}
        return self.predict_action(obs)["action"]

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        cond = self._encode_obs(nobs)
        B = cond.shape[0]
        device = cond.device
        dtype = cond.dtype
        naction = self._sample_hybrid_bfn(B, self.horizon, cond, device, dtype)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = naction[:, start:end]
        action_unnorm = action.clone()
        if action.shape[-1] > 1:
            full_unnorm = self.normalizer["action"].unnormalize(action.clone())
            action_unnorm[:, :, 1:] = full_unnorm[:, :, 1:]
        return {"action": action_unnorm, "action_pred": naction}

    @torch.no_grad()
    def _sample_hybrid_bfn(self, B, T, cond, device, dtype):
        n_steps = self.n_timesteps
        cont_dim = self.continuous_dim
        disc_configs = self.discrete_configs

        mu_cont = torch.zeros(B, T, cont_dim, device=device, dtype=dtype)
        rho_cont = 1.0
        theta_list = [
            torch.full((B, T, n), 1.0 / n, device=device, dtype=dtype) for _, n in disc_configs
        ]

        for i in range(1, n_steps + 1):
            t_val = (i - 1) / n_steps
            t_batch = torch.full((B,), t_val, device=device, dtype=dtype)
            net_input = torch.cat([mu_cont, *theta_list], dim=-1) if theta_list else mu_cont
            out_flat = self.unet_wrapper(net_input.reshape(B, -1), t_batch, cond=cond)
            out = out_flat.reshape(B, T, -1)

            x_cont_pred = out[:, :, :cont_dim]
            alpha_cont = (self.sigma_1 ** (-2.0 * i / n_steps)) * (1.0 - self.sigma_1 ** (2.0 / n_steps))
            sender_std = 1.0 / (alpha_cont ** 0.5 + 1e-8)
            y_cont = x_cont_pred + sender_std * torch.randn_like(x_cont_pred)
            new_rho = rho_cont + alpha_cont
            mu_cont = (rho_cont * mu_cont + alpha_cont * y_cont) / new_rho
            rho_cont = new_rho

            alpha_disc = self.beta_1 * (2 * i - 1) / (n_steps ** 2)
            offset = cont_dim
            new_theta_list = []
            for j, (_, n_classes) in enumerate(disc_configs):
                logits = out[:, :, offset:offset + n_classes]
                probs = torch.softmax(logits, dim=-1)
                probs_flat = probs.reshape(-1, n_classes)
                k_samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1).reshape(B, T)
                e_k = F.one_hot(k_samples, num_classes=n_classes).float()
                y_mean = alpha_disc * (n_classes * e_k - 1)
                y_std = (alpha_disc * n_classes + 1e-8) ** 0.5
                y_disc = y_mean + y_std * torch.randn_like(y_mean)
                log_theta = torch.log(theta_list[j] + 1e-8)
                theta_new = torch.softmax(log_theta + y_disc, dim=-1)
                new_theta_list.append(theta_new)
                offset += n_classes
            theta_list = new_theta_list

        # Final
        t_final = torch.ones(B, device=device, dtype=dtype)
        net_input = torch.cat([mu_cont, *theta_list], dim=-1) if theta_list else mu_cont
        out_final = self.unet_wrapper(net_input.reshape(B, -1), t_final, cond=cond).reshape(B, T, -1)
        x_cont_final = out_final[:, :, :cont_dim].clamp(-1.0, 1.0)

        disc_values = []
        offset = cont_dim
        for j, (_, n_classes) in enumerate(disc_configs):
            logits = out_final[:, :, offset:offset + n_classes]
            disc_values.append(logits.argmax(dim=-1).float().unsqueeze(-1))
            offset += n_classes

        if disc_values:
            return torch.cat([torch.cat(disc_values, dim=-1), x_cont_final], dim=-1)
        return x_cont_final

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        nobs = self.normalizer.normalize(batch["obs"])
        cond = self._encode_obs(nobs)
        raw_action = batch["action"]
        discrete_k = raw_action[:, :, 0].long()
        naction = self.normalizer["action"].normalize(raw_action)
        continuous_x = naction[:, :, 1:]

        B = raw_action.shape[0]
        T = self.horizon
        device = raw_action.device
        dtype = raw_action.dtype

        t = torch.rand(B, device=device, dtype=dtype).clamp(min=1e-5, max=1.0 - 1e-5)
        t_exp = t.view(B, 1, 1)
        gamma = 1.0 - (self.sigma_1 ** (2.0 * t_exp))
        var = gamma * (1.0 - gamma)
        std = (var + 1e-8).sqrt()
        mu_cont = gamma * continuous_x + std * torch.randn_like(continuous_x)

        beta = self.beta_1 * t_exp.pow(2.0)
        theta_list = []
        disc_targets = []
        for j, (_, n) in enumerate(self.discrete_configs):
            d = discrete_k.clamp(0, n - 1)
            disc_targets.append(d)
            e_x = F.one_hot(d, num_classes=n).float()
            mean = beta * (n * e_x - 1)
            std_disc = (beta * n + 1e-8).sqrt()
            y = mean + std_disc * torch.randn_like(mean)
            theta_list.append(torch.softmax(y, dim=-1))

        net_input = torch.cat([mu_cont, *theta_list], dim=-1) if theta_list else mu_cont
        out_flat = self.unet_wrapper(net_input.reshape(B, -1), t, cond=cond)
        out = out_flat.reshape(B, T, -1)

        x_cont_pred = out[:, :, :self.continuous_dim]
        cont_loss = (gamma * (continuous_x - x_cont_pred).pow(2.0)).mean()

        disc_loss = 0.0
        offset = self.continuous_dim
        for j, (_, n) in enumerate(self.discrete_configs):
            logits = out[:, :, offset:offset + n]
            disc_loss = disc_loss + F.cross_entropy(
                logits.reshape(-1, n), disc_targets[j].reshape(-1)
            )
            offset += n

        return cont_loss + disc_loss

    def state_dict(self):
        return {
            "obs_encoder": self.obs_encoder.state_dict(),
            "model": self.model.state_dict(),
            "normalizer": self.normalizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.obs_encoder.load_state_dict(state_dict["obs_encoder"])
        self.model.load_state_dict(state_dict["model"])
        if "normalizer" in state_dict:
            self.normalizer.load_state_dict(state_dict["normalizer"])

    def set_actions(self, action: torch.Tensor):
        pass

    def reset(self):
        pass
