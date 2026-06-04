"""Conditional 1D U-Net for Trajectory Diffusion/BFN.

This module implements a 1D U-Net backbone that operates on temporal sequences
(Batch, Horizon, Dim). It supports FiLM-based conditioning for global 
contexts (observations, class labels, etc.).

Adapted from diffusion_policy structure to be standalone and backward compatible.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from typing import Union, List, Optional, Tuple

from networks.base import BFNetwork, SinusoidalPosEmb
from utils.bfn_utils import default

logger = logging.getLogger(__name__)

__all__ = ["Unet"]


# --- Components ---


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, in_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels

        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )

        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


# --- Main Model ---


class Unet(BFNetwork):
    """Conditional 1D U-Net compatible with BFN/Diffusion Policy."""

    def __init__(
        self,
        input_dim: Optional[int] = None,  # New name
        channels: Optional[int] = None,  # Legacy name for input_dim
        dim: Optional[int] = None,  # Legacy name for base hidden dim
        cond_dim: Optional[int] = 0,  # Changed hint to Optional
        local_cond_dim: Optional[int] = None,
        global_cond_dim: Optional[int] = None,
        # Architecture args
        diffusion_step_embed_dim: int = 256,
        down_dims: Optional[List[int]] = None,
        dim_mults: Optional[List[int]] = None,  # Legacy for calculating down_dims
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
        # Catch-all for other legacy args (e.g. flash_attn) to prevent crashes
        **kwargs,
    ):
        super().__init__(is_conditional_model=True)

        # --- Compatibility Logic ---
        # 1. Resolve Input Dimension
        self.input_dim = input_dim or channels
        if self.input_dim is None:
            raise ValueError(
                "Unet requires `input_dim` (or `channels` in legacy configs)."
            )

        # --- FIX: Alias input_dim to 'dim' and 'action_dim' for BFN validation ---
        self.dim = self.input_dim
        self.action_dim = self.input_dim

        # 2. Resolve Hidden Dimensions
        # If down_dims is not provided, try to construct it from dim + dim_mults
        if down_dims is None:
            if dim is not None:
                mults = dim_mults or [1, 2, 4]
                down_dims = [dim * m for m in mults]
            else:
                # Default fallback
                down_dims = [256, 512, 1024]

        # --- Safely handle None types from Hydra config ---
        if cond_dim is None:
            cond_dim = 0

        # 3. Resolve Conditioning Dimension
        if global_cond_dim is None and cond_dim > 0:
            global_cond_dim = cond_dim

        # Expose BFN-specific attributes
        self.cond_dim = global_cond_dim if global_cond_dim is not None else 0
        self.cond_is_discrete = False  # Explicitly mark as continuous

        # ---------------------------

        all_dims = [self.input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # Total conditioning dimension for FiLM layers
        # Time embedding + Global Conditioning
        fiLm_cond_dim = dsed
        if self.cond_dim > 0:
            fiLm_cond_dim += self.cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Local Conditioning (e.g. sequence of observations)
        self.local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            self.local_cond_encoder = nn.ModuleList(
                [
                    # down encoder
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=fiLm_cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                    # up encoder
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=fiLm_cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                ]
            )

        # Mid Blocks
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=fiLm_cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=fiLm_cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]
        )

        # Down Blocks
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=fiLm_cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=fiLm_cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Up Blocks
        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=fiLm_cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=fiLm_cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, self.input_dim, 1),
        )

        logger.info(
            f"Initialized Unet with input_dim={self.input_dim}, down_dims={down_dims}"
        )

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        local_cond: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            x: Input sample (B, T, input_dim) or (B, input_dim)
            time: Diffusion/Flow step (B,) or int
            cond: Global conditioning (B, global_cond_dim)
            local_cond: Local conditioning sequence (B, T, local_cond_dim)
        """
        # 1. Shape Normalization
        # Check if input is 2D (B, D) which implies Horizon=1
        is_flat_input = x.ndim == 2
        if is_flat_input:
            x = x.unsqueeze(1)

        # Rearrange for 1D Conv: (B, T, D) -> (B, D, T)
        x = einops.rearrange(x, "b t d -> b d t")

        # Capture original T for final resizing
        original_t = x.shape[-1]

        # 2. Time Embedding
        if not torch.is_tensor(time):
            time = torch.tensor([time], dtype=torch.long, device=x.device)
        else:
            # FIX: Ensure time is on the same device as input
            time = time.to(x.device)

        if time.ndim == 0:
            time = time.unsqueeze(0)
        if time.shape[0] != x.shape[0]:
            time = time.expand(x.shape[0])

        global_feature = self.diffusion_step_encoder(time)

        # 3. Global Conditioning (e.g. Observations)
        if cond is not None:
            # Ensure cond is on same device
            cond = cond.to(x.device)
            global_feature = torch.cat([global_feature, cond], dim=-1)

        # 4. Local Conditioning Encoder
        h_local = list()
        if local_cond is not None and self.local_cond_encoder is not None:
            # Ensure local_cond is on same device
            local_cond = local_cond.to(x.device)
            # (B, T, D) -> (B, D, T)
            local_cond = einops.rearrange(local_cond, "b t d -> b d t")
            resnet, resnet2 = self.local_cond_encoder
            lc = resnet(local_cond, global_feature)
            h_local.append(lc)
            lc = resnet2(local_cond, global_feature)
            h_local.append(lc)

        # 5. U-Net Backbone
        h = []

        # Down
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)

            # Inject local cond at start
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]

            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # Mid
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Up
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            # Shape Mismatch Handling
            skip = h.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")

            x = torch.cat((x, skip), dim=1)

            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        # Final
        x = self.final_conv(x)

        # Final Resize to Original Horizon
        if x.shape[-1] != original_t:
            x = F.interpolate(x, size=original_t, mode="nearest")

        # (B, D, T) -> (B, T, D)
        x = einops.rearrange(x, "b d t -> b t d")

        # --- FIX: Restore flat shape if input was flat ---
        if is_flat_input:
            x = x.squeeze(1)

        return x
