"""Consistency Model 1D U-Net.

This module implements a 1D U-Net backbone for Consistency Policy training.
The key difference from standard diffusion U-Nets is the additional stop-time
embedding for CTM (Consistency Training Model) distillation.

Adapted from: https://github.com/Aaditya-Prasad/consistency-policy
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from networks.base import SinusoidalPosEmb
from networks.unet import Conv1dBlock, Downsample1d, Upsample1d, ConditionalResidualBlock1D

logger = logging.getLogger(__name__)

__all__ = ["ConsistencyUnet1D"]


class ConsistencyResidualBlock1D(nn.Module):
    """Residual block with optional dropout for consistency training.

    Same as ConditionalResidualBlock1D but with configurable dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels

        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        # Dropout for consistency training
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, C_in, T]
            cond: Conditioning [B, cond_dim]

        Returns:
            Output [B, C_out, T]
        """
        out = self.blocks[0](x)
        out = self.dropout(out)
        embed = self.cond_encoder(cond)

        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed

        out = self.blocks[1](out)
        out = self.dropout(out)
        out = out + self.residual_conv(x)
        return out


class ConsistencyUnet1D(nn.Module):
    """1D U-Net with stop-time embedding for Consistency Policy.

    This network takes both a timestep t and a stop-time s, allowing it to
    predict denoising trajectories to arbitrary stop points. The stop-time
    embedding is zero-initialized to enable warm-starting from a teacher model.

    Args:
        input_dim: Input/output dimension (action dim).
        local_cond_dim: Local conditioning dimension (per-timestep).
        global_cond_dim: Global conditioning dimension (e.g., obs features).
        diffusion_step_embed_dim: Dimension of time embeddings.
        down_dims: List of channel dimensions for down blocks.
        kernel_size: Convolution kernel size.
        n_groups: Number of groups for GroupNorm.
        cond_predict_scale: Use FiLM scale+shift (vs just shift).
        dropout_rate: Dropout probability (important for consistency training).
    """

    def __init__(
        self,
        input_dim: int,
        local_cond_dim: Optional[int] = None,
        global_cond_dim: Optional[int] = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: List[int] = [256, 512, 1024],
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim

        # Time embedding encoder
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # Stop-time embedding encoder (zero-initialized for warm-start)
        self.stop_time_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # Zero-initialize stop-time encoder for warm-starting from teacher
        with torch.no_grad():
            self.stop_time_encoder[-1].weight.zero_()
            self.stop_time_encoder[-1].bias.zero_()

        # Combined conditioning dimension: time + stop_time + global_cond
        cond_dim = dsed * 2  # For both timestep and stoptime
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Local conditioning encoder
        self.local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            self.local_cond_encoder = nn.ModuleList(
                [
                    ConsistencyResidualBlock1D(
                        local_cond_dim,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                    ConsistencyResidualBlock1D(
                        local_cond_dim,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                ]
            )

        # Mid blocks
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConsistencyResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConsistencyResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]
        )

        # Down blocks
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConsistencyResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            dropout_rate=dropout_rate,
                        ),
                        ConsistencyResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            dropout_rate=dropout_rate,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Up blocks
        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConsistencyResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            dropout_rate=dropout_rate,
                        ),
                        ConsistencyResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            dropout_rate=dropout_rate,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Final convolution
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        logger.info(
            f"Initialized ConsistencyUnet1D with {sum(p.numel() for p in self.parameters())} parameters"
        )

    def prepare_drop_generators(self) -> None:
        """Prepare dropout generators with fixed seed for reproducibility."""
        dropout_generator = torch.Generator().manual_seed(42)
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.generator = dropout_generator
                if self.dropout_rate == 0.0:
                    module.generator = None

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        stoptime: Union[torch.Tensor, float, int],
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            sample: Input [B, T, input_dim] or [B, input_dim, T]
            timestep: Diffusion timestep [B] or scalar
            stoptime: Target stop time [B] or scalar
            local_cond: Local conditioning [B, T, local_cond_dim]
            global_cond: Global conditioning [B, global_cond_dim]

        Returns:
            Output [B, T, input_dim]
        """
        # Rearrange to [B, C, T] for convolutions
        sample = einops.rearrange(sample, "b h t -> b t h")

        # Encode time and stop-time
        encoded_times = self.diffusion_step_encoder(timestep)
        encoded_stops = self.stop_time_encoder(stoptime)

        global_feature = torch.cat([encoded_times, encoded_stops], dim=-1)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        # Local conditioning
        h_local = []
        if local_cond is not None and self.local_cond_encoder is not None:
            local_cond = einops.rearrange(local_cond, "b h t -> b t h")
            resnet, resnet2 = self.local_cond_encoder
            h_local.append(resnet(local_cond, global_feature))
            h_local.append(resnet2(local_cond, global_feature))

        # U-Net forward pass
        x = sample
        h = []

        # Down
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
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
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) - 1 and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # Rearrange back to [B, T, C]
        x = einops.rearrange(x, "b t h -> b h t")
        return x
