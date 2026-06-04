"""Conditional ResNet (MLP) for Vector/State-based BFN.

This network is designed for low-dimensional action spaces (e.g. dimension 2).
It uses Residual MLP blocks with time and observation conditioning.
"""

import torch
import torch.nn as nn
import logging

from networks.base import BFNetwork, SinusoidalPosEmb

log = logging.getLogger(__name__)

__all__ = ["Resnet"]


class ResidualBlock(nn.Module):
    """MLP Residual Block with Time/Cond modulation (FiLM)."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 2, dim)

    def forward(self, x, scale_shift=None):
        # Save residual
        residual = x

        h = self.norm1(x)

        # FiLM conditioning (Scale/Shift) applied after norm
        if scale_shift is not None:
            scale, shift = scale_shift
            h = h * (scale + 1) + shift

        h = self.linear1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h)

        return residual + h


class Resnet(BFNetwork):
    """Conditional ResNet (MLP) for BFN.

    Args:
        dim: Input dimension (Action dim).
        cond_dim: Conditioning dimension.
        hidden_dim: Hidden layer dimension.
        depth: Number of residual blocks.
        dropout: Dropout probability.
        time_dim: Time embedding dimension.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int = 0,
        hidden_dim: int = 256,
        depth: int = 3,
        dropout: float = 0.0,
        time_dim: int = 128,
    ):
        super().__init__(is_conditional_model=True)

        log.info(
            f"Initializing ResNet: dim={dim}, cond_dim={cond_dim}, hidden={hidden_dim}"
        )

        self.dim = dim
        self.action_dim = dim
        self.cond_dim = cond_dim
        self.cond_is_discrete = False

        # 1. Input Projection
        # We add LayerNorm here to help with unnormalized inputs
        self.input_norm = nn.LayerNorm(dim)
        self.input_proj = nn.Linear(dim, hidden_dim)

        # 2. Time Embedding
        # We assume SinusoidalPosEmb takes a tensor of shape (B, D) or (B,)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 3. Condition Embedding
        if cond_dim > 0:
            self.cond_norm = nn.LayerNorm(cond_dim)
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # 4. Residual Blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(depth)]
        )

        # FiLM generator
        # We use 2*hidden_dim input because we will CONCATENATE time and cond
        # instead of adding them. This is more robust.
        context_dim = hidden_dim
        if cond_dim > 0:
            context_dim = hidden_dim * 2

        self.film_gen = nn.Linear(context_dim, 2 * hidden_dim * depth)

        # 5. Output Projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, dim)

    def forward(
        self, x: torch.Tensor, time: torch.Tensor, cond: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy input (B, D)
            time: Time steps (B,) in [0, 1]
            cond: Conditioning (B, C)
        """
        # Flatten if necessary
        if x.ndim == 1 and self.dim == 1:
            x = x.unsqueeze(-1)
        original_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])

        B_eff = x.shape[0]
        B_orig = time.shape[0]

        # 1. Embed inputs
        x_emb = self.input_proj(self.input_norm(x))

        # 2. Embed Time
        # CRITICAL FIX: Scale time by 1000.0 to ensure Sinusoidal embeddings vary
        # significantly for t in [0, 1].
        t_scaled = time * 1000.0
        t_emb = self.time_mlp(t_scaled)  # (B, H)

        if B_eff != B_orig:
            t_emb = t_emb.repeat_interleave(B_eff // B_orig, dim=0)

        # 3. Embed Condition & Fuse
        if self.cond_dim > 0:
            if cond is None:
                cond = torch.zeros((B_orig, self.cond_dim), device=x.device)
            if cond.dtype != torch.float32:
                cond = cond.float()

            c_emb = self.cond_mlp(self.cond_norm(cond))

            if B_eff != B_orig:
                c_emb = c_emb.repeat_interleave(B_eff // B_orig, dim=0)

            # Concatenate Context (Robust Fusion)
            context = torch.cat([t_emb, c_emb], dim=-1)
        else:
            context = t_emb

        # 4. FiLM Parameters
        film_params = self.film_gen(context)
        film_params = film_params.view(B_eff, len(self.blocks), 2, -1)

        # 5. Apply Blocks
        h = x_emb
        for i, block in enumerate(self.blocks):
            scale, shift = film_params[:, i, 0], film_params[:, i, 1]
            h = block(h, scale_shift=(scale, shift))

        # 6. Output
        h = self.output_norm(h)
        out = self.output_proj(h)

        if len(original_shape) > 2:
            out = out.view(original_shape)

        return out
