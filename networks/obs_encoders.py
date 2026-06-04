"""Factory for creating observation encoders.

Supports:
1. 'identity': Flattens input (default for vector states).
2. 'mlp': Projects vector state to latent embedding.
3. 'multi_image': Uses MultiImageObsEncoder for visual tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Dict, Union, Optional

# Import MultiImageObsEncoder safely
try:
    from networks.multi_image_obs_encoder import MultiImageObsEncoder
except ImportError:
    MultiImageObsEncoder = None


class IdentityEncoder(nn.Module):
    """Pass-through encoder that simply flattens inputs."""

    def __init__(self):
        super().__init__()

    def forward(self, obs: Union[torch.Tensor, Dict[str, Any]]) -> torch.Tensor:
        if isinstance(obs, dict):
            # Deterministic concatenation by sorted keys
            # Flattens (B, ...) -> (B, -1)
            tensors = [obs[k].flatten(start_dim=1) for k in sorted(obs.keys())]
            return torch.cat(tensors, dim=1)
        return obs.flatten(start_dim=1)


class MLPEncoder(nn.Module):
    """Projects flattened observation to a latent embedding."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int = 256, depth: int = 2
    ):
        super().__init__()
        layers = []
        curr_dim = input_dim

        for _ in range(depth - 1):
            layers.extend([nn.Linear(curr_dim, hidden_dim), nn.Mish()])
            curr_dim = hidden_dim

        layers.append(nn.Linear(curr_dim, output_dim))
        # Optional: Add LayerNorm at output? BFN usually handles it in backbone.
        self.net = nn.Sequential(*layers)

    def forward(self, obs: Union[torch.Tensor, Dict[str, Any]]) -> torch.Tensor:
        if isinstance(obs, dict):
            tensors = [obs[k].flatten(start_dim=1) for k in sorted(obs.keys())]
            x = torch.cat(tensors, dim=1)
        else:
            x = obs.flatten(start_dim=1)
        return self.net(x)


def get_obs_encoder(type_str: str = "identity", **kwargs) -> nn.Module:
    """Factory method to create observation encoders."""
    if type_str == "identity":
        return IdentityEncoder()

    elif type_str == "mlp":
        return MLPEncoder(**kwargs)

    elif type_str == "multi_image":
        if MultiImageObsEncoder is None:
            raise ImportError(
                "MultiImageObsEncoder not found. Ensure networks/multi_image_obs_encoder.py exists."
            )
        return MultiImageObsEncoder(**kwargs)

    else:
        raise ValueError(
            f"Unknown obs_encoder type: '{type_str}'. Supported: identity, mlp, multi_image"
        )
