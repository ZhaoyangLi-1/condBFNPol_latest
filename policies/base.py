"""Base class for all robotics policies.

This module defines the abstract base class `BasePolicy`, which standardizes
the interface for environment interaction (`act`), model inference (`forward`),
training (`compute_loss`), and device/data management.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from utils.bfn_utils import str_to_torch_dtype

__all__ = ["BasePolicy"]


class BasePolicy(nn.Module, ABC):
    """Abstract base class for robotics policies.

    Provides shared functionality for:
    1. Device and dtype management.
    2. Data conversion (Numpy <-> Torch).
    3. Automatic batch dimension handling.
    4. Action clipping and normalization hooks.

    Subclasses must implement:
    - `forward(obs, ...)`: The core PyTorch inference logic.
    - `compute_loss(batch)`: The training logic (optional, raises NotImplementedError by default).
    """

    def __init__(
        self,
        action_space: Any,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        clip_actions: bool = True,
        normalizer: Optional[Any] = None,
    ):
        """Initializes the BasePolicy.

        Args:
            action_space: The Gym action space (used for clipping).
            device: Default device to place tensors on ('cpu', 'cuda').
            dtype: Default dtype for tensor creation ('float32', 'float16').
            clip_actions: Whether to clamp output actions to the action_space bounds.
            normalizer: Optional normalization module (e.g., LinearNormalizer).
        """
        super().__init__()
        self.action_space = action_space
        self._device = torch.device(device)
        self._dtype = str_to_torch_dtype(dtype)
        self.clip_actions = clip_actions
        self.normalizer = normalizer

    # --- Properties ---

    @property
    def device(self) -> torch.device:
        """Returns the current device of the policy.

        Infers device from the first parameter if available (robust to .to() calls),
        otherwise falls back to the initialization value.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Returns the current dtype of the policy."""
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return self._dtype

    # --- Public Interface ---

    def set_normalizer(self, normalizer: Any) -> None:
        """Updates the normalizer used by the policy."""
        self.normalizer = normalizer

    @abstractmethod
    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        *,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Core inference method returning a batch of actions.

        Args:
            obs: Observations with leading batch dimension [B, ...].
            deterministic: Whether to sample deterministically (policy dependent).
            **kwargs: Additional arguments (e.g., conditioning info).

        Returns:
            Action tensor of shape [B, ActionDim].
        """
        raise NotImplementedError

    def compute_loss(self, batch: Any) -> torch.Tensor:
        """Computes training loss for the policy.

        Args:
            batch: A batch of data (usually containing 'obs', 'action').

        Returns:
            Scalar loss tensor.

        Raises:
            NotImplementedError: If the policy does not support internal training logic.
        """
        raise NotImplementedError(
            f"compute_loss is not implemented for {self.__class__.__name__}."
        )

    @torch.inference_mode()
    def act(
        self,
        obs: Any,
        *,
        deterministic: bool = False,
        return_torch: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Convenience wrapper for environment interaction.

        Handles:
        1. conversion of numpy obs -> torch tensors.
        2. adding batch dimension if missing.
        3. inference via `forward`.
        4. clipping actions.
        5. conversion of torch output -> numpy action (optional).

        Args:
            obs: Observation from the environment (Numpy array, Dict, or Tensor).
            deterministic: Whether to use deterministic mode.
            return_torch: If True, returns a Tensor on device; else returns Numpy array.
            **kwargs: Passed to `forward`.

        Returns:
            Action (Numpy array or Tensor).
        """
        # 1. Convert to Tensor
        obs_t = self._to_tensor(obs)

        # 2. Auto-Batching
        # We assume environment interaction usually provides a single unbatched observation
        obs_t, batch_added = self._maybe_add_batch_dim(obs_t)

        # 3. Inference
        action = self.forward(obs_t, deterministic=deterministic, **kwargs)

        # 4. Clipping
        action = self._clip_actions(action)

        # 5. Return
        if return_torch:
            return action

        # Remove batch dim if we added it
        if batch_added:
            action = action.squeeze(0)

        return action.detach().cpu().numpy()

    # --- Internal Helpers ---

    def _to_tensor(self, data: Any) -> Any:
        """Recursively converts input data to tensors on the correct device/dtype."""
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device, dtype=self.dtype)

        if isinstance(data, Mapping):
            return {k: self._to_tensor(v) for k, v in data.items()}

        if isinstance(data, (list, tuple)):
            return type(data)(self._to_tensor(v) for v in data)

        # Fallback for numpy arrays / scalars
        return torch.as_tensor(data, device=self.device, dtype=self.dtype)

    def _maybe_add_batch_dim(self, obs: Any) -> Tuple[Any, bool]:
        """Adds a leading batch dimension if the input appears to be unbatched.

        Note: This uses a heuristic. If the input is a Tensor, we assume it is
        unbatched if it matches the observation space shape (not implemented here generic enough)
        OR we rely on the caller context (usually `act` is single-step).

        Here, we unconditionally unsqueeze dim 0 for `act` convenience.
        """
        batch_added = False

        if isinstance(obs, torch.Tensor):
            # Heuristic: We assume `act` is called with single observations.
            # For robust batch detection, one would check obs_space.shape.
            # Here we simply unsqueeze to ensure [1, ...] shape.
            obs = obs.unsqueeze(0)
            batch_added = True

        elif isinstance(obs, Mapping):
            # Handle Dict inputs (e.g. {'image': ..., 'state': ...})
            # Only unsqueeze tensors.
            new_obs = {}
            for k, v in obs.items():
                if isinstance(v, torch.Tensor):
                    new_obs[k] = v.unsqueeze(0)
                    batch_added = True  # Mark true if ANY tensor was unsqueezed
                else:
                    new_obs[k] = v
            obs = new_obs

        return obs, batch_added

    def _clip_actions(self, action: torch.Tensor) -> torch.Tensor:
        """Clips actions to the environment bounds if `clip_actions` is True."""
        if not self.clip_actions:
            return action

        # Check if action space has bounds
        if not hasattr(self.action_space, "low") or not hasattr(
            self.action_space, "high"
        ):
            return action

        # Create tensor bounds on the fly (caching could be an optimization)
        low = torch.as_tensor(
            self.action_space.low, device=action.device, dtype=action.dtype
        )
        high = torch.as_tensor(
            self.action_space.high, device=action.device, dtype=action.dtype
        )

        return torch.clamp(action, low, high)
