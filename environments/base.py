"""Base environment abstraction for robotics tasks.

This module provides a base class for robotics environments that is compatible
with the Gymnasium API.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Tuple, Optional

import numpy as np

try:  # Prefer gymnasium if available, else fallback to gym
    import gymnasium as gym
except ImportError:  # pragma: no cover - dependency optional
    import gym  # type: ignore


class BaseEnv(gym.Env, abc.ABC):
    """A minimal Gymnasium-compatible interface for robotics environments.

    This class provides a minimal, Gymnasium-compatible interface for robotics
    environments. It is an abstract base class that should be inherited by all
    environments.
    """

    observation_space: Any
    action_space: Any

    @abc.abstractmethod
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to its initial state.

        Args:
            seed: The seed that is used to initialize the environment's PRNG.
            options: Additional information to specify how the environment is reset.

        Returns:
            A tuple containing the initial observation and an info dict.
        """
        # Call the parent gym.Env reset to handle seeding the internal np_random
        super().reset(seed=seed, options=options)
        # We don't return anything useful here because the subclass must implement
        # the actual logic to generate the observation.
        return np.array([]), {}

    @abc.abstractmethod
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Takes a step in the environment.

        Args:
            action: The action to take.

        Returns:
            A tuple containing the next observation, the reward, a boolean
            indicating whether the episode has terminated, a boolean indicating
            whether the episode has been truncated, and an info dict.
        """
        raise NotImplementedError

    def close(self):
        """Cleans up the environment's resources."""
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
