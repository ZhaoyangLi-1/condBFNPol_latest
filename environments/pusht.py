"""Wrapper for the PushT environment from gym_pusht.

This module provides a wrapper for the PushT environment from the gym_pusht
library, making it compatible with the BaseEnv interface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple, Optional, Union

import numpy as np
import gymnasium as gym

# Import BaseEnv (assuming it exists in your project structure)
try:
    from environments.base import BaseEnv
except ImportError:
    # Fallback for standalone testing if environments.base is missing
    class BaseEnv(gym.Env):
        pass


log = logging.getLogger(__name__)


# --- Auto-Registration Logic ---
def _ensure_registered():
    """Ensures gym_pusht is imported so the environment is registered."""
    # We only need to try importing it. The package registers itself on import.
    try:
        import gym_pusht  # noqa: F401
    except ImportError:
        # We log a warning, but we don't register a mock anymore.
        log.warning(
            "gym_pusht not installed. Environment 'gym_pusht/PushT-v0' will likely fail to load. "
            "Install via `pip install gym_pusht`."
        )


_ensure_registered()


class PushTEnv(BaseEnv):
    """A thin wrapper around the PushT-v0 environment from gym_pusht.

    Attributes:
        observation_space: The observation space of the environment.
        action_space: The action space of the environment.
    """

    def __init__(
        self, env_name: str = "gym_pusht/PushT-v0", render_mode: str | None = None
    ):
        """Initializes a new PushTEnv environment.

        Args:
            env_name: The name of the PushT environment. Defaults to 'gym_pusht/PushT-v0'.
            render_mode: 'rgb_array' or 'human'.
        """
        if "gym_pusht/" not in env_name:
            env_name = f"gym_pusht/{env_name}"

        log.info(f"Initializing PushTEnv with Gym ID: {env_name}")

        try:
            self._env = gym.make(env_name, render_mode=render_mode)
        except gym.error.NameNotFound as e:
            raise gym.error.NameNotFound(
                f"Environment {env_name} not found. Please install gym_pusht."
            ) from e

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        """Resets the environment to its initial state."""
        obs, info = self._env.reset()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Takes a step in the environment."""
        return self._env.step(action)

    def render(self) -> np.ndarray | None:
        """Renders the environment."""
        return self._env.render()

    def close(self):
        """Cleans up the environment's resources."""
        return self._env.close()
