"""Data collection utilities for BFN Policy training.

This module handles generating expert demonstrations for environments that
provide an oracle (like PointMazeEnv). BFNs require this data for Behavior Cloning.
"""

import torch
import numpy as np
from typing import Tuple, Any


def collect_expert_dataset(
    env: Any, episodes: int = 50, max_steps: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """Collects a dataset of observations and actions using the env's expert.

    Args:
        env: The environment instance (must have get_expert_action).
        episodes: Number of episodes to collect.
        max_steps: Max steps per episode.

    Returns:
        obs_data: Array of observations [N, ObsDim]
        act_data: Array of actions [N, ActionDim]
    """
    all_obs = []
    all_acts = []

    success_count = 0

    print(f"Collecting {episodes} expert episodes...")

    for _ in range(episodes):
        obs, _ = env.reset()
        episode_obs = []
        episode_acts = []
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Check if environment has an oracle/expert
            if hasattr(env, "get_expert_action"):
                # Use the Oracle
                action = env.get_expert_action(obs)
            else:
                # Fallback (Warning: BFN won't learn anything useful from random actions)
                action = env.action_space.sample()

            episode_obs.append(obs)
            episode_acts.append(action)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            if terminated:
                success_count += 1

        # Only save data if the episode was successful or valid
        # For PointMaze, we might want to keep everything, but successful is better.
        if len(episode_obs) > 0:
            all_obs.append(np.stack(episode_obs))
            all_acts.append(np.stack(episode_acts))

    print(f"Collection complete. Success rate: {success_count}/{episodes}")

    # Flatten into a single dataset of transitions [TotalSteps, Dim]
    # This is for simple MLP training. For sequence models (U-Net), we might keep episodes.
    obs_data = np.concatenate(all_obs, axis=0).astype(np.float32)
    act_data = np.concatenate(all_acts, axis=0).astype(np.float32)

    return obs_data, act_data
