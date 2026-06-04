"""Catch Point environment wrapper for hybrid action policy training.

Catch Point (simple_catch-v0):
- 2 discrete actions: 0=move (with angle param), 1=catch (no param)
- 1 continuous param (only meaningful for move): angle in [-1, 1] mapped to [-π, π]
- State: 4D agent obs + valid_time + timestep = 6D

Hybrid action format:
    BFN: [k, angle] (2D)
    Diffusion: [one_hot(2), angle] (3D)
"""

import math
import sys
from pathlib import Path

import numpy as np

# Add DLPA path so we can use multiagent env
DLPA_PATH = Path(__file__).parent.parent / "_external" / "DLPA"
sys.path.insert(0, str(DLPA_PATH))


class CatchPointEnv:
    """Wrapper for Catch Point: 2 discrete + 1 continuous params."""

    NUM_DISCRETE = 2
    MAX_PARAM_DIM = 1
    OBS_DIM = 6  # 4 + valid_time + timestep

    def __init__(self, max_steps: int = 50):
        from multiagent.environment import MultiAgentEnv
        import multiagent.scenarios as scenarios

        scenario = scenarios.load("simple_catch.py").Scenario()
        world = scenario.make_world()
        self._env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
        )
        self.max_steps = max_steps
        self.step_count = 0
        self.valid_time = -1.0
        self.timestep = -1.0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self._env.reset()
        self.step_count = 0
        self.valid_time = -1.0
        self.timestep = -1.0
        # State = obs + valid_time + timestep (matching DLPA)
        state = list(obs[0]) + [self.valid_time, self.timestep]
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        """Take a step.

        Args:
            action: dict {'k': int, 'x_k': [angle]}, or array [k, angle] (BFN),
                    or array [one_hot(2), angle] (diffusion)
        """
        if isinstance(action, dict):
            k = int(action['k'])
            x_k = float(np.array(action['x_k']).flatten()[0])
        elif len(action) == 2:
            # BFN format: [k, angle]
            k = int(np.clip(np.round(action[0]), 0, self.NUM_DISCRETE - 1))
            x_k = float(action[1])
        else:
            # Diffusion: [one_hot(2), angle]
            k = int(np.argmax(action[:self.NUM_DISCRETE]))
            x_k = float(action[self.NUM_DISCRETE])

        x_k = np.clip(x_k, -1.0, 1.0)

        # DLPA's action format for catch:
        # if k == 0 (move): action = [1, angle*pi, 1, 0] (move flag, angle, move=1, catch=0)
        # if k == 1 (catch): action = [1, 0, 0, 1] (move flag, no angle, move=0, catch=1)
        if k == 0:
            env_action = [np.hstack(([1], [x_k * math.pi], [1], [0]))]
        else:
            env_action = [np.hstack(([1], [0], [0], [1]))]

        ret = self._env.step(env_action)
        next_state, reward_n, done_n, _ = ret
        next_state_obs = next_state[0]
        reward = float(reward_n[0])
        terminal = bool(done_n[0]) if done_n else False

        # Track valid_time (matching DLPA logic)
        # If agent is moving toward target and catch action is taken when close
        prev_valid_time = self.valid_time
        self.timestep += 1.0 / 12.0

        # Check if catch was triggered and agent is close enough
        if k == 1:  # catch action
            # Check if close to target (use velocity magnitude as proxy)
            agent_to_target_sq = np.sum(np.square(next_state_obs[2:4]))
            if prev_valid_time <= 2.0 / 3.0 and agent_to_target_sq > 0.04:
                self.valid_time = prev_valid_time + 1.0 / 6.0

        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        # Episode termination conditions (per DLPA):
        # reward > 4 or reward == 0 or timestep >= episode_length
        if reward > 4 or reward == 0 or self.step_count >= self.max_steps:
            terminal = True

        # Build state with valid_time + timestep
        state = list(next_state_obs) + [self.valid_time, self.timestep]
        return np.array(state, dtype=np.float32), reward, terminal, truncated, {}

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    @property
    def action_space_info(self):
        return {
            'type': 'hybrid',
            'num_discrete': self.NUM_DISCRETE,
            'continuous_dim': self.MAX_PARAM_DIM,
        }


def test_catch_point():
    print("Testing Catch Point environment...")
    env = CatchPointEnv()
    obs, _ = env.reset(seed=42)
    print(f"Initial obs: {obs}")
    print(f"Obs dim: {obs.shape}")
    print(f"Action space: {env.action_space_info}")

    total_reward = 0
    for step in range(env.max_steps):
        k = np.random.randint(0, env.NUM_DISCRETE)
        angle = np.random.uniform(-1, 1)
        obs, reward, terminated, truncated, _ = env.step({'k': k, 'x_k': [angle]})
        total_reward += reward
        if terminated or truncated:
            break
    print(f"Episode finished: steps={step+1}, total_reward={total_reward:.2f}")
    env.close()
    print("Test passed!")


if __name__ == "__main__":
    test_catch_point()
