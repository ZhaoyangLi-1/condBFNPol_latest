"""Hard Move environment wrapper for hybrid action policy training.

Hard Move is a PAMDP benchmark where an agent must move to a target position.
The action space consists of:
- 2^n discrete actions (movement directions)
- 1 continuous parameter per action (movement distance)

This wrapper provides a consistent interface for training BFN and Diffusion policies.
"""

import numpy as np
import sys
from pathlib import Path

# Add DLPA path for multiagent environment
DLPA_PATH = Path(__file__).parent.parent / "_external" / "DLPA"
sys.path.insert(0, str(DLPA_PATH))

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios


class HardMoveEnv:
    """Wrapper for Hard Move environment with hybrid action space."""

    def __init__(self, n_actuators: int = 4, max_steps: int = 25):
        """
        Args:
            n_actuators: Number of actuators, giving 2^n discrete actions
            max_steps: Maximum steps per episode
        """
        self.n_actuators = n_actuators
        self.num_discrete = 2 ** n_actuators
        self.continuous_dim = 1  # One continuous param per action
        self.max_steps = max_steps

        # Create environment
        scenario = scenarios.load("simple_move_4_direction_v1.py").Scenario()
        world = scenario.make_world()
        self._env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation
        )

        # Compute motion directions for each discrete action
        self._compute_motions()

        self.obs_dim = 4  # velocity (2D) + relative position (2D)
        self.step_count = 0

    def _compute_motions(self):
        """Compute movement directions for each of 2^n discrete actions."""
        n_actions = self.n_actuators
        x = np.linspace(0, 2 * np.pi, self.num_discrete + 1)[:-1]
        motion_x = np.cos(x)
        motion_y = np.sin(x)
        self.movements = np.vstack((motion_x, motion_y)).T

        # Generate all 2^n direction combinations
        shape = (self.num_discrete, 2)
        self.motions = np.zeros(shape)
        for idx in range(self.num_discrete):
            # Binary encoding: each bit controls a direction
            action = self._binary_encoding(idx, n_actions)
            self.motions[idx] = np.dot(action, self.movements[:n_actions])

        # Normalize
        max_dist = np.max(np.linalg.norm(self.motions, ord=2, axis=-1))
        if max_dist > 0:
            self.motions /= max_dist

    def _binary_encoding(self, idx: int, n_bits: int) -> np.ndarray:
        """Convert index to binary array."""
        return np.array([(idx >> i) & 1 for i in range(n_bits)])

    def reset(self, seed=None):
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
        obs = self._env.reset()
        self.step_count = 0
        return np.array(obs[0], dtype=np.float32), {}

    def step(self, action):
        """
        Take a step with hybrid action.

        Args:
            action: dict with 'k' (discrete action index) and 'x_k' (continuous param)
                   or array [k, x_k] for BFN format
                   or array [one_hot_k, x_k] for diffusion format
        """
        # Parse action format
        if isinstance(action, dict):
            k = int(action['k'])
            x_k = float(action['x_k'][0]) if hasattr(action['x_k'], '__len__') else float(action['x_k'])
        elif len(action) == 2:
            # BFN format: [discrete_class, continuous_param]
            k = int(np.clip(np.round(action[0]), 0, self.num_discrete - 1))
            x_k = float(action[1])
        else:
            # Diffusion format: [one_hot, continuous_param]
            k = int(np.argmax(action[:self.num_discrete]))
            x_k = float(action[self.num_discrete])

        # Clip continuous param to [-1, 1]
        x_k = np.clip(x_k, -1.0, 1.0)

        # Match DLPA's pad_action format for simple_move_4_direction_v1:
        # action = [[8, discrete_idx, n_dim, [param_for_each_action]]]
        # Where param_for_each_action is a list with x_k at index k, zeros elsewhere
        params_per_action = [0.0] * self.num_discrete
        params_per_action[k] = float(x_k)  # signed value matters!
        action_for_env = [[8, k, self.n_actuators, params_per_action]]

        obs_n, reward_n, done_n, info = self._env.step(action_for_env)

        self.step_count += 1
        obs = np.array(obs_n[0], dtype=np.float32)
        reward = float(reward_n[0])

        # Check termination
        terminated = reward > 4  # Reached target
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render environment."""
        return self._env.render()

    def close(self):
        """Close environment."""
        pass

    @property
    def action_space_info(self):
        """Return action space information."""
        return {
            'type': 'hybrid',
            'num_discrete': self.num_discrete,
            'continuous_dim': self.continuous_dim,
            'n_actuators': self.n_actuators,
        }


def test_hard_move():
    """Test the Hard Move environment."""
    print("Testing Hard Move environment...")

    for n in [4, 6, 8]:
        print(f"\n--- n={n}, num_discrete=2^{n}={2**n} ---")
        env = HardMoveEnv(n_actuators=n)

        obs, _ = env.reset(seed=42)
        print(f"Initial obs: {obs}")
        print(f"Obs dim: {env.obs_dim}")
        print(f"Action space: {env.action_space_info}")

        # Random episode
        total_reward = 0
        for step in range(25):
            k = np.random.randint(0, env.num_discrete)
            x_k = np.random.uniform(-1, 1)
            action = {'k': k, 'x_k': [x_k]}
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        print(f"Episode finished: steps={step+1}, total_reward={total_reward:.2f}")
        env.close()

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_hard_move()
