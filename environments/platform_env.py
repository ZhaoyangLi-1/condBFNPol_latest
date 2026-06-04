"""Platform environment wrapper for hybrid action policy training.

Platform-v0 (gym-platform):
- 3 discrete actions: 0=run, 1=hop, 2=leap
- 1 continuous param per action (force/distance)
- 9D state (player x/y/vx, platform info)

Hybrid action format:
    BFN: [k, force] (2D)
    Diffusion: [one_hot(3), force] (4D)
"""

import sys
from pathlib import Path

import gym
import numpy as np

DLPA_PATH = Path(__file__).parent.parent / "_external" / "DLPA"
sys.path.insert(0, str(DLPA_PATH))


class PlatformEnv:
    """Wrapper for Platform: 3 discrete + 1 continuous param each."""

    NUM_DISCRETE = 3
    MAX_PARAM_DIM = 1
    OBS_DIM = 9

    def __init__(self, max_steps: int = 250):
        import gym_platform
        from common.platform_domain import PlatformFlattenedActionWrapper
        from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper

        env = gym.make("Platform-v0")
        env = ScaledStateWrapper(env)
        env = PlatformFlattenedActionWrapper(env)
        env = ScaledParameterisedActionWrapper(env)
        self._env = env
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        ret = self._env.reset()
        if isinstance(ret, tuple):
            state = ret[0]
        else:
            state = ret
        self.step_count = 0
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        """Take a step.

        Args:
            action: dict {'k': int, 'x_k': [force]}, or array [k, force] (BFN),
                    or array [one_hot(3), force] (diffusion)
        """
        if isinstance(action, dict):
            k = int(action['k'])
            x_k = float(np.array(action['x_k']).flatten()[0])
        elif len(action) == 2:
            k = int(np.clip(np.round(action[0]), 0, self.NUM_DISCRETE - 1))
            x_k = float(action[1])
        else:
            k = int(np.argmax(action[:self.NUM_DISCRETE]))
            x_k = float(action[self.NUM_DISCRETE])

        x_k = np.clip(x_k, -1.0, 1.0)

        # Platform-v0 action format after flattening: (act, [params])
        params = [np.zeros(1, dtype=np.float32) for _ in range(self.NUM_DISCRETE)]
        params[k] = np.array([x_k], dtype=np.float32)
        env_action = (k, params)

        ret = self._env.step(env_action)
        if isinstance(ret[0], tuple):
            (next_state, _), reward, terminal, info = ret
        else:
            next_state, reward, terminal, info = ret

        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        return np.array(next_state, dtype=np.float32), float(reward), bool(terminal), bool(truncated), info

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


def test_platform():
    print("Testing Platform environment...")
    env = PlatformEnv()
    obs, _ = env.reset(seed=42)
    print(f"Initial obs: {obs}")
    print(f"Obs dim: {obs.shape}")

    total_reward = 0
    for step in range(env.max_steps):
        k = np.random.randint(0, env.NUM_DISCRETE)
        x_k = np.random.uniform(-1, 1)
        obs, reward, terminated, truncated, _ = env.step({'k': k, 'x_k': [x_k]})
        total_reward += reward
        if terminated or truncated:
            break
    print(f"Episode finished: steps={step+1}, total_reward={total_reward:.2f}")
    env.close()
    print("Test passed!")


if __name__ == "__main__":
    test_platform()
