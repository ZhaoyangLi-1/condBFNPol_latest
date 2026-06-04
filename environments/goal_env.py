"""Goal environment wrapper for hybrid action policy training.

Goal-v0 (gym-goal):
- 3 discrete actions: 0=KICK_TO, 1=SHOOT_LEFT, 2=SHOOT_RIGHT
- Variable continuous params: par_size=[2, 1, 1] (KICK_TO uses 2, SHOOT use 1)
- 17D state

Hybrid action format (padded to max=2):
    BFN: [k, p1, p2] (3D)
    Diffusion: [one_hot(3), p1, p2] (5D)
"""

import sys
from pathlib import Path

import gym
import numpy as np

DLPA_PATH = Path(__file__).parent.parent / "_external" / "DLPA"
sys.path.insert(0, str(DLPA_PATH))


class GoalEnv:
    """Wrapper for Goal-v0: 3 discrete + variable continuous params."""

    NUM_DISCRETE = 3
    MAX_PARAM_DIM = 2
    OBS_DIM = 17
    PAR_SIZE = [2, 1, 1]

    def __init__(self, max_steps: int = 100):
        from common.goal_domain import GoalFlattenedActionWrapper, GoalObservationWrapper
        from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper

        env = gym.make('Goal-v0')
        env = GoalObservationWrapper(env)
        env = GoalFlattenedActionWrapper(env)
        env = ScaledParameterisedActionWrapper(env)
        env = ScaledStateWrapper(env)
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
        """Take a step with hybrid action.

        Args:
            action: dict {'k': int, 'x_k': [p1, p2]}, or array [k, p1, p2] (BFN),
                    or array [one_hot(3), p1, p2] (diffusion)
        """
        if isinstance(action, dict):
            k = int(action['k'])
            params = np.array(action['x_k'], dtype=np.float32).flatten()
            if len(params) < 2:
                params = np.concatenate([params, np.zeros(2 - len(params))])
        elif len(action) == 3:
            k = int(np.clip(np.round(action[0]), 0, self.NUM_DISCRETE - 1))
            params = np.array([action[1], action[2]], dtype=np.float32)
        else:
            k = int(np.argmax(action[:self.NUM_DISCRETE]))
            params = np.array(action[self.NUM_DISCRETE:self.NUM_DISCRETE + 2], dtype=np.float32)

        params = np.clip(params, -1.0, 1.0)

        # Goal-v0 expects: (act, [params_for_act_0, params_for_act_1, params_for_act_2])
        # KICK_TO: 2D, SHOOT_LEFT: 1D, SHOOT_RIGHT: 1D
        out_params = [
            np.zeros(2, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
        ]
        if k == 0:
            out_params[0][0] = float(params[0])
            out_params[0][1] = float(params[1])
        elif k == 1:
            out_params[1][0] = float(params[0])
        else:
            out_params[2][0] = float(params[0])

        env_action = (k, out_params)

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


def test_goal():
    print("Testing Goal environment...")
    env = GoalEnv()
    obs, _ = env.reset(seed=42)
    print(f"Initial obs: {obs}")
    print(f"Obs dim: {obs.shape}")

    total_reward = 0
    for step in range(env.max_steps):
        k = np.random.randint(0, env.NUM_DISCRETE)
        params = np.random.uniform(-1, 1, size=2)
        obs, reward, terminated, truncated, _ = env.step({'k': k, 'x_k': params})
        total_reward += reward
        if terminated or truncated:
            break
    print(f"Episode finished: steps={step+1}, total_reward={total_reward:.2f}")
    env.close()
    print("Test passed!")


if __name__ == "__main__":
    test_goal()
