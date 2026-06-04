"""Hard Goal environment wrapper for hybrid action policy training.

Hard Goal extends gym_goal's Goal-v0 (3 actions) to 11 actions by subdividing
the SHOOT actions into 5+5 sub-ranges. This makes the discrete action space larger
while maintaining a meaningful continuous parameter space.

Discrete actions (11):
    0: KICK_TO (2 continuous params: target_x, target_y)
    1-5: SHOOT_GOAL_LEFT with 5 sub-ranges (1 continuous param: angle within sub-range)
    6-10: SHOOT_GOAL_RIGHT with 5 sub-ranges (1 continuous param: angle within sub-range)

Hybrid action format (padded to max=2 continuous params):
    BFN: [k, p1, p2] (3D)
    Diffusion: [one_hot(11), p1, p2] (13D)
"""

import sys
from pathlib import Path

import gym
import numpy as np

# Add DLPA path so we can use its wrappers
DLPA_PATH = Path(__file__).parent.parent / "_external" / "DLPA"
sys.path.insert(0, str(DLPA_PATH))


# Sub-range table for SHOOT actions (matches HyAR/DLPA convention)
SHOOT_C_RATE = [[-1.0, -0.6], [-0.6, -0.2], [-0.2, 0.2], [0.2, 0.6], [0.6, 1.0]]


def hard_goal_to_env_action(k: int, params: np.ndarray):
    """Convert hybrid action (k in [0,10], padded params of size 2) to Goal-v0 format.

    Returns: (act_3, [kick_params, shoot_left_params, shoot_right_params])
    """
    k = int(k)
    out_params = [np.zeros(2, dtype=np.float32),
                  np.zeros(1, dtype=np.float32),
                  np.zeros(1, dtype=np.float32)]

    if k == 0:
        # KICK_TO: use both continuous params
        out_params[0][0] = float(params[0])
        out_params[0][1] = float(params[1])
        return (0, out_params)
    elif 1 <= k <= 5:
        sub_idx = k - 1
        lo, hi = SHOOT_C_RATE[sub_idx]
        median = (hi - lo) / 2.0
        offset = (hi + lo) / 2.0
        angle = float(params[0]) * median + offset
        out_params[1] = np.array([angle], dtype=np.float32)
        return (1, out_params)
    else:  # 6 <= k <= 10
        sub_idx = k - 6
        lo, hi = SHOOT_C_RATE[sub_idx]
        median = (hi - lo) / 2.0
        offset = (hi + lo) / 2.0
        angle = float(params[0]) * median + offset
        out_params[2] = np.array([angle], dtype=np.float32)
        return (2, out_params)


class HardGoalEnv:
    """Wrapper for Hard Goal: 11 discrete actions + variable continuous params."""

    NUM_DISCRETE = 11
    MAX_PARAM_DIM = 2  # max continuous params (KICK_TO uses 2, others use 1)
    OBS_DIM = 17

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
        # Goal-v0 reset returns (state, steps_info)
        if isinstance(ret, tuple) and len(ret) == 2 and not isinstance(ret[0], (list, tuple)):
            state = ret[0]
        else:
            state, _ = ret
        self.step_count = 0
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        """Take a step with hybrid action.

        Args:
            action: dict {'k': int, 'x_k': [p1, p2]}, or array [k, p1, p2] (BFN),
                    or array [one_hot(11), p1, p2] (diffusion)
        """
        if isinstance(action, dict):
            k = int(action['k'])
            params = np.array(action['x_k'], dtype=np.float32).flatten()
            if len(params) < 2:
                params = np.concatenate([params, np.zeros(2 - len(params))])
        elif len(action) == 3:
            # BFN format: [k, p1, p2]
            k = int(np.clip(np.round(action[0]), 0, self.NUM_DISCRETE - 1))
            params = np.array([action[1], action[2]], dtype=np.float32)
        else:
            # Diffusion format: [one_hot(11), p1, p2]
            k = int(np.argmax(action[:self.NUM_DISCRETE]))
            params = np.array(action[self.NUM_DISCRETE:self.NUM_DISCRETE + 2], dtype=np.float32)

        params = np.clip(params, -1.0, 1.0)
        env_action = hard_goal_to_env_action(k, params)

        ret = self._env.step(env_action)
        # Goal-v0 returns ((state, steps), reward, terminal, info)
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

    @property
    def action_space_info(self):
        return {
            'type': 'hybrid',
            'num_discrete': self.NUM_DISCRETE,
            'continuous_dim': self.MAX_PARAM_DIM,
        }


def test_hard_goal():
    """Test the Hard Goal environment."""
    print("Testing Hard Goal environment...")
    env = HardGoalEnv()
    obs, _ = env.reset(seed=42)
    print(f"Observation: {obs}")
    print(f"Obs dim: {obs.shape}")
    print(f"Action space: {env.action_space_info}")

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
    test_hard_goal()
