"""符合 Gymnasium API 的一维 LineWorld 环境。"""

from typing import Any

import gymnasium as gym


class LineWorldEnv(gym.Env):
    """从状态0移动到目标状态的离散环境。"""

    metadata = {"render_modes": []}

    LEFT = 0
    RIGHT = 1

    def __init__(self, goal_state: int = 4, max_steps: int | None = None) -> None:
        if goal_state < 1:
            raise ValueError("goal_state必须大于等于1")
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps必须大于等于1")

        self.goal_state = goal_state
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Discrete(goal_state + 1)
        self.action_space = gym.spaces.Discrete(2)
        self.state = 0
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        self.state = 0
        self.steps = 0
        return self.state, {"distance_to_goal": self.goal_state}

    def step(
        self,
        action: int,
    ) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError("action必须是0或1")

        if action == self.LEFT:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(self.goal_state, self.state + 1)

        self.steps += 1
        terminated = self.state == self.goal_state
        truncated = (
            self.max_steps is not None
            and self.steps >= self.max_steps
            and not terminated
        )
        reward = 10.0 if terminated else -1.0
        info = {"distance_to_goal": self.goal_state - self.state}

        return self.state, reward, terminated, truncated, info
