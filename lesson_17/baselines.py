"""LineWorld 的规则基线与随机基线。"""

import random

from line_world_env import LineWorldEnv


class AlwaysRightPolicy:
    """无论观察是什么，始终向右移动。"""

    def select_action(self, observation: int) -> int:
        return LineWorldEnv.RIGHT


class RandomPolicy:
    """使用独立随机数生成器选择合法动作。"""

    def __init__(self, seed: int) -> None:
        self.random_generator = random.Random(seed)

    def select_action(self, observation: int) -> int:
        return self.random_generator.choice(
            [
                LineWorldEnv.LEFT,
                LineWorldEnv.RIGHT,
            ]
        )
