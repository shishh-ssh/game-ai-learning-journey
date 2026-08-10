"""第 6 课：可复现的 epsilon-greedy 决策。"""

import numpy as np


def choose_epsilon_greedy(
    score_table: list[list[float]],
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """为每个状态返回一个动作索引。"""
    score_table_np = np.array(score_table, dtype=np.float64)
    explore_mask = rng.random(score_table_np.shape[0]) < epsilon
    random_actions = rng.integers(score_table_np.shape[1], size=score_table_np.shape[0])
    greedy_actions = score_table_np.argmax(axis=1)
    return np.where(
        explore_mask,
        random_actions,
        greedy_actions
    )
