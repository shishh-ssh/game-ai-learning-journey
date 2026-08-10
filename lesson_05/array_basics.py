"""第 5 课：NumPy 数组与向量化运算。"""

import numpy as np


def scale_rewards(
    rewards: list[float],
    factor: float,
) -> np.ndarray:
    """将奖励转换为一维浮点数组并逐元素缩放。"""
    scaled_rewards = np.array(rewards, dtype=np.float64) * factor
    return scaled_rewards


def select_state_scores(
    score_table: list[list[float]],
    state_index: int,
) -> np.ndarray:
    """从二维状态分数表中返回指定状态的一行动作分数。"""
    score_table_np = np.array(score_table, dtype=np.float64)
    state_scores = score_table_np[state_index]
    return state_scores


def compute_score_means(
    score_table: list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """返回每个动作的平均分和每个状态的平均分。"""
    score_table_np = np.array(score_table, dtype=np.float64)
    action_means = score_table_np.mean(axis=0)
    state_means = score_table_np.mean(axis=1)
    return action_means, state_means


def select_best_actions(
    score_table: list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """返回每个状态的最佳动作索引及其最高分。"""
    score_table_np = np.array(score_table, dtype=np.float64)
    best_actions = score_table_np.argmax(axis=1)
    best_scores = score_table_np.max(axis=1)
    return best_actions, best_scores


def apply_action_bonus(
    score_table: list[list[float]],
    action_bonus: list[float],
) -> np.ndarray:
    """使用广播为每个状态的同一动作增加相同修正值。"""
    score_table_np = np.array(score_table, dtype=np.float64)
    action_bonus_np = np.array(action_bonus, dtype=np.float64)
    return score_table_np + action_bonus_np


def build_adjusted_policy(
    score_table: list[list[float]],
    action_bonus: list[float],
    state_bonus: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回调整表、每个状态的最佳动作及其最高分。"""
    score_table_np = np.array(score_table, dtype=np.float64)
    action_bonus_np = np.array(action_bonus, dtype=np.float64)
    state_bonus_np = np.array(state_bonus, dtype=np.float64)
    state_bonus_column = state_bonus_np.reshape(len(state_bonus), 1)
    adjusted_scores = score_table_np + action_bonus_np + state_bonus_column
    best_actions = adjusted_scores.argmax(axis=1)
    best_scores = adjusted_scores.max(axis=1)
    return (adjusted_scores, best_actions, best_scores)
