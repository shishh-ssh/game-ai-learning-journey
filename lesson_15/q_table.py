"""使用 Torch Q 表执行单步 Q-learning 更新。"""

import torch


def update_q_value(
    q_table: torch.Tensor,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    done: bool,
    gamma: float,
    alpha: float,
) -> float:
    """原地更新 Q(state, action)，并返回更新后的 Python 浮点数。"""
    old_q = q_table[state, action].item()
    if done:
        target = reward
    else:
        next_max_q_value = torch.max(q_table[next_state]).item()
        target = reward + gamma * next_max_q_value
    new_q = old_q + alpha * (target - old_q)
    q_table[state, action] = new_q
    return new_q
