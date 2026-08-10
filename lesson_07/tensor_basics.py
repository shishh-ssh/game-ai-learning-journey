"""第 7 课：PyTorch 张量基础。"""

import torch


def state_to_tensor(state: list[float]) -> torch.Tensor:
    """将单个状态转换为带 batch 维度的 float32 张量。"""
    state_tensor = torch.tensor(state, dtype=torch.float32)
    state_tensor = state_tensor.reshape([1, len(state)])
    return state_tensor


def select_greedy_actions(q_values: list[list[float]]) -> torch.Tensor:
    """返回每个状态中 Q 值最高的动作索引。"""
    q_values_tensor = torch.tensor(q_values, dtype=torch.float32)
    best_actions = q_values_tensor.argmax(dim=1)
    return best_actions
