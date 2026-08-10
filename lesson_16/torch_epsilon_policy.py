"""基于 Torch Q 表的 epsilon-greedy 动作选择。"""

import torch


def select_action(
    q_table: torch.Tensor,
    state: int,
    epsilon: float,
    generator: torch.Generator,
) -> int:
    """以 epsilon 概率探索，否则返回最大 Q 值动作索引。"""
    roll = torch.rand(
        (),
        generator=generator,
    ).item()
    if roll < epsilon:
        action_count = q_table.shape[1]
        return torch.randint(
            low=0,
            high=action_count,
            size=(),
            generator=generator,
        ).item()
    q_values = q_table[state]
    return torch.argmax(q_values).item()
