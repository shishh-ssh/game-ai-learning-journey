"""第 19 课 Day 2：构造最小 XOR 分类数据。"""

import torch


def make_xor_data() -> tuple[torch.Tensor, torch.Tensor]:
    """返回四个 XOR 样本及其类别标签。"""
    inputs = torch.tensor(
        [[0, 0], 
         [0, 1],
         [1, 0], 
         [1, 1],],
        dtype=torch.float32,
    )
    labels = torch.tensor([
        0,
        1,
        1,
        0,
    ],
        dtype=torch.int64,
    )
    return inputs, labels
