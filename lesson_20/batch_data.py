"""第 20 课 Day 1：使用 TensorDataset 和 DataLoader 组织 batch。"""

import torch
from torch.utils.data import DataLoader


def make_data_loader(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
) -> DataLoader:
    """返回可选择打乱且保留最后小 batch 的 DataLoader。"""
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator,
    )
