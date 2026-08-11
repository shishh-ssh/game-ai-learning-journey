"""第 20 课 Day 4：可复现的训练集与验证集划分。"""

import torch


def split_train_validation(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    train_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """随机划分配对的 inputs 与 labels。"""
    sample_count = inputs.shape[0]
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        sample_count,
        generator=generator,
    )
    train_size = int(sample_count * train_fraction)
    train_indices = indices[:train_size]
    validation_indices = indices[train_size:]

    train_inputs = inputs[train_indices]
    train_labels = labels[train_indices]

    validation_inputs = inputs[validation_indices]
    validation_labels = labels[validation_indices]
    return (train_inputs, train_labels, validation_inputs, validation_labels)
