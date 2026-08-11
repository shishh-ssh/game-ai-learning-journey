"""第 20 课 Day 3：使用 DataLoader 训练一个 epoch。"""

import torch
from torch.utils.data import DataLoader

from lesson_19.mlp_classifier import MLPClassifier
from lesson_19.training_step import train_one_step


def train_one_epoch(
    model: MLPClassifier,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
) -> float:
    """遍历一个 DataLoader，并返回按样本数加权的平均 loss。"""
    total_loss = 0.0
    total_samples = 0
    for batch_inputs, batch_labels in loader:
        batch_loss = train_one_step(
            model=model,
            optimizer=optimizer,
            inputs=batch_inputs,
            labels=batch_labels,
        )
        batch_size = batch_inputs.shape[0]
        total_loss += batch_loss * batch_size
        total_samples += batch_size
    return total_loss / total_samples
