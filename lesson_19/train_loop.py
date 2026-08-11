"""第 19 课 Day 5：重复执行分类训练 step。"""

import torch

from lesson_19.mlp_classifier import MLPClassifier
from lesson_19.training_step import train_one_step


def train_for_steps(
    model: MLPClassifier,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
) -> list[float]:
    """执行指定次数的训练并返回每一步更新前的 loss。"""
    losses= []
    for _ in range(steps):
        loss = train_one_step(model, optimizer, inputs, labels)
        losses.append(loss)
    return losses

