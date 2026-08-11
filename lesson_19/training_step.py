"""第 19 课 Day 4：执行一个分类训练 step。"""

import torch

from lesson_19.mlp_classifier import MLPClassifier


def train_one_step(
    model: MLPClassifier,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """执行一次完整参数更新并返回更新前的交叉熵损失。"""
    model.train()
    optimizer.zero_grad()
    logits = model(inputs)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    loss_value = loss.item()
    loss.backward()
    optimizer.step()
    return loss_value
