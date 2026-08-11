"""第 19 课 Day 3：分类 logits 与交叉熵损失。"""

import torch

from lesson_19.mlp_classifier import MLPClassifier


def forward_and_loss(
    model: MLPClassifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """执行一次前向计算并返回 logits 与标量交叉熵损失。"""
    logits = model(inputs)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    return logits, loss
