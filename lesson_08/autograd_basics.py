"""第 8 课：PyTorch 自动求导与单步更新。"""

import torch


def square_gradient_step(
    value: float,
    learning_rate: float,
) -> tuple[float, float, float]:
    """对 loss=x**2 执行一次梯度下降并返回关键数值。"""
    x = torch.tensor(value, dtype=torch.float32, requires_grad=True)
    loss = x ** 2
    loss.backward()
    loss_value = loss.item()
    gradient_value = x.grad.item()
    with torch.no_grad():
        x -= learning_rate * x.grad
    updated_value = x.item()
    return loss_value, gradient_value, updated_value


def train_square_parameter(
    value: float,
    learning_rate: float,
    steps: int,
) -> tuple[list[float], float]:
    """重复最小化 loss=x**2，并返回每轮损失及最终参数。"""
    x = torch.tensor(value, dtype=torch.float32, requires_grad=True)
    losses = []
    for _ in range(steps):
        loss = x ** 2
        losses.append(loss.item())
        loss.backward()
        with torch.no_grad():
            x -= learning_rate * x.grad
        x.grad.zero_()
    return losses, x.item()
