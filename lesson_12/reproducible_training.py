"""使用固定 PyTorch 随机种子进行可复现训练。"""

import torch


def train_reproducible(
    inputs: list[float],
    targets: list[float],
    learning_rate: float,
    epochs: int,
    seed: int,
) -> tuple[list[float], float, float]:
    """返回训练损失历史以及最终权重和偏置。"""
    torch.manual_seed(seed)
    layer = torch.nn.Linear(1, 1)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).reshape(-1, 1)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(layer.parameters(), lr=learning_rate)
    loss_history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = layer(inputs_tensor)
        loss = loss_fn(predictions, targets_tensor)
        loss.backward()
        loss_history.append(loss.item())
        optimizer.step()
    return loss_history, layer.weight.item(), layer.bias.item()
