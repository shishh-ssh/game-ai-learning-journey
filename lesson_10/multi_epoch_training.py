"""多轮线性模型训练。"""

import torch


def train_linear_many(
    inputs: list[float],
    targets: list[float],
    validation_inputs: list[float],
    validation_targets: list[float],
    learning_rate: float,
    epochs: int,
) -> tuple[list[float], float, float, float]:
    """训练多个 epoch，并返回损失历史和最终参数。"""
    layer = torch.nn.Linear(1, 1)
    with torch.no_grad():
        layer.weight.fill_(0.0)
        layer.bias.fill_(0.0)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).reshape(-1, 1)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
    validation_inputs_tensor = torch.tensor(
        validation_inputs, dtype=torch.float32
    ).reshape(-1, 1)
    validation_targets_tensor = torch.tensor(
        validation_targets, dtype=torch.float32
    ).reshape(-1, 1)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(
        layer.parameters(),
        lr=learning_rate,
    )
    loss_history = []
    layer.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = layer(inputs_tensor)
        loss = loss_fn(predictions, targets_tensor)
        loss.backward()
        loss_history.append(loss.item())
        optimizer.step()
    layer.eval()
    with torch.no_grad():
        validation_predictions = layer(validation_inputs_tensor)
        validation_loss = loss_fn(validation_predictions, validation_targets_tensor)
    return (
        loss_history,
        validation_loss.item(),
        layer.weight.item(),
        layer.bias.item(),
)
