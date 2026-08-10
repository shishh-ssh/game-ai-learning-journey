"""第 9 课：线性模型、损失函数与优化器。"""

import torch


def train_linear_once(
    inputs: list[float],
    targets: list[float],
    learning_rate: float,
) -> tuple[float, float, float]:
    """从零参数开始执行一次线性模型训练。"""
    layer = torch.nn.Linear(
        in_features=1,
        out_features=1,
    )

    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.0]]))
        layer.bias.copy_(torch.tensor([0.0]))

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).reshape(-1, 1)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(layer.parameters(), lr=learning_rate)

    optimizer.zero_grad()
    predictions = layer(inputs_tensor)
    loss = loss_fn(predictions, targets_tensor)
    loss_value = loss.item()
    loss.backward()
    optimizer.step()

    bias_value = layer.bias.item()
    weight_value = layer.weight.item()
    return loss_value, weight_value, bias_value
