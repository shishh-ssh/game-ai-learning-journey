"""完整、可复现的线性模型训练实验。"""

import json

import torch


def run_training_experiment(
    train_inputs: list[float],
    train_targets: list[float],
    validation_inputs: list[float],
    validation_targets: list[float],
    learning_rate: float,
    epochs: int,
    seed: int,
    model_path: str,
    record_path: str,
) -> dict:
    """运行训练和验证，保存模型与实验记录，并返回记录。"""
    torch.manual_seed(seed)
    layer = torch.nn.Linear(1, 1)
    train_inputs_tensor = torch.tensor(
        train_inputs, dtype=torch.float32
    ).reshape(-1, 1)
    train_targets_tensor = torch.tensor(
        train_targets, dtype=torch.float32
    ).reshape(-1, 1)
    validation_inputs_tensor = torch.tensor(
        validation_inputs, dtype=torch.float32
    ).reshape(-1, 1)
    validation_targets_tensor = torch.tensor(
        validation_targets, dtype=torch.float32
    ).reshape(-1, 1)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(layer.parameters(), lr=learning_rate)
    training_loss_history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = layer(train_inputs_tensor)
        loss = loss_fn(predictions, train_targets_tensor)
        loss.backward()
        training_loss_history.append(loss.item())
        optimizer.step()
    layer.eval()
    with torch.no_grad():
        validation_predictions = layer(validation_inputs_tensor)
        validation_loss = loss_fn(validation_predictions, validation_targets_tensor).item()
    torch.save(layer.state_dict(), model_path)
    record = {
        "seed": seed,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "initial_training_loss": training_loss_history[0],
        "training_loss": training_loss_history[-1],
        "validation_loss": validation_loss,
        "weight": layer.weight.item(),
        "bias": layer.bias.item(),
        "model_path": model_path,
    }
    with open(record_path, "w", encoding="utf-8") as file:
        json.dump(
            record,
            file,
            ensure_ascii=False,
            indent=2,
        )
    return record
