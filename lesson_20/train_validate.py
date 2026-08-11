"""第 20 课 Day 5：训练集更新与验证集评估闭环。"""

import torch
from torch.utils.data import DataLoader

from lesson_19.evaluation import evaluate_classifier
from lesson_19.mlp_classifier import MLPClassifier
from lesson_20.train_epoch import train_one_epoch


def train_and_validate(
    model: MLPClassifier,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    validation_inputs: torch.Tensor,
    validation_labels: torch.Tensor,
    epochs: int,
) -> dict[str, list[float]]:
    """训练多个 epoch，并记录训练 loss 与验证指标。"""
    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }
    for _ in range(epochs):
        train_loss = train_one_epoch(
            model,
            optimizer,
            train_loader,
        )

        validation_metrics = evaluate_classifier(
            model,
            validation_inputs,
            validation_labels,
        )

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_metrics["loss"])
        history["validation_accuracy"].append(validation_metrics["accuracy"])

    return history
