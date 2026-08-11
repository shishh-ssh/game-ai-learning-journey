"""第 19 课 Day 6：分类模型独立评估。"""

import torch

from lesson_19.mlp_classifier import MLPClassifier


def evaluate_classifier(
    model: MLPClassifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """只读评估模型并返回交叉熵与准确率。"""
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
    return {
        "loss": loss.item(),
        "accuracy": accuracy.item(),
    }
