"""第 19 课 Day 7：MLP state_dict 保存与加载。"""

import torch

from lesson_19.mlp_classifier import MLPClassifier


def save_mlp_model(model: MLPClassifier, path: str) -> None:
    """只保存 MLP 的 state_dict。"""
    torch.save(model.state_dict(), path)



def load_mlp_model(
    path: str,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
) -> MLPClassifier:
    """加载参数并返回处于评估模式的 MLP。"""
    model = MLPClassifier(input_dim, hidden_dim, num_classes)
    state_dict = torch.load(path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model
