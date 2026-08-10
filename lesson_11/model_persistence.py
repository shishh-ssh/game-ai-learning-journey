"""线性模型参数的保存与加载。"""

import torch


def save_linear_model(layer: torch.nn.Linear, path: str) -> None:
    """将线性模型的参数保存到 path。"""
    torch.save(
        layer.state_dict(),
        path,
)


def load_linear_model(
    path: str,
    in_features: int,
    out_features: int,
) -> torch.nn.Linear:
    """从 path 加载参数并返回处于评估模式的 Linear(in_features, out_features)。"""
    layer = torch.nn.Linear(in_features, out_features)
    state_dict = torch.load(
        path,
        weights_only=True,
    )
    layer.load_state_dict(state_dict)
    layer.eval()
    return layer
