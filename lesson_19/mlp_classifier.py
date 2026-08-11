"""第 19 课：两层 MLP 分类器。"""

import torch


class MLPClassifier(torch.nn.Module):
    """将二维或一般特征输入映射为多分类 logits。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.ReLU()
        self.output_layer = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """返回每个样本的类别 logits，形状为 [batch, num_classes]。"""
        hidden = self.hidden_layer(inputs)
        activation = self.activation(hidden)
        logits = self.output_layer(activation)
        return logits
