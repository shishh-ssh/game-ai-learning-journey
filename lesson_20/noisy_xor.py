"""第 20 课 Day 6：生成可复现的带噪声 XOR 数据。"""

import torch


def make_noisy_xor(
    samples_per_corner: int,
    noise_std: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """围绕四个 XOR 中心生成二维分类数据。"""
    centers = torch.tensor(
    [
        [-1.0, -1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 1.0],
    ],
    dtype=torch.float32,
)
    corner_labels = torch.tensor(
    [0, 1, 1, 0],
    dtype=torch.int64,
)
    inputs = centers.repeat_interleave(
    samples_per_corner,
    dim=0,
)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(
            inputs.shape,
            generator=generator,
            dtype=inputs.dtype,
    )
    inputs += noise * noise_std
    labels = corner_labels.repeat_interleave(
    samples_per_corner,
)
    return inputs, labels
