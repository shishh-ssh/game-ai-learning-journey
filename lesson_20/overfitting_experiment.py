"""第 20 课 Day 7：观察训练集与验证集的性能分叉。"""

import torch

from lesson_19.mlp_classifier import MLPClassifier
from lesson_20.batch_data import make_data_loader
from lesson_20.data_split import split_train_validation
from lesson_20.noisy_xor import make_noisy_xor
from lesson_20.train_validate import train_and_validate


def run_overfitting_experiment() -> dict[str, list[float]]:
    """用少量、高噪声数据训练较大的模型。"""
    torch.manual_seed(42)

    inputs, labels = make_noisy_xor(
        samples_per_corner=4,
        noise_std=0.9,
        seed=42,
    )
    train_inputs, train_labels, validation_inputs, validation_labels = (
        split_train_validation(
            inputs,
            labels,
            train_fraction=0.5,
            seed=42,
        )
    )
    train_loader = make_data_loader(
        train_inputs,
        train_labels,
        batch_size=8,
        shuffle=True,
        seed=42,
    )

    model = MLPClassifier(
        input_dim=2,
        hidden_dim=128,
        num_classes=2,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    return train_and_validate(
        model,
        optimizer,
        train_loader,
        validation_inputs,
        validation_labels,
        epochs=500,
    )


if __name__ == "__main__":
    history = run_overfitting_experiment()
    print(f"final train loss: {history['train_loss'][-1]:.4f}")
    print(f"final validation loss: {history['validation_loss'][-1]:.4f}")
    print(f"final validation accuracy: {history['validation_accuracy'][-1]:.4f}")
