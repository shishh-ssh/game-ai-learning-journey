from unittest.mock import patch

import pytest
import torch

from lesson_19.mlp_classifier import MLPClassifier
from lesson_20.batch_data import make_data_loader
from lesson_20.train_epoch import train_one_epoch


def make_training_objects() -> tuple[
    MLPClassifier,
    torch.optim.Optimizer,
    torch.utils.data.DataLoader,
]:
    torch.manual_seed(42)
    inputs = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    labels = torch.arange(10, dtype=torch.int64) % 2
    model = MLPClassifier(input_dim=2, hidden_dim=8, num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = make_data_loader(inputs, labels, batch_size=4, shuffle=False)
    return model, optimizer, loader


def test_epoch_returns_float_and_updates_model() -> None:
    model, optimizer, loader = make_training_objects()
    original_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    loss = train_one_epoch(model, optimizer, loader)

    assert type(loss) is float
    assert any(
        not torch.equal(parameter, original)
        for parameter, original in zip(model.parameters(), original_parameters)
    )


def test_epoch_uses_sample_weighted_mean_for_uneven_batches() -> None:
    model, optimizer, loader = make_training_objects()

    with patch(
        "lesson_20.train_epoch.train_one_step",
        side_effect=[1.0, 1.0, 3.0],
    ) as train_step:
        loss = train_one_epoch(model, optimizer, loader)

    assert loss == pytest.approx((1.0 * 4 + 1.0 * 4 + 3.0 * 2) / 10)
    assert train_step.call_count == 3


def test_multiple_epochs_reduce_loss_on_a_simple_dataset() -> None:
    model, optimizer, loader = make_training_objects()
    losses = [train_one_epoch(model, optimizer, loader) for _ in range(30)]

    assert losses[-1] < losses[0]


def test_epoch_does_not_modify_loader_dataset_tensors() -> None:
    model, optimizer, loader = make_training_objects()
    original_inputs = loader.dataset.tensors[0].clone()
    original_labels = loader.dataset.tensors[1].clone()

    train_one_epoch(model, optimizer, loader)

    assert torch.equal(loader.dataset.tensors[0], original_inputs)
    assert torch.equal(loader.dataset.tensors[1], original_labels)
