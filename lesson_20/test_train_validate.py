from unittest.mock import patch

import torch

from lesson_19.mlp_classifier import MLPClassifier
from lesson_20.batch_data import make_data_loader
from lesson_20.train_validate import train_and_validate


def make_objects() -> tuple[
    MLPClassifier,
    torch.optim.Optimizer,
    torch.utils.data.DataLoader,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(42)
    inputs = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    labels = torch.arange(10, dtype=torch.int64) % 2
    model = MLPClassifier(input_dim=2, hidden_dim=8, num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_loader = make_data_loader(inputs[:8], labels[:8], batch_size=4)
    return model, optimizer, train_loader, inputs[8:], labels[8:]


def test_zero_epochs_returns_three_empty_histories() -> None:
    model, optimizer, train_loader, validation_inputs, validation_labels = (
        make_objects()
    )

    history = train_and_validate(
        model,
        optimizer,
        train_loader,
        validation_inputs,
        validation_labels,
        epochs=0,
    )

    assert history == {
        "train_loss": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }


def test_each_epoch_records_train_and_validation_metrics() -> None:
    model, optimizer, train_loader, validation_inputs, validation_labels = (
        make_objects()
    )

    with (
        patch(
            "lesson_20.train_validate.train_one_epoch",
            side_effect=[1.5, 1.0],
        ) as train_epoch,
        patch(
            "lesson_20.train_validate.evaluate_classifier",
            side_effect=[
                {"loss": 1.2, "accuracy": 0.5},
                {"loss": 0.8, "accuracy": 1.0},
            ],
        ) as evaluate,
    ):
        history = train_and_validate(
            model,
            optimizer,
            train_loader,
            validation_inputs,
            validation_labels,
            epochs=2,
        )

    assert history == {
        "train_loss": [1.5, 1.0],
        "validation_loss": [1.2, 0.8],
        "validation_accuracy": [0.5, 1.0],
    }
    assert train_epoch.call_count == 2
    assert evaluate.call_count == 2


def test_real_training_updates_model_and_returns_validation_history() -> None:
    model, optimizer, train_loader, validation_inputs, validation_labels = (
        make_objects()
    )
    original_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    history = train_and_validate(
        model,
        optimizer,
        train_loader,
        validation_inputs,
        validation_labels,
        epochs=3,
    )

    assert len(history["train_loss"]) == 3
    assert len(history["validation_loss"]) == 3
    assert len(history["validation_accuracy"]) == 3
    assert any(
        not torch.equal(parameter, original)
        for parameter, original in zip(model.parameters(), original_parameters)
    )
