from unittest.mock import patch

import pytest
import torch

from lesson_19.mlp_classifier import MLPClassifier
from lesson_19.nonlinear_data import make_xor_data
from lesson_19.training_step import train_one_step


def make_training_objects() -> tuple[
    MLPClassifier,
    torch.optim.Optimizer,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(42)
    model = MLPClassifier(input_dim=2, hidden_dim=8, num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs, labels = make_xor_data()
    return model, optimizer, inputs, labels


def test_returns_the_loss_before_parameter_update() -> None:
    model, optimizer, inputs, labels = make_training_objects()
    expected_loss = torch.nn.functional.cross_entropy(model(inputs), labels).item()

    loss = train_one_step(model, optimizer, inputs, labels)

    assert type(loss) is float
    assert loss == pytest.approx(expected_loss)


def test_one_step_updates_at_least_one_model_parameter() -> None:
    model, optimizer, inputs, labels = make_training_objects()
    original_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    train_one_step(model, optimizer, inputs, labels)

    assert any(
        not torch.equal(parameter, original)
        for parameter, original in zip(model.parameters(), original_parameters)
    )


def test_training_step_calls_zero_grad_and_step_once() -> None:
    model, optimizer, inputs, labels = make_training_objects()

    with (
        patch.object(optimizer, "zero_grad", wraps=optimizer.zero_grad) as zero_grad,
        patch.object(optimizer, "step", wraps=optimizer.step) as step,
    ):
        train_one_step(model, optimizer, inputs, labels)

    zero_grad.assert_called_once_with()
    step.assert_called_once_with()


def test_training_step_switches_model_back_to_train_mode() -> None:
    model, optimizer, inputs, labels = make_training_objects()
    model.eval()
    assert model.training is False

    train_one_step(model, optimizer, inputs, labels)

    assert model.training is True


def test_training_step_does_not_modify_inputs_or_labels() -> None:
    model, optimizer, inputs, labels = make_training_objects()
    original_inputs = inputs.clone()
    original_labels = labels.clone()

    train_one_step(model, optimizer, inputs, labels)

    assert torch.equal(inputs, original_inputs)
    assert torch.equal(labels, original_labels)
