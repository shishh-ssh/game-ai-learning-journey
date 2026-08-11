import pytest
import torch

from lesson_19.evaluation import evaluate_classifier
from lesson_19.mlp_classifier import MLPClassifier
from lesson_19.nonlinear_data import make_xor_data
from lesson_19.train_loop import train_for_steps


def make_trained_model() -> tuple[MLPClassifier, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    model = MLPClassifier(input_dim=2, hidden_dim=8, num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs, labels = make_xor_data()
    train_for_steps(model, optimizer, inputs, labels, steps=100)
    return model, inputs, labels


def test_evaluation_reports_loss_and_accuracy_as_floats() -> None:
    model, inputs, labels = make_trained_model()

    metrics = evaluate_classifier(model, inputs, labels)

    assert metrics.keys() == {"loss", "accuracy"}
    assert type(metrics["loss"]) is float
    assert type(metrics["accuracy"]) is float
    assert metrics["loss"] < 0.5
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_evaluation_switches_mode_and_disables_grad() -> None:
    model, inputs, labels = make_trained_model()
    grad_enabled_during_forward: list[bool] = []
    hook = model.register_forward_hook(
        lambda _module, _inputs, _output: grad_enabled_during_forward.append(
            torch.is_grad_enabled()
        )
    )

    evaluate_classifier(model, inputs, labels)
    hook.remove()

    assert model.training is False
    assert grad_enabled_during_forward == [False]


def test_evaluation_does_not_change_parameters_or_gradients() -> None:
    model, inputs, labels = make_trained_model()
    original_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    original_gradients = [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in model.parameters()
    ]

    evaluate_classifier(model, inputs, labels)

    assert all(
        torch.equal(parameter, original)
        for parameter, original in zip(model.parameters(), original_parameters)
    )
    assert all(
        (gradient is None and original is None)
        or (
            gradient is not None
            and original is not None
            and torch.equal(gradient, original)
        )
        for gradient, original in zip(
            (parameter.grad for parameter in model.parameters()), original_gradients
        )
    )


def test_evaluation_does_not_modify_inputs_or_labels() -> None:
    model, inputs, labels = make_trained_model()
    original_inputs = inputs.clone()
    original_labels = labels.clone()

    evaluate_classifier(model, inputs, labels)

    assert torch.equal(inputs, original_inputs)
    assert torch.equal(labels, original_labels)
