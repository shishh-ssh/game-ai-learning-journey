import torch

from lesson_19.mlp_classifier import MLPClassifier
from lesson_19.nonlinear_data import make_xor_data
from lesson_19.train_loop import train_for_steps


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


def test_zero_steps_returns_empty_history() -> None:
    model, optimizer, inputs, labels = make_training_objects()

    losses = train_for_steps(model, optimizer, inputs, labels, steps=0)

    assert losses == []


def test_history_has_one_float_per_training_step() -> None:
    model, optimizer, inputs, labels = make_training_objects()

    losses = train_for_steps(model, optimizer, inputs, labels, steps=5)

    assert len(losses) == 5
    assert all(type(loss) is float for loss in losses)


def test_loss_decreases_and_xor_predictions_become_correct() -> None:
    model, optimizer, inputs, labels = make_training_objects()

    losses = train_for_steps(model, optimizer, inputs, labels, steps=100)

    assert losses[-1] < losses[0]
    assert model(inputs).argmax(dim=1).tolist() == [0, 1, 1, 0]


def test_training_loop_does_not_modify_inputs_or_labels() -> None:
    model, optimizer, inputs, labels = make_training_objects()
    original_inputs = inputs.clone()
    original_labels = labels.clone()

    train_for_steps(model, optimizer, inputs, labels, steps=5)

    assert torch.equal(inputs, original_inputs)
    assert torch.equal(labels, original_labels)
