import torch

from lesson_19.classification_loss import forward_and_loss
from lesson_19.mlp_classifier import MLPClassifier
from lesson_19.nonlinear_data import make_xor_data


def make_model_and_data() -> tuple[MLPClassifier, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    model = MLPClassifier(input_dim=2, hidden_dim=8, num_classes=2)
    inputs, labels = make_xor_data()
    return model, inputs, labels


def test_returns_batch_logits_and_scalar_loss() -> None:
    model, inputs, labels = make_model_and_data()

    logits, loss = forward_and_loss(model, inputs, labels)

    assert logits.shape == (4, 2)
    assert loss.shape == ()
    assert loss.dtype == torch.float32


def test_loss_matches_cross_entropy_applied_directly_to_logits() -> None:
    model, inputs, labels = make_model_and_data()
    expected_logits = model(inputs)
    expected_loss = torch.nn.functional.cross_entropy(expected_logits, labels)

    logits, loss = forward_and_loss(model, inputs, labels)

    assert torch.equal(logits, expected_logits)
    assert torch.allclose(loss, expected_loss)


def test_loss_is_connected_to_all_model_parameters() -> None:
    model, inputs, labels = make_model_and_data()
    _, loss = forward_and_loss(model, inputs, labels)

    loss.backward()

    assert all(parameter.grad is not None for parameter in model.parameters())


def test_forward_and_loss_does_not_update_parameters() -> None:
    model, inputs, labels = make_model_and_data()
    original_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    forward_and_loss(model, inputs, labels)

    assert all(
        torch.equal(parameter, original)
        for parameter, original in zip(model.parameters(), original_parameters)
    )
