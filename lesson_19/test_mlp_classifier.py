import torch

from lesson_19.mlp_classifier import MLPClassifier


def test_is_module_and_registers_two_linear_layers() -> None:
    model = MLPClassifier(input_dim=2, hidden_dim=8, num_classes=2)

    assert isinstance(model, torch.nn.Module)
    assert isinstance(model.hidden_layer, torch.nn.Linear)
    assert isinstance(model.activation, torch.nn.ReLU)
    assert isinstance(model.output_layer, torch.nn.Linear)
    assert model.hidden_layer.in_features == 2
    assert model.hidden_layer.out_features == 8
    assert model.output_layer.in_features == 8
    assert model.output_layer.out_features == 2
    assert sum(parameter.numel() for parameter in model.parameters()) == 42


def test_forward_preserves_batch_dimension_and_returns_class_logits() -> None:
    model = MLPClassifier(input_dim=2, hidden_dim=8, num_classes=2)
    inputs = torch.zeros(4, 2)

    logits = model(inputs)

    assert logits.shape == (4, 2)
    assert logits.dtype == torch.float32


def test_forward_accepts_a_different_batch_size() -> None:
    model = MLPClassifier(input_dim=2, hidden_dim=8, num_classes=3)

    assert model(torch.randn(7, 2)).shape == (7, 3)


def test_output_is_connected_to_autograd() -> None:
    model = MLPClassifier(input_dim=2, hidden_dim=8, num_classes=2)
    logits = model(torch.randn(5, 2))
    loss = logits.sum()

    loss.backward()

    assert all(parameter.grad is not None for parameter in model.parameters())


def test_model_does_not_change_input_tensor() -> None:
    model = MLPClassifier(input_dim=2, hidden_dim=8, num_classes=2)
    inputs = torch.randn(3, 2)
    original = inputs.clone()

    _ = model(inputs)

    assert torch.equal(inputs, original)
