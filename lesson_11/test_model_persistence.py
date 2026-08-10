import torch

from model_persistence import load_linear_model, save_linear_model


def test_loaded_model_matches_original_predictions(tmp_path):
    original = torch.nn.Linear(1, 1)
    with torch.no_grad():
        original.weight.fill_(2.0)
        original.bias.fill_(1.0)

    model_path = str(tmp_path / "linear_model.pt")
    save_linear_model(original, model_path)
    loaded = load_linear_model(model_path, in_features=1, out_features=1)

    inputs = torch.tensor([[5.0], [6.0]], dtype=torch.float32)
    with torch.no_grad():
        original_predictions = original(inputs)
        loaded_predictions = loaded(inputs)

    assert torch.allclose(original_predictions, loaded_predictions)
    assert loaded.training is False


def test_loads_linear_layer_with_different_shape(tmp_path):
    original = torch.nn.Linear(2, 3)
    model_path = str(tmp_path / "linear_2_to_3.pt")

    save_linear_model(original, model_path)
    loaded = load_linear_model(model_path, in_features=2, out_features=3)

    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    with torch.no_grad():
        original_predictions = original(inputs)
        loaded_predictions = loaded(inputs)

    assert loaded.weight.shape == (3, 2)
    assert loaded.bias.shape == (3,)
    assert torch.allclose(original_predictions, loaded_predictions)
    assert loaded.training is False
