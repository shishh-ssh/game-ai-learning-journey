from pathlib import Path

import torch

from lesson_19.checkpoint import load_mlp_model, save_mlp_model
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


def save_and_load(tmp_path: Path) -> tuple[
    MLPClassifier,
    MLPClassifier,
    torch.Tensor,
    torch.Tensor,
]:
    original, inputs, labels = make_trained_model()
    path = str(tmp_path / "mlp_state_dict.pt")
    save_mlp_model(original, path)
    loaded = load_mlp_model(
        path,
        input_dim=2,
        hidden_dim=8,
        num_classes=2,
    )
    return original, loaded, inputs, labels


def test_checkpoint_file_is_created(tmp_path: Path) -> None:
    model, _, _ = make_trained_model()
    path = tmp_path / "mlp_state_dict.pt"

    save_mlp_model(model, str(path))

    assert path.is_file()


def test_loaded_model_matches_original_predictions(tmp_path: Path) -> None:
    original, loaded, inputs, _ = save_and_load(tmp_path)

    original.eval()
    with torch.no_grad():
        original_logits = original(inputs)
        loaded_logits = loaded(inputs)

    assert torch.equal(loaded_logits, original_logits)


def test_loaded_model_is_in_eval_mode_and_matches_metrics(tmp_path: Path) -> None:
    original, loaded, inputs, labels = save_and_load(tmp_path)

    original_metrics = evaluate_classifier(original, inputs, labels)
    loaded_metrics = evaluate_classifier(loaded, inputs, labels)

    assert loaded.training is False
    assert loaded_metrics == original_metrics
