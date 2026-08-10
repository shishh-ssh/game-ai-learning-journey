import json

import torch

from training_experiment import run_training_experiment


def test_runs_and_saves_reproducible_experiment(tmp_path):
    model_path = str(tmp_path / "model.pt")
    record_path = str(tmp_path / "record.json")
    arguments = {
        "train_inputs": [1.0, 2.0, 3.0, 4.0],
        "train_targets": [3.0, 5.0, 7.0, 9.0],
        "validation_inputs": [5.0, 6.0],
        "validation_targets": [11.0, 13.0],
        "learning_rate": 0.01,
        "epochs": 100,
        "seed": 42,
        "model_path": model_path,
        "record_path": record_path,
    }

    first = run_training_experiment(**arguments)
    second = run_training_experiment(**arguments)

    with open(record_path, "r", encoding="utf-8") as file:
        saved_record = json.load(file)

    loaded_layer = torch.nn.Linear(1, 1)
    loaded_layer.load_state_dict(torch.load(model_path, weights_only=True))

    assert first == second == saved_record
    assert first["seed"] == 42
    assert first["epochs"] == 100
    assert first["training_loss"] < first["initial_training_loss"]
    assert first["validation_loss"] < 0.1
    assert torch.allclose(loaded_layer.weight, torch.tensor([[first["weight"]]]))
    assert torch.allclose(loaded_layer.bias, torch.tensor([first["bias"]]))
