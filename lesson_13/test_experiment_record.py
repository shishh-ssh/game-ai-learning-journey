import json

from experiment_record import load_experiment, save_experiment


def test_saves_readable_json_and_loads_same_record(tmp_path):
    record = {
        "seed": 42,
        "learning_rate": 0.01,
        "epochs": 100,
        "final_loss": 0.0076,
        "weight": 2.07,
        "bias": 0.79,
    }
    path = str(tmp_path / "experiment.json")

    save_experiment(record, path)
    loaded = load_experiment(path)

    with open(path, "r", encoding="utf-8") as file:
        raw_json = json.load(file)

    assert loaded == record
    assert raw_json == record
