import torch

from lesson_20.batch_data import make_data_loader


def make_ten_samples() -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    labels = torch.arange(10, dtype=torch.int64)
    return inputs, labels


def collect(loader) -> tuple[torch.Tensor, torch.Tensor]:
    batches = list(loader)
    inputs = torch.cat([batch_inputs for batch_inputs, _ in batches])
    labels = torch.cat([batch_labels for _, batch_labels in batches])
    return inputs, labels


def test_shuffle_false_keeps_original_order() -> None:
    inputs, labels = make_ten_samples()
    loader = make_data_loader(
        inputs,
        labels,
        batch_size=4,
        shuffle=False,
        seed=42,
    )

    loaded_inputs, loaded_labels = collect(loader)

    assert torch.equal(loaded_inputs, inputs)
    assert torch.equal(loaded_labels, labels)


def test_same_seed_reproduces_first_epoch_order() -> None:
    inputs, labels = make_ten_samples()
    first_loader = make_data_loader(inputs, labels, 4, shuffle=True, seed=42)
    second_loader = make_data_loader(inputs, labels, 4, shuffle=True, seed=42)

    _, first_labels = collect(first_loader)
    _, second_labels = collect(second_loader)

    assert torch.equal(first_labels, second_labels)
    assert not torch.equal(first_labels, labels)


def test_different_seeds_can_change_shuffle_order() -> None:
    inputs, labels = make_ten_samples()
    first_loader = make_data_loader(inputs, labels, 4, shuffle=True, seed=42)
    second_loader = make_data_loader(inputs, labels, 4, shuffle=True, seed=7)

    _, first_labels = collect(first_loader)
    _, second_labels = collect(second_loader)

    assert not torch.equal(first_labels, second_labels)


def test_shuffling_keeps_each_input_paired_with_its_label() -> None:
    inputs, labels = make_ten_samples()
    loader = make_data_loader(inputs, labels, 4, shuffle=True, seed=42)

    loaded_inputs, loaded_labels = collect(loader)
    labels_recovered_from_inputs = (loaded_inputs[:, 0] / 2).to(torch.int64)

    assert torch.equal(labels_recovered_from_inputs, loaded_labels)
