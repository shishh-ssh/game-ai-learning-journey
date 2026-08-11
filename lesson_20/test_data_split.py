import torch

from lesson_20.data_split import split_train_validation


def make_ten_samples() -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    labels = torch.arange(10, dtype=torch.int64)
    return inputs, labels


def split(seed: int = 42):
    inputs, labels = make_ten_samples()
    return split_train_validation(inputs, labels, train_fraction=0.8, seed=seed)


def test_split_uses_expected_train_and_validation_sizes() -> None:
    train_inputs, train_labels, validation_inputs, validation_labels = split()

    assert train_inputs.shape == (8, 2)
    assert train_labels.shape == (8,)
    assert validation_inputs.shape == (2, 2)
    assert validation_labels.shape == (2,)


def test_split_keeps_inputs_paired_with_labels() -> None:
    train_inputs, train_labels, validation_inputs, validation_labels = split()

    recovered_train_labels = (train_inputs[:, 0] / 2).to(torch.int64)
    recovered_validation_labels = (validation_inputs[:, 0] / 2).to(torch.int64)

    assert torch.equal(recovered_train_labels, train_labels)
    assert torch.equal(recovered_validation_labels, validation_labels)


def test_same_seed_reproduces_the_same_split() -> None:
    first = split(seed=42)
    second = split(seed=42)

    assert all(
        torch.equal(first_tensor, second_tensor)
        for first_tensor, second_tensor in zip(first, second)
    )


def test_train_and_validation_are_disjoint_and_cover_all_samples() -> None:
    _, train_labels, _, validation_labels = split()
    combined = torch.cat([train_labels, validation_labels])

    assert len(torch.unique(combined)) == 10
    assert torch.equal(combined.sort().values, torch.arange(10, dtype=torch.int64))
