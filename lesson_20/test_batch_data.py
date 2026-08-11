import torch

from lesson_20.batch_data import make_data_loader


def make_ten_samples() -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    labels = torch.arange(10, dtype=torch.int64)
    return inputs, labels


def test_loader_splits_ten_samples_into_four_four_two() -> None:
    inputs, labels = make_ten_samples()
    loader = make_data_loader(inputs, labels, batch_size=4)

    batches = list(loader)

    assert [batch_inputs.shape[0] for batch_inputs, _ in batches] == [4, 4, 2]
    assert [batch_labels.shape[0] for _, batch_labels in batches] == [4, 4, 2]


def test_loader_preserves_sample_order() -> None:
    inputs, labels = make_ten_samples()
    loader = make_data_loader(inputs, labels, batch_size=4)

    loaded_inputs = torch.cat([batch_inputs for batch_inputs, _ in loader])
    loaded_labels = torch.cat([batch_labels for _, batch_labels in loader])

    assert torch.equal(loaded_inputs, inputs)
    assert torch.equal(loaded_labels, labels)


def test_each_batch_preserves_feature_and_label_shapes() -> None:
    inputs, labels = make_ten_samples()
    loader = make_data_loader(inputs, labels, batch_size=4)

    for batch_inputs, batch_labels in loader:
        assert batch_inputs.ndim == 2
        assert batch_inputs.shape[1] == 2
        assert batch_labels.ndim == 1


def test_loader_preserves_dtypes_and_dataset_length() -> None:
    inputs, labels = make_ten_samples()
    loader = make_data_loader(inputs, labels, batch_size=4)
    first_inputs, first_labels = next(iter(loader))

    assert len(loader.dataset) == 10
    assert first_inputs.dtype == torch.float32
    assert first_labels.dtype == torch.int64
