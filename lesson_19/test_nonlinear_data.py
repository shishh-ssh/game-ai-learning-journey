import torch

from lesson_19.nonlinear_data import make_xor_data


def test_xor_data_has_expected_shapes() -> None:
    inputs, labels = make_xor_data()

    assert inputs.shape == (4, 2)
    assert labels.shape == (4,)


def test_xor_data_uses_training_compatible_dtypes() -> None:
    inputs, labels = make_xor_data()

    assert inputs.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_xor_inputs_are_in_a_fixed_order() -> None:
    inputs, _ = make_xor_data()
    expected = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=torch.float32,
    )

    assert torch.equal(inputs, expected)


def test_xor_labels_are_one_only_when_inputs_differ() -> None:
    _, labels = make_xor_data()

    assert torch.equal(labels, torch.tensor([0, 1, 1, 0], dtype=torch.int64))
