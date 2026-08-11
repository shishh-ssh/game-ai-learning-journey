import torch

from lesson_20.noisy_xor import make_noisy_xor


def test_dataset_shapes_and_dtypes_match_requested_size() -> None:
    inputs, labels = make_noisy_xor(samples_per_corner=5, noise_std=0.2, seed=42)

    assert inputs.shape == (20, 2)
    assert labels.shape == (20,)
    assert inputs.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_zero_noise_repeats_exact_xor_centers_and_labels() -> None:
    inputs, labels = make_noisy_xor(samples_per_corner=2, noise_std=0.0, seed=42)
    expected_inputs = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    expected_labels = torch.tensor([0, 0, 1, 1, 1, 1, 0, 0], dtype=torch.int64)

    assert torch.equal(inputs, expected_inputs)
    assert torch.equal(labels, expected_labels)


def test_same_seed_reproduces_the_same_noisy_samples() -> None:
    first = make_noisy_xor(samples_per_corner=5, noise_std=0.2, seed=42)
    second = make_noisy_xor(samples_per_corner=5, noise_std=0.2, seed=42)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_different_seed_changes_noise_but_not_labels() -> None:
    first_inputs, first_labels = make_noisy_xor(5, noise_std=0.2, seed=42)
    second_inputs, second_labels = make_noisy_xor(5, noise_std=0.2, seed=7)

    assert not torch.equal(first_inputs, second_inputs)
    assert torch.equal(first_labels, second_labels)
