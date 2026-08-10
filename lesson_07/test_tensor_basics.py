import torch

from lesson_07.tensor_basics import state_to_tensor


def test_returns_tensor_with_batch_dimension() -> None:
    result = state_to_tensor([0.2, 0.7, 1.0, -0.5])

    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([1, 4])


def test_uses_float32_dtype() -> None:
    result = state_to_tensor([1, 2, 3])

    assert result.dtype == torch.float32


def test_preserves_values_in_order() -> None:
    result = state_to_tensor([4.0, 6.0, 2.0])

    torch.testing.assert_close(
        result,
        torch.tensor([[4.0, 6.0, 2.0]], dtype=torch.float32),
    )


def test_single_feature_state_still_has_two_dimensions() -> None:
    result = state_to_tensor([8.0])

    assert result.shape == torch.Size([1, 1])
    torch.testing.assert_close(result, torch.tensor([[8.0]], dtype=torch.float32))


def test_input_list_is_not_modified() -> None:
    state = [0.0, 1.0, 2.0]
    original = state.copy()

    state_to_tensor(state)

    assert state == original
