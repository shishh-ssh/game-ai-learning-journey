import torch

from lesson_07.tensor_basics import select_greedy_actions


def test_selects_best_action_for_each_state() -> None:
    result = select_greedy_actions(
        [
            [1.0, 4.0, 3.0, 2.0],
            [8.0, 6.0, 9.0, 0.0],
            [5.0, 7.0, 2.0, 1.0],
        ]
    )

    torch.testing.assert_close(result, torch.tensor([1, 2, 1]))


def test_non_square_table_keeps_state_count() -> None:
    result = select_greedy_actions(
        [
            [3.0, 8.0],
            [9.0, 1.0],
            [4.0, 6.0],
            [7.0, 2.0],
        ]
    )

    assert result.shape == torch.Size([4])
    torch.testing.assert_close(result, torch.tensor([1, 0, 1, 0]))


def test_returns_integer_index_tensor() -> None:
    result = select_greedy_actions([[2.0, 5.0, 1.0]])

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.int64
    assert result.shape == torch.Size([1])


def test_supports_all_negative_q_values() -> None:
    result = select_greedy_actions(
        [
            [-8.0, -2.0, -5.0],
            [-1.0, -7.0, -3.0],
        ]
    )

    torch.testing.assert_close(result, torch.tensor([1, 0]))


def test_input_table_is_not_modified() -> None:
    q_values = [[2.0, 8.0], [6.0, 1.0]]
    original = [row.copy() for row in q_values]

    select_greedy_actions(q_values)

    assert q_values == original
