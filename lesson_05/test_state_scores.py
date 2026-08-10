import numpy as np

from array_basics import select_state_scores


def test_selects_requested_state_row() -> None:
    result = select_state_scores(
        [
            [1.0, 4.0, 7.0],
            [2.0, 5.0, 8.0],
        ],
        1,
    )

    np.testing.assert_array_equal(result, np.array([2.0, 5.0, 8.0]))


def test_uses_state_index_instead_of_fixed_row() -> None:
    score_table = [
        [10.0, 11.0],
        [20.0, 21.0],
        [30.0, 31.0],
    ]

    np.testing.assert_array_equal(
        select_state_scores(score_table, 0),
        np.array([10.0, 11.0]),
    )


def test_returns_one_dimensional_float64_array() -> None:
    result = select_state_scores([[1, 2, 3], [4, 5, 6]], 0)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == (3,)


def test_does_not_modify_input_table() -> None:
    score_table = [[1.0, 2.0], [3.0, 4.0]]

    select_state_scores(score_table, 1)

    assert score_table == [[1.0, 2.0], [3.0, 4.0]]
