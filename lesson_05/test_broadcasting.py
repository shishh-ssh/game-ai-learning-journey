import numpy as np

from array_basics import apply_action_bonus


def test_applies_each_bonus_to_the_matching_action_column() -> None:
    result = apply_action_bonus(
        [
            [2.0, 4.0],
            [6.0, 8.0],
            [10.0, 12.0],
        ],
        [1.0, -2.0],
    )

    np.testing.assert_allclose(
        result,
        np.array(
            [
                [3.0, 2.0],
                [7.0, 6.0],
                [11.0, 10.0],
            ]
        ),
    )


def test_broadcasts_across_a_different_non_square_table() -> None:
    result = apply_action_bonus(
        [
            [1.0, 4.0, 7.0],
            [2.0, 5.0, 8.0],
        ],
        [0.5, -1.0, 2.0],
    )

    np.testing.assert_allclose(
        result,
        np.array(
            [
                [1.5, 3.0, 9.0],
                [2.5, 4.0, 10.0],
            ]
        ),
    )


def test_returns_float64_array_with_original_table_shape() -> None:
    result = apply_action_bonus(
        [[1, 2], [3, 4], [5, 6]],
        [10, 20],
    )

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == (3, 2)


def test_does_not_modify_inputs() -> None:
    score_table = [[1.0, 2.0], [3.0, 4.0]]
    action_bonus = [0.5, -0.5]

    apply_action_bonus(score_table, action_bonus)

    assert score_table == [[1.0, 2.0], [3.0, 4.0]]
    assert action_bonus == [0.5, -0.5]
