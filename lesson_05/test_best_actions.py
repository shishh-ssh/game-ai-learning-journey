import numpy as np

from array_basics import select_best_actions


def test_returns_best_action_indices_and_scores() -> None:
    best_actions, best_scores = select_best_actions(
        [
            [1.0, 7.0, 3.0],
            [9.0, 2.0, 5.0],
            [4.0, 6.0, 1.0],
        ]
    )

    np.testing.assert_array_equal(best_actions, np.array([1, 0, 1]))
    np.testing.assert_allclose(best_scores, np.array([7.0, 9.0, 6.0]))


def test_handles_negative_scores_and_different_shape() -> None:
    best_actions, best_scores = select_best_actions(
        [
            [-5.0, -1.0, -3.0, -4.0],
            [-2.0, -7.0, -6.0, -8.0],
        ]
    )

    np.testing.assert_array_equal(best_actions, np.array([1, 0]))
    np.testing.assert_allclose(best_scores, np.array([-1.0, -2.0]))


def test_returns_expected_dtypes_and_shapes() -> None:
    best_actions, best_scores = select_best_actions(
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
    )

    assert isinstance(best_actions, np.ndarray)
    assert isinstance(best_scores, np.ndarray)
    assert np.issubdtype(best_actions.dtype, np.integer)
    assert best_scores.dtype == np.float64
    assert best_actions.shape == (3,)
    assert best_scores.shape == (3,)


def test_does_not_modify_input_table() -> None:
    score_table = [[1.0, 3.0], [4.0, 2.0]]

    select_best_actions(score_table)

    assert score_table == [[1.0, 3.0], [4.0, 2.0]]
