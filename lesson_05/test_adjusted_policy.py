import numpy as np

from array_basics import build_adjusted_policy


def test_applies_both_bonus_directions_and_selects_actions() -> None:
    adjusted_scores, best_actions, best_scores = build_adjusted_policy(
        [
            [1.0, 4.0],
            [6.0, 2.0],
            [3.0, 5.0],
        ],
        [0.5, -1.0],
        [10.0, 20.0, 30.0],
    )

    np.testing.assert_allclose(
        adjusted_scores,
        np.array(
            [
                [11.5, 13.0],
                [26.5, 21.0],
                [33.5, 34.0],
            ]
        ),
    )
    np.testing.assert_array_equal(best_actions, np.array([1, 0, 1]))
    np.testing.assert_allclose(best_scores, np.array([13.0, 26.5, 34.0]))


def test_handles_a_different_non_square_shape() -> None:
    adjusted_scores, best_actions, best_scores = build_adjusted_policy(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        [10.0, 0.0, -10.0],
        [100.0, 200.0],
    )

    np.testing.assert_allclose(
        adjusted_scores,
        np.array(
            [
                [111.0, 102.0, 93.0],
                [214.0, 205.0, 196.0],
            ]
        ),
    )
    np.testing.assert_array_equal(best_actions, np.array([0, 0]))
    np.testing.assert_allclose(best_scores, np.array([111.0, 214.0]))


def test_returns_expected_dtypes_and_shapes() -> None:
    adjusted_scores, best_actions, best_scores = build_adjusted_policy(
        [[1, 2], [3, 4], [5, 6]],
        [0, 1],
        [10, 20, 30],
    )

    assert adjusted_scores.dtype == np.float64
    assert np.issubdtype(best_actions.dtype, np.integer)
    assert best_scores.dtype == np.float64
    assert adjusted_scores.shape == (3, 2)
    assert best_actions.shape == (3,)
    assert best_scores.shape == (3,)


def test_does_not_modify_inputs() -> None:
    score_table = [[1.0, 2.0], [3.0, 4.0]]
    action_bonus = [0.5, -0.5]
    state_bonus = [10.0, 20.0]

    build_adjusted_policy(score_table, action_bonus, state_bonus)

    assert score_table == [[1.0, 2.0], [3.0, 4.0]]
    assert action_bonus == [0.5, -0.5]
    assert state_bonus == [10.0, 20.0]
